import os
import json
from pathlib import Path
from typing import Optional
import torch
import torchio as tio
from torchio import Subject
import hydra
from omegaconf import DictConfig
import logging
import pandas as pd

from vascular_superenhancement.training.model_factory import build_generator
from vascular_superenhancement.training.datasets import make_subject
from vascular_superenhancement.training.transforms import build_transforms
from vascular_superenhancement.utils.path_config import load_path_config
from vascular_superenhancement.data_management.patients import Patient

logger = logging.getLogger(__name__)

class VascularSuperenhancer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # get the checkpoint path
        self.checkpoint_path = Path(self.cfg.inference.checkpoint_path)
        self.inference_name = self.cfg.inference.inference_name
        self.output_dir = Path(self.cfg.inference.output_dir) 
    
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file {self.checkpoint_path} not found")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Inference will be saved in {self.output_dir}")
        
        # load model
        self.generator = build_generator(cfg).to(self.device)
        self._load_checkpoint()
        self.generator.eval()
        
        self.transforms = build_transforms(cfg, train=False)
          
    def _load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
        
    def predict_single(self, patient_id: str, time_point: int = 3) -> Path:
        # load the patient
        path_config = load_path_config(self.cfg.path_config.path_config_name)
        patient = Patient(
            path_config=path_config,
            phonetic_id=patient_id,
            debug=False
        )
        
        # load the subject
        subject = make_subject(patient, time_point, transforms=self.transforms)
        
        prediction = self._predict_subject(subject)
        
        # save the prediction
        output_dir = self._save_prediction(prediction, patient_id, time_point)
        logger.info(f"Prediction completed and saved for patient {patient_id} at time point {time_point}")
        return output_dir
        
        
        
    def _predict_subject(self, subject: tio.Subject)-> tio.ScalarImage:
        sampler = tio.inference.GridSampler(
            subject, 
            patch_size=self.cfg.inference.patch_size,
            patch_overlap=self.cfg.inference.patch_overlap
        )
        
        loader = torch.utils.data.DataLoader(sampler, batch_size=self.cfg.inference.batch_size, num_workers=self.cfg.inference.num_workers)
        aggregator = tio.inference.GridAggregator(
            sampler,
            overlap_mode=self.cfg.inference.patch_aggregation_overlap_mode
        )
        with torch.no_grad():
            for batch in loader:
                mag = batch["mag"][tio.DATA].to(self.device)
                fvx = batch["flow_vx"][tio.DATA].to(self.device)
                fvy = batch["flow_vy"][tio.DATA].to(self.device)
                fvz = batch["flow_vz"][tio.DATA].to(self.device)
                
                speed = torch.sqrt(fvx ** 2 + fvy ** 2 + fvz ** 2)
                input_base = torch.cat([mag, speed], dim=1)
                
                prediction = self.generator(input_base)
                aggregator.add_batch(prediction.cpu(), batch[tio.LOCATION])

        pred_tensor = aggregator.get_output_tensor()
        result = tio.ScalarImage(tensor=pred_tensor, affine=subject["mag"][tio.AFFINE])
        return result

        
    def _save_prediction(self, prediction: tio.ScalarImage, patient_id: str, time_point: int) -> Path:
        output_path = self.output_dir / f"pred_{patient_id}_t{time_point:02d}_overlap_{self.cfg.inference.patch_overlap}_overlap-mode_{self.cfg.inference.patch_aggregation_overlap_mode}.nii.gz"
        prediction.save(output_path)
        logger.info(f"Saved prediction to {output_path}")
        return output_path


@hydra.main(
    version_base="1.1",
    config_path=str((Path(__file__).resolve().parents[3] / "hydra_configs").as_posix()),
    config_name="config"
)
def main(cfg: DictConfig):
    # Check for required parameters
    if not cfg.inference.get('patient_id'):
        logger.error("patient_id is required but not provided")
        raise ValueError("patient_id is required")
    
    if not cfg.inference.get('time_point'):
        logger.error("time_point is required but not provided")
        raise ValueError("time_point is required")

    logger.info(f"Starting inference for patient_id: {cfg.inference.patient_id}, time_point: {cfg.inference.time_point}")
    
    try:
        superenhancer = VascularSuperenhancer(cfg)
        output_dir = superenhancer.predict_single(cfg.inference.patient_id, cfg.inference.time_point)
        logger.info(f"Inference completed successfully. Output saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
        