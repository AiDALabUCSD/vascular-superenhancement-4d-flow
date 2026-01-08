from pathlib import Path
from typing import Optional
import torch
import torchio as tio
import hydra
from omegaconf import DictConfig
import logging
import pandas as pd

from vascular_superenhancement.training.model_factory import build_generator
from vascular_superenhancement.training.transforms import build_transforms
from vascular_superenhancement.utils.path_config import load_path_config
from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.inferencing.datasets import make_subject_full_fov

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
    
    def _ensure_full_fov_files_exist(self, patient: Patient) -> None:
        """
        Ensure that full FOV per-timepoint files exist for the patient.
        If they don't exist, build them.
        
        Args:
            patient: Patient object to check and build files for
        """
        # Check if full FOV per-timepoint files exist
        full_fov_dirs = [
            patient.flow_mag_per_timepoint_full_fov_dir,
            patient.flow_vx_per_timepoint_full_fov_dir,
            patient.flow_vy_per_timepoint_full_fov_dir,
            patient.flow_vz_per_timepoint_full_fov_dir,
        ]
        
        # Check if all directories have files
        all_exist = all(
            len(list(directory.glob("*.nii.gz"))) > 0
            for directory in full_fov_dirs
        )
        
        if not all_exist:
            logger.info(f"Full FOV per-timepoint files not found for patient {patient.identifier}, building them...")
            try:
                patient.build_4d_flow_per_timepoint_full_fov()
                logger.info(f"Successfully built full FOV per-timepoint files for patient {patient.identifier}")
            except Exception as e:
                logger.error(f"Error building full FOV per-timepoint files: {str(e)}", exc_info=True)
                raise
        else:
            logger.debug(f"Full FOV per-timepoint files already exist for patient {patient.identifier}")
          
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
        
        # Ensure full FOV per-timepoint files exist
        self._ensure_full_fov_files_exist(patient)
        
        # load the subject using full FOV data
        subject = make_subject_full_fov(patient, time_point, transforms=self.transforms)
        
        prediction = self._predict_subject(subject)
        
        # save the prediction
        output_dir = self._save_prediction(prediction, patient_id, time_point)
        logger.info(f"Prediction completed and saved for patient {patient_id} at time point {time_point}")
        return output_dir
    
    def predict_all_timepoints(self, patient_id: str) -> Path:
        """
        Run inference for all timepoints for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Path to patient-specific directory containing all predictions
        """
        # load the patient
        path_config = load_path_config(self.cfg.path_config.path_config_name)
        patient = Patient(
            path_config=path_config,
            phonetic_id=patient_id,
            debug=False
        )
        
        num_timepoints = patient.num_timepoints
        logger.info(f"Running inference for {num_timepoints} timepoints for patient {patient_id}")
        
        # Ensure full FOV per-timepoint files exist
        self._ensure_full_fov_files_exist(patient)
        
        # Create patient-specific output directory
        patient_output_dir = self.output_dir / patient_id / "predictions"
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Patient predictions will be saved in {patient_output_dir}")
        
        for time_point in range(num_timepoints):
            logger.info(f"Processing timepoint {time_point}/{num_timepoints-1} for patient {patient_id}")
            try:
                # load the subject using full FOV data
                subject = make_subject_full_fov(patient, time_point, transforms=self.transforms)
                
                prediction = self._predict_subject(subject)
                
                # save the prediction in patient-specific directory
                self._save_prediction(prediction, patient_id, time_point, output_dir=patient_output_dir)
                logger.info(f"Prediction completed for timepoint {time_point}")
            except Exception as e:
                logger.error(f"Error during inference for timepoint {time_point}: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"Completed inference for all timepoints for patient {patient_id}")
        return patient_output_dir
        
        
        
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

        
    def _save_prediction(self, prediction: tio.ScalarImage, patient_id: str, time_point: int, output_dir: Optional[Path] = None) -> Path:
        """
        Save prediction to disk.
        
        Args:
            prediction: The prediction ScalarImage to save
            patient_id: Patient identifier
            time_point: Timepoint index
            output_dir: Optional output directory. If None, uses self.output_dir
            
        Returns:
            Path to saved prediction file
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        output_path = output_dir / f"pred_{patient_id}_{self.cfg.inference.inference_name}_t{time_point:02d}_overlap_{self.cfg.inference.patch_overlap}_overlap-mode_{self.cfg.inference.patch_aggregation_overlap_mode}.nii.gz"
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
    if not cfg.inference.get('patient_id') and not cfg.inference.get('all_test_patients'):
        logger.error("patient_id is required but not provided")
        raise ValueError("patient_id is required")
    
    # Check if we should process all timepoints
    all_timepoints = cfg.inference.get('all_timepoints', False)
    
    if not all_timepoints:
        if not cfg.inference.get('time_point'):
            logger.error("time_point is required when all_timepoints is False")
            raise ValueError("time_point is required")
    
    # if not cfg.inference.get('all_test_patients'):
    #     logger.error("all_test_patients is required but not provided")
    #     raise ValueError("all_test_patients is required")

    if cfg.inference.get('all_test_patients'):
        logger.info("Starting inference for all test patients")
        
        df = pd.read_csv(cfg.data.splits_path)
        patient_ids = df[df.split == 'test'].patient_id.tolist()
        
        superenhancer = VascularSuperenhancer(cfg)
        for patient_id in patient_ids:
            logger.info(f"Starting inference for patient_id: {patient_id}")
            try:
                if all_timepoints:
                    output_dir = superenhancer.predict_all_timepoints(patient_id)
                else:
                    output_dir = superenhancer.predict_single(patient_id, cfg.inference.time_point)
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}", exc_info=True)
                continue
            logger.info(f"Inference completed successfully. Output saved to: {output_dir}")

    else:
        logger.info(f"Starting inference for patient_id: {cfg.inference.patient_id}")
        
        try:
            superenhancer = VascularSuperenhancer(cfg)
            if all_timepoints:
                output_dir = superenhancer.predict_all_timepoints(cfg.inference.patient_id)
            else:
                output_dir = superenhancer.predict_single(cfg.inference.patient_id, cfg.inference.time_point)
            logger.info(f"Inference completed successfully. Output saved to: {output_dir}")
            
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    main()
        