from pathlib import Path
from typing import Optional, List
import torch
import torchio as tio
import hydra
from omegaconf import DictConfig
import logging
import pandas as pd

from vascular_superenhancement.training.model_factory import build_generator
from vascular_superenhancement.training.transforms import build_transforms, build_multi_timepoint_transforms
from vascular_superenhancement.utils.path_config import load_path_config
from vascular_superenhancement.utils.logger import setup_patient_logger
from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.inferencing.datasets import make_subject_full_fov, make_multi_timepoint_subject_full_fov

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

        # Multi-timepoint configuration
        self.use_multi_timepoint = cfg.train.get('use_multi_timepoint', False)
        self.temporal_window_size = cfg.train.get('temporal_window_size', 5)
        self.center_idx = self.temporal_window_size // 2
        self.prediction_mode = cfg.train.get('inference_prediction_mode', 'center')
        self.temporal_weights = cfg.train.get('inference_temporal_weights', None)

        if self.use_multi_timepoint:
            logger.info(f"Multi-timepoint inference enabled:")
            logger.info(f"  - Window size: {self.temporal_window_size}")
            logger.info(f"  - Center index: {self.center_idx}")
            logger.info(f"  - Prediction mode: {self.prediction_mode}")

        # load model
        self.generator = build_generator(cfg).to(self.device)
        self._load_checkpoint()
        self.generator.eval()

        # Build transforms based on mode
        if self.use_multi_timepoint:
            self.transforms = build_multi_timepoint_transforms(
                cfg, train=False, window_size=self.temporal_window_size
            )
        else:
            self.transforms = build_transforms(cfg, train=False)
    
    def _ensure_full_fov_files_exist(self, patient: Patient, patient_logger: Optional[logging.Logger] = None) -> None:
        """
        Ensure that full FOV per-timepoint files exist for the patient.
        If they don't exist, build them.
        
        Args:
            patient: Patient object to check and build files for
            patient_logger: Optional patient-specific logger. If None, uses module-level logger.
        """
        log = patient_logger if patient_logger is not None else logger
        
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
            log.info(f"Full FOV per-timepoint files not found for patient {patient.identifier}, building them...")
            try:
                patient.build_4d_flow_per_timepoint_full_fov()
                log.info(f"Successfully built full FOV per-timepoint files for patient {patient.identifier}")
            except Exception as e:
                log.error(f"Error building full FOV per-timepoint files: {str(e)}", exc_info=True)
                raise
        else:
            log.debug(f"Full FOV per-timepoint files already exist for patient {patient.identifier}")
          
    def _load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
    
    def _get_prediction_path(self, patient_id: str, time_point: int, output_dir: Optional[Path] = None) -> Path:
        """
        Get the expected path for a prediction file.
        
        Args:
            patient_id: Patient identifier
            time_point: Timepoint index
            output_dir: Optional output directory. If None, uses self.output_dir
            
        Returns:
            Path to expected prediction file
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        output_path = output_dir / f"pred_{patient_id}_{self.cfg.inference.inference_name}_t{time_point:02d}_overlap_{self.cfg.inference.patch_overlap}_overlap-mode_{self.cfg.inference.patch_aggregation_overlap_mode}.nii.gz"
        return output_path
    
    def _prediction_exists(self, patient_id: str, time_point: int, output_dir: Optional[Path] = None) -> bool:
        """
        Check if a prediction already exists for a given patient and timepoint.
        
        Args:
            patient_id: Patient identifier
            time_point: Timepoint index
            output_dir: Optional output directory. If None, uses self.output_dir
            
        Returns:
            True if prediction exists, False otherwise
        """
        prediction_path = self._get_prediction_path(patient_id, time_point, output_dir)
        return prediction_path.exists()
    
    def _all_timepoints_predictions_exist(self, patient_id: str, num_timepoints: int) -> bool:
        """
        Check if predictions exist for all timepoints for a patient.
        
        Args:
            patient_id: Patient identifier
            num_timepoints: Number of timepoints to check
            
        Returns:
            True if all timepoints have predictions, False otherwise
        """
        patient_output_dir = self.output_dir / patient_id / "predictions"
        for time_point in range(num_timepoints):
            if not self._prediction_exists(patient_id, time_point, output_dir=patient_output_dir):
                return False
        return True
        
    def predict_single(self, patient_id: str, time_point: int = 3) -> Path:
        # Load path config first to ensure we use the correct config
        config_name = self.cfg.path_config.path_config_name
        path_config_obj = load_path_config(config_name)
        
        # Set up patient-specific logger using the same config
        patient_logger = setup_patient_logger(
            patient_id,
            config=config_name
        )
        patient_logger.debug(f"Using path_config: {config_name}, dataset: {path_config_obj.dataset}")
        
        # Check if prediction already exists
        overwrite = self.cfg.inference.get('overwrite', False)
        if not overwrite and self._prediction_exists(patient_id, time_point):
            existing_path = self._get_prediction_path(patient_id, time_point)
            patient_logger.info(f"Prediction already exists for patient {patient_id} at time point {time_point}: {existing_path}. Skipping (overwrite=False).")
            return existing_path.parent
        
        # load the patient
        patient = Patient(
            path_config=path_config_obj,
            phonetic_id=patient_id,
            debug=False,
            config=config_name
        )
        
        # Ensure full FOV per-timepoint files exist
        self._ensure_full_fov_files_exist(patient, patient_logger=patient_logger)
        
        # load the subject using full FOV data
        subject = make_subject_full_fov(patient, time_point, transforms=self.transforms)
        
        prediction = self._predict_subject(subject)
        
        # save the prediction
        output_dir = self._save_prediction(prediction, patient_id, time_point, patient_logger=patient_logger)
        patient_logger.info(f"Prediction completed and saved for patient {patient_id} at time point {time_point}")
        return output_dir
    
    def predict_all_timepoints(self, patient_id: str) -> Path:
        """
        Run inference for all timepoints for a patient.

        Supports both single-timepoint and multi-timepoint modes.
        For multi-timepoint, uses temporal sliding window for each prediction.

        Args:
            patient_id: Patient identifier

        Returns:
            Path to patient-specific directory containing all predictions
        """
        # Load path config first to ensure we use the correct config
        config_name = self.cfg.path_config.path_config_name
        path_config_obj = load_path_config(config_name)

        # Set up patient-specific logger using the same config
        patient_logger = setup_patient_logger(
            patient_id,
            config=config_name
        )
        patient_logger.debug(f"Using path_config: {config_name}, dataset: {path_config_obj.dataset}")

        # load the patient
        patient = Patient(
            path_config=path_config_obj,
            phonetic_id=patient_id,
            debug=False,
            config=config_name
        )

        num_timepoints = patient.num_timepoints
        patient_logger.info(f"Running inference for {num_timepoints} timepoints for patient {patient_id}")
        if self.use_multi_timepoint:
            patient_logger.info(f"Using multi-timepoint mode with window size {self.temporal_window_size}")

        # Check if all predictions already exist
        overwrite = self.cfg.inference.get('overwrite', False)
        if not overwrite and self._all_timepoints_predictions_exist(patient_id, num_timepoints):
            patient_output_dir = self.output_dir / patient_id / "predictions"
            patient_logger.info(f"All predictions already exist for patient {patient_id}. Skipping (overwrite=False).")
            return patient_output_dir

        # Ensure full FOV per-timepoint files exist
        self._ensure_full_fov_files_exist(patient, patient_logger=patient_logger)

        # Create patient-specific output directory
        patient_output_dir = self.output_dir / patient_id / "predictions"
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        patient_logger.info(f"Patient predictions will be saved in {patient_output_dir}")

        for time_point in range(num_timepoints):
            # Check if this specific timepoint prediction exists
            if not overwrite and self._prediction_exists(patient_id, time_point, output_dir=patient_output_dir):
                existing_path = self._get_prediction_path(patient_id, time_point, output_dir=patient_output_dir)
                patient_logger.info(f"Prediction already exists for timepoint {time_point}: {existing_path}. Skipping (overwrite=False).")
                continue

            patient_logger.info(f"Processing timepoint {time_point}/{num_timepoints-1} for patient {patient_id}")
            try:
                # Load subject using appropriate method based on mode
                if self.use_multi_timepoint:
                    subject = make_multi_timepoint_subject_full_fov(
                        patient,
                        center_time_index=time_point,
                        window_size=self.temporal_window_size,
                        transforms=self.transforms
                    )
                else:
                    subject = make_subject_full_fov(patient, time_point, transforms=self.transforms)

                prediction = self._predict_subject(subject)

                # save the prediction in patient-specific directory
                self._save_prediction(prediction, patient_id, time_point, output_dir=patient_output_dir, patient_logger=patient_logger)
                patient_logger.info(f"Prediction completed for timepoint {time_point}")
            except Exception as e:
                patient_logger.error(f"Error during inference for timepoint {time_point}: {str(e)}", exc_info=True)
                continue

        patient_logger.info(f"Completed inference for all timepoints for patient {patient_id}")
        return patient_output_dir
        
        
        
    def _predict_subject(self, subject: tio.Subject) -> tio.ScalarImage:
        """
        Run patch-based inference on a subject.

        Supports both single-timepoint and multi-timepoint modes.
        For multi-timepoint, extracts the center prediction based on prediction_mode.

        Args:
            subject: TorchIO Subject with input data

        Returns:
            Predicted ScalarImage
        """
        sampler = tio.inference.GridSampler(
            subject,
            patch_size=self.cfg.inference.patch_size,
            patch_overlap=self.cfg.inference.patch_overlap
        )

        loader = torch.utils.data.DataLoader(
            sampler,
            batch_size=self.cfg.inference.batch_size,
            num_workers=self.cfg.inference.num_workers
        )
        aggregator = tio.inference.GridAggregator(
            sampler,
            overlap_mode=self.cfg.inference.patch_aggregation_overlap_mode
        )

        with torch.no_grad():
            for batch in loader:
                if self.use_multi_timepoint:
                    # Multi-timepoint inference
                    input_tensor = self._prepare_multi_timepoint_batch(batch)
                    prediction = self.generator(input_tensor)  # [B, window_size, D, H, W]

                    # Extract output based on prediction mode
                    if self.prediction_mode == 'weighted':
                        final_pred = self._get_weighted_prediction(prediction)
                    else:  # 'center'
                        final_pred = prediction[:, self.center_idx:self.center_idx + 1, ...]
                else:
                    # Single-timepoint inference
                    mag = batch["mag"][tio.DATA].to(self.device)
                    fvx = batch["flow_vx"][tio.DATA].to(self.device)
                    fvy = batch["flow_vy"][tio.DATA].to(self.device)
                    fvz = batch["flow_vz"][tio.DATA].to(self.device)

                    speed = torch.sqrt(fvx ** 2 + fvy ** 2 + fvz ** 2)
                    input_tensor = torch.cat([mag, speed], dim=1)

                    final_pred = self.generator(input_tensor)

                aggregator.add_batch(final_pred.cpu(), batch[tio.LOCATION])

        pred_tensor = aggregator.get_output_tensor()

        # Get affine from appropriate source
        if self.use_multi_timepoint:
            affine = subject[f"mag_t{self.center_idx}"][tio.AFFINE]
        else:
            affine = subject["mag"][tio.AFFINE]

        result = tio.ScalarImage(tensor=pred_tensor, affine=affine)
        return result

    def _prepare_multi_timepoint_batch(self, batch: dict) -> torch.Tensor:
        """
        Prepare multi-timepoint batch for inference.

        Args:
            batch: Batch dictionary from dataloader

        Returns:
            Input tensor of shape [B, 2*window_size, D, H, W]
        """
        mag_tensors = []
        speed_tensors = []

        for i in range(self.temporal_window_size):
            suffix = f'_t{i}'
            mag = batch[f'mag{suffix}'][tio.DATA].to(self.device)
            fvx = batch[f'flow_vx{suffix}'][tio.DATA].to(self.device)
            fvy = batch[f'flow_vy{suffix}'][tio.DATA].to(self.device)
            fvz = batch[f'flow_vz{suffix}'][tio.DATA].to(self.device)

            speed = torch.sqrt(fvx ** 2 + fvy ** 2 + fvz ** 2)
            mag_tensors.append(mag)
            speed_tensors.append(speed)

        all_mags = torch.cat(mag_tensors, dim=1)
        all_speeds = torch.cat(speed_tensors, dim=1)
        return torch.cat([all_mags, all_speeds], dim=1)

    def _get_weighted_prediction(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Get weighted average prediction from multi-channel output.

        Args:
            pred: Prediction tensor of shape [B, window_size, D, H, W]

        Returns:
            Weighted average prediction of shape [B, 1, D, H, W]
        """
        if self.temporal_weights is not None:
            weights = list(self.temporal_weights)
        else:
            # Default triangular weights
            half = self.temporal_window_size // 2
            weights = []
            for i in range(self.temporal_window_size):
                dist = abs(i - half)
                w = 1.0 / (dist + 1)
                weights.append(w)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        weights_tensor = torch.tensor(weights, device=pred.device, dtype=pred.dtype)
        weights_tensor = weights_tensor.view(1, self.temporal_window_size, 1, 1, 1)

        weighted = (pred * weights_tensor).sum(dim=1, keepdim=True)
        return weighted

        
    def _save_prediction(self, prediction: tio.ScalarImage, patient_id: str, time_point: int, output_dir: Optional[Path] = None, patient_logger: Optional[logging.Logger] = None) -> Path:
        """
        Save prediction to disk.
        
        Args:
            prediction: The prediction ScalarImage to save
            patient_id: Patient identifier
            time_point: Timepoint index
            output_dir: Optional output directory. If None, uses self.output_dir
            patient_logger: Optional patient-specific logger. If None, uses module-level logger.
            
        Returns:
            Path to saved prediction file
        """
        log = patient_logger if patient_logger is not None else logger
        output_path = self._get_prediction_path(patient_id, time_point, output_dir)
        prediction.save(output_path)
        log.info(f"Saved prediction to {output_path}")
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
        
        logger.info(f"Found {len(patient_ids)} test patients")
        
        logger.info(f"Patient IDs: {patient_ids}")
        superenhancer = VascularSuperenhancer(cfg)
        overwrite = cfg.inference.get('overwrite', False)
        
        skipped_count = 0
        processed_count = 0
        
        for patient_id in patient_ids:
            logger.info(f"Starting inference for patient_id: {patient_id}")
            
            # Quick check: skip entire patient if predictions already exist and overwrite=False
            if not overwrite:
                if all_timepoints:
                    try:
                        path_config = load_path_config(cfg.path_config.path_config_name)
                        patient = Patient(
                            path_config=path_config,
                            phonetic_id=patient_id,
                            debug=True,
                            config=cfg.path_config.path_config_name
                        )
                        num_timepoints = patient.num_timepoints
                        if superenhancer._all_timepoints_predictions_exist(patient_id, num_timepoints):
                            logger.info(f"Skipping patient {patient_id}: all {num_timepoints} timepoint predictions already exist (overwrite=False)")
                            skipped_count += 1
                            continue
                    except Exception as e:
                        logger.warning(f"Could not check existing predictions for {patient_id}, will attempt inference: {str(e)}")
                else:
                    # Check if single timepoint prediction exists
                    if superenhancer._prediction_exists(patient_id, cfg.inference.time_point):
                        existing_path = superenhancer._get_prediction_path(patient_id, cfg.inference.time_point)
                        logger.info(f"Skipping patient {patient_id}: prediction already exists at {existing_path} (overwrite=False)")
                        skipped_count += 1
                        continue
            
            try:
                if all_timepoints:
                    output_dir = superenhancer.predict_all_timepoints(patient_id)
                else:
                    output_dir = superenhancer.predict_single(patient_id, cfg.inference.time_point)
                processed_count += 1
                logger.info(f"Inference completed successfully. Output saved to: {output_dir}")
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"Inference summary: {processed_count} patients processed, {skipped_count} patients skipped (already had predictions)")

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
        