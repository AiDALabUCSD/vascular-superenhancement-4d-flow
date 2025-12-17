"""Visualization callback for saving predictions and images."""

from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING, List
import torch
import torchio as tio
import logging
from omegaconf import DictConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Avoid circular imports
if TYPE_CHECKING:
    from ..trainers.base_trainer import BaseTrainer

from .base_callback import Callback
from ..datasets import make_subject
from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.utils.path_config import load_path_config

logger = logging.getLogger(__name__)


class VisualizationCallback(Callback):
    """Callback for visualizing and saving model predictions."""
    
    def __init__(
        self, 
        cfg: DictConfig
    ):
        """Initialize visualization callback.
        
        Args:
            cfg: Hydra configuration
        """
        self.cfg = cfg
        
        # Get visualization settings from config
        self.output_dir = Path.cwd() / "visualizations"
        self.num_samples = self.cfg.train.num_sample_predictions
        self.save_original = self.cfg.train.visualization_save_original
        self.save_frequency = self.cfg.train.visualization_save_frequency    
        
        # W&B settings
        self.wandb_enabled = self.cfg.wandb.enabled
        self.visualization_log_to_wandb = self.cfg.wandb.log_images
        self.visualization_log_frequency = self.cfg.wandb.log_images_frequency
        
        # Patch-based inference settings
        self.patch_size = self.cfg.train.patch_size
        self.patch_overlap = self.cfg.train.patch_overlap
        self.patch_aggregation_overlap_mode = self.cfg.train.patch_aggregation_overlap_mode
        
        # Track subjects for visualization
        self.subjects_to_visualize = None
        self.inference_only_subjects: List[tio.Subject] = []
        self.original_saved = False
        self.inference_cfg = self.cfg.train.get('validation_inference', None)
        
        logger.info("VisualizationCallback initialized:")
        logger.info(f"  - Save frequency: every {self.save_frequency} epochs")
        logger.info(f"  - W&B visualization logging: {'enabled' if self.visualization_log_to_wandb else 'disabled'}")
        logger.info(f"  - Output directory: {self.output_dir}")
        
    def on_fit_start(self, trainer: 'BaseTrainer') -> None:
        """Select subjects for visualization at training start."""
        self.subjects_to_visualize = []

        if trainer.val_subjects is None:
            logger.warning("No validation subjects available for visualization")
        else:
            # Iterate through dataset to get TRANSFORMED subjects
            for i, subject in enumerate(trainer.val_dataset):
                if i >= self.num_samples and self.num_samples > 0:
                    break
                self.subjects_to_visualize.append(subject)

            logger.info(f"Selected {len(self.subjects_to_visualize)} subjects for visualization")
            for subject in self.subjects_to_visualize:
                logger.info(f"  - Patient {subject.patient_id}")

        # Load inference-only subjects that should be visualized but not part of validation metrics
        self.inference_only_subjects = self._load_inference_only_subjects(trainer)
        if self.inference_only_subjects:
            logger.info(f"Added {len(self.inference_only_subjects)} inference-only subjects for visualization")
            for subject in self.inference_only_subjects:
                logger.info(f"  - Inference-only patient {subject.patient_id}")
            self.subjects_to_visualize.extend(self.inference_only_subjects)

        if not self.subjects_to_visualize:
            logger.warning("No subjects available for visualization after including inference-only patients")

    
    def on_validation_epoch_end(self, trainer: 'BaseTrainer', epoch: int,
                     metrics: Dict[str, float]) -> None:
        """Generate and save visualizations at epoch end."""
        
        is_last_epoch = epoch == trainer.cfg.train.num_epochs - 1
        is_improving = metrics[trainer.monitor_metric] < trainer.best_val_metric_moving_average * (1 - trainer.cfg.train.get('early_stop_threshold', 0.33))
        
        should_visualize = (
            epoch % self.save_frequency == 0 or
            epoch <=10 or 
            is_last_epoch or 
            is_improving
        )
        
        # Only visualize at specified frequency
        if not should_visualize:
            return

        if not self.subjects_to_visualize:
            logger.debug("Skipping visualization because no subjects are available")
            return
        
        # Get generator model (assuming GAN trainer)
        if 'generator' not in trainer.models:
            logger.warning("No generator model found for visualization")
            return
        
        generator = trainer.models['generator']
        generator.eval()
        
        wandb_images = {}
        
        with torch.no_grad():
            for subject in self.subjects_to_visualize:
                patient_id = getattr(subject, 'patient_id', 'unknown')
                
                # Save original images on first visualization
                if epoch == 0 and self.save_original and not self.original_saved:
                    self._save_original_images(subject, patient_id)
                
                # Generate prediction
                prediction = self._generate_prediction(
                    subject, generator, trainer.device
                )
                
                # Save prediction
                self._save_prediction(
                    prediction, subject, patient_id, epoch
                )
                
                # Log to W&B if enabled
                if self.visualization_log_to_wandb:
                    image_key = f"validation/{patient_id}/center_slice"
                    wandb_images[image_key] = self._prepare_wandb_image(
                        prediction, patient_id, epoch, trainer.global_step, metrics
                    )
        
        # Mark original as saved
        if epoch == 0 and self.save_original:
            self.original_saved = True
        
        # Log all images to W&B at once
        if self.visualization_log_to_wandb:
            wandb.log(wandb_images, step=trainer.global_step)
    
    def _save_original_images(self, subject: tio.Subject, patient_id: str) -> None:
        """Save original images from subject.
        
        Args:
            subject: TorchIO subject containing images
            patient_id: Patient identifier
        """
        output_dir = self.output_dir / patient_id / "original"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define images to save
        images_to_save = {
            'cine': 'cine',
            'mag': 'mag',
            'flow_vx': 'fvx',
            'flow_vy': 'fvy',
            'flow_vz': 'fvz',
        }
        
        for key, prefix in images_to_save.items():
            if key in subject:
                data = subject[key][tio.DATA]
                affine = subject[key][tio.AFFINE]
                path = output_dir / f"{prefix}_{patient_id}.nii.gz"
                
                image = tio.ScalarImage(tensor=data, affine=affine)
                image.save(path)
                logger.debug(f"Saved {prefix} to {path}")
        
        # Save computed speed
        if all(k in subject for k in ['flow_vx', 'flow_vy', 'flow_vz']):
            speed_data = torch.sqrt(
                subject["flow_vx"][tio.DATA] ** 2 +
                subject["flow_vy"][tio.DATA] ** 2 +
                subject["flow_vz"][tio.DATA] ** 2
            )
            speed_affine = subject["flow_vx"][tio.AFFINE]
            speed_path = output_dir / f"speed_{patient_id}.nii.gz"
            
            tio.ScalarImage(tensor=speed_data, affine=speed_affine).save(speed_path)
            logger.debug(f"Saved speed to {speed_path}")
    
    def _generate_prediction(
        self, 
        subject: tio.Subject,
        generator: torch.nn.Module,
        device: torch.device
    ) -> torch.Tensor:
        """Generate prediction for a subject using patch-based inference.
        
        Args:
            subject: TorchIO subject
            generator: Generator model
            device: Device to run on
            
        Returns:
            Generated prediction tensor
        """
        # Debug the input subject
        logger.info("Subject data shapes:")
        self._log_tensor_stats('mag', subject['mag'][tio.DATA])
        if 'cine' in subject:
            self._log_tensor_stats('cine', subject['cine'][tio.DATA])
        else:
            logger.info("  cine: not available (inference-only subject)")

        
        # Create sampler for patch-based inference
        sampler = tio.inference.GridSampler(
            subject,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap
        )
        
        # Create data loader
        loader = torch.utils.data.DataLoader(sampler, batch_size=1)
        
        # Create aggregator
        aggregator = tio.inference.GridAggregator(
            sampler,
            overlap_mode=self.patch_aggregation_overlap_mode
        )
        
        # Process patches
        for batch in loader:
            # Prepare input
            mag = batch["mag"][tio.DATA].to(device)
            fvx = batch["flow_vx"][tio.DATA].to(device)
            fvy = batch["flow_vy"][tio.DATA].to(device)
            fvz = batch["flow_vz"][tio.DATA].to(device)
                        
            speed = torch.sqrt(fvx ** 2 + fvy ** 2 + fvz ** 2)
            input_tensor = torch.cat([mag, speed], dim=1)
            
            # Generate prediction
            prediction = generator(input_tensor)
            
            # Add to aggregator
            aggregator.add_batch(prediction.cpu(), batch[tio.LOCATION])
        
        pred_tensor = aggregator.get_output_tensor()
        logger.info(f"Prediction stats: shape={pred_tensor.shape}, min={pred_tensor.min():.3f}, max={pred_tensor.max():.3f}, mean={pred_tensor.mean():.3f}")
        return pred_tensor
    
    def _save_prediction(
        self,
        prediction: torch.Tensor,
        subject: tio.Subject,
        patient_id: str,
        epoch: int
    ) -> Path:
        """Save prediction as NIfTI file.
        
        Args:
            prediction: Prediction tensor
            subject: Original subject for affine matrix
            patient_id: Patient identifier
            epoch: Current epoch
            
        Returns:
            Path to saved file
        """
        output_dir = self.output_dir / patient_id / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"pred_epoch_{epoch:04d}_{patient_id}.nii.gz"
        
        # Use affine from original image
        affine = subject["mag"][tio.AFFINE] if "mag" in subject else torch.eye(4)
        
        # Save prediction
        output_image = tio.ScalarImage(tensor=prediction, affine=affine)
        output_image.save(output_path)
        
        logger.info(f"Saved prediction to {output_path} with shape {prediction.shape}")
        return output_path
    
    def _prepare_wandb_image(
        self,
        prediction: torch.Tensor,
        patient_id: str,
        epoch: int,
        global_step: int,
        metrics: Dict[str, float]
    ) -> Any:
        """Prepare image for W&B logging.
        
        Args:
            prediction: Prediction tensor
            patient_id: Patient identifier
            epoch: Current epoch
            global_step: Current global step
            metrics: Current metrics
            
        Returns:
            W&B Image object
        """
        # Get center slice
        z_middle = prediction.shape[-1] // 2
        center_slice = prediction[0, :, :, z_middle].cpu().numpy()
        
        # Rotate for proper orientation if needed
        # center_slice = np.rot90(center_slice, k=1)
        
        # Create caption with metrics
        # caption = f"e {epoch:04d}, g {global_step:04d}, p {subject.patient_id}, z {z_middle}, g_gan {scalar_loss_generator_gan_val:.4f}, g_l1 {scalar_loss_generator_l1_val:.4f}, g_ssim {scalar_loss_generator_ssim_val:.4f}, g {scalar_loss_generator_val:.4f}, d {scalar_loss_discriminator_val:.4f}"
        caption = (
            f"e {epoch:04d}, g {global_step:04d}, p {patient_id}, z {z_middle}, "
            f"g_gan {metrics.get('val/loss_generator_gan', 0):.4f}, "
            f"g_l1 {metrics.get('val/loss_generator_l1', 0):.4f}, "
            f"g_ssim {metrics.get('val/loss_generator_ssim', 0):.4f}, "
            f"g {metrics.get('val/loss_generator', 0):.4f}, "
            f"d {metrics.get('val/loss_discriminator', 0):.4f}"
        )
        
        return wandb.Image(center_slice, caption=caption)

    def _load_inference_only_subjects(self, trainer: 'BaseTrainer') -> List[tio.Subject]:
        """Load additional subjects for visualization that are not part of the validation loop."""
        if not self.inference_cfg:
            return []

        patient_ids = self.inference_cfg.get('patient_ids', [])
        if not patient_ids:
            return []

        # Determine which time index to use; default to validation time index
        time_index = self.inference_cfg.get('time_index')
        if time_index is None:
            time_index = self.cfg.train.validation_time_index

        # Get the validation transform (same one used by validation dataset)
        val_transform = getattr(trainer.val_dataset, 'transform', None)
        
        # Load path config and create subjects directly with transforms applied
        # This matches how the inference script works - transforms are applied in make_subject
        path_config = load_path_config(self.cfg.path_config.path_config_name)
        
        subjects: List[tio.Subject] = []
        for pid in patient_ids:
            try:
                patient = Patient(
                    path_config=path_config,
                    phonetic_id=pid,
                    debug=self.cfg.train.debug
                )
                # Call make_subject directly with transforms, matching inference script pattern
                subject = make_subject(
                    patient,
                    time_index,
                    transforms=val_transform,
                    peak_systolic_only=self.cfg.train.get('peak_systolic_only', False),
                    inference_mode=True
                )
                subjects.append(subject)
            except Exception as exc:
                logger.error(f"Failed to load inference-only subject for patient {pid}: {exc}")

        return subjects

    @staticmethod
    def _log_tensor_stats(name: str, tensor: torch.Tensor) -> None:
        """Safely log summary statistics for a tensor, casting to float if needed."""
        try:
            stats_tensor = tensor.to(torch.float32)
        except Exception:
            logger.debug(f"Unable to cast tensor '{name}' to float32 for stats; using original dtype {tensor.dtype}")
            stats_tensor = tensor

        logger.info(
            "  %s: %s, min=%.3f, max=%.3f, mean=%.3f",
            name,
            tuple(stats_tensor.shape),
            stats_tensor.min().item(),
            stats_tensor.max().item(),
            stats_tensor.mean().item()
        )