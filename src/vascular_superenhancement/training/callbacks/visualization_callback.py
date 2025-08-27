"""Visualization callback for saving predictions and images."""

from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import torch
import torchio as tio
# import numpy as np
import logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Avoid circular imports
if TYPE_CHECKING:
    from ..trainers.base_trainer import BaseTrainer

from .base_callback import Callback

logger = logging.getLogger(__name__)


class VisualizationCallback(Callback):
    """Callback for visualizing and saving model predictions."""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        num_samples: int = 3,
        save_frequency: int = 5,
        save_original: bool = True,
        log_to_wandb: bool = True,
        patch_size: Optional[List[int]] = None,
        patch_overlap: int = 24,
        patch_aggregation_mode: str = 'hann',
    ):
        """Initialize visualization callback.
        
        Args:
            output_dir: Directory to save visualizations
            num_samples: Number of validation samples to visualize
            save_frequency: Save visualizations every N epochs
            save_original: Whether to save original images (first epoch only)
            log_to_wandb: Whether to log images to W&B if available
            patch_size: Size of patches for inference
            patch_overlap: Overlap between patches
            patch_aggregation_mode: Mode for aggregating overlapping patches
        """
        self.output_dir = Path(output_dir or Path.cwd() / "visualizations")
        self.num_samples = num_samples
        self.save_frequency = save_frequency
        self.save_original = save_original
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        
        # Patch-based inference settings
        self.patch_size = patch_size or [48, 48, 48]
        self.patch_overlap = patch_overlap
        self.patch_aggregation_mode = patch_aggregation_mode
        
        # Track subjects for visualization
        self.subjects_to_visualize = []
        self.original_saved = False
        
    def on_train_begin(self, trainer: 'BaseTrainer') -> None:
        """Select subjects for visualization at training start."""
        if trainer.val_loader is None:
            logger.warning("No validation loader available for visualization")
            return
        
        # Select subjects from validation dataset
        # Note: This assumes the dataset has a way to iterate through subjects
        # You may need to adapt this based on your actual dataset structure
        val_dataset = trainer.val_loader.dataset
        
        # For SubjectsDataset from TorchIO
        if hasattr(val_dataset, '_subjects'):
            subjects = val_dataset._subjects[:self.num_samples]
            self.subjects_to_visualize = subjects
            logger.info(f"Selected {len(subjects)} subjects for visualization")
            
            for subject in subjects:
                if hasattr(subject, 'patient_id'):
                    logger.info(f"  - Patient {subject.patient_id}")
    
    def on_epoch_end(self, trainer: 'BaseTrainer', epoch: int,
                     metrics: Dict[str, float]) -> None:
        """Generate and save visualizations at epoch end."""
        
        # Only visualize at specified frequency
        if epoch % self.save_frequency != 0:
            return
        
        if not self.subjects_to_visualize:
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
                output_path = self._save_prediction(
                    prediction, subject, patient_id, epoch
                )
                
                # Log to W&B if enabled
                if self.log_to_wandb and wandb.run is not None:
                    image_key = f"val/predictions/{patient_id}"
                    wandb_images[image_key] = self._prepare_wandb_image(
                        prediction, patient_id, epoch, metrics
                    )
        
        # Mark original as saved
        if epoch == 0 and self.save_original:
            self.original_saved = True
        
        # Log all images to W&B at once
        if wandb_images and wandb.run is not None:
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
            overlap_mode=self.patch_aggregation_mode
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
        
        # Get aggregated output
        return aggregator.get_output_tensor()
    
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
        
        logger.debug(f"Saved prediction to {output_path}")
        return output_path
    
    def _prepare_wandb_image(
        self,
        prediction: torch.Tensor,
        patient_id: str,
        epoch: int,
        metrics: Dict[str, float]
    ) -> Any:
        """Prepare image for W&B logging.
        
        Args:
            prediction: Prediction tensor
            patient_id: Patient identifier
            epoch: Current epoch
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
        caption_parts = [
            f"Epoch {epoch:04d}",
            f"Patient {patient_id}",
            f"Slice {z_middle}",
        ]
        
        # Add key metrics
        for key in ['val/loss_generator', 'val/loss_discriminator']:
            if key in metrics:
                caption_parts.append(f"{key.split('/')[-1]}: {metrics[key]:.4f}")
        
        caption = ", ".join(caption_parts)
        
        return wandb.Image(center_slice, caption=caption)