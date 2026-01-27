"""Visualization callback for saving predictions and images."""

from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING, List, Tuple
import gc
import torch
import torchio as tio
import logging
from omegaconf import DictConfig
# #region agent log
import json
import psutil
import time as time_module
DEBUG_LOG_PATH = "/home/ayeluru/vascular-superenhancement-4d-flow/.cursor/debug.log"
def _debug_log_viz(location: str, message: str, hypothesis_id: str, extra_data: dict = None):
    try:
        proc = psutil.Process()
        mem = proc.memory_info()
        children_mem = sum(c.memory_info().rss for c in proc.children(recursive=True))
        sys_mem = psutil.virtual_memory()
        data = {"main_rss_gb": mem.rss / 1e9, "children_rss_gb": children_mem / 1e9, "total_proc_gb": (mem.rss + children_mem) / 1e9, "sys_avail_gb": sys_mem.available / 1e9, "sys_used_pct": sys_mem.percent}
        if torch.cuda.is_available():
            data["gpu_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
        if extra_data:
            data.update(extra_data)
        entry = {"timestamp": int(time_module.time() * 1000), "location": location, "message": message, "hypothesisId": hypothesis_id, "data": data, "sessionId": "debug-session"}
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
# #endregion

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Avoid circular imports
if TYPE_CHECKING:
    from ..trainers.base_trainer import BaseTrainer

from .base_callback import Callback
from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.utils.path_config import load_path_config
from vascular_superenhancement.training.transforms import build_inference_transforms
from vascular_superenhancement.inferencing.datasets import make_subject_full_fov, make_multi_timepoint_subject_full_fov

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

        # Multi-timepoint settings
        self.use_multi_timepoint = cfg.train.get('use_multi_timepoint', False)
        self.temporal_window_size = cfg.train.get('temporal_window_size', 5)
        self.center_idx = self.temporal_window_size // 2

        # Track subjects for visualization - store only metadata, not data
        self.visualization_patient_info: List[Tuple[str, bool]] = []  # List of (patient_id, is_inference_only)
        self.original_saved = False
        self.inference_cfg = self.cfg.train.get('validation_inference', None)

        logger.info("VisualizationCallback initialized:")
        logger.info(f"  - Save frequency: every {self.save_frequency} epochs")
        logger.info(f"  - W&B visualization logging: {'enabled' if self.visualization_log_to_wandb else 'disabled'}")
        logger.info(f"  - Output directory: {self.output_dir}")
        if self.use_multi_timepoint:
            logger.info(f"  - Multi-timepoint mode: window_size={self.temporal_window_size}")
        
    def on_fit_start(self, trainer: 'BaseTrainer') -> None:
        """Select subjects for visualization at training start.
        
        Only collects patient IDs, does not load actual subject data to save memory.
        Subjects are loaded on-demand during visualization.
        """
        self.visualization_patient_info = []

        if trainer.val_subjects is None:
            logger.warning("No validation subjects available for visualization")
        else:
            # Use dry_iter to get patient IDs without loading data
            count = 0
            for subject in trainer.val_dataset.dry_iter():
                if count >= self.num_samples and self.num_samples > 0:
                    break
                patient_id = getattr(subject, 'patient_id', None)
                if patient_id:
                    self.visualization_patient_info.append((patient_id, False))
                    count += 1

            logger.info(f"Selected {count} subjects for visualization")
            for patient_id, _ in self.visualization_patient_info:
                logger.info(f"  - Patient {patient_id}")

        # Add inference-only patient IDs (not loading them yet)
        if self.inference_cfg:
            inference_patient_ids = self.inference_cfg.get('patient_ids', [])
            if inference_patient_ids:
                logger.info(f"Added {len(inference_patient_ids)} inference-only subjects for visualization")
                for pid in inference_patient_ids:
                    self.visualization_patient_info.append((pid, True))
                    logger.info(f"  - Inference-only patient {pid}")

        if not self.visualization_patient_info:
            logger.warning("No subjects available for visualization after including inference-only patients")

    
    def on_validation_epoch_end(self, trainer: 'BaseTrainer', epoch: int,
                     metrics: Dict[str, float]) -> None:
        """Generate and save visualizations at epoch end.
        
        Loads subjects on-demand and releases them after processing to minimize memory usage.
        """
        
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

        if not self.visualization_patient_info:
            logger.debug("Skipping visualization because no subjects are available")
            return
        
        # Get generator model (assuming GAN trainer)
        if 'generator' not in trainer.models:
            logger.warning("No generator model found for visualization")
            return
        
        generator = trainer.models['generator']
        generator.eval()
        
        wandb_images = {}
        # #region agent log
        _debug_log_viz(f"visualization_callback.py:epoch_{epoch}_viz_start", f"Visualization starting for epoch {epoch}", "E", {"epoch": epoch, "num_patients": len(self.visualization_patient_info)})
        # #endregion
        
        with torch.no_grad():
            for patient_idx, (patient_id, is_inference_only) in enumerate(self.visualization_patient_info):
                try:
                    # #region agent log
                    _debug_log_viz(f"visualization_callback.py:epoch_{epoch}_patient_{patient_idx}_load_start", f"Loading patient {patient_id}", "E", {"epoch": epoch, "patient_id": patient_id, "patient_idx": patient_idx})
                    # #endregion
                    # Load subject on-demand
                    subject = self._load_subject_for_visualization(patient_id)
                    # #region agent log
                    _debug_log_viz(f"visualization_callback.py:epoch_{epoch}_patient_{patient_idx}_load_end", f"Loaded patient {patient_id}", "E", {"epoch": epoch, "patient_id": patient_id})
                    # #endregion

                    # Save original images on first visualization
                    if epoch == 0 and self.save_original and not self.original_saved:
                        self._save_original_images(subject, patient_id)

                    # Generate prediction
                    prediction = self._generate_prediction(
                        subject, generator, trainer.device
                    )
                    # #region agent log
                    _debug_log_viz(f"visualization_callback.py:epoch_{epoch}_patient_{patient_idx}_pred_end", f"Generated prediction for {patient_id}", "E", {"epoch": epoch, "patient_id": patient_id})
                    # #endregion

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

                except Exception as exc:
                    logger.error(f"Failed to visualize patient {patient_id}: {exc}")
                finally:
                    # Explicitly release memory after each subject
                    if 'subject' in locals():
                        del subject
                    if 'prediction' in locals():
                        del prediction
                    gc.collect()
                    torch.cuda.empty_cache()
                    # #region agent log
                    _debug_log_viz(f"visualization_callback.py:epoch_{epoch}_patient_{patient_idx}_cleanup", f"Cleaned up after {patient_id}", "E", {"epoch": epoch, "patient_id": patient_id})
                    # #endregion
        
        # Mark original as saved
        if epoch == 0 and self.save_original:
            self.original_saved = True
        
        # Log all images to W&B at once
        if self.visualization_log_to_wandb:
            wandb.log(wandb_images, step=trainer.global_step)
    
    def _save_original_images(self, subject: tio.Subject, patient_id: str) -> None:
        """Save original images from subject.

        Handles both single-timepoint and multi-timepoint subjects.
        For multi-timepoint, saves the center timepoint images.

        Args:
            subject: TorchIO subject containing images
            patient_id: Patient identifier
        """
        output_dir = self.output_dir / patient_id / "original"
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.use_multi_timepoint:
            # Multi-timepoint: save center timepoint images
            suffix = f'_t{self.center_idx}'
            images_to_save = {
                f'cine{suffix}': 'cine',
                f'mag{suffix}': 'mag',
                f'flow_vx{suffix}': 'fvx',
                f'flow_vy{suffix}': 'fvy',
                f'flow_vz{suffix}': 'fvz',
            }
            flow_keys = [f'flow_vx{suffix}', f'flow_vy{suffix}', f'flow_vz{suffix}']
        else:
            # Single-timepoint
            images_to_save = {
                'cine': 'cine',
                'mag': 'mag',
                'flow_vx': 'fvx',
                'flow_vy': 'fvy',
                'flow_vz': 'fvz',
            }
            flow_keys = ['flow_vx', 'flow_vy', 'flow_vz']

        for key, prefix in images_to_save.items():
            if key in subject:
                data = subject[key][tio.DATA]
                affine = subject[key][tio.AFFINE]
                path = output_dir / f"{prefix}_{patient_id}.nii.gz"

                image = tio.ScalarImage(tensor=data, affine=affine)
                image.save(path)
                logger.debug(f"Saved {prefix} to {path}")

        # Save computed speed
        if all(k in subject for k in flow_keys):
            speed_data = torch.sqrt(
                subject[flow_keys[0]][tio.DATA] ** 2 +
                subject[flow_keys[1]][tio.DATA] ** 2 +
                subject[flow_keys[2]][tio.DATA] ** 2
            )
            speed_affine = subject[flow_keys[0]][tio.AFFINE]
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

        Supports both single-timepoint and multi-timepoint modes.

        Args:
            subject: TorchIO subject
            generator: Generator model
            device: Device to run on

        Returns:
            Generated prediction tensor
        """
        # Debug the input subject
        logger.info("Subject data shapes:")
        if self.use_multi_timepoint:
            self._log_tensor_stats(f'mag_t{self.center_idx}', subject[f'mag_t{self.center_idx}'][tio.DATA])
            if f'cine_t{self.center_idx}' in subject:
                self._log_tensor_stats(f'cine_t{self.center_idx}', subject[f'cine_t{self.center_idx}'][tio.DATA])
            else:
                logger.info(f"  cine_t{self.center_idx}: not available (inference-only subject)")
        else:
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
            if self.use_multi_timepoint:
                # Multi-timepoint mode
                input_tensor = self._prepare_multi_timepoint_input(batch, device)
                prediction = generator(input_tensor)  # [B, window_size, D, H, W]
                # Extract center prediction
                final_pred = prediction[:, self.center_idx:self.center_idx + 1, ...]
            else:
                # Single-timepoint mode
                mag = batch["mag"][tio.DATA].to(device)
                fvx = batch["flow_vx"][tio.DATA].to(device)
                fvy = batch["flow_vy"][tio.DATA].to(device)
                fvz = batch["flow_vz"][tio.DATA].to(device)

                speed = torch.sqrt(fvx ** 2 + fvy ** 2 + fvz ** 2)
                input_tensor = torch.cat([mag, speed], dim=1)

                final_pred = generator(input_tensor)

            # Add to aggregator
            aggregator.add_batch(final_pred.cpu(), batch[tio.LOCATION])

        pred_tensor = aggregator.get_output_tensor()
        logger.info(f"Prediction stats: shape={pred_tensor.shape}, min={pred_tensor.min():.3f}, max={pred_tensor.max():.3f}, mean={pred_tensor.mean():.3f}")
        return pred_tensor

    def _prepare_multi_timepoint_input(self, batch: dict, device: torch.device) -> torch.Tensor:
        """Prepare multi-timepoint input tensor from batch.

        Args:
            batch: Batch dictionary from dataloader
            device: Device to move tensors to

        Returns:
            Input tensor of shape [B, 2*window_size, D, H, W]
        """
        mag_tensors = []
        speed_tensors = []

        for i in range(self.temporal_window_size):
            suffix = f'_t{i}'
            mag = batch[f'mag{suffix}'][tio.DATA].to(device)
            fvx = batch[f'flow_vx{suffix}'][tio.DATA].to(device)
            fvy = batch[f'flow_vy{suffix}'][tio.DATA].to(device)
            fvz = batch[f'flow_vz{suffix}'][tio.DATA].to(device)

            speed = torch.sqrt(fvx ** 2 + fvy ** 2 + fvz ** 2)
            mag_tensors.append(mag)
            speed_tensors.append(speed)

        all_mags = torch.cat(mag_tensors, dim=1)
        all_speeds = torch.cat(speed_tensors, dim=1)
        return torch.cat([all_mags, all_speeds], dim=1)
    
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

        # Use affine from appropriate source based on mode
        if self.use_multi_timepoint:
            affine_key = f"mag_t{self.center_idx}"
        else:
            affine_key = "mag"
        affine = subject[affine_key][tio.AFFINE] if affine_key in subject else torch.eye(4)

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

    def _load_subject_for_visualization(self, patient_id: str) -> tio.Subject:
        """Load a single subject on-demand for visualization.

        Args:
            patient_id: Patient identifier to load

        Returns:
            Loaded and transformed TorchIO Subject
        """
        # Build inference transforms (for velocity data, not precomputed speed)
        transforms = build_inference_transforms(
            self.cfg,
            multi_timepoint=self.use_multi_timepoint
        )

        # Load path config and create patient
        path_config = load_path_config(self.cfg.path_config.path_config_name)
        patient = Patient(
            path_config=path_config,
            phonetic_id=patient_id,
            debug=self.cfg.train.debug
        )

        # Ensure full-FOV per-timepoint files exist; build only if missing
        full_fov_dirs = [
            patient.flow_mag_per_timepoint_full_fov_dir,
            patient.flow_vx_per_timepoint_full_fov_dir,
            patient.flow_vy_per_timepoint_full_fov_dir,
            patient.flow_vz_per_timepoint_full_fov_dir,
        ]
        missing_full_fov = any(not any(d.glob("*.nii.gz")) for d in full_fov_dirs)
        if missing_full_fov:
            patient.build_4d_flow_per_timepoint_full_fov()

        # Determine time index
        time_index = self.cfg.train.validation_time_index

        # Build subject using appropriate function based on mode
        if self.use_multi_timepoint:
            subject = make_multi_timepoint_subject_full_fov(
                patient,
                center_time_index=time_index,
                window_size=self.temporal_window_size,
                transforms=transforms,
            )
        else:
            subject = make_subject_full_fov(
                patient,
                time_index,
                transforms=transforms,
            )

        return subject

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