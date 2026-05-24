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

import numpy as np

from .base_callback import Callback
from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.utils.path_config import load_path_config
from vascular_superenhancement.training.transforms import build_inference_transforms, build_downsampled_transforms
from vascular_superenhancement.training.datasets import make_downsampled_subject, make_downsampled_subject_inference, get_downsampled_mag_keys
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
        # ``log_images`` is the per-feature gate; ``enabled`` is the master
        # switch. Without both, calling ``wandb.log`` raises because
        # ``wandb.init`` was never called.
        self.visualization_log_to_wandb = bool(
            self.cfg.wandb.enabled and self.cfg.wandb.log_images
        )
        self.visualization_log_frequency = self.cfg.wandb.log_images_frequency

        # Trainer type (dual_task, generator, gan)
        self.trainer_type = cfg.train.get('trainer_type', 'gan')

        # Patch-based inference settings (not used in dual_task mode)
        self.patch_size = cfg.train.get('patch_size', None)
        self.patch_overlap = cfg.train.get('patch_overlap', 0)
        self.patch_aggregation_overlap_mode = cfg.train.get('patch_aggregation_overlap_mode', 'hann')

        # Multi-timepoint settings (GAN / generator modes)
        self.use_multi_timepoint = cfg.train.get('use_multi_timepoint', False)
        self.temporal_window_size = cfg.train.get('temporal_window_size', 5)
        self.center_idx = self.temporal_window_size // 2

        # Dual-task settings
        if self.trainer_type == 'dual_task':
            self.temporal_mag_offsets = list(cfg.train.temporal_mag_offsets)
            self.mag_keys = get_downsampled_mag_keys(self.temporal_mag_offsets)
            self.visualization_patient_ids = list(cfg.train.get('visualization_patient_ids', []))

        # Track subjects for visualization - store only metadata, not data
        self.visualization_patient_info: List[Tuple[str, bool]] = []  # List of (patient_id, is_inference_only)
        self.original_saved = False
        self.inference_cfg = self.cfg.train.get('validation_inference', None)

        logger.info("VisualizationCallback initialized:")
        logger.info(f"  - Trainer type: {self.trainer_type}")
        logger.info(f"  - Save frequency: every {self.save_frequency} epochs")
        logger.info(f"  - W&B visualization logging: {'enabled' if self.visualization_log_to_wandb else 'disabled'}")
        logger.info(f"  - Output directory: {self.output_dir}")
        if self.trainer_type == 'dual_task':
            logger.info(f"  - Dual-task mode: mag offsets={self.temporal_mag_offsets}")
        elif self.use_multi_timepoint:
            logger.info(f"  - Multi-timepoint mode: window_size={self.temporal_window_size}")
        
    def on_fit_start(self, trainer: 'BaseTrainer') -> None:
        """Select subjects for visualization at training start.
        
        Only collects patient IDs, does not load actual subject data to save memory.
        Subjects are loaded on-demand during visualization.
        """
        if not trainer.is_main_process:
            return

        self.visualization_patient_info = []

        if trainer.val_subjects is None:
            logger.warning("No validation subjects available for visualization")
        else:
            # Use dry_iter to get unique patient IDs without loading data
            seen_ids = set()
            count = 0
            for subject in trainer.val_dataset.dry_iter():
                if count >= self.num_samples and self.num_samples > 0:
                    break
                patient_id = getattr(subject, 'patient_id', None)
                if patient_id and patient_id not in seen_ids:
                    seen_ids.add(patient_id)
                    self.visualization_patient_info.append((patient_id, False))
                    count += 1

            logger.info(f"Selected {count} unique patients for visualization")
            for patient_id, _ in self.visualization_patient_info:
                logger.info(f"  - Patient {patient_id}")

        # Add visualization-only patient IDs (dual_task mode)
        if self.trainer_type == 'dual_task' and self.visualization_patient_ids:
            seen_ids = {pid for pid, _ in self.visualization_patient_info}
            added = 0
            for pid in self.visualization_patient_ids:
                if pid not in seen_ids:
                    self.visualization_patient_info.append((pid, True))
                    seen_ids.add(pid)
                    added += 1
                    logger.info(f"  - Visualization-only (inference) patient {pid}")
                else:
                    logger.info(f"  - Skipping {pid} (already selected from validation)")
            if added:
                logger.info(f"Added {added} visualization-only patients")

        if not self.visualization_patient_info:
            logger.warning("No subjects available for visualization after including inference-only patients")

    
    def on_validation_epoch_end(self, trainer: 'BaseTrainer', epoch: int,
                     metrics: Dict[str, float]) -> None:
        """Generate and save visualizations at epoch end.
        
        Loads subjects on-demand and releases them after processing to minimize memory usage.
        """
        if not trainer.is_main_process:
            return
        
        is_last_epoch = epoch == trainer.cfg.train.num_epochs - 1
        
        should_visualize = (
            epoch % self.save_frequency == 0 or
            epoch <= 10 or 
            is_last_epoch or 
            trainer.is_improving
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
                    subject = self._load_subject_for_visualization(patient_id, is_inference_only)
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
                        venc = subject["venc"] if "venc" in subject else None
                        image_key = f"validation/{patient_id}/center_slice"
                        wandb_images[image_key] = self._prepare_wandb_image(
                            prediction, patient_id, epoch, trainer.global_step, metrics, venc=venc
                        )
                        if self.trainer_type == 'dual_task' and not is_inference_only:
                            dd_key = f"validation_delta-delta/{patient_id}/delta_delta"
                            wandb_images[dd_key] = self._prepare_wandb_delta_delta(
                                prediction, subject, venc, epoch, trainer.global_step, metrics
                            )
                            gt_resid_key = f"validation_gt-residual/{patient_id}/gt_and_residual"
                            wandb_images[gt_resid_key] = self._prepare_wandb_gt_and_residual(
                                prediction, subject, venc, epoch, trainer.global_step, metrics
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

        Handles dual-task, multi-timepoint, and single-timepoint subjects.

        Args:
            subject: TorchIO subject containing images
            patient_id: Patient identifier
        """
        output_dir = self.output_dir / patient_id / "original"
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.trainer_type == 'dual_task':
            # Dual-task: save center mag, uncorrected vel, cine, gt corrections, cine mask
            images_to_save = {
                'mag_center': 'mag_center',
                'cine': 'cine',
                'cine_mask': 'cine_mask',
                'uncorrected_vx': 'uncorrected_vx',
                'uncorrected_vy': 'uncorrected_vy',
                'uncorrected_vz': 'uncorrected_vz',
                'gt_correction_vx': 'gt_correction_vx',
                'gt_correction_vy': 'gt_correction_vy',
                'gt_correction_vz': 'gt_correction_vz',
            }
            for key, prefix in images_to_save.items():
                if key in subject:
                    data = subject[key][tio.DATA]
                    affine = subject[key][tio.AFFINE]
                    path = output_dir / f"{prefix}_{patient_id}.nii.gz"
                    tio.ScalarImage(tensor=data, affine=affine).save(path)
                    logger.debug(f"Saved {prefix} to {path}")
            return

        # --- Existing modes (GAN / generator) ---
        if self.use_multi_timepoint:
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
                tio.ScalarImage(tensor=data, affine=affine).save(path)
                logger.debug(f"Saved {prefix} to {path}")

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
        """Generate prediction for a subject.

        In dual-task mode performs a single forward pass on the full volume.
        In patch-based modes uses GridSampler/GridAggregator.

        Args:
            subject: TorchIO subject
            generator: Generator model
            device: Device to run on

        Returns:
            Generated prediction tensor
        """
        if self.trainer_type == 'dual_task':
            return self._generate_prediction_dual_task(subject, generator, device)

        # ---------- Existing patch-based path ----------
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

        sampler = tio.inference.GridSampler(
            subject,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap
        )
        loader = torch.utils.data.DataLoader(sampler, batch_size=1)
        aggregator = tio.inference.GridAggregator(
            sampler,
            overlap_mode=self.patch_aggregation_overlap_mode
        )

        for batch in loader:
            if self.use_multi_timepoint:
                input_tensor = self._prepare_multi_timepoint_input(batch, device)
                prediction = generator(input_tensor)
                final_pred = prediction[:, self.center_idx:self.center_idx + 1, ...]
            else:
                mag = batch["mag"][tio.DATA].to(device)
                fvx = batch["flow_vx"][tio.DATA].to(device)
                fvy = batch["flow_vy"][tio.DATA].to(device)
                fvz = batch["flow_vz"][tio.DATA].to(device)
                speed = torch.sqrt(fvx ** 2 + fvy ** 2 + fvz ** 2)
                input_tensor = torch.cat([mag, speed], dim=1)
                final_pred = generator(input_tensor)

            aggregator.add_batch(final_pred.cpu(), batch[tio.LOCATION])

        pred_tensor = aggregator.get_output_tensor()
        logger.info(f"Prediction stats: shape={pred_tensor.shape}, min={pred_tensor.min():.3f}, max={pred_tensor.max():.3f}, mean={pred_tensor.mean():.3f}")
        return pred_tensor

    def _generate_prediction_dual_task(
        self,
        subject: tio.Subject,
        generator: torch.nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate dual-task prediction via a single forward pass on the full volume.

        Mirrors the input assembly logic of ``DualTaskTrainer.prepare_batch``
        but for a single subject (no batch dim from dataloader).

        Returns:
            Prediction tensor of shape ``[4, D, H, W]`` (1 cine + 3 corrections).
        """
        # Assemble magnitude channels
        mag_tensors = [subject[k][tio.DATA].to(device) for k in self.mag_keys]

        # Normalise velocity by VENC and clamp
        venc = torch.tensor(subject["venc"], dtype=torch.float32, device=device)
        vel_tensors = []
        for comp in ("vx", "vy", "vz"):
            vel = subject[f"uncorrected_{comp}"][tio.DATA].to(device)
            vel = (vel / venc).clamp(-1.0, 1.0)
            vel_tensors.append(vel)

        input_tensor = torch.cat(mag_tensors + vel_tensors, dim=0).unsqueeze(0)  # [1, C, D, H, W]

        logger.info(f"Dual-task input shape: {input_tensor.shape}")
        self._log_tensor_stats("mag_center", subject["mag_center"][tio.DATA])
        if "cine" in subject:
            self._log_tensor_stats("cine", subject["cine"][tio.DATA])

        pred = generator(input_tensor)  # [1, 4, D, H, W]
        pred_tensor = pred.squeeze(0).cpu()  # [4, D, H, W]
        logger.info(
            f"Dual-task prediction stats: shape={pred_tensor.shape}, "
            f"min={pred_tensor.min():.3f}, max={pred_tensor.max():.3f}, mean={pred_tensor.mean():.3f}"
        )
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
        """Save prediction as NIfTI file(s).

        In dual-task mode the 4-channel prediction is split into separate files:
        ``pred_mag_*``, ``pred_correction_vx_*``, ``pred_correction_vy_*``,
        ``pred_correction_vz_*``.

        Args:
            prediction: Prediction tensor ([C, D, H, W] for dual-task, [1, D, H, W] otherwise)
            subject: Original subject for affine matrix
            patient_id: Patient identifier
            epoch: Current epoch

        Returns:
            Path to saved file (or to the cine prediction in dual-task mode)
        """
        output_dir = self.output_dir / patient_id / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.trainer_type == 'dual_task':
            affine = subject["mag_center"][tio.AFFINE] if "mag_center" in subject else torch.eye(4)

            channel_names = ["mag", "correction_vx", "correction_vy", "correction_vz"]
            first_path = None
            for ch_idx, name in enumerate(channel_names):
                ch_path = output_dir / f"pred_{name}_{patient_id}_epoch_{epoch:04d}.nii.gz"
                ch_data = prediction[ch_idx:ch_idx + 1]  # [1, D, H, W]
                tio.ScalarImage(tensor=ch_data, affine=affine).save(ch_path)
                logger.info(f"Saved {name} prediction to {ch_path}")
                if first_path is None:
                    first_path = ch_path
            return first_path

        # --- Existing single-prediction path ---
        output_path = output_dir / f"pred_epoch_{epoch:04d}_{patient_id}.nii.gz"

        if self.use_multi_timepoint:
            affine_key = f"mag_t{self.center_idx}"
        else:
            affine_key = "mag"
        affine = subject[affine_key][tio.AFFINE] if affine_key in subject else torch.eye(4)

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
        metrics: Dict[str, float],
        venc: float = None,
    ) -> Any:
        """Prepare image for W&B logging.

        In dual-task mode creates a 4-panel image (cine + 3 corrections).
        Correction channels are scaled to physical units (cm/s) via VENC and
        displayed with a fixed [-300, 300] range for consistent comparison.
        
        Args:
            prediction: Prediction tensor
            patient_id: Patient identifier
            epoch: Current epoch
            global_step: Current global step
            metrics: Current metrics
            venc: Velocity encoding (cm/s) for scaling correction channels
            
        Returns:
            W&B Image object
        """
        z_middle = prediction.shape[-1] // 2

        if self.trainer_type == 'dual_task':
            panels = []
            for ch in range(min(4, prediction.shape[0])):
                panel = prediction[ch, :, :, z_middle].cpu().numpy()
                if ch == 0:
                    # Cine channel: normalise to [0, 1] for display
                    pmin, pmax = panel.min(), panel.max()
                    if pmax - pmin > 1e-8:
                        panel = (panel - pmin) / (pmax - pmin)
                else:
                    # Correction channels: scale to cm/s, map fixed [-300, 300] → [0, 1]
                    if venc is not None:
                        panel = panel * venc
                    panel = np.clip(panel, -300.0, 300.0)
                    panel = (panel + 300.0) / 600.0
                panels.append(panel)

            # Stack horizontally
            multi_panel = np.concatenate(panels, axis=1)  # [H, 4*W]

            caption = (
                f"e {epoch:04d}, g {global_step:04d}, p {patient_id}, z {z_middle}, "
                f"l1_cine {metrics.get('val/loss_cine_l1', 0):.4f}, "
                f"ssim_cine {metrics.get('val/loss_cine_ssim', 0):.4f}, "
                f"mse_corr {metrics.get('val/loss_correction_mse', 0):.4f}, "
                f"total {metrics.get('val/loss_generator', 0):.4f}"
            )
            return wandb.Image(multi_panel, caption=caption)

        # --- Existing single-panel path ---
        center_slice = prediction[0, :, :, z_middle].cpu().numpy()

        caption = (
            f"e {epoch:04d}, g {global_step:04d}, p {patient_id}, z {z_middle}, "
            f"g_gan {metrics.get('val/loss_generator_gan', 0):.4f}, "
            f"g_l1 {metrics.get('val/loss_generator_l1', 0):.4f}, "
            f"g_ssim {metrics.get('val/loss_generator_ssim', 0):.4f}, "
            f"g {metrics.get('val/loss_generator', 0):.4f}, "
            f"d {metrics.get('val/loss_discriminator', 0):.4f}"
        )
        
        return wandb.Image(center_slice, caption=caption)

    def _prepare_wandb_delta_delta(
        self,
        prediction: torch.Tensor,
        subject: tio.Subject,
        venc: float,
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
    ) -> Any:
        """Prepare CNN improvement (delta-delta) image for W&B logging.

        Computes ``|GT| - |GT - CNN_pred|`` per correction component at the
        centre z-slice.  Positive values mean the CNN prediction is closer to
        GT than zero-correction (helped); negative means it hurt.

        Uses a diverging RdBu_r colourmap with a fixed [-300, 300] cm/s scale.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        z_middle = prediction.shape[-1] // 2
        comps = ["vx", "vy", "vz"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        im = None
        vlim = 300.0

        for i, comp in enumerate(comps):
            gt = subject[f"gt_correction_{comp}"][tio.DATA][0, :, :, z_middle].cpu().numpy()
            cnn = prediction[i + 1, :, :, z_middle].cpu().numpy()

            # Both to physical units (cm/s)
            if venc is not None:
                cnn = cnn * venc

            improvement = np.abs(gt) - np.abs(gt - cnn)

            im = axes[i].imshow(
                improvement.T, origin="upper", cmap="RdBu_r",
                vmin=-vlim, vmax=vlim,
            )
            axes[i].set_title(f"|GT−uncorr|−|GT−CNN| {comp}")
            axes[i].axis("off")

        fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.04,
                      label="+CNN helped / −CNN hurt")
        fig.suptitle(
            f"e {epoch:04d}  g {global_step:04d}  "
            f"mse_corr {metrics.get('val/loss_correction_mse', 0):.4f}",
            fontsize=10,
        )

        wandb_img = wandb.Image(fig)
        plt.close(fig)
        return wandb_img

    def _prepare_wandb_gt_and_residual(
        self,
        prediction: torch.Tensor,
        subject: tio.Subject,
        venc: float,
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
    ) -> Any:
        """Prepare GT correction + signed residual image for W&B logging.

        Creates a 2x3 figure:
          - Top row: ground-truth correction field (cm/s) for vx, vy, vz
          - Bottom row: signed residual ``GT - CNN`` (cm/s) for vx, vy, vz

        Both rows use a fixed [-300, 300] cm/s scale with RdBu_r colourmap.
        As training progresses the bottom row should converge toward white.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        z_middle = prediction.shape[-1] // 2
        comps = ["vx", "vy", "vz"]
        vlim = 300.0

        fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
        im = None

        for i, comp in enumerate(comps):
            gt = subject[f"gt_correction_{comp}"][tio.DATA][0, :, :, z_middle].cpu().numpy()
            cnn = prediction[i + 1, :, :, z_middle].cpu().numpy()
            if venc is not None:
                cnn = cnn * venc

            residual = gt - cnn

            im = axes[0, i].imshow(
                gt.T, origin="upper", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
            )
            axes[0, i].set_title(f"GT correction {comp}")
            axes[0, i].axis("off")

            axes[1, i].imshow(
                residual.T, origin="upper", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
            )
            axes[1, i].set_title(f"Residual (GT−CNN) {comp}")
            axes[1, i].axis("off")

        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04, label="cm/s")
        fig.suptitle(
            f"e {epoch:04d}  g {global_step:04d}  "
            f"mse_corr {metrics.get('val/loss_correction_mse', 0):.4f}",
            fontsize=10,
        )

        wandb_img = wandb.Image(fig)
        plt.close(fig)
        return wandb_img

    def _load_subject_for_visualization(self, patient_id: str, is_inference_only: bool = False) -> tio.Subject:
        """Load a single subject on-demand for visualization.

        Branches based on ``self.trainer_type``:
          - ``"dual_task"``: uses ``make_downsampled_subject`` (or the inference-only
            variant when ``is_inference_only=True``) + ``build_downsampled_transforms``
          - others: uses the existing full-FOV inference subject loaders

        Args:
            patient_id: Patient identifier to load
            is_inference_only: If True, load only model inputs (no ground-truth
                targets).  Used for patients that lack cine / correction data.

        Returns:
            Loaded and transformed TorchIO Subject
        """
        path_config = load_path_config(self.cfg.path_config.path_config_name)
        patient = Patient(
            path_config=path_config,
            phonetic_id=patient_id,
            debug=self.cfg.train.debug,
        )
        time_index = self.cfg.train.validation_time_index

        if self.trainer_type == 'dual_task':
            # Build val transforms (no augmentation)
            transforms = build_downsampled_transforms(self.cfg, train=False)
            if is_inference_only:
                subject = make_downsampled_subject_inference(
                    patient,
                    center_time_index=time_index,
                    temporal_mag_offsets=self.temporal_mag_offsets,
                    downsampled_folder=self.cfg.data.downsampled_folder,
                )
            else:
                subject = make_downsampled_subject(
                    patient,
                    center_time_index=time_index,
                    temporal_mag_offsets=self.temporal_mag_offsets,
                    downsampled_folder=self.cfg.data.downsampled_folder,
                )
            # Apply transforms
            subject = transforms(subject)
            return subject

        # --- Existing patch-based modes ---
        transforms = build_inference_transforms(
            self.cfg,
            multi_timepoint=self.use_multi_timepoint,
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