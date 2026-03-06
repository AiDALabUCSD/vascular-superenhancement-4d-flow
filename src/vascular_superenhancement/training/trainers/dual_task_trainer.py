"""Dual-task trainer for simultaneous cine enhancement and phase error correction.

This trainer operates on full downsampled volumes (128x128x64) without
patching, producing a 4-channel output: 1 enhanced cine channel + 3 phase
error correction channels (vx, vy, vz).

Losses:
  - Cine (inside mask): masked L1 + bounding-box SSIM (masked by cine_mask)
  - Cine (outside mask): L1 vs input magnitude (passthrough regulariser)
  - Correction: tissue-masked MSE (magnitude-thresholded to exclude air)
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torchio as tio
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from omegaconf import DictConfig
import logging
from torchinfo import summary

from .base_trainer import BaseTrainer
from ..callbacks.base_callback import Callback
from ..model_factory import build_generator
from ..losses import masked_l1_loss, bbox_ssim_loss, tissue_masked_mse_loss, outside_mask_l1_loss, masked_sobel_loss
from ..datasets import get_downsampled_mag_keys

logger = logging.getLogger(__name__)


class DualTaskTrainer(BaseTrainer):
    """Trainer for dual-task full-volume training.

    Combines vascular superenhancement (mag -> cine) with phase error
    correction (velocity -> correction fields) in a single shared-backbone
    model with two output heads.
    """

    def __init__(
        self,
        cfg: DictConfig,
        train_loader,
        train_dataset: Optional[tio.SubjectsDataset] = None,
        val_loader=None,
        val_dataset: Optional[tio.SubjectsDataset] = None,
        test_loader=None,
        test_dataset: Optional[tio.SubjectsDataset] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        super().__init__(
            cfg=cfg,
            train_loader=train_loader,
            train_dataset=train_dataset,
            val_loader=val_loader,
            val_dataset=val_dataset,
            test_loader=test_loader,
            test_dataset=test_dataset,
            callbacks=callbacks,
        )

        # Loss weights
        self.lambda_l1_cine = cfg.model.generator.weights.l1_cine
        self.lambda_ssim_cine = cfg.model.generator.weights.ssim_cine
        self.lambda_mse_correction = cfg.model.generator.weights.mse_correction
        self.lambda_outside_mask = cfg.model.generator.weights.l1_outside_mask
        self.lambda_sobel_cine = cfg.model.generator.weights.sobel_cine

        # Temporal offset config (for ordering mag channels)
        self.temporal_mag_offsets = list(cfg.train.temporal_mag_offsets)
        self.mag_keys = get_downsampled_mag_keys(self.temporal_mag_offsets)

        self.use_amp = cfg.train.get("use_amp", False) and torch.cuda.is_available()
        self.scaler = GradScaler("cuda") if self.use_amp else None
        if self.use_amp:
            logger.info("Mixed precision (AMP) enabled")

        logger.info("DualTaskTrainer initialized:")
        logger.info(f"  - Mag keys: {self.mag_keys}")
        logger.info(f"  - L1 cine weight: {self.lambda_l1_cine}")
        logger.info(f"  - SSIM cine weight: {self.lambda_ssim_cine}")
        logger.info(f"  - Sobel cine weight: {self.lambda_sobel_cine}")
        logger.info(f"  - MSE correction weight: {self.lambda_mse_correction}")
        logger.info(f"  - L1 outside-mask weight: {self.lambda_outside_mask}")

    # ------------------------------------------------------------------
    # Model / optimiser / scheduler setup
    # ------------------------------------------------------------------

    def build_models(self) -> Dict[str, nn.Module]:
        """Build the dual-head generator model."""
        models = {"generator": build_generator(self.cfg)}

        num_params = sum(p.numel() for p in models["generator"].parameters())
        logger.info(f"Built dual-head generator with {num_params:,} parameters")

        input_channels = self.cfg.model.generator.in_channels
        logger.info(f"Generator input channels: {input_channels}")
        logger.info(
            f"Generator output: {self.cfg.model.generator.cine_out_channels} cine + "
            f"{self.cfg.model.generator.correction_out_channels} correction"
        )

        try:
            summary_str = summary(
                models["generator"],
                input_size=(self.cfg.train.batch_size, input_channels, 128, 128, 64),
                depth=10,
                verbose=0,
            )
            logger.info(f"Generator summary:\n{summary_str}")
        except Exception as e:
            logger.warning(f"Could not generate model summary: {e}")

        return models

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Build Adam optimiser for the generator."""
        return {
            "generator": torch.optim.Adam(
                self.models["generator"].parameters(),
                lr=self.cfg.model.generator.optimizer.lr,
                betas=tuple(self.cfg.model.generator.optimizer.betas),
            )
        }

    def build_schedulers(self) -> Dict[str, Any]:
        """Build LR schedulers if configured."""
        schedulers = {}
        if self.cfg.train.get("use_lr_scheduler", False):

            def lambda_rule(epoch):
                start_decay = self.cfg.train.get("lr_decay_start", 100)
                if epoch < start_decay:
                    return 1.0
                return 1.0 - max(0, epoch - start_decay) / float(
                    self.cfg.train.num_epochs - start_decay
                )

            schedulers["generator"] = torch.optim.lr_scheduler.LambdaLR(
                self.optimizers["generator"], lr_lambda=lambda_rule
            )
        return schedulers

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def prepare_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Assemble the input / target tensors from a TorchIO batch.

        Steps:
          1. Concatenate magnitude images in offset order (already [0,1]).
          2. Normalise velocity by per-patient VENC → [-1, 1] then clamp.
          3. Concatenate mags + velocities into the model input.
          4. Extract cine target, correction target, and cine mask.

        Returns:
            Dictionary with keys ``input``, ``cine_target``,
            ``correction_target``, ``cine_mask``, ``mag_center``.
        """
        # -- Mags (already normalised [0,1] by transforms) -----------------
        mag_tensors = [batch[k][tio.DATA].to(self.device) for k in self.mag_keys]

        # -- Per-patient VENC normalisation of velocity --------------------
        # TorchIO collates scalar metadata as a plain list, not a tensor
        venc = torch.tensor(batch["venc"], dtype=torch.float32, device=self.device)  # [B]
        venc = venc.view(-1, 1, 1, 1, 1)     # [B, 1, 1, 1, 1] for broadcasting

        vel_tensors = []
        for comp in ("vx", "vy", "vz"):
            vel = batch[f"uncorrected_{comp}"][tio.DATA].to(self.device)
            vel = (vel / venc).clamp(-1.0, 1.0)
            vel_tensors.append(vel)

        # -- Build input tensor [B, num_mag + 3, D, H, W] -----------------
        input_tensor = torch.cat(mag_tensors + vel_tensors, dim=1)

        # -- Centre magnitude (for outside-mask loss target) ---------------
        mag_center = mag_tensors[self.mag_keys.index("mag_center")]  # [B, 1, D, H, W]

        # -- Targets -------------------------------------------------------
        cine_target = batch["cine"][tio.DATA].to(self.device)       # [B, 1, D, H, W]
        cine_mask = batch["cine_mask"][tio.DATA].to(self.device)    # [B, 1, D, H, W]

        correction_targets = [
            batch[f"gt_correction_{c}"][tio.DATA].to(self.device) for c in ("vx", "vy", "vz")
        ]
        correction_target = torch.cat(correction_targets, dim=1)    # [B, 3, D, H, W]

        return {
            "input": input_tensor,
            "cine_target": cine_target,
            "correction_target": correction_target,
            "cine_mask": cine_mask,
            "mag_center": mag_center,
        }

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        data = self.prepare_batch(batch)
        input_tensor = data["input"]
        cine_target = data["cine_target"]
        correction_target = data["correction_target"]
        cine_mask = data["cine_mask"]
        mag_center = data["mag_center"]

        self.optimizers["generator"].zero_grad()

        # Forward pass → [B, 4, D, H, W] (with autocast when AMP enabled)
        with autocast("cuda", enabled=self.use_amp):
            pred = self.models["generator"](input_tensor)
            pred_cine = pred[:, 0:1]
            pred_correction = pred[:, 1:4]

            # --- Cine losses (masked) ---
            loss_l1_cine_uw = masked_l1_loss(pred_cine, cine_target, cine_mask)
            loss_ssim_cine_uw = bbox_ssim_loss(pred_cine, cine_target, cine_mask)
            loss_sobel_cine_uw = masked_sobel_loss(pred_cine, cine_target, cine_mask)
            loss_l1_cine = self.lambda_l1_cine * loss_l1_cine_uw
            loss_ssim_cine = self.lambda_ssim_cine * loss_ssim_cine_uw
            loss_sobel_cine = self.lambda_sobel_cine * loss_sobel_cine_uw

            # --- Outside-mask loss (mag passthrough) ---
            loss_outside_uw = outside_mask_l1_loss(pred_cine, mag_center, cine_mask)
            loss_outside = self.lambda_outside_mask * loss_outside_uw

            # --- Correction loss (tissue-masked MSE) ---
            loss_mse_corr_uw = tissue_masked_mse_loss(pred_correction, correction_target, mag_center)
            loss_mse_corr = self.lambda_mse_correction * loss_mse_corr_uw

            # --- Total ---
            loss_total = loss_l1_cine + loss_ssim_cine + loss_sobel_cine + loss_outside + loss_mse_corr

        if self.scaler is not None:
            self.scaler.scale(loss_total).backward()
            self.scaler.unscale_(self.optimizers["generator"])
            torch.nn.utils.clip_grad_norm_(self.models["generator"].parameters(), max_norm=1.0)
            self.scaler.step(self.optimizers["generator"])
            self.scaler.update()
        else:
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.models["generator"].parameters(), max_norm=1.0)
            self.optimizers["generator"].step()

        logger.info(
            f"e {self.current_epoch:04d}, b {batch_idx:04d}, g {self.global_step:04d}: "
            f"l1_cine {loss_l1_cine.item():.4f}, ssim_cine {loss_ssim_cine.item():.4f}, "
            f"sobel_cine {loss_sobel_cine.item():.4f}, outside {loss_outside.item():.4f}, "
            f"mse_corr {loss_mse_corr.item():.4f}, total {loss_total.item():.4f}"
        )

        return {
            "loss_generator": loss_total,
            "loss_cine_l1": loss_l1_cine,
            "loss_cine_ssim": loss_ssim_cine,
            "loss_cine_sobel": loss_sobel_cine,
            "loss_outside_mask": loss_outside,
            "loss_correction_mse": loss_mse_corr,
            "loss_cine_l1_unweighted": loss_l1_cine_uw,
            "loss_cine_ssim_unweighted": loss_ssim_cine_uw,
            "loss_cine_sobel_unweighted": loss_sobel_cine_uw,
            "loss_outside_mask_unweighted": loss_outside_uw,
            "loss_correction_mse_unweighted": loss_mse_corr_uw,
        }

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        data = self.prepare_batch(batch)
        input_tensor = data["input"]
        cine_target = data["cine_target"]
        correction_target = data["correction_target"]
        cine_mask = data["cine_mask"]
        mag_center = data["mag_center"]

        with autocast("cuda", enabled=self.use_amp):
            pred = self.models["generator"](input_tensor)
            pred_cine = pred[:, 0:1]
            pred_correction = pred[:, 1:4]

            loss_l1_cine_uw = masked_l1_loss(pred_cine, cine_target, cine_mask)
            loss_ssim_cine_uw = bbox_ssim_loss(pred_cine, cine_target, cine_mask)
            loss_sobel_cine_uw = masked_sobel_loss(pred_cine, cine_target, cine_mask)
            loss_l1_cine = self.lambda_l1_cine * loss_l1_cine_uw
            loss_ssim_cine = self.lambda_ssim_cine * loss_ssim_cine_uw
            loss_sobel_cine = self.lambda_sobel_cine * loss_sobel_cine_uw

            loss_outside_uw = outside_mask_l1_loss(pred_cine, mag_center, cine_mask)
            loss_outside = self.lambda_outside_mask * loss_outside_uw

            loss_mse_corr_uw = tissue_masked_mse_loss(pred_correction, correction_target, mag_center)
            loss_mse_corr = self.lambda_mse_correction * loss_mse_corr_uw

            loss_total = loss_l1_cine + loss_ssim_cine + loss_sobel_cine + loss_outside + loss_mse_corr

        return {
            "loss_generator": loss_total,
            "loss_cine_l1": loss_l1_cine,
            "loss_cine_ssim": loss_ssim_cine,
            "loss_cine_sobel": loss_sobel_cine,
            "loss_outside_mask": loss_outside,
            "loss_correction_mse": loss_mse_corr,
            "loss_cine_l1_unweighted": loss_l1_cine_uw,
            "loss_cine_ssim_unweighted": loss_ssim_cine_uw,
            "loss_cine_sobel_unweighted": loss_sobel_cine_uw,
            "loss_outside_mask_unweighted": loss_outside_uw,
            "loss_correction_mse_unweighted": loss_mse_corr_uw,
        }
