"""Generator-only trainer for multi-timepoint temporal training.

This trainer trains only the generator network without a discriminator,
using L1 and SSIM losses for reconstruction. It supports multi-timepoint
input/output where temporal context from neighboring frames improves
prediction consistency.
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torchio as tio
from omegaconf import DictConfig
import logging
from torchinfo import summary

from .base_trainer import BaseTrainer
from ..callbacks.base_callback import Callback
from ..model_factory import build_generator
from ..losses import generator_l1_loss, generator_ssim_loss

logger = logging.getLogger(__name__)


class GeneratorTrainer(BaseTrainer):
    """Trainer for generator-only training without discriminator.

    This trainer is designed for multi-timepoint temporal training where:
    - Input: Multiple timepoints of mag + speed data (e.g., 5 timepoints = 10 channels)
    - Output: Multiple timepoints of cine predictions (e.g., 5 channels)
    - Loss: L1 + SSIM reconstruction losses

    The temporal context from neighboring frames helps produce more consistent
    predictions across timepoints compared to single-frame training.
    """

    def __init__(
        self,
        cfg: DictConfig,
        train_loader: tio.SubjectsLoader,
        train_dataset: Optional[tio.SubjectsDataset] = None,
        val_loader: Optional[tio.SubjectsLoader] = None,
        val_dataset: Optional[tio.SubjectsDataset] = None,
        test_loader: Optional[tio.SubjectsLoader] = None,
        test_dataset: Optional[tio.SubjectsDataset] = None,
        callbacks: Optional[List[Callback]] = None
    ):
        """Initialize the generator trainer.

        Args:
            cfg: Hydra configuration
            train_loader: Training data loader
            train_dataset: Training dataset
            val_loader: Validation data loader
            val_dataset: Validation dataset
            test_loader: Test data loader
            test_dataset: Test dataset
            callbacks: List of callback objects
        """
        super().__init__(
            cfg=cfg,
            train_loader=train_loader,
            train_dataset=train_dataset,
            val_loader=val_loader,
            val_dataset=val_dataset,
            test_loader=test_loader,
            test_dataset=test_dataset,
            callbacks=callbacks
        )

        # Loss weights
        self.lambda_l1 = cfg.model.generator.weights.l1
        self.lambda_ssim = cfg.model.generator.weights.ssim

        # Temporal window configuration
        self.window_size = cfg.train.get('temporal_window_size', 5)
        self.center_idx = self.window_size // 2

        logger.info(f"GeneratorTrainer initialized with temporal window size: {self.window_size}")
        logger.info(f"  - Center index: {self.center_idx}")
        logger.info(f"  - L1 weight: {self.lambda_l1}")
        logger.info(f"  - SSIM weight: {self.lambda_ssim}")

    def build_models(self) -> Dict[str, nn.Module]:
        """Build generator model only (no discriminator)."""
        models = {
            'generator': build_generator(self.cfg)
        }

        num_params = sum(p.numel() for p in models['generator'].parameters())
        logger.info(f"Built generator with {num_params:,} parameters")

        # Log model summary
        input_channels = self.cfg.model.generator.in_channels
        patch_size = self.cfg.train.patch_size
        logger.info(f"Generator input shape: [B, {input_channels}, {patch_size[0]}, {patch_size[1]}, {patch_size[2]}]")
        logger.info(f"Generator output channels: {self.cfg.model.generator.out_channels}")

        try:
            summary_str = summary(
                models['generator'],
                input_size=(self.cfg.train.batch_size, input_channels, *patch_size),
                depth=10,
                verbose=0
            )
            logger.info(f"Generator summary:\n{summary_str}")
        except Exception as e:
            logger.warning(f"Could not generate model summary: {e}")

        return models

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Build optimizer for generator only."""
        return {
            'generator': torch.optim.Adam(
                self.models['generator'].parameters(),
                lr=self.cfg.model.generator.optimizer.lr,
                betas=tuple(self.cfg.model.generator.optimizer.betas)
            )
        }

    def prepare_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Prepare multi-timepoint batch for training.

        Concatenates mag and precomputed speed from all timepoints into a single input tensor.
        The ordering is: [mag_t0, mag_t1, ..., mag_tN, speed_t0, speed_t1, ..., speed_tN]

        Args:
            batch: Raw batch from dataloader containing multi-timepoint data

        Returns:
            Dictionary with:
                - 'input': Concatenated input tensor [B, 2*window_size, D, H, W]
                - 'target': Concatenated target tensor [B, window_size, D, H, W]
                - 'batch_info': Original batch data for callbacks
        """
        mag_tensors = []
        speed_tensors = []
        cine_tensors = []

        for i in range(self.window_size):
            suffix = f'_t{i}'

            # Get mag and precomputed speed for this timepoint
            mag = batch[f'mag{suffix}'][tio.DATA].to(self.device)
            speed = batch[f'speed{suffix}'][tio.DATA].to(self.device)

            mag_tensors.append(mag)
            speed_tensors.append(speed)

            # Get cine target if available
            cine_key = f'cine{suffix}'
            if cine_key in batch:
                cine = batch[cine_key][tio.DATA].to(self.device)
                cine_tensors.append(cine)

        # Concatenate all tensors in one step to avoid intermediate allocations
        # Each tensor is [B, 1, D, H, W], result is [B, 2*window_size, D, H, W]
        # Ordering: [mag_t0, ..., mag_tN, speed_t0, ..., speed_tN]
        input_tensor = torch.cat(mag_tensors + speed_tensors, dim=1)

        # Target: [B, window_size, D, H, W] if cine available
        target = torch.cat(cine_tensors, dim=1) if cine_tensors else None

        return {
            'input': input_tensor,
            'target': target,
            'batch_info': batch
        }

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Execute one training step.

        Args:
            batch: Input batch from dataloader
            batch_idx: Index of current batch

        Returns:
            Dictionary containing losses and outputs
        """
        data = self.prepare_batch(batch)
        input_tensor = data['input']
        target = data['target']

        self.optimizers['generator'].zero_grad()

        # Forward pass
        pred = self.models['generator'](input_tensor)  # [B, window_size, D, H, W]

        # Compute losses across all output channels
        loss_l1_unweighted = generator_l1_loss(pred, target)
        loss_ssim_unweighted = generator_ssim_loss(pred, target)

        loss_l1 = self.lambda_l1 * loss_l1_unweighted
        loss_ssim = self.lambda_ssim * loss_ssim_unweighted
        loss_total = loss_l1 + loss_ssim

        loss_total.backward()
        self.optimizers['generator'].step()

        logger.info(
            f"e {self.current_epoch:04d}, b {batch_idx:04d}, g {self.global_step:04d}: "
            f"l1 {loss_l1.item():.4f}, ssim {loss_ssim.item():.4f}, total {loss_total.item():.4f}"
        )

        return {
            'loss_generator': loss_total,
            'loss_generator_l1': loss_l1,
            'loss_generator_ssim': loss_ssim,
            'loss_generator_l1_unweighted': loss_l1_unweighted,
            'loss_generator_ssim_unweighted': loss_ssim_unweighted,
            'generated': pred.detach(),
            'input': input_tensor,
            'target': target,
            'batch_info': data['batch_info']
        }

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Execute one validation step.

        Args:
            batch: Input batch from dataloader
            batch_idx: Index of current batch

        Returns:
            Dictionary containing losses and outputs
        """
        data = self.prepare_batch(batch)
        input_tensor = data['input']
        target = data['target']

        # Forward pass (no gradients)
        pred = self.models['generator'](input_tensor)

        # Compute losses
        loss_l1_unweighted = generator_l1_loss(pred, target)
        loss_ssim_unweighted = generator_ssim_loss(pred, target)

        loss_l1 = self.lambda_l1 * loss_l1_unweighted
        loss_ssim = self.lambda_ssim * loss_ssim_unweighted
        loss_total = loss_l1 + loss_ssim

        return {
            'loss_generator': loss_total,
            'loss_generator_l1': loss_l1,
            'loss_generator_ssim': loss_ssim,
            'loss_generator_l1_unweighted': loss_l1_unweighted,
            'loss_generator_ssim_unweighted': loss_ssim_unweighted,
            'generated': pred.detach(),
            'input': input_tensor,
            'target': target,
            'batch_info': data['batch_info']
        }

    def build_schedulers(self) -> Dict[str, Any]:
        """Build learning rate schedulers if configured."""
        schedulers = {}

        if self.cfg.train.get('use_lr_scheduler', False):
            def lambda_rule(epoch):
                start_decay = self.cfg.train.get('lr_decay_start', 100)
                if epoch < start_decay:
                    return 1.0
                else:
                    return 1.0 - max(0, epoch - start_decay) / float(self.cfg.train.num_epochs - start_decay)

            schedulers['generator'] = torch.optim.lr_scheduler.LambdaLR(
                self.optimizers['generator'], lr_lambda=lambda_rule
            )

        return schedulers

    def get_center_prediction(self, pred: torch.Tensor) -> torch.Tensor:
        """Extract the center timepoint prediction from multi-channel output.

        Args:
            pred: Prediction tensor of shape [B, window_size, D, H, W]

        Returns:
            Center timepoint prediction of shape [B, 1, D, H, W]
        """
        return pred[:, self.center_idx:self.center_idx + 1, ...]

    def get_weighted_prediction(
        self,
        pred: torch.Tensor,
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Get weighted average prediction from multi-channel output.

        Args:
            pred: Prediction tensor of shape [B, window_size, D, H, W]
            weights: Optional weights for each channel. If None, uses
                    triangular weights centered on the middle channel.

        Returns:
            Weighted average prediction of shape [B, 1, D, H, W]
        """
        if weights is None:
            # Triangular weights: [0.1, 0.2, 0.4, 0.2, 0.1] for window_size=5
            half = self.window_size // 2
            weights = []
            for i in range(self.window_size):
                dist = abs(i - half)
                w = 1.0 / (dist + 1)
                weights.append(w)
            # Normalize
            total = sum(weights)
            weights = [w / total for w in weights]

        # Apply weights
        weights_tensor = torch.tensor(weights, device=pred.device, dtype=pred.dtype)
        weights_tensor = weights_tensor.view(1, self.window_size, 1, 1, 1)

        weighted = (pred * weights_tensor).sum(dim=1, keepdim=True)
        return weighted
