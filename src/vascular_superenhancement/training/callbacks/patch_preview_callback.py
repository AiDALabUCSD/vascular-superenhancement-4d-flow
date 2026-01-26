"""Patch preview callback for dual-panel pre|post augmentation logging."""
from typing import Any, Dict, TYPE_CHECKING
import logging
import numpy as np
import torchio as tio

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

if TYPE_CHECKING:
    from ..trainers.base_trainer import BaseTrainer

from .base_callback import Callback

logger = logging.getLogger(__name__)


class PatchPreviewCallback(Callback):
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = bool(getattr(cfg.wandb, "enabled", False)) and bool(
            getattr(cfg.wandb, "log_patch_previews", False)
        )
        self.max_patches = int(getattr(cfg.wandb, "patch_preview_count", 4))

        # Multi-timepoint settings
        self.use_multi_timepoint = cfg.train.get('use_multi_timepoint', False)
        self.temporal_window_size = cfg.train.get('temporal_window_size', 5)
        self.center_idx = self.temporal_window_size // 2
        
        # Track last logged epoch to ensure once-per-epoch logging
        self._last_logged_epoch = -1

    def on_train_batch_end(self, trainer: "BaseTrainer", batch: Any, batch_idx: int,
                           outputs: Dict[str, Any]) -> None:
        # Log once per effective epoch (not once per batch_idx==0)
        if not self.enabled or not WANDB_AVAILABLE:
            return
        
        # Skip if we already logged for this epoch
        if trainer.current_epoch == self._last_logged_epoch:
            return
        self._last_logged_epoch = trainer.current_epoch

        # Determine the mag key based on mode
        if self.use_multi_timepoint:
            mag_key = f"mag_t{self.center_idx}"
        else:
            mag_key = "mag"

        if mag_key not in batch or tio.DATA not in batch[mag_key]:
            return

        post = batch[mag_key][tio.DATA]  # [B, 1, D, H, W]
        pre = None
        if "mag_pre_aug" in batch and isinstance(batch["mag_pre_aug"], dict):
            pre = batch["mag_pre_aug"].get(tio.DATA, None)

        bsz = post.shape[0]
        count = min(bsz, self.max_patches)
        zmid = post.shape[-1] // 2

        images = {}
        for i in range(count):
            post_slice = post[i, 0, :, :, zmid].detach().cpu().numpy()
            if pre is not None:
                pre_slice = pre[i, 0, :, :, zmid].detach().cpu().numpy()
            else:
                pre_slice = np.zeros_like(post_slice)

            pre_slice = np.clip(pre_slice, 0, 1)
            post_slice = np.clip(post_slice, 0, 1)

            dual = np.concatenate([pre_slice, post_slice], axis=1)
            images[f"training_data_patch_preview/epoch_{trainer.current_epoch}_sample{i}_z{zmid}"] = wandb.Image(
                dual, caption=f"epoch {trainer.current_epoch} step {trainer.global_step} sample {i}"
            )

        if images:
            wandb.log(images, step=trainer.global_step)
