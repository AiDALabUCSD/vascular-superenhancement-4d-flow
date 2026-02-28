"""Checkpoint management callback."""

from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING
import torch
import torch.nn as nn
import logging
from omegaconf import DictConfig
# #region agent log
import gc
import json
import psutil
import time
DEBUG_LOG_PATH = "/home/ayeluru/vascular-superenhancement-4d-flow/.cursor/debug.log"
def _debug_log_ckpt(location: str, message: str, hypothesis_id: str, extra_data: dict = None):
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
        entry = {"timestamp": int(time.time() * 1000), "location": location, "message": message, "hypothesisId": hypothesis_id, "data": data, "sessionId": "debug-session"}
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
# #endregion

# Avoid circular imports
if TYPE_CHECKING:
    from ..trainers.base_trainer import BaseTrainer

from .base_callback import Callback

logger = logging.getLogger(__name__)


class CheckpointCallback(Callback):
    """Callback for saving model checkpoints during training."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize checkpoint callback.
        
        Args:
            cfg: Hydra configuration
        """
        self.cfg = cfg
        
        # Checkpoint settings from config
        self.checkpoint_dir = Path.cwd() / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_checkpoint_dir = Path.cwd() / "best_checkpoints"
        self.best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_interval = cfg.train.get('checkpoint_interval', 5)
        
        logger.info("CheckpointCallback initialized:")
        logger.info(f"  - Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"  - Best checkpoint directory: {self.best_checkpoint_dir}")
        logger.info(f"  - Checkpoint interval: every {self.checkpoint_interval} epochs")
    
    def on_validation_epoch_end(self, trainer: 'BaseTrainer', epoch: int, 
                                metrics: Dict[str, float]) -> None:
        """Save checkpoints at end of validation epoch."""
        
        # Save best checkpoint if the monitored metric improved
        if trainer.is_improving:
            current_metric = metrics[trainer.monitor_metric]
            logger.info(
                f"New best {trainer.monitor_metric}: {current_metric:.6f} — "
                f"saving best checkpoint for epoch {epoch}"
            )
            
            checkpoint_path = self.best_checkpoint_dir / f"best_epoch_{epoch:04d}.pt"
            self._save_checkpoint(trainer, epoch, metrics, checkpoint_path, is_best=True)
            
            # Also save as "latest_best.pt" for easy loading
            latest_best_path = self.best_checkpoint_dir / "latest_best.pt"
            self._save_checkpoint(trainer, epoch, metrics, latest_best_path, is_best=True)
        
        # Save regular checkpoint at intervals or last epoch
        is_last_epoch = (epoch == trainer.cfg.train.num_epochs - 1)
        if epoch % self.checkpoint_interval == 0 or is_last_epoch:
            logger.info(f"Saving regular checkpoint for epoch {epoch}")
            checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
            self._save_checkpoint(trainer, epoch, metrics, checkpoint_path, is_best=False)
    
    def _save_checkpoint(
        self, 
        trainer: 'BaseTrainer',
        epoch: int,
        metrics: Dict[str, float],
        checkpoint_path: Path,
        is_best: bool = False
    ) -> None:
        """Save a checkpoint.
        
        Args:
            trainer: Trainer instance with models and optimizers
            epoch: Current epoch number
            metrics: Current metrics
            checkpoint_path: Path where to save checkpoint
            is_best: Whether this is the best checkpoint
        """
        # #region agent log
        _debug_log_ckpt(f"checkpoint_callback.py:epoch_{epoch}_ckpt_start", f"Checkpoint save starting", "D", {"epoch": epoch, "is_best": is_best})
        # #endregion
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'global_step': trainer.global_step,
            'best_val_metric': trainer.best_val_metric,
        }
        
        # Add all validation metrics
        for key, value in metrics.items():
            checkpoint[key] = value
        
        # Save model states (use .module for DataParallel to get clean keys for inference)
        for name, model in trainer.models.items():
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            checkpoint[f'{name}_state_dict'] = state_dict
        
        # Save optimizer states
        for name, optimizer in trainer.optimizers.items():
            checkpoint[f'optimizer_{name}_state_dict'] = optimizer.state_dict()
        
        # Save scheduler states if they exist
        if trainer.schedulers:
            for name, scheduler in trainer.schedulers.items():
                checkpoint[f'scheduler_{name}_state_dict'] = scheduler.state_dict()
        
        # Add best flag
        if is_best:
            checkpoint['is_best'] = True
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        # #region agent log
        del checkpoint
        gc.collect()
        _debug_log_ckpt(f"checkpoint_callback.py:epoch_{epoch}_ckpt_end", f"Checkpoint saved and cleaned", "D", {"epoch": epoch, "is_best": is_best})
        # #endregion
        logger.info(f"Saved checkpoint to {checkpoint_path}")