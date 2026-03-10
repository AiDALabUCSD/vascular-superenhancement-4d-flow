"""Checkpoint management callback."""

from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
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
        
        # Multi-metric best checkpoint tracking
        # Each entry is a metric key (e.g. "val/loss_cine_l1") to independently
        # track and save best checkpoints for.
        self.monitor_metrics: List[str] = list(cfg.train.get('checkpoint_monitor_metrics', []))
        self.best_per_metric: Dict[str, float] = {m: float('inf') for m in self.monitor_metrics}
        
        logger.info("CheckpointCallback initialized:")
        logger.info(f"  - Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"  - Best checkpoint directory: {self.best_checkpoint_dir}")
        logger.info(f"  - Checkpoint interval: every {self.checkpoint_interval} epochs")
        if self.monitor_metrics:
            logger.info(f"  - Per-metric best checkpoints: {self.monitor_metrics}")
    
    def on_validation_epoch_end(self, trainer: 'BaseTrainer', epoch: int, 
                                metrics: Dict[str, float]) -> None:
        """Save checkpoints at end of validation epoch."""
        if not trainer.is_main_process:
            return
        
        # Save best checkpoint if the primary monitored metric improved
        if trainer.is_improving:
            current_metric = metrics[trainer.monitor_metric]
            logger.info(
                f"New best {trainer.monitor_metric}: {current_metric:.6f} — "
                f"saving best checkpoint for epoch {epoch}"
            )
            
            latest_best_path = self.best_checkpoint_dir / "latest_best.pt"
            self._save_checkpoint(trainer, epoch, metrics, latest_best_path, is_best=True)
        
        # Per-metric best checkpoints: check which metrics improved, save one
        # checkpoint with all improvement info encoded in the filename.
        if self.monitor_metrics:
            improved = []
            for metric_key in self.monitor_metrics:
                if metric_key not in metrics:
                    continue
                current = metrics[metric_key]
                if current < self.best_per_metric[metric_key]:
                    self.best_per_metric[metric_key] = current
                    improved.append(metric_key)

            if improved:
                # Build filename: epoch + marker per metric (Y=improved, N=not)
                parts = []
                for metric_key in self.monitor_metrics:
                    short = metric_key.split("/")[-1]
                    marker = "Y" if metric_key in improved else "N"
                    parts.append(f"{short}={marker}")
                tag = "_".join(parts)
                ckpt_path = self.best_checkpoint_dir / f"e{epoch:04d}_{tag}.pt"
                self._save_checkpoint(trainer, epoch, metrics, ckpt_path, is_best=True)
                for m in improved:
                    logger.info(f"New best {m}: {metrics[m]:.6f} at epoch {epoch}")
                logger.info(f"Saved multi-metric best checkpoint: {ckpt_path.name}")
        
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
        
        for name, model in trainer.models.items():
            target = model.module if isinstance(model, (nn.DataParallel, DDP)) else model
            checkpoint[f'{name}_state_dict'] = target.state_dict()
        
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