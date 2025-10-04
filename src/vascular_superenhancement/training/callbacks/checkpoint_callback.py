"""Checkpoint management callback."""

from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING
import torch
import logging
from omegaconf import DictConfig

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
        
        # Check if this is an improving epoch using moving average logic
        current_metric = metrics[trainer.monitor_metric]
        threshold = trainer.best_val_metric_moving_average * (1 - trainer.cfg.train.get('early_stop_threshold', 0.33))
        is_improving = current_metric < threshold
        
        # Save best checkpoint if improving
        if is_improving:
            logger.info(f"Validation metric dipped below its moving avg {trainer.best_val_metric_moving_average:.6f} to {current_metric:.6f}")
            logger.info(f"Saving best checkpoint for epoch {epoch}")
            
            checkpoint_path = self.best_checkpoint_dir / f"best_epoch_{epoch:04d}.pt"
            self._save_checkpoint(trainer, epoch, metrics, checkpoint_path, is_best=True)
            
            # Also save as "latest_best.pt" for easy loading
            latest_best_path = self.best_checkpoint_dir / "latest_best.pt"
            self._save_checkpoint(trainer, epoch, metrics, latest_best_path, is_best=True)
            logger.info(f"Latest best checkpoint saved to {latest_best_path}")
        
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
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'global_step': trainer.global_step,
            'best_val_metric': trainer.best_val_metric,
            'best_val_metric_moving_average': trainer.best_val_metric_moving_average,
        }
        
        # Add all validation metrics
        for key, value in metrics.items():
            checkpoint[key] = value
        
        # Save model states
        for name, model in trainer.models.items():
            checkpoint[f'{name}_state_dict'] = model.state_dict()
        
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
        logger.info(f"Saved checkpoint to {checkpoint_path}")