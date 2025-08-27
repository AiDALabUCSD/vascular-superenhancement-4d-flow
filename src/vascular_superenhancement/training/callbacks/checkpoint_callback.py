"""Checkpoint management callback."""

from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING
import torch
import logging

# Avoid circular imports
if TYPE_CHECKING:
    from ..trainers.base_trainer import BaseTrainer

from .base_callback import Callback

logger = logging.getLogger(__name__)


class CheckpointCallback(Callback):
    """Callback for saving model checkpoints during training."""
    
    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        save_frequency: int = 1,
        save_best: bool = True,
        save_last: bool = True,
        monitor_metric: str = 'val/loss_total',
        monitor_mode: str = 'min',
        keep_n_best: int = 3,
        keep_n_recent: int = 2,
    ):
        """Initialize checkpoint callback.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_frequency: Save checkpoint every N epochs
            save_best: Whether to save best model based on metric
            save_last: Whether to always save the last checkpoint
            monitor_metric: Metric to monitor for best model
            monitor_mode: 'min' or 'max' for metric optimization
            keep_n_best: Number of best checkpoints to keep
            keep_n_recent: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir or Path.cwd() / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_frequency = save_frequency
        self.save_best = save_best
        self.save_last = save_last
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.keep_n_best = keep_n_best
        self.keep_n_recent = keep_n_recent
        
        # Tracking
        self.best_metric = float('inf') if monitor_mode == 'min' else float('-inf')
        self.best_checkpoints = []  # List of (metric_value, path) tuples
        self.recent_checkpoints = []  # List of paths
        
    def on_epoch_end(self, trainer: 'BaseTrainer', epoch: int, 
                     metrics: Dict[str, float]) -> None:
        """Save checkpoint at end of epoch if conditions are met."""
        
        # Save regular checkpoint at specified frequency
        if epoch % self.save_frequency == 0:
            self._save_checkpoint(
                trainer, epoch, metrics, 
                prefix='epoch', is_best=False
            )
        
        # Save best checkpoint if metric improved
        if self.save_best and self.monitor_metric in metrics:
            current_metric = metrics[self.monitor_metric]
            is_best = self._is_better(current_metric, self.best_metric)
            
            if is_best:
                self.best_metric = current_metric
                checkpoint_path = self._save_checkpoint(
                    trainer, epoch, metrics, 
                    prefix='best', is_best=True
                )
                
                # Track best checkpoints
                self.best_checkpoints.append((current_metric, checkpoint_path))
                self.best_checkpoints.sort(
                    key=lambda x: x[0], 
                    reverse=(self.monitor_mode == 'max')
                )
                
                # Remove old best checkpoints
                while len(self.best_checkpoints) > self.keep_n_best:
                    _, old_path = self.best_checkpoints.pop()
                    if old_path.exists():
                        old_path.unlink()
                        logger.info(f"Removed old best checkpoint: {old_path}")
    
    def on_train_end(self, trainer: 'BaseTrainer') -> None:
        """Save final checkpoint at end of training."""
        if self.save_last:
            metrics = {**trainer.train_metrics, **trainer.val_metrics}
            self._save_checkpoint(
                trainer, trainer.current_epoch, metrics,
                prefix='last', is_best=False
            )
    
    def _save_checkpoint(
        self, 
        trainer: 'BaseTrainer',
        epoch: int,
        metrics: Dict[str, float],
        prefix: str = 'epoch',
        is_best: bool = False
    ) -> Path:
        """Save a checkpoint.
        
        Args:
            trainer: Trainer instance with models and optimizers
            epoch: Current epoch number
            metrics: Current metrics
            prefix: Prefix for checkpoint filename
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'global_step': trainer.global_step,
            'best_val_metric': trainer.best_val_metric,
            'metrics': metrics,
        }
        
        # Save model states
        for name, model in trainer.models.items():
            checkpoint[f'{name}_state_dict'] = model.state_dict()
        
        # Save optimizer states
        for name, optimizer in trainer.optimizers.items():
            checkpoint[f'optimizer_{name}_state_dict'] = optimizer.state_dict()
        
        # Save scheduler states if they exist
        if hasattr(trainer, 'schedulers'):
            for name, scheduler in trainer.schedulers.items():
                checkpoint[f'scheduler_{name}_state_dict'] = scheduler.state_dict()
        
        # Determine filename
        if is_best:
            metric_value = metrics.get(self.monitor_metric, 0)
            filename = f"{prefix}_epoch_{epoch:04d}_{self.monitor_metric.replace('/', '_')}_{metric_value:.4f}.pt"
        else:
            filename = f"{prefix}_epoch_{epoch:04d}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Track recent checkpoints
        if prefix == 'epoch':
            self.recent_checkpoints.append(checkpoint_path)
            
            # Remove old recent checkpoints
            while len(self.recent_checkpoints) > self.keep_n_recent:
                old_path = self.recent_checkpoints.pop(0)
                if old_path.exists():
                    old_path.unlink()
                    logger.info(f"Removed old checkpoint: {old_path}")
        
        return checkpoint_path
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best metric.
        
        Args:
            current: Current metric value
            best: Best metric value so far
            
        Returns:
            True if current is better than best
        """
        if self.monitor_mode == 'min':
            return current < best
        else:
            return current > best