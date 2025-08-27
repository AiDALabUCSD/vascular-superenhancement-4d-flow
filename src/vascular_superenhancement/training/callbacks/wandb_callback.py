"""Weights & Biases integration callback."""

from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING
import wandb
import omegaconf
import logging
import os

# Avoid circular imports
if TYPE_CHECKING:
    from ..trainers.base_trainer import BaseTrainer

from .base_callback import Callback

logger = logging.getLogger(__name__)


class WandbCallback(Callback):
    """Callback for Weights & Biases logging and tracking."""
    
    def __init__(self, wandb_cfg: Dict[str, Any], full_cfg: omegaconf.DictConfig):
        """Initialize W&B callback.
        
        Args:
            wandb_cfg: W&B-specific configuration
            full_cfg: Full Hydra configuration for logging
        """
        self.wandb_cfg = wandb_cfg
        self.full_cfg = full_cfg
        self.enabled = wandb_cfg.get('enabled', False)
        
        if not self.enabled:
            logger.info("W&B logging is disabled")
            return
        
        # Configuration
        self.project = wandb_cfg.get('project', 'vascular-superenhancement')
        self.entity = wandb_cfg.get('entity', None)
        self.name = wandb_cfg.get('name', None)
        self.mode = wandb_cfg.get('mode', 'online')
        self.log_frequency = wandb_cfg.get('log_frequency', 1)
        self.log_images = wandb_cfg.get('log_images', True)
        self.log_gradients = wandb_cfg.get('log_gradients', False)
        
        # Tracking
        self.step = 0
    
    def on_train_begin(self, trainer: 'BaseTrainer') -> None:
        """Initialize W&B run at training start."""
        if not self.enabled:
            return
        
        # Convert config to dict for W&B
        config = omegaconf.OmegaConf.to_container(
            self.full_cfg, resolve=True, throw_on_missing=True
        )
        
        # Initialize W&B
        wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            mode=self.mode,
            config=config,
        )
        
        logger.info(f"W&B run initialized: {wandb.run.name}")
        
        # Log code if specified
        if self.wandb_cfg.get('log_code', True):
            code_dir = Path(os.getcwd()).resolve().parents[4] / "src"
            if code_dir.exists():
                wandb.run.log_code(str(code_dir))
                logger.info(f"Logged code from {code_dir} to W&B")
        
        # Watch models if gradient logging is enabled
        if self.log_gradients:
            for name, model in trainer.models.items():
                wandb.watch(model, log='all', log_freq=100)
                logger.info(f"W&B watching model: {name}")
    
    def on_batch_end(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int,
                     outputs: Dict[str, Any]) -> None:
        """Log training batch metrics."""
        if not self.enabled:
            return
        
        # Only log at specified frequency
        if trainer.global_step % self.log_frequency != 0:
            return
        
        # Prepare metrics
        metrics = {}
        for key, value in outputs.items():
            if key.startswith('loss') or key.startswith('metric'):
                if hasattr(value, 'item'):
                    metrics[f'train/{key}'] = value.item()
                else:
                    metrics[f'train/{key}'] = value
        
        # Add global step
        metrics['global_step'] = trainer.global_step
        
        # Log to W&B
        wandb.log(metrics, step=trainer.global_step)
    
    def on_epoch_end(self, trainer: 'BaseTrainer', epoch: int, 
                     metrics: Dict[str, float]) -> None:
        """Log epoch-level metrics."""
        if not self.enabled:
            return
        
        # Log all epoch metrics
        epoch_metrics = {
            'epoch': epoch,
            **metrics
        }
        
        # Add learning rates
        for name, optimizer in trainer.optimizers.items():
            for i, param_group in enumerate(optimizer.param_groups):
                epoch_metrics[f'lr/{name}_group_{i}'] = param_group['lr']
        
        wandb.log(epoch_metrics, step=trainer.global_step)