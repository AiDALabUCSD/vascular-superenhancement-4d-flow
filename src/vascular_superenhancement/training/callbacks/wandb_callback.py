"""Weights & Biases integration callback."""

from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING
import wandb
import omegaconf
from omegaconf import DictConfig
import logging
import os

# Avoid circular imports
if TYPE_CHECKING:
    from ..trainers.base_trainer import BaseTrainer

from .base_callback import Callback

logger = logging.getLogger(__name__)


class WandbCallback(Callback):
    """Callback for Weights & Biases logging and tracking."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize W&B callback.
        
        Args:
            cfg: Full Hydra configuration containing wandb settings
        """
        self.cfg = cfg
        self.enabled = cfg.wandb.enabled
        
        if not self.enabled:
            logger.info("W&B logging is disabled")
            return
        else:
            logger.info("W&B logging is enabled")
        
        # Configuration
        self.project = self.cfg.wandb.project
        self.entity = self.cfg.wandb.entity
        self.name = self.cfg.wandb.name
        self.mode = self.cfg.wandb.mode
        self.log_frequency = self.cfg.wandb.log_frequency
        # self.log_images = self.cfg.wandb.log_images
        self.log_gradients = self.cfg.wandb.log_gradients
        self.log_code = self.cfg.wandb.log_code
    
    # finished looking at this function
    def on_fit_start(self, trainer: 'BaseTrainer') -> None:
        """Initialize W&B run at training start."""
        if not self.enabled:
            return
        
        # Convert config to dict for W&B
        config = omegaconf.OmegaConf.to_container(
            self.cfg, resolve=True, throw_on_missing=True
        )
        
        # Initialize W&B
        wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            mode=self.mode,
            config=config,
        )
        
        # log the wandb generated name of this run
        logger.info(f"W&B initialized with name: {wandb.run.name}")
        # create a simple file in the directory titled {wandb.run.name}.txt
        (Path(os.getcwd()) / f"{wandb.run.name}.txt").touch()
        
        # Log code if specified
        if self.log_code:
            code_dir = Path(os.getcwd()).resolve().parents[4] / "src"
            if code_dir.exists():
                wandb.run.log_code(str(code_dir))
                logger.info(f"Logged code from {code_dir} to W&B")
        
        # Watch models if gradient logging is enabled
        if self.log_gradients:
            for name, model in trainer.models.items():
                wandb.watch(model, log='all', log_freq=self.log_frequency)
                logger.info(f"W&B watching model: {name}")
    
    # finished looking at this function
    def on_fit_end(self, trainer: 'BaseTrainer') -> None:
        """Log end of training."""
        if not self.enabled:
            return
        
        wandb.finish()
        logger.info("W&B finished")
    
    # finished looking at this function
    def on_train_batch_end(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int,
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
    
    def on_validation_epoch_end(self, trainer: 'BaseTrainer', epoch: int, 
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