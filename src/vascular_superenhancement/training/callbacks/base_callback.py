"""Base callback system for training hooks."""

from abc import ABC
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging

# Avoid circular imports
if TYPE_CHECKING:
    from ..trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base class for training callbacks.
    
    Callbacks provide hooks into the training process, allowing for
    modular implementation of logging, checkpointing, visualization, etc.
    """
    
    def on_fit_start(self, trainer: 'BaseTrainer') -> None:
        """Called when the training run starts."""
        pass
    
    def on_fit_end(self, trainer: 'BaseTrainer') -> None:
        """Called when the training run ends."""
        pass
    
    def on_train_epoch_start(self, trainer: 'BaseTrainer', epoch: int) -> None:
        """Called at the start of the training epoch."""
        pass
    
    def on_train_epoch_end(self, trainer: 'BaseTrainer', epoch: int, metrics: Dict[str, float]) -> None:
        """Called at the end of the training epoch.
        
        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            metrics: Training metrics for this epoch
        """
        pass
    
    def on_train_batch_start(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int) -> None:
        """Called before processing each training batch."""
        pass
    
    def on_train_batch_end(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int, 
                            outputs: Dict[str, Any]) -> None:
        """Called after processing each training batch."""
        pass
    
    def on_validation_epoch_start(self, trainer: 'BaseTrainer', epoch: int) -> None:
        """Called at the start of the validation epoch."""
        pass
    
    def on_validation_epoch_end(self, trainer: 'BaseTrainer', epoch: int, 
                                metrics: Dict[str, float]) -> None:
        """Called at the end of the validation epoch.
        
        This is also effectively the end of the full epoch (train + val).
        
        Args:
            trainer: The trainer instance
            epoch: Current epoch number
            metrics: Validation metrics for this epoch
        """
        pass
    
    def on_validation_batch_start(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int) -> None:
        """Called before processing each validation batch."""
        pass
    
    def on_validation_batch_end(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int,
                                outputs: Dict[str, Any]) -> None:
        """Called after processing each validation batch."""
        pass


class CallbackList:
    """Container for managing multiple callbacks."""
    
    def __init__(self, callbacks: Optional[list] = None):
        self.callbacks = callbacks or []
    
    def on_fit_start(self, trainer: 'BaseTrainer') -> None:
        for callback in self.callbacks:
            callback.on_fit_start(trainer)
    
    def on_fit_end(self, trainer: 'BaseTrainer') -> None:
        for callback in self.callbacks:
            callback.on_fit_end(trainer)
    
    def on_train_epoch_start(self, trainer: 'BaseTrainer', epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_train_epoch_start(trainer, epoch)
    
    def on_train_epoch_end(self, trainer: 'BaseTrainer', epoch: int, metrics: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_train_epoch_end(trainer, epoch, metrics)
    
    def on_train_batch_start(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_start(trainer, batch, batch_idx)
    
    def on_train_batch_end(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int,
                            outputs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_end(trainer, batch, batch_idx, outputs)
    
    def on_validation_epoch_start(self, trainer: 'BaseTrainer', epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_validation_epoch_start(trainer, epoch)
    
    def on_validation_epoch_end(self, trainer: 'BaseTrainer', epoch: int, 
                                metrics: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_validation_epoch_end(trainer, epoch, metrics)
    
    def on_validation_batch_start(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int) -> None:
        for callback in self.callbacks:
            callback.on_validation_batch_start(trainer, batch, batch_idx)
    
    def on_validation_batch_end(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int,
                                outputs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_validation_batch_end(trainer, batch, batch_idx, outputs)