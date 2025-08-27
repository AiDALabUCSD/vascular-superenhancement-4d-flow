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
    
    def on_train_begin(self, trainer: 'BaseTrainer') -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer: 'BaseTrainer') -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer: 'BaseTrainer', epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: 'BaseTrainer', epoch: int, metrics: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int) -> None:
        """Called before processing each batch."""
        pass
    
    def on_batch_end(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int, 
                     outputs: Dict[str, Any]) -> None:
        """Called after processing each batch."""
        pass
    
    def on_validation_begin(self, trainer: 'BaseTrainer') -> None:
        """Called at the beginning of validation."""
        pass
    
    def on_validation_end(self, trainer: 'BaseTrainer', metrics: Dict[str, float]) -> None:
        """Called at the end of validation."""
        pass
    
    def on_validation_batch_begin(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int) -> None:
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
    
    def on_train_begin(self, trainer: 'BaseTrainer') -> None:
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer: 'BaseTrainer') -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer: 'BaseTrainer', epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(self, trainer: 'BaseTrainer', epoch: int, metrics: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, metrics)
    
    def on_batch_begin(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch, batch_idx)
    
    def on_batch_end(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int,
                     outputs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch, batch_idx, outputs)
    
    def on_validation_begin(self, trainer: 'BaseTrainer') -> None:
        for callback in self.callbacks:
            callback.on_validation_begin(trainer)
    
    def on_validation_end(self, trainer: 'BaseTrainer', metrics: Dict[str, float]) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(trainer, metrics)
    
    def on_validation_batch_begin(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int) -> None:
        for callback in self.callbacks:
            callback.on_validation_batch_begin(trainer, batch, batch_idx)
    
    def on_validation_batch_end(self, trainer: 'BaseTrainer', batch: Any, batch_idx: int,
                                outputs: Dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.on_validation_batch_end(trainer, batch, batch_idx, outputs)