"""Base trainer class for all training paradigms."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from omegaconf import DictConfig
import logging
import torchio as tio

from ..callbacks.base_callback import CallbackList, Callback

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base class for all trainers.
    
    Provides common functionality for training loops, device management,
    and callback handling. Subclasses should implement the abstract methods
    for their specific training paradigm.
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
        """Initialize the base trainer.
        
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
        self.cfg = cfg
        self.train_loader = train_loader
        self.train_dataset = train_dataset
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.test_loader = test_loader
        self.test_dataset = test_dataset
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU specs: {torch.cuda.get_device_properties(self.device)}")
        
        # Setup callbacks
        self.callbacks = CallbackList(callbacks or [])
        
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.monitor_metric = self.cfg.train.get('early_stop_metric', 'val/loss_generator')
        self.best_val_metric = float('inf')
        self.best_val_metric_list = []
        self.best_val_metric_moving_average = float('inf')
        self.early_stop_counter = 0
        
        # Models and optimizers (to be set by subclasses)
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
        
    @abstractmethod
    def build_models(self) -> Dict[str, nn.Module]:
        """Build and return model(s) for training.
        
        Returns:
            Dictionary of models with descriptive keys
        """
        pass
    
    @abstractmethod
    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Build and return optimizer(s) for training.
        
        Returns:
            Dictionary of optimizers with descriptive keys
        """
        pass
    
    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Execute one training step.
        
        Args:
            batch: Input batch from dataloader
            batch_idx: Index of current batch
            
        Returns:
            Dictionary containing loss values and any other outputs
        """
        pass
    
    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Execute one validation step.
        
        Args:
            batch: Input batch from dataloader
            batch_idx: Index of current batch
            
        Returns:
            Dictionary containing loss values and any other outputs
        """
        pass
    
    @property
    def val_subjects(self) -> Optional[List[tio.Subject]]:
        """Get the subjects from the validation dataset."""
        if self.val_dataset is not None:
            return self.val_dataset._subjects
        else:
            return None
    
    @property
    def test_subjects(self) -> Optional[List[tio.Subject]]:
        """Get the subjects from the test dataset."""
        if self.test_dataset is not None:
            return self.test_dataset._subjects
        else:
            return None
    
    @property
    def train_subjects(self) -> Optional[List[tio.Subject]]:
        """Get the subjects from the training dataset."""
        if self.train_dataset is not None:
            return self.train_dataset._subjects
        else:
            return None
    
    def fit(self):
        """Main training loop."""
        # Setup
        self.models = self.build_models()
        self.optimizers = self.build_optimizers()
        self.schedulers = self.build_schedulers() if hasattr(self, 'build_schedulers') else {}
        
        # Move models to device
        for name, model in self.models.items():
            self.models[name] = model.to(self.device)
            logger.info(f"Model '{name}' moved to {self.device}")
        
        # Training begins
        self.callbacks.on_fit_start(self)
        
        try:
            for epoch in range(self.cfg.train.num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                self.train_metrics = self._train_epoch()
                
                # Validation phase
                if self.val_loader is not None:
                    self.val_metrics = self._validate_epoch()
                    
                    # Early stopping check
                    if self._check_early_stopping():
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
                
                # Learning rate scheduling
                for scheduler in self.schedulers.values():
                    scheduler.step()
                                
        finally:
            # Training ends
            self.callbacks.on_fit_end(self)
    
    # finished looking at this function
    def _train_epoch(self) -> Dict[str, float]:
        """Execute one training epoch.
        
        Returns:
            Dictionary of average training metrics
        """
        # Set models to appropriate mode (can be overridden by subclasses)
        self.set_training_mode(True)
        
        # Training epoch begins
        self.callbacks.on_train_epoch_start(self, self.current_epoch)
        
        # Accumulate metrics
        metric_accumulator = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Batch begins
            self.callbacks.on_train_batch_start(self, batch, batch_idx)
            
            # Training step
            outputs = self.training_step(batch, batch_idx)
            self.global_step += 1
            
            # Accumulate metrics
            for key, value in outputs.items():
                if key.startswith('loss') or key.startswith('metric'):
                    if key not in metric_accumulator:
                        metric_accumulator[key] = []
                    metric_accumulator[key].append(value.item() if torch.is_tensor(value) else value)
            
            # Add global step to outputs
            outputs['global_step'] = self.global_step
            
            # Batch ends
            self.callbacks.on_train_batch_end(self, batch, batch_idx, outputs)
        
        # Compute average metrics
        avg_metrics = {}
        for key, values in metric_accumulator.items():
            avg_metrics[f'train/{key}'] = sum(values) / len(values)
        
        # Training epoch ends
        self.callbacks.on_train_epoch_end(self, self.current_epoch, avg_metrics)
        
        return avg_metrics
    
    # finished looking at this function
    def _validate_epoch(self) -> Dict[str, float]:
        """Execute one validation epoch.
        
        Returns:
            Dictionary of average validation metrics
        """
        # Set models to appropriate mode (can be overridden by subclasses)
        self.set_training_mode(False)
        
        # Validation begins
        self.callbacks.on_validation_epoch_start(self, self.current_epoch)
        
        # Accumulate metrics
        metric_accumulator = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Validation batch begins
                self.callbacks.on_validation_batch_start(self, batch, batch_idx)
                
                # Validation step
                outputs = self.validation_step(batch, batch_idx)
                
                # Accumulate metrics
                for key, value in outputs.items():
                    if key.startswith('loss') or key.startswith('metric'):
                        if key not in metric_accumulator:
                            metric_accumulator[key] = []
                        metric_accumulator[key].append(value.item() if torch.is_tensor(value) else value)
                
                # Validation batch ends
                self.callbacks.on_validation_batch_end(self, batch, batch_idx, outputs)
        
        # Compute average metrics
        avg_metrics = {}
        for key, values in metric_accumulator.items():
            avg_metrics[f'val/{key}'] = sum(values) / len(values)
            
        # update the moving average of the metric being monitored
        self.best_val_metric_list.append(avg_metrics[self.monitor_metric])
        self.best_val_metric_moving_average = sum(self.best_val_metric_list) / len(self.best_val_metric_list)
        
        # Validation ends
        self.callbacks.on_validation_epoch_end(self, self.current_epoch, avg_metrics)
        
        return avg_metrics
    
    # finished looking at this function
    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria is met.
        
        Returns:
            True if training should stop, False otherwise
        """
        # Get the metric to monitor
        mode = self.cfg.train.get('early_stop_mode', 'min')
        patience = self.cfg.train.get('early_stop_patience', 10)
        
        if self.monitor_metric not in self.val_metrics:
            return False
        
        current_metric = self.val_metrics[self.monitor_metric]
        
        # Check if metric improved
        # improved = False
        # if mode == 'min':
        #     improved = current_metric < self.best_val_metric
        # elif mode == 'max':
        #     improved = current_metric > self.best_val_metric
        
        # if improved:
        #     self.best_val_metric = current_metric
        #     self.early_stop_counter = 0
        #     logger.info(f"New best {monitor_metric}: {current_metric:.4f}")
        # else:
        #     self.early_stop_counter += 1
        #     logger.info(f"No improvement in {monitor_metric} for {self.early_stop_counter} epochs")
        
        
        
        # early stopping occurs when some criteria has not been met for some number of epochs. the criteria
        # i am using is:
        # a good epoch is one where the metric being monitored dropped below some threshold percentage
        # of its moving average. if this is the case, the early stop counter is reset. otherwise, the
        # early stop counter is incremented. if the early stop counter is greater than the patience,
        # early stopping is triggered.
        if current_metric < self.best_val_metric_moving_average * (1 - self.cfg.train.get('early_stop_threshold', 0.33)):
            self.early_stop_counter = 0
            logger.info(f"New best {self.monitor_metric}: {current_metric:.4f}")
        else:
            self.early_stop_counter += 1
            logger.info(f"No improvement in {self.monitor_metric} for {self.early_stop_counter} epochs")
        if self.early_stop_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {self.current_epoch}")
            return True
        else:
            return False
    
    # finished looking at this function
    def set_training_mode(self, mode: bool) -> None:
        """Set training/evaluation mode for models.
        
        This method can be overridden by subclasses that need different
        behavior (e.g., keeping some models in eval mode during training).
        
        Args:
            mode: True for training mode, False for evaluation mode
        """
        for model in self.models.values():
            model.train(mode)
    
    # (TODO) have not yet figured out wandb compatible checkpoint loading. maybe thats unecessary tbh
    # and i should first at least do loading from a checkpoint and training like a new run
    def load_checkpoint(self, checkpoint_path: Path):
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        for name, model in self.models.items():
            if f'{name}_state_dict' in checkpoint:
                model.load_state_dict(checkpoint[f'{name}_state_dict'])
                logger.info(f"Loaded {name} state")
        
        # Load optimizer states
        for name, optimizer in self.optimizers.items():
            if f'optimizer_{name}_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint[f'optimizer_{name}_state_dict'])
                logger.info(f"Loaded {name} optimizer state")
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_metric = checkpoint.get('best_val_metric', float('inf'))
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")