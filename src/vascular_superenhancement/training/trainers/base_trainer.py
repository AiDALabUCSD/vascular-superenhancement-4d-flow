"""Base trainer class for all training paradigms."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from omegaconf import DictConfig
import logging
import torchio as tio

from ..dataloading import build_train_loader
# #region agent log
import gc
import json
import psutil
import time
DEBUG_LOG_PATH = "/home/ayeluru/vascular-superenhancement-4d-flow/.cursor/debug.log"
def _debug_log_memory(location: str, message: str, hypothesis_id: str, extra_data: dict = None):
    try:
        proc = psutil.Process()
        mem = proc.memory_info()
        children_mem = sum(c.memory_info().rss for c in proc.children(recursive=True))
        sys_mem = psutil.virtual_memory()
        data = {"main_rss_gb": mem.rss / 1e9, "children_rss_gb": children_mem / 1e9, "total_proc_gb": (mem.rss + children_mem) / 1e9, "sys_avail_gb": sys_mem.available / 1e9, "sys_used_pct": sys_mem.percent}
        if torch.cuda.is_available():
            data["gpu_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
            data["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        if extra_data:
            data.update(extra_data)
        entry = {"timestamp": int(time.time() * 1000), "location": location, "message": message, "hypothesisId": hypothesis_id, "data": data, "sessionId": "debug-session"}
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
# #endregion

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
        
        # Batch-based validation configuration
        # If timepoints_as_augmentation is enabled, validate every N batches (1/num_timepoints of epoch)
        # Otherwise, validate at the end of each epoch (validation_batch_interval = 0 means epoch-based)
        if cfg.train.get('timepoints_as_augmentation', False):
            num_timepoints = cfg.train.get('num_timepoints', 20)
            self.validation_batch_interval = max(1, len(train_loader) // num_timepoints)
            logger.info(f"Batch-based validation enabled: validating every {self.validation_batch_interval} batches")
        else:
            self.validation_batch_interval = 0  # 0 means epoch-based validation
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU specs: {torch.cuda.get_device_properties(self.device)}")
            logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        
        # Setup callbacks
        self.callbacks = CallbackList(callbacks or [])
        
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.monitor_metric = self.cfg.train.get('early_stop_metric', 'val/loss_generator')
        self.early_stop_mode = self.cfg.train.get('early_stop_mode', 'min')
        self.best_val_metric = float('inf') if self.early_stop_mode == 'min' else float('-inf')
        self.early_stop_counter = 0
        self.is_improving = False  # set each validation epoch; used by callbacks
        
        # Models and optimizers (to be set by subclasses)
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
    
    def _wrap_models_data_parallel(self) -> None:
        """Wrap models with nn.DataParallel when num_gpus > 1 and multiple GPUs available."""
        num_gpus = self.cfg.train.get('num_gpus', 1)
        if self.device.type != "cuda" or num_gpus <= 1:
            return
        device_count = torch.cuda.device_count()
        if device_count < 2:
            logger.info(f"num_gpus={num_gpus} requested but only {device_count} GPU(s) available; using single GPU")
            return
        n_use = min(num_gpus, device_count)
        device_ids = list(range(n_use))
        for name, model in self.models.items():
            if isinstance(model, nn.DataParallel):
                continue  # Already wrapped (e.g. from load_checkpoint)
            self.models[name] = nn.DataParallel(model, device_ids=device_ids)
            logger.info(f"Model '{name}' wrapped with DataParallel on devices {device_ids}")
        
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
        """Main training loop.
        
        Supports two validation modes:
        1. Batch-based validation (validation_batch_interval > 0): 
           Runs validation every N batches for faster feedback with large datasets
        2. Epoch-based validation (validation_batch_interval == 0):
           Runs validation at the end of each epoch (traditional approach)
        """
        # Setup - only build if not already built (e.g., by load_checkpoint)
        if not self.models:
            self.models = self.build_models()
            # Move models to device
            for name, model in self.models.items():
                self.models[name] = model.to(self.device)
                logger.info(f"Model '{name}' moved to {self.device}")
            self._wrap_models_data_parallel()
        
        if not hasattr(self, 'optimizers') or not self.optimizers:
            self.optimizers = self.build_optimizers()
        
        self.schedulers = self.build_schedulers() if hasattr(self, 'build_schedulers') else {}
        
        # Training begins
        self.callbacks.on_fit_start(self)
        
        try:
            # Determine starting epoch
            start_epoch = self.current_epoch if hasattr(self, 'current_epoch') and self.current_epoch > 0 else 0
            if start_epoch > 0:
                logger.info(f"Resuming training from epoch {start_epoch}")
            
            # #region agent log
            _debug_log_memory("base_trainer.py:fit:baseline", "Baseline before training loop", "A", {"start_epoch": start_epoch})
            # #endregion
            
            if self.validation_batch_interval > 0:
                # Batch-based validation mode
                self._fit_batch_based(start_epoch)
            else:
                # Epoch-based validation mode (traditional)
                self._fit_epoch_based(start_epoch)
                                
        finally:
            # Training ends
            self.callbacks.on_fit_end(self)
    
    def _fit_epoch_based(self, start_epoch: int):
        """Traditional epoch-based training loop."""
        for epoch in range(start_epoch, self.cfg.train.num_epochs):
            self.current_epoch = epoch

            # #region agent log
            _debug_log_memory(f"base_trainer.py:fit:epoch_{epoch}_start", f"Epoch {epoch} starting", "A", {"epoch": epoch})
            # #endregion

            # Training phase
            self.train_metrics = self._train_epoch()
            # #region agent log
            _debug_log_memory(f"base_trainer.py:fit:epoch_{epoch}_train_end", f"Epoch {epoch} training ended", "B", {"epoch": epoch})
            # #endregion

            # Validation phase
            if self.val_loader is not None:
                # #region agent log
                _debug_log_memory(f"base_trainer.py:fit:epoch_{epoch}_val_start", f"Epoch {epoch} validation starting", "C", {"epoch": epoch})
                # #endregion
                self.val_metrics = self._validate_epoch()
                # #region agent log
                _debug_log_memory(f"base_trainer.py:fit:epoch_{epoch}_val_end", f"Epoch {epoch} validation ended", "C", {"epoch": epoch})
                gc.collect()
                torch.cuda.empty_cache()
                _debug_log_memory(f"base_trainer.py:fit:epoch_{epoch}_after_gc", f"Epoch {epoch} after gc.collect()", "C", {"epoch": epoch})
                # #endregion

                # Early stopping check
                if self._check_early_stopping():
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

            # Learning rate scheduling
            for scheduler in self.schedulers.values():
                scheduler.step()
                                
    def _fit_batch_based(self, start_epoch: int):
        """Batch-based training loop with validation every N batches.

        This mode is useful when training on all timepoints shuffled together,
        providing faster validation feedback without waiting for all timepoints.

        An "effective epoch" is defined as validation_batch_interval batches.
        """
        # Track effective epoch for callbacks (validation_batch_interval batches = 1 effective epoch)
        effective_epoch = start_epoch
        batches_since_validation = 0
        num_effective_epochs = self.cfg.train.num_epochs
        dataloader_iteration = 0

        logger.info(f"Starting batch-based training: {num_effective_epochs} effective epochs, "
                   f"validation every {self.validation_batch_interval} batches")

        # Set models to training mode
        self.set_training_mode(True)
        self.callbacks.on_train_epoch_start(self, effective_epoch)

        # Training loop - iterate indefinitely over the loader
        metric_accumulator = {}
        should_stop = False

        while effective_epoch < num_effective_epochs and not should_stop:
            # Log memory at start of dataloader iteration
            _debug_log_memory(
                f"base_trainer.py:dataloader_iter_{dataloader_iteration}_start",
                f"Starting dataloader iteration {dataloader_iteration}",
                "DATALOADER_BOUNDARY",
                {"dataloader_iteration": dataloader_iteration, "effective_epoch": effective_epoch}
            )

            # Iterate through one full pass of the data
            for batch_idx, batch in enumerate(self.train_loader):
                self.current_epoch = effective_epoch  # For callbacks/checkpointing

                # Batch begins
                self.callbacks.on_train_batch_start(self, batch, batch_idx)

                # Training step
                outputs = self.training_step(batch, batch_idx)
                self.global_step += 1
                batches_since_validation += 1

                # Accumulate metrics
                for key, value in outputs.items():
                    if key.startswith('loss') or key.startswith('metric'):
                        if key not in metric_accumulator:
                            metric_accumulator[key] = []
                        metric_accumulator[key].append(value.item() if torch.is_tensor(value) else value)

                outputs['global_step'] = self.global_step
                self.callbacks.on_train_batch_end(self, batch, batch_idx, outputs)

                # Check if it's time for validation
                if batches_since_validation >= self.validation_batch_interval:
                    # Compute average training metrics
                    self.train_metrics = {
                        key: sum(values) / len(values)
                        for key, values in metric_accumulator.items()
                    }
                    metric_accumulator = {}  # Reset accumulator

                    # End training "epoch"
                    self.callbacks.on_train_epoch_end(self, effective_epoch, self.train_metrics)

                    # Validation phase
                    if self.val_loader is not None:
                        self.val_metrics = self._validate_epoch()
                        gc.collect()
                        torch.cuda.empty_cache()

                        # Early stopping check
                        if self._check_early_stopping():
                            logger.info(f"Early stopping triggered at effective epoch {effective_epoch}")
                            should_stop = True
                            break

                    # Learning rate scheduling
                    for scheduler in self.schedulers.values():
                        scheduler.step()

                    # Increment effective epoch
                    effective_epoch += 1
                    batches_since_validation = 0

                    # Log progress
                    if effective_epoch % 10 == 0:
                        logger.info(f"Completed effective epoch {effective_epoch}/{num_effective_epochs}")

                    # Check if done
                    if effective_epoch >= num_effective_epochs:
                        break

                    # Start new "epoch"
                    self.set_training_mode(True)
                    self.callbacks.on_train_epoch_start(self, effective_epoch)

            # === DATALOADER EXHAUSTED - CLEANUP AND RECREATE ===
            # This is the critical point where OOM can occur if we don't properly
            # clean up before recreating the loader/queue

            _debug_log_memory(
                f"base_trainer.py:dataloader_iter_{dataloader_iteration}_exhausted",
                f"Dataloader iteration {dataloader_iteration} exhausted, cleaning up",
                "DATALOADER_BOUNDARY",
                {"dataloader_iteration": dataloader_iteration, "effective_epoch": effective_epoch}
            )

            # Explicitly delete the old loader to release Queue workers
            old_loader = self.train_loader
            del old_loader
            del self.train_loader

            # Force garbage collection to free Queue worker memory
            gc.collect()
            torch.cuda.empty_cache()

            # Conservative delay for memory cleanup - give workers time to fully terminate
            # and OS time to reclaim memory before recreating loader
            cleanup_delay = self.cfg.train.get('dataloader_cleanup_delay', 60)
            logger.info(f"Waiting {cleanup_delay}s for memory cleanup before recreating loader...")
            time.sleep(cleanup_delay)
            
            # Second GC pass after delay
            gc.collect()
            torch.cuda.empty_cache()

            _debug_log_memory(
                f"base_trainer.py:dataloader_iter_{dataloader_iteration}_after_gc",
                f"After GC, before recreating loader",
                "DATALOADER_BOUNDARY",
                {"dataloader_iteration": dataloader_iteration, "effective_epoch": effective_epoch}
            )

            # Recreate the loader with fresh Queue and workers
            self.train_loader = build_train_loader(self.train_dataset, self.cfg, subject_sampler=None, train=True)

            _debug_log_memory(
                f"base_trainer.py:dataloader_iter_{dataloader_iteration}_recreated",
                f"Loader recreated for iteration {dataloader_iteration + 1}",
                "DATALOADER_BOUNDARY",
                {"dataloader_iteration": dataloader_iteration + 1, "effective_epoch": effective_epoch}
            )

            dataloader_iteration += 1
            logger.info(f"Dataloader iteration {dataloader_iteration}: recreated loader after exhaustion")
    
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
            # #region agent log
            # Log memory at EVERY batch to catch gradual accumulation
            _debug_log_memory(f"base_trainer.py:e{self.current_epoch}_b{batch_idx}", f"train batch", "A", {"e": self.current_epoch, "b": batch_idx})
            # #endregion
            # Batch begins
            self.callbacks.on_train_batch_start(self, batch, batch_idx)
            
            # Training step
            outputs = self.training_step(batch, batch_idx)
            self.global_step += 1
            # #region agent log
            _debug_log_memory(f"base_trainer.py:e{self.current_epoch}_b{batch_idx}_post", f"after step", "B", {"e": self.current_epoch, "b": batch_idx})
            # #endregion
            
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
                # #region agent log
                # Log memory at EVERY validation batch
                _debug_log_memory(f"base_trainer.py:e{self.current_epoch}_val_b{batch_idx}", f"val batch", "C", {"e": self.current_epoch, "vb": batch_idx})
                # #endregion
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
            # #region agent log
            _debug_log_memory(f"base_trainer.py:epoch_{self.current_epoch}_val_batches_done", f"All validation batches done", "C", {"epoch": self.current_epoch})
            # #endregion
        
        # Compute average metrics
        avg_metrics = {}
        for key, values in metric_accumulator.items():
            avg_metrics[f'val/{key}'] = sum(values) / len(values)

        # Check if monitored metric improved
        current = avg_metrics[self.monitor_metric]
        min_delta = self.cfg.train.get('early_stop_min_delta', 0.0)
        if self.early_stop_mode == 'min':
            self.is_improving = current < self.best_val_metric - min_delta
        else:
            self.is_improving = current > self.best_val_metric + min_delta

        if self.is_improving:
            self.best_val_metric = current

        # Validation ends
        self.callbacks.on_validation_epoch_end(self, self.current_epoch, avg_metrics)
        
        return avg_metrics
    
    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria is met.

        Uses ``self.is_improving`` which is set during ``_validate_epoch``.
        If the monitored metric has not improved for ``early_stop_patience``
        consecutive validation epochs the method returns ``True``.
        """
        patience = self.cfg.train.get('early_stop_patience', 10)

        if self.monitor_metric not in self.val_metrics:
            return False

        if self.is_improving:
            self.early_stop_counter = 0
            logger.info(
                f"New best {self.monitor_metric}: {self.best_val_metric:.4f}"
            )
        else:
            self.early_stop_counter += 1
            logger.info(
                f"No improvement in {self.monitor_metric} for "
                f"{self.early_stop_counter} epochs "
                f"(best: {self.best_val_metric:.4f})"
            )

        if self.early_stop_counter >= patience:
            logger.info(
                f"Early stopping triggered at epoch {self.current_epoch}"
            )
            return True
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
    def load_checkpoint(self, checkpoint_path: Path, resume_training_state: bool = True):
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            resume_training_state: If True, restore epoch/global_step; if False, only load model/optimizer weights
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Build models if they haven't been built yet
        if not self.models:
            self.models = self.build_models()
            # Move models to device
            for name, model in self.models.items():
                self.models[name] = model.to(self.device)
                logger.info(f"Model '{name}' moved to {self.device}")
            self._wrap_models_data_parallel()
        
        # Load model states (load into .module when DataParallel for clean checkpoint keys)
        for name, model in self.models.items():
            if f'{name}_state_dict' in checkpoint:
                target = model.module if isinstance(model, nn.DataParallel) else model
                target.load_state_dict(checkpoint[f'{name}_state_dict'])
                logger.info(f"Loaded {name} state")
        
        # Build optimizers if they haven't been built yet (needed for loading optimizer state)
        if not hasattr(self, 'optimizers') or not self.optimizers:
            self.optimizers = self.build_optimizers()
        
        # Load optimizer states
        for name, optimizer in self.optimizers.items():
            if f'optimizer_{name}_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint[f'optimizer_{name}_state_dict'])
                logger.info(f"Loaded {name} optimizer state")
        
        # Load training state only if resuming
        if resume_training_state:
            self.current_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            default_best = float('inf') if self.early_stop_mode == 'min' else float('-inf')
            self.best_val_metric = checkpoint.get('best_val_metric', default_best)
            logger.info(f"Loaded checkpoint from epoch {self.current_epoch}, resuming training (best metric: {self.best_val_metric:.4f})")
        else:
            # Reset training state but keep model/optimizer weights
            self.current_epoch = 0
            self.global_step = 0
            self.best_val_metric = float('inf') if self.early_stop_mode == 'min' else float('-inf')
            logger.info(f"Loaded checkpoint weights but starting training from epoch 0")