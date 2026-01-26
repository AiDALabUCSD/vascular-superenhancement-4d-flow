"""New training script developed in the modularize-training-script branch
    which utilizes base_trainer and base_callbacks.

Supports two training modes:
1. GAN training (default): Uses GanTrainer with discriminator
2. Generator-only training: Uses GeneratorTrainer without discriminator

And two data modes:
1. Single-timepoint (default): Each subject has data from one timepoint
2. Multi-timepoint (temporal): Each subject has data from a window of timepoints
"""

from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging

from vascular_superenhancement.training.datasets import (
    build_subjects_dataset,
    build_multi_timepoint_subjects_dataset,
)
from vascular_superenhancement.training.transforms import build_transforms, build_multi_timepoint_transforms
from vascular_superenhancement.training.dataloading import build_train_loader
from vascular_superenhancement.training.trainers.gan_trainer import GanTrainer
from vascular_superenhancement.training.trainers.generator_trainer import GeneratorTrainer
from vascular_superenhancement.training.callbacks.wandb_callback import WandbCallback
from vascular_superenhancement.training.callbacks.checkpoint_callback import CheckpointCallback
from vascular_superenhancement.training.callbacks.visualization_callback import VisualizationCallback
from vascular_superenhancement.training.callbacks.patch_preview_callback import PatchPreviewCallback

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.1",
    config_path=str((Path(__file__).resolve().parents[3] / "hydra_configs").as_posix()),
    config_name="config"
)
def train_model(cfg: DictConfig):
    """Main training function using callback-based trainer architecture.

    Supports two modes based on configuration:
    1. Standard mode (trainer_type='gan'): Single-timepoint GAN training
    2. Temporal mode (trainer_type='generator', use_multi_timepoint=True):
       Multi-timepoint generator-only training
    """

    # Set up logging level based on debug flag
    if cfg.train.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    else:
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)

    logger.info("Setting up training...")
    logger.info(f"Current working directory: {Path.cwd()}")

    # Determine training mode
    trainer_type = cfg.train.get('trainer_type', 'gan')
    use_multi_timepoint = cfg.train.get('use_multi_timepoint', False)
    temporal_window_size = cfg.train.get('temporal_window_size', 5)

    logger.info(f"Training mode: trainer_type={trainer_type}, multi_timepoint={use_multi_timepoint}")
    if use_multi_timepoint:
        logger.info(f"Temporal window size: {temporal_window_size}")

    # Build transforms based on mode
    if use_multi_timepoint:
        training_transforms = build_multi_timepoint_transforms(
            cfg, train=True, window_size=temporal_window_size
        )
        validation_transforms = build_multi_timepoint_transforms(
            cfg, train=False, window_size=temporal_window_size
        )
    else:
        training_transforms = build_transforms(cfg, train=True)
        validation_transforms = build_transforms(cfg, train=False)

    logger.info(f"Training transforms: {training_transforms}")
    logger.info(f"Validation transforms: {validation_transforms}")

    # Build training dataset
    if use_multi_timepoint:
        # Multi-timepoint mode: each subject contains a window of timepoints
        training_dataset = build_multi_timepoint_subjects_dataset(
            "train",
            Path(cfg.data.splits_path),
            cfg.path_config.path_config_name,
            window_size=temporal_window_size,
            transforms=training_transforms,
            debug=cfg.train.debug,
            include_all_timepoints=True,
            peak_systolic_only=cfg.train.get('peak_systolic_only', False),
        )
        logger.info(f"Training dataset length (multi-timepoint): {len(training_dataset)}")
    else:
        # Standard single-timepoint mode
        training_dataset = build_subjects_dataset(
            "train",
            Path(cfg.data.splits_path),
            cfg.path_config.path_config_name,
            transforms=training_transforms,
            debug=cfg.train.debug,
            include_all_timepoints=cfg.train.timepoints_as_augmentation,
            peak_systolic_only=cfg.train.get('peak_systolic_only', False),
        )
        logger.info(f"Training dataset length: {len(training_dataset)}")

    # Build dataloader (all timepoints shuffled together, no cycling sampler)
        training_loader = build_train_loader(training_dataset, cfg, subject_sampler=None, train=True)

    logger.info(f"Number of batches in training loader: {len(training_loader)}")
    
    # If using timepoints_as_augmentation, log effective epoch info
    if cfg.train.timepoints_as_augmentation:
        num_timepoints = cfg.train.get('num_timepoints', 20)
        batches_per_effective_epoch = len(training_loader) // num_timepoints
        logger.info(f"Using batch-based validation with {num_timepoints} timepoints")
        logger.info(f"Batches per effective epoch: {batches_per_effective_epoch}")

    # Build validation dataset and dataloader
    if use_multi_timepoint:
        validation_dataset = build_multi_timepoint_subjects_dataset(
            "validation",
            Path(cfg.data.splits_path),
            cfg.path_config.path_config_name,
            window_size=temporal_window_size,
            transforms=validation_transforms,
            debug=cfg.train.debug,
            time_index=cfg.train.validation_time_index,
        )
        logger.info(f"Validation dataset length (multi-timepoint, center timepoint {cfg.train.validation_time_index}): {len(validation_dataset)}")
    else:
        validation_dataset = build_subjects_dataset(
            "validation",
            Path(cfg.data.splits_path),
            cfg.path_config.path_config_name,
            transforms=validation_transforms,
            debug=cfg.train.debug,
            time_index=cfg.train.validation_time_index,
        )
        logger.info(f"Validation dataset length (timepoint {cfg.train.validation_time_index}): {len(validation_dataset)}")

    validation_loader = build_train_loader(validation_dataset, cfg, train=False)
    logger.info(f"Number of batches in validation loader: {len(validation_loader)}")

    # Build callbacks
    callbacks = [
        WandbCallback(cfg),
        CheckpointCallback(cfg),
        VisualizationCallback(cfg),
        PatchPreviewCallback(cfg),
    ]

    # Build trainer based on configuration
    if trainer_type == 'generator':
        logger.info("Using GeneratorTrainer (generator-only, no discriminator)")
        trainer = GeneratorTrainer(
            cfg=cfg,
            train_loader=training_loader,
            train_dataset=training_dataset,
            val_loader=validation_loader,
            val_dataset=validation_dataset,
            callbacks=callbacks
        )
    else:
        logger.info("Using GanTrainer (generator + discriminator)")
        trainer = GanTrainer(
            cfg=cfg,
            train_loader=training_loader,
            train_dataset=training_dataset,
            val_loader=validation_loader,
            val_dataset=validation_dataset,
            callbacks=callbacks
        )

    # Load checkpoint if specified
    checkpoint_path = cfg.train.get('checkpoint_path', None)
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            resume_training_state = cfg.train.get('resume_from_checkpoint_epoch', True)
            trainer.load_checkpoint(checkpoint_path, resume_training_state=resume_training_state)
            if resume_training_state:
                logger.info(f"Checkpoint loaded. Resuming from epoch {trainer.current_epoch}, global_step {trainer.global_step}")
            else:
                logger.info("Checkpoint loaded. Starting new training from epoch 0 with pretrained weights")
        else:
            logger.warning(f"Checkpoint path specified but file not found: {checkpoint_path}")

    # Train
    logger.info("Starting training...")
    trainer.fit()

    logger.info("Training completed successfully")


if __name__ == "__main__":
    train_model()