"""New training script developed in the modularize-training-script branch
    which utilizes base_trainer and base_callbacks."""

from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging

from vascular_superenhancement.training.datasets import build_subjects_dataset, TimepointCyclingSampler
from vascular_superenhancement.training.transforms import build_transforms
from vascular_superenhancement.training.dataloading import build_train_loader
from vascular_superenhancement.training.trainers.gan_trainer import GanTrainer
from vascular_superenhancement.training.callbacks.wandb_callback import WandbCallback
from vascular_superenhancement.training.callbacks.checkpoint_callback import CheckpointCallback
from vascular_superenhancement.training.callbacks.visualization_callback import VisualizationCallback

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.1",
    config_path=str((Path(__file__).resolve().parents[3] / "hydra_configs").as_posix()),
    config_name="config"
)
def train_model(cfg: DictConfig):
    """Main training function using callback-based trainer architecture."""
    
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
    
    # Build transforms
    training_transforms = build_transforms(cfg, train=True)
    validation_transforms = build_transforms(cfg, train=False)
    logger.info(f"Training transforms: {training_transforms}")
    logger.info(f"Validation transforms: {validation_transforms}")
    
    # Build training dataset and dataloader
    if cfg.train.timepoints_as_augmentation:
        training_dataset = build_subjects_dataset(
            "train",
            Path(cfg.data.splits_path),
            cfg.path_config.path_config_name,
            transforms=training_transforms,
            debug=cfg.train.debug,
            include_all_timepoints=True,
            peak_systolic_only=cfg.train.get('peak_systolic_only', False),
        )
        logger.info(f"Training dataset length: {len(training_dataset)}")
        
        subject_sampler = TimepointCyclingSampler(
            training_dataset, 
            num_timepoints=20, 
            shuffle_within_timepoint=True
        )
        training_loader = build_train_loader(training_dataset, cfg, subject_sampler=subject_sampler, train=True)
    else:
        training_dataset = build_subjects_dataset(
            "train",
            Path(cfg.data.splits_path),
            cfg.path_config.path_config_name,
            transforms=training_transforms,
            debug=cfg.train.debug,
            include_all_timepoints=False,
        )
        logger.info(f"Training dataset length: {len(training_dataset)}")
        training_loader = build_train_loader(training_dataset, cfg, subject_sampler=None, train=True)
    
    logger.info(f"Number of batches in training loader: {len(training_loader)}")
    
    # Build validation dataset and dataloader
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
    ]
    
    # Build trainer
    trainer = GanTrainer(
        cfg=cfg,
        train_loader=training_loader,
        train_dataset=training_dataset,
        val_loader=validation_loader,
        val_dataset=validation_dataset,
        callbacks=callbacks
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit()
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    train_model()