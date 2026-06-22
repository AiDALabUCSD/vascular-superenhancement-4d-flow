"""Training script using callback-based trainer architecture.

Supports three training modes:
1. GAN training (trainer_type='gan'): Single-timepoint GAN training
2. Generator-only training (trainer_type='generator'): Multi-timepoint generator-only
3. Dual-task training (trainer_type='dual_task'): Full-volume downsampled cine
   enhancement + phase error correction with shared backbone and dual output heads

And two data paradigms:
1. Patch-based (modes 1 & 2): TorchIO Queue + UniformSampler on resampled data
2. Full-volume (mode 3): Standard DataLoader on precomputed downsampled 128x128x64 data
"""

from pathlib import Path
from datetime import timedelta
import os
import hydra
from omegaconf import DictConfig
import logging
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from vascular_superenhancement.training.datasets import (
    build_subjects_dataset,
    build_multi_timepoint_subjects_dataset,
    build_downsampled_dataset,
)
from vascular_superenhancement.training.transforms import (
    build_transforms,
    build_multi_timepoint_transforms,
    build_downsampled_transforms,
)
from vascular_superenhancement.training.dataloading import build_train_loader, build_standard_loader
from vascular_superenhancement.training.trainers.gan_trainer import GanTrainer
from vascular_superenhancement.training.trainers.generator_trainer import GeneratorTrainer
from vascular_superenhancement.training.trainers.dual_task_trainer import DualTaskTrainer
from vascular_superenhancement.training.callbacks.wandb_callback import WandbCallback
from vascular_superenhancement.training.callbacks.checkpoint_callback import CheckpointCallback
from vascular_superenhancement.training.callbacks.visualization_callback import VisualizationCallback
from vascular_superenhancement.training.callbacks.flow_validation_callback import FlowValidationCallback
from vascular_superenhancement.training.callbacks.batch_metrics_callback import BatchMetricsCallback
from vascular_superenhancement.training.callbacks.patch_preview_callback import PatchPreviewCallback

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.1",
    config_path=str((Path(__file__).resolve().parents[3] / "hydra_configs").as_posix()),
    config_name="config"
)
def train_model(cfg: DictConfig):
    """Main training function using callback-based trainer architecture.

    Supports three modes based on configuration:
    1. Standard mode (trainer_type='gan'): Single-timepoint GAN training
    2. Temporal mode (trainer_type='generator', use_multi_timepoint=True):
       Multi-timepoint generator-only training
    3. Dual-task mode (trainer_type='dual_task'):
       Full-volume downsampled cine enhancement + phase error correction
    """

    # DDP: detect torchrun environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1

    if is_ddp:
        # Rank-0-only callbacks (visualization, in-loop flow validation) can keep
        # the main process busy for many minutes while the other ranks idle at the
        # next collective. The default NCCL watchdog timeout (10 min) trips in that
        # window and SIGABRTs the idle ranks, so give the process group generous
        # headroom. Override via DDP_TIMEOUT_MIN if needed.
        ddp_timeout_min = int(os.environ.get("DDP_TIMEOUT_MIN", "120"))
        dist.init_process_group(
            backend="nccl", timeout=timedelta(minutes=ddp_timeout_min)
        )
        torch.cuda.set_device(local_rank)

    is_main_process = (rank == 0)

    # Set up logging level based on debug flag
    if cfg.train.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        if is_main_process:
            logger.info("Debug logging enabled")
    else:
        log_level = logging.INFO if is_main_process else logging.WARNING
        logging.getLogger().setLevel(log_level)
        logger.setLevel(log_level)

    if is_main_process:
        logger.info("Setting up training...")
        logger.info(f"Current working directory: {Path.cwd()}")
        if is_ddp:
            logger.info(f"DDP enabled: world_size={world_size}, backend=nccl")

    # Determine training mode
    trainer_type = cfg.train.get('trainer_type', 'gan')
    use_multi_timepoint = cfg.train.get('use_multi_timepoint', False)

    logger.info(f"Training mode: trainer_type={trainer_type}, multi_timepoint={use_multi_timepoint}")

    # Apply patient limit if set (for debugging / smoke tests).
    # Two mutually-exclusive ways to subset the training set:
    #   * ``debug_train_patient_ids`` (list[str]): explicit IDs, in order.
    #     Takes precedence when both are set. Use this to deliberately mix
    #     specific subjects (e.g. some with cine and some without).
    #   * ``debug_max_train_patients`` (int): take the first N from the
    #     split CSV.
    explicit_train_ids = cfg.train.get('debug_train_patient_ids', None)
    train_patient_limit = cfg.train.get('debug_max_train_patients', None)
    if explicit_train_ids:
        limited_patient_ids = list(explicit_train_ids)
        logger.warning(
            f"DEBUG: Restricting training to explicit patient list "
            f"({len(limited_patient_ids)} patients): {limited_patient_ids}"
        )
    elif train_patient_limit is not None:
        import pandas as pd
        df = pd.read_csv(cfg.data.splits_path)
        all_train_ids = df[df.split == "train"].patient_id.tolist()
        limited_patient_ids = all_train_ids[:train_patient_limit]
        logger.warning(f"DEBUG: Limiting training to {len(limited_patient_ids)} patients: {limited_patient_ids}")
    else:
        limited_patient_ids = None

    # Mirror subsetting for validation (smoke runs benefit from a tiny val
    # set too). Only ``debug_validation_patient_ids`` is supported -- a
    # count-based variant wasn't needed historically.
    explicit_val_ids = cfg.train.get('debug_validation_patient_ids', None)
    limited_val_ids = list(explicit_val_ids) if explicit_val_ids else None
    if limited_val_ids:
        logger.warning(
            f"DEBUG: Restricting validation to explicit patient list "
            f"({len(limited_val_ids)} patients): {limited_val_ids}"
        )

    # =====================================================================
    # Dual-task mode: full-volume downsampled training
    # =====================================================================
    if trainer_type == 'dual_task':
        logger.info("Using DualTaskTrainer (cine enhancement + phase error correction)")
        logger.info(f"Temporal mag offsets: {list(cfg.train.temporal_mag_offsets)}")

        # Transforms
        training_transforms = build_downsampled_transforms(cfg, train=True)
        validation_transforms = build_downsampled_transforms(cfg, train=False)

        # Datasets (debug patient limit only applies to training, not validation)
        # Drop patients with known-corrupt correction targets (isolated >VENC
        # speckle in the diff field) so they don't inject loss/grad spikes.
        exclude_train_ids = list(cfg.train.get("exclude_train_patient_ids", []) or [])
        if exclude_train_ids:
            logger.info(f"Excluding {len(exclude_train_ids)} patient(s) from training: {exclude_train_ids}")
        training_dataset = build_downsampled_dataset(
            cfg, split="train", transforms=training_transforms, patient_ids=limited_patient_ids,
            exclude_patient_ids=exclude_train_ids if exclude_train_ids else None,
        )
        logger.info(f"Training dataset length (dual-task): {len(training_dataset)}")

        # Exclude visualization-only patients (they lack ground-truth targets)
        viz_only_ids = list(cfg.train.get('visualization_patient_ids', []))
        validation_dataset = build_downsampled_dataset(
            cfg, split="validation", transforms=validation_transforms,
            time_index=cfg.train.validation_time_index,
            patient_ids=limited_val_ids,
            exclude_patient_ids=viz_only_ids if viz_only_ids else None,
        )
        logger.info(f"Validation dataset length (dual-task): {len(validation_dataset)}")

        # Loaders (no Queue, no patching)
        train_sampler = DistributedSampler(training_dataset, shuffle=True) if is_ddp else None
        training_loader = build_standard_loader(training_dataset, cfg, train=True, sampler=train_sampler)
        validation_loader = build_standard_loader(validation_dataset, cfg, train=False)

        if is_main_process:
            logger.info(f"Training batches: {len(training_loader)}, Validation batches: {len(validation_loader)}")

        # Callbacks (no PatchPreviewCallback -- no patches or sphere inversion).
        # FlowValidationCallback self-disables unless cfg.train.flow_validation.enabled.
        callbacks = [
            WandbCallback(cfg),
            CheckpointCallback(cfg),
            VisualizationCallback(cfg),
            FlowValidationCallback(cfg),
            BatchMetricsCallback(cfg),
        ]

        trainer = DualTaskTrainer(
            cfg=cfg,
            train_loader=training_loader,
            train_dataset=training_dataset,
            val_loader=validation_loader,
            val_dataset=validation_dataset,
            callbacks=callbacks,
            local_rank=local_rank,
            rank=rank,
            world_size=world_size,
            train_sampler=train_sampler,
        )

    # =====================================================================
    # Existing modes: GAN / Generator-only
    # =====================================================================
    else:
        if use_multi_timepoint:
            logger.info(f"Temporal window size: {cfg.train.temporal_window_size}")

        # Build transforms based on mode
        if use_multi_timepoint:
            training_transforms = build_multi_timepoint_transforms(cfg, train=True)
            validation_transforms = build_multi_timepoint_transforms(cfg, train=False)
        else:
            training_transforms = build_transforms(cfg, train=True)
            validation_transforms = build_transforms(cfg, train=False)

        logger.info(f"Training transforms: {training_transforms}")
        logger.info(f"Validation transforms: {validation_transforms}")

        # Build training dataset
        if use_multi_timepoint:
            training_dataset = build_multi_timepoint_subjects_dataset(
                cfg,
                split="train",
                transforms=training_transforms,
                include_all_timepoints=True,
                patient_ids=limited_patient_ids,
            )
            logger.info(f"Training dataset length (multi-timepoint): {len(training_dataset)}")
        else:
            training_dataset = build_subjects_dataset(
                cfg,
                split="train",
                transforms=training_transforms,
                include_all_timepoints=cfg.train.timepoints_as_augmentation,
                patient_ids=limited_patient_ids,
            )
            logger.info(f"Training dataset length: {len(training_dataset)}")

        # Build dataloader
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
                cfg,
                split="validation",
                transforms=validation_transforms,
                time_index=cfg.train.validation_time_index,
            )
            logger.info(f"Validation dataset length (multi-timepoint, center timepoint {cfg.train.validation_time_index}): {len(validation_dataset)}")
        else:
            validation_dataset = build_subjects_dataset(
                cfg,
                split="validation",
                transforms=validation_transforms,
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

        # Build trainer
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
            resume_training_state = cfg.train.get('resume_from_checkpoint_epoch', False)
            trainer.load_checkpoint(checkpoint_path, resume_training_state=resume_training_state)
            if resume_training_state:
                logger.info(f"Checkpoint loaded. Resuming from epoch {trainer.current_epoch}, global_step {trainer.global_step}")
            else:
                logger.info("Checkpoint loaded. Starting new training from epoch 0 with pretrained weights")
        else:
            logger.warning(f"Checkpoint path specified but file not found: {checkpoint_path}")

    # Train
    if is_main_process:
        logger.info("Starting training...")
    try:
        trainer.fit()
        if is_main_process:
            logger.info("Training completed successfully")
    finally:
        if is_ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    train_model()