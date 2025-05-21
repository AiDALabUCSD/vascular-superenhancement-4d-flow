import torch
import torchio as tio
from torch.utils.data import DataLoader


def build_train_loader(dataset: tio.SubjectsDataset, cfg) -> DataLoader:
    """
    Build a TorchIO patch-based DataLoader using GridSampler and Queue.
    The GridSampler will create patches of size patch_size from each subject,
    regardless of the subject's original shape.
    """
    # Create a sampler that will be applied to each subject
    sampler = tio.data.GridSampler(
        patch_size=cfg.train.patch_size
    )

    queue = tio.Queue(
        subjects_dataset=dataset,
        max_length=cfg.train.queue_length,
        samples_per_volume=cfg.train.samples_per_volume,
        sampler=sampler,
        num_workers=cfg.train.num_queue_workers,
        shuffle_subjects=True,
        shuffle_patches=True
    )

    loader = DataLoader(
        queue,
        pin_memory=True,
        num_workers=cfg.train.num_loader_workers,
        batch_size=cfg.train.batch_size
    )

    return loader


# Example config values expected in cfg.train:
# patch_size: [96, 96, 96]
# batch_size: 2
# queue_length: 100
# samples_per_volume: 8
# num_workers: 4
