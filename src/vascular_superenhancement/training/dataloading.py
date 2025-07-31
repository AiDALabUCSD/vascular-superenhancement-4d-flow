# import torch
import torchio as tio
from torchio import SubjectsLoader
from .datasets import TimepointCyclingSampler
from typing import Optional
from torch.utils.data.sampler import Sampler


def build_train_loader(dataset: tio.SubjectsDataset, cfg, subject_sampler: Optional[Sampler] = None) -> SubjectsLoader:
    """
    Build a TorchIO patch-based DataLoader using UniformSampler and Queue.
    The UniformSampler will randomly sample patches of size patch_size from each subject,
    providing more variety in the training data.
    """
    # Create a sampler that will be applied to each subject
    patch_sampler = tio.UniformSampler(
        patch_size=cfg.train.patch_size
    )
    
    if subject_sampler is not None:
        shuffle_subjects = False
    else:
        shuffle_subjects = cfg.train.shuffle_subjects
    

    queue = tio.Queue(
        subjects_dataset=dataset,
        max_length=cfg.train.queue_length,
        samples_per_volume=cfg.train.samples_per_volume,
        sampler=patch_sampler,
        subject_sampler=subject_sampler,
        num_workers=cfg.train.num_queue_workers,
        shuffle_subjects=shuffle_subjects,
        shuffle_patches=cfg.train.shuffle_patches,
    )

    loader = SubjectsLoader(
        queue,
        pin_memory=cfg.train.pin_memory,
        num_workers=cfg.train.num_loader_workers,
        batch_size=cfg.train.batch_size,
    )

    return loader
