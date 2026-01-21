# import torch
import torchio as tio
from torchio import SubjectsLoader
from typing import Optional, Iterator
from torch.utils.data.sampler import Sampler
from vascular_superenhancement.training.transforms import apply_sphere_inversion_to_patch


class PatchAugmentedLoader:
    """
    Wrapper around SubjectsLoader that applies patch-level sphere inversion augmentation
    to batches after they're loaded.
    
    This wrapper applies augmentation only to the mag image in each batch, with configurable
    parameters for radius range, alpha blending, center jitter, and probability.
    """
    
    def __init__(
        self,
        loader: SubjectsLoader,
        radius_range: tuple = (5, 15),
        alpha: float = 0.7,
        center_jitter: int = 5,
        p: float = 0.5,
        enable_previews: bool = True
    ):
        """
        Initialize the patch augmented loader.
        
        Args:
            loader: The underlying SubjectsLoader to wrap
            radius_range: Tuple of (min_radius, max_radius) in voxels
            alpha: Alpha blending factor (0-1)
            center_jitter: Maximum jitter from patch center in voxels
            p: Probability of applying augmentation to each patch
        """
        self.loader = loader
        self.radius_range = radius_range
        self.alpha = alpha
        self.center_jitter = center_jitter
        self.p = p
        self.enable_previews = enable_previews
    
    def __iter__(self) -> Iterator:
        """Return an iterator that applies augmentation to each batch."""
        for batch in self.loader:
            # Apply augmentation to mag image if it exists in the batch
            if "mag" in batch and tio.DATA in batch["mag"]:
                mag_tensor = batch["mag"][tio.DATA]
                # Save pre-augmentation copy for previews
                if self.enable_previews:
                    batch["mag_pre_aug"] = {
                        tio.DATA: mag_tensor.clone(),
                        tio.AFFINE: batch["mag"][tio.AFFINE],
                    }
                augmented_mag = apply_sphere_inversion_to_patch(
                    mag_tensor=mag_tensor,
                    radius_range=self.radius_range,
                    alpha=self.alpha,
                    center_jitter=self.center_jitter,
                    p=self.p
                )
                # Update the batch with augmented mag
                batch["mag"][tio.DATA] = augmented_mag
            
            yield batch
    
    def __len__(self) -> int:
        """Return the length of the underlying loader."""
        return len(self.loader)


def build_train_loader(dataset: tio.SubjectsDataset, cfg, subject_sampler: Optional[Sampler] = None, train: bool = True):
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
    
    # Apply patch-level sphere inversion augmentation during training
    if train and cfg.train.get('sphere_inversion_probability', 0) > 0:
        loader = PatchAugmentedLoader(
            loader=loader,
            radius_range=(
                cfg.train.get('sphere_inversion_radius_min', 5),
                cfg.train.get('sphere_inversion_radius_max', 15)
            ),
            alpha=cfg.train.get('sphere_inversion_alpha', 0.7),
            center_jitter=cfg.train.get('sphere_inversion_center_jitter', 5),
            p=cfg.train.get('sphere_inversion_probability', 0.5),
            enable_previews=cfg.wandb.get('log_patch_previews', False)
        )

    return loader
