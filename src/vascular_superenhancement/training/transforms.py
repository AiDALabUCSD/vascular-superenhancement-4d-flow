import torchio as tio

def build_transforms(cfg):
    """
    Build a TorchIO transform pipeline for 3D flow + cine data.
    Applies:
    - Resampling to fixed physical spacing
    - Intensity normalization
    - Spatial padding/cropping to ensure patch compatibility
    - Optional augmentations (can be added later)
    """
    patch_size = cfg.train.patch_size  # e.g., [96, 96, 96]
    spacing = cfg.data.spacing         # e.g., [1.4, 1.4, 1.4]

    transforms = tio.Compose([
        tio.Resample(spacing),
        tio.ZNormalization(),
        tio.CropOrPad(patch_size),
        # You can add augmentations here later, like:
        # tio.RandomFlip(axes=('LR',)),
        # tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
    ])

    return transforms