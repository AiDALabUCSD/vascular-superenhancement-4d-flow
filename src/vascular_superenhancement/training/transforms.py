import torchio as tio

def build_transforms(cfg, train: bool = True):
    """
    Build a TorchIO transform pipeline for 3D flow + cine data.
    Applies:
    - Resampling to fixed physical spacing
    - Intensity normalization
    - Spatial padding/cropping to ensure patch compatibility
    - Optional augmentations (can be added later)
    """
    spacing = cfg.data.spacing         # e.g., [1.4, 1.4, 1.4]

    # Preprocessing transforms
    transforms = [
        tio.Resample(spacing),
        tio.RescaleIntensity(out_min_max=(0, 1), include=["cine", "mag"]),
        tio.RescaleIntensity(out_min_max=(-1, 1), include=["flow_vx", "flow_vy", "flow_vz"]),
        # tio.ZNormalization(),
        # tio.CropOrPad(patch_size),
        # You can add augmentations here later, like:
        # tio.RandomFlip(axes=('Left',), flip_probability=0.5),
        # tio.RandomBlur(p=0.5),
        # tio.RandomGhosting(p=0.5),
        # tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
    ]
    
    # Subject Level Augmentation transforms
    if train:
        transforms += [
            # tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
            # tio.RandomFlip(axes=('Left',), flip_probability=0.5),
            tio.RandomBlur(p=0.5),
            # tio.RandomGhosting(p=0.5),
            tio.Clamp(out_min=0, out_max=1, include=["cine", "mag"]),
            tio.Clamp(out_min=-1, out_max=1, include=["flow_vx", "flow_vy", "flow_vz"]),
        ]
        
    return tio.Compose(transforms)


# how to use the transforms
# transforms = build_transforms(cfg)
# subject = transforms(subject)