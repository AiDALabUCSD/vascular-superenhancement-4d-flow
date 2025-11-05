import torchio as tio
import logging
import torch

# get the logger
logger = logging.getLogger(__name__)

class MaskedContrastInversion(tio.Transform):
    """
    Create a mask from the speed image and invert contrast in the magnitude image
    within the masked region.
    
    Args:
        speed_threshold: Threshold value for creating the binary mask
        invert_inside: If True, invert inside the mask (speed > threshold)
                      If False, invert outside the mask (speed <= threshold)
        p: Probability of applying the transform (for augmentation)
    """
    def __init__(
        self, 
        speed_threshold: float = 0.1, 
        invert_inside: bool = True,
        p: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.speed_threshold = speed_threshold
        self.invert_inside = invert_inside
        self.p = p
    
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # Skip with probability (1 - p)
        if torch.rand(1).item() > self.p:
            return subject
        
        # Get flow components
        fvx = subject['flow_vx'].data
        fvy = subject['flow_vy'].data
        fvz = subject['flow_vz'].data
        mag = subject['mag'].data.clone()  # Clone to avoid modifying original
        
        # Compute speed
        speed = torch.sqrt(fvx**2 + fvy**2 + fvz**2)
        
        # Create binary mask
        if self.invert_inside:
            mask = speed > self.speed_threshold
        else:
            mask = speed <= self.speed_threshold
        
        # Invert contrast in masked region
        # Assuming mag is normalized to [0, 1] range
        # If not normalized, adjust accordingly
        mag[mask] = 1.0 - mag[mask]
        
        # Update the subject with modified magnitude
        subject['mag'].set_data(mag)
        
        return subject

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
        
        # (TODO #4): if flow data augmentation is needed, add an if statement to check if we need training or validation data
        # and then apply the rescale intensity transform accordingly. ie if train, then apply the rescale intensity transform
        # in the train section below, and if not, apply it in the base transforms above.
        tio.RescaleIntensity(out_min_max=(-1, 1),
            in_min_max=(-1*cfg.data.vel_cap, cfg.data.vel_cap),
            include=["flow_vx", "flow_vy", "flow_vz"]
        ), 
        
        
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
            # tio.RandomBlur(p=0.5),
            # tio.RandomGhosting(p=0.5),
            # elastic deformation
            # tio.RandomGamma(
            #     log_gamma=(cfg.train.log_gamma_min, cfg.train.log_gamma_max),
            #     include=["mag"],
            #     p=cfg.train.gamma_probability
            # ),
            # Mutually exclusive magnitude augmentations
            tio.OneOf({
                MaskedContrastInversion(
                    speed_threshold=cfg.train.contrast_inversion_threshold,
                    invert_inside=cfg.train.contrast_inversion_inside,
                    p=1.0  # Always apply if chosen
                ): 1,  # Equal weight
                tio.RandomGamma(
                    log_gamma=(cfg.train.log_gamma_min, cfg.train.log_gamma_max),
                    include=["mag"],
                    p=1.0  # Always apply if chosen
                ): 1,  # Equal weight
            }, p=cfg.train.mag_contrast_aug_prob),  # 50% chance one is applied

            
            tio.RandomElasticDeformation(
                num_control_points=cfg.train.num_control_points,
                max_displacement=cfg.train.max_displacement,
                p=cfg.train.elastic_deformation_probability),
            
            tio.Clamp(out_min=0, out_max=1, include=["cine", "mag"]),
            tio.Clamp(out_min=-1, out_max=1, include=["flow_vx", "flow_vy", "flow_vz"]),
        ]
    
    for i, transform in enumerate(transforms):
        logger.info(f"Transform {i}: {transform}")
        if isinstance(transform, tio.RandomElasticDeformation):
            logger.info(f"Number of control points: {transform.num_control_points}")
            logger.info(f"Maximum displacement: {transform.max_displacement}")
    
    return tio.Compose(transforms)


# how to use the transforms
# transforms = build_transforms(cfg)
# subject = transforms(subject)