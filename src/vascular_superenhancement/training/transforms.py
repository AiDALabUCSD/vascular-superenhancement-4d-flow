import torchio as tio
import logging
import torch
import random
import math

# get the logger
logger = logging.getLogger(__name__)

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
        tio.RescaleIntensity(out_min_max=(-1, 1), in_min_max=(-1*cfg.data.vel_cap, cfg.data.vel_cap), include=["flow_vx", "flow_vy", "flow_vz"]), 
        
        
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
            tio.RandomGamma(
                log_gamma=(cfg.train.log_gamma_min, cfg.train.log_gamma_max),
                include=["mag"],
                p=cfg.train.gamma_probability
            ),
            
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


def apply_sphere_inversion_to_patch(
    mag_tensor: torch.Tensor,
    radius_range: tuple = (30, 60),
    alpha: float = 1.0,
    center_jitter: int = 5,
    p: float = 0.5,
    falloff_fraction: float = 0.5
) -> torch.Tensor:
    """
    Apply sphere inversion augmentation to a patch's mag image.
    
    This function creates a sphere near the center of the patch, inverts the values
    within the sphere, and alpha blends the inverted sphere with the original patch.
    Uses a smooth cosine falloff at the sphere boundary for natural-looking results.
    
    Supports both single patches [1, D, H, W] and batched patches [B, 1, D, H, W].
    Each patch in the batch is processed independently.
    
    Args:
        mag_tensor: Mag image tensor of shape [1, D, H, W] or [B, 1, D, H, W]
        radius_range: Tuple of (min_radius, max_radius) in voxels
        alpha: Alpha blending factor (0-1), where 1.0 means fully inverted, 0.0 means original
        center_jitter: Maximum jitter in each dimension from patch center (in voxels)
        p: Probability of applying the augmentation to each patch (0-1)
        falloff_fraction: Fraction of radius to use for the smooth falloff (0-1)
        
    Returns:
        Augmented mag tensor of the same shape as input
    """
    
    # Handle both batched and unbatched tensors
    if mag_tensor.dim() == 4:
        is_batched = False
        mag_tensor = mag_tensor.unsqueeze(0)
    elif mag_tensor.dim() == 5:
        is_batched = True
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {mag_tensor.shape}")
    
    batch_size = mag_tensor.shape[0]
    _, depth, height, width = mag_tensor.shape[1:]
    
    # Process each patch in the batch independently
    augmented_patches = []
    for b in range(batch_size):
        patch = mag_tensor[b:b+1]  # [1, D, H, W]
        
        # Apply augmentation with probability p for each patch
        if random.random() > p:
            augmented_patches.append(patch)
            continue
        
        # Calculate patch center
        center_d = depth / 2.0
        center_h = height / 2.0
        center_w = width / 2.0
        
        # Add random jitter to center
        jitter_d = random.uniform(-center_jitter, center_jitter)
        jitter_h = random.uniform(-center_jitter, center_jitter)
        jitter_w = random.uniform(-center_jitter, center_jitter)
        
        sphere_center = (
            center_d + jitter_d,
            center_h + jitter_h,
            center_w + jitter_w
        )
        
        # Random radius within range
        radius = random.uniform(radius_range[0], radius_range[1])
        
        # Create coordinate grids
        d_coords = torch.arange(depth, dtype=torch.float32, device=mag_tensor.device)
        h_coords = torch.arange(height, dtype=torch.float32, device=mag_tensor.device)
        w_coords = torch.arange(width, dtype=torch.float32, device=mag_tensor.device)
        
        # Create meshgrid
        d_grid, h_grid, w_grid = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
        
        # Calculate distances from sphere center
        distances = torch.sqrt(
            (d_grid - sphere_center[0]) ** 2 +
            (h_grid - sphere_center[1]) ** 2 +
            (w_grid - sphere_center[2]) ** 2
        )
        
        # Cosine falloff for smooth transition
        falloff_width = radius * falloff_fraction
        inner_radius = radius - falloff_width
        
        # Normalized distance through the falloff region (0 at inner_radius, 1 at radius)
        normalized = torch.clamp((distances - inner_radius) / falloff_width, 0, 1)
        
        # Cosine falloff: 1 at inner_radius, 0 at radius, smooth transition between
        sphere_mask = 0.5 * (1 + torch.cos(math.pi * normalized))
        
        # Ensure fully 1 inside inner radius (handles numerical precision)
        sphere_mask = torch.where(distances < inner_radius, torch.ones_like(sphere_mask), sphere_mask)
        
        # Ensure mask has the same shape as patch [1, D, H, W]
        sphere_mask = sphere_mask.unsqueeze(0)
        
        # Invert the mag values (assuming normalized to [0, 1])
        inverted_patch = 1.0 - patch
        
        # Alpha blend: result = alpha * inverted + (1 - alpha) * original
        # Only apply blending within the sphere
        augmented_patch = patch * (1 - sphere_mask * alpha) + inverted_patch * (sphere_mask * alpha)
        
        augmented_patches.append(augmented_patch)
    
    # Concatenate all patches back into a batch
    result = torch.cat(augmented_patches, dim=0)
    
    # Remove batch dimension if input was unbatched
    if not is_batched:
        result = result.squeeze(0)
    
    return result