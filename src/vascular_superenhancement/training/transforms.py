import torchio as tio
import logging
import torch
import random
import math
from typing import List, Tuple

# get the logger
logger = logging.getLogger(__name__)


def get_multi_timepoint_image_keys(window_size: int = 5) -> Tuple[List[str], List[str], List[str]]:
    """
    Get the image key names for multi-timepoint subjects.

    Args:
        window_size: Number of timepoints in the window

    Returns:
        Tuple of (mag_keys, cine_keys, speed_keys) where each is a list of strings
    """
    mag_keys = [f'mag_t{i}' for i in range(window_size)]
    cine_keys = [f'cine_t{i}' for i in range(window_size)]
    speed_keys = [f'speed_t{i}' for i in range(window_size)]
    return mag_keys, cine_keys, speed_keys


def build_multi_timepoint_transforms(cfg, train: bool = True):
    """
    Build a TorchIO transform pipeline for multi-timepoint 3D flow + cine data.

    This function creates transforms that apply consistently across all timepoints
    in a temporal window. Spatial transforms (like elastic deformation) apply the
    same deformation to all images in a subject, ensuring temporal consistency.

    Args:
        cfg: Hydra configuration object
        train: Whether to include training augmentations

    Returns:
        tio.Compose transform pipeline
    """
    spacing = cfg.data.spacing
    window_size = cfg.train.get('temporal_window_size', 5)

    # Get image keys for all timepoints
    mag_keys, cine_keys, speed_keys = get_multi_timepoint_image_keys(window_size)
    
    # Speed cap: max possible speed is sqrt(vx^2 + vy^2 + vz^2) where each can be vel_cap
    # So max speed = sqrt(3) * vel_cap, but we use 1.5 * vel_cap as a practical cap
    speed_cap = cfg.data.get('speed_cap', 1.5 * cfg.data.vel_cap)

    # Preprocessing transforms
    transforms = [
        # tio.Resample(spacing),
        tio.RescaleIntensity(out_min_max=(0, 1), include=cine_keys + mag_keys),
        # Speed is non-negative, rescale from [0, speed_cap] to [0, 1]
        tio.RescaleIntensity(
            out_min_max=(0, 1),
            in_min_max=(0, speed_cap),
            include=speed_keys
        ),
    ]

    # Training augmentations
    if train:
        transforms += [
            # Gamma augmentation - apply to all mag images
            tio.RandomGamma(
                log_gamma=(cfg.train.log_gamma_min, cfg.train.log_gamma_max),
                include=mag_keys,
                p=cfg.train.gamma_probability
            ),

            # Elastic deformation - applies consistently to ALL images in subject
            # This is critical for temporal consistency
            tio.RandomElasticDeformation(
                num_control_points=cfg.train.num_control_points,
                max_displacement=cfg.train.max_displacement,
                p=cfg.train.elastic_deformation_probability
            ),

            # Clamp values to valid ranges
            tio.Clamp(out_min=0, out_max=1, include=cine_keys + mag_keys + speed_keys),
        ]

    for i, transform in enumerate(transforms):
        logger.info(f"Multi-timepoint Transform {i}: {transform}")
        if isinstance(transform, tio.RandomElasticDeformation):
            logger.info(f"  Number of control points: {transform.num_control_points}")
            logger.info(f"  Maximum displacement: {transform.max_displacement}")

    return tio.Compose(transforms)


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
    
    # Speed cap: max possible speed is sqrt(vx^2 + vy^2 + vz^2) where each can be vel_cap
    # So max speed = sqrt(3) * vel_cap, but we use 1.5 * vel_cap as a practical cap
    speed_cap = cfg.data.get('speed_cap', 1.5 * cfg.data.vel_cap)

    # Preprocessing transforms
    transforms = [
        # tio.Resample(spacing),
        tio.RescaleIntensity(out_min_max=(0, 1), include=["cine", "mag"]),
        # Speed is non-negative, rescale from [0, speed_cap] to [0, 1]
        tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(0, speed_cap), include=["speed"]), 
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
            
            tio.Clamp(out_min=0, out_max=1, include=["cine", "mag", "speed"]),
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


def get_multi_timepoint_inference_keys(window_size: int = 5) -> Tuple[List[str], List[str]]:
    """
    Get the image key names for multi-timepoint inference subjects.

    Args:
        window_size: Number of timepoints in the window

    Returns:
        Tuple of (mag_keys, flow_keys) where each is a list of strings
    """
    mag_keys = [f'mag_t{i}' for i in range(window_size)]
    flow_keys = []
    for i in range(window_size):
        flow_keys.extend([f'flow_vx_t{i}', f'flow_vy_t{i}', f'flow_vz_t{i}'])
    return mag_keys, flow_keys


def build_inference_transforms(cfg, multi_timepoint: bool = True):
    """
    Build a TorchIO transform pipeline for inference/visualization.
    
    For inference, we use velocity data (vx, vy, vz) and compute speed on the fly.
    This is different from training transforms which use precomputed speed.

    Args:
        cfg: Hydra configuration object
        multi_timepoint: Whether to use multi-timepoint mode

    Returns:
        tio.Compose transform pipeline
    """
    spacing = cfg.data.spacing
    window_size = cfg.train.get('temporal_window_size', 5)

    if multi_timepoint:
        mag_keys, flow_keys = get_multi_timepoint_inference_keys(window_size)
    else:
        mag_keys = ['mag']
        flow_keys = ['flow_vx', 'flow_vy', 'flow_vz']

    # Preprocessing transforms only (no augmentation for inference)
    transforms = [
        # tio.Resample(spacing),
        tio.RescaleIntensity(out_min_max=(0, 1), include=mag_keys),
        # Rescale velocity components from [-vel_cap, vel_cap] to [-1, 1]
        tio.RescaleIntensity(
            out_min_max=(-1, 1),
            in_min_max=(-cfg.data.vel_cap, cfg.data.vel_cap),
            include=flow_keys
        ),
    ]

    for i, transform in enumerate(transforms):
        logger.info(f"Inference Transform {i}: {transform}")

    return tio.Compose(transforms)