import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import SSIMLoss


# =============================================================================
# Existing single-task losses (backwards compatible)
# =============================================================================


def discriminator_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    """
    Standard BCE-based PatchGAN discriminator loss.
    real_pred: output of D(real_input) → should be all 1s
    fake_pred: output of D(fake_input) → should be all 0s
    """
    loss_real = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
    loss_fake = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
    return 0.5 * (loss_real + loss_fake)


def generator_gan_loss(fake_pred: torch.Tensor) -> torch.Tensor:
    """
    GAN loss for generator: tries to fool discriminator (i.e., make it predict all 1s)
    """
    return F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))


def generator_l1_loss(fake_img: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
    """
    L1 reconstruction loss between generated and real cine images.
    """
    return F.l1_loss(fake_img, real_img)


def generator_ssim_loss(fake_img: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
    """
    SSIM-based reconstruction loss using MONAI's 3D SSIM implementation.
    
    Args:
        fake_img: Generated image tensor [B, C, D, H, W]
        real_img: Target image tensor [B, C, D, H, W]
    
    Returns:
        SSIM loss value (1 - SSIM for minimization)
    """
    return SSIMLoss(spatial_dims=3)(fake_img, real_img)


# =============================================================================
# Dual-task losses (cine enhancement + phase error correction)
# =============================================================================


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked L1 loss normalised by the number of masked voxels.

    Only voxels where ``mask > 0`` contribute to the loss, and the result is
    divided by ``mask.sum()`` (not the total number of voxels) so the loss
    magnitude is independent of the mask size.

    Args:
        pred: Predicted tensor ``[B, C, D, H, W]``
        target: Ground-truth tensor ``[B, C, D, H, W]``
        mask: Binary mask tensor ``[B, 1, D, H, W]`` (broadcastable to pred)

    Returns:
        Scalar masked L1 loss.
    """
    mask_sum = mask.sum().clamp(min=1.0)
    return (torch.abs(pred - target) * mask).sum() / mask_sum


def outside_mask_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """L1 loss over voxels *outside* the mask, normalised by their count.

    Complement of :func:`masked_l1_loss`: only voxels where ``mask == 0``
    contribute.  Useful for regularising predictions in regions that lack
    ground-truth cine data (e.g. superior slices above the cine FOV).

    Args:
        pred: Predicted tensor ``[B, C, D, H, W]``
        target: Pseudo-target tensor ``[B, C, D, H, W]`` (e.g. input magnitude)
        mask: Binary mask tensor ``[B, 1, D, H, W]`` (broadcastable to pred)

    Returns:
        Scalar L1 loss over the unmasked region.
    """
    inv_mask = 1.0 - mask
    inv_mask_sum = inv_mask.sum().clamp(min=1.0)
    return (torch.abs(pred - target) * inv_mask).sum() / inv_mask_sum


def bbox_ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """SSIM loss computed on the tight 3D bounding box of a binary mask.

    SSIM uses local sliding windows, so naively zeroing out-of-mask voxels
    corrupts boundary statistics.  Instead, this function extracts the axis-
    aligned bounding box of the mask and computes SSIM on the clean rectangular
    crop only.

    Args:
        pred: Predicted tensor ``[B, 1, D, H, W]``
        target: Ground-truth tensor ``[B, 1, D, H, W]``
        mask: Binary mask tensor ``[B, 1, D, H, W]``

    Returns:
        Scalar SSIM loss (1 − SSIM) over the bounding-box region.
    """
    # Collapse batch and channel dims for bbox extraction
    mask_3d = mask.squeeze(1).sum(dim=0) > 0  # [D, H, W], True where any sample has mask
    nonzero = torch.nonzero(mask_3d, as_tuple=True)  # (d_idxs, h_idxs, w_idxs)

    if nonzero[0].numel() == 0:
        # Mask is empty — fall back to full-volume SSIM
        return SSIMLoss(spatial_dims=3)(pred, target)

    d_min, d_max = nonzero[0].min(), nonzero[0].max() + 1
    h_min, h_max = nonzero[1].min(), nonzero[1].max() + 1
    w_min, w_max = nonzero[2].min(), nonzero[2].max() + 1

    pred_crop = pred[:, :, d_min:d_max, h_min:h_max, w_min:w_max]
    target_crop = target[:, :, d_min:d_max, h_min:h_max, w_min:w_max]

    return SSIMLoss(spatial_dims=3)(pred_crop, target_crop)


def correction_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Standard MSE loss for phase error correction fields.

    Follows the labmate's validated ``corrected_mse_vol`` approach: mean
    squared error across all voxels and all 3 correction components.

    With values in [-1, 1], per-voxel errors are sub-unit so MSE produces
    smaller values than L1 for the same error magnitude.

    Args:
        pred: Predicted corrections ``[B, 3, D, H, W]``
        target: Ground-truth corrections ``[B, 3, D, H, W]``

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(pred, target)


def weighted_correction_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
) -> torch.Tensor:
    """Spatially-weighted MSE loss for phase error correction fields.

    Each voxel's squared error is multiplied by the corresponding weight,
    then normalised by the total weight volume (× number of channels) so
    the loss magnitude stays comparable regardless of the weight
    distribution.

    Typical usage: ``weight_map = 1.0 + α * tissue_mask`` to upweight
    tissue voxels while retaining a non-zero loss in air.

    When ``weight_map`` is all 1s this is equivalent to
    :func:`correction_mse_loss` (plain MSE).

    Args:
        pred: Predicted corrections ``[B, 3, D, H, W]``
        target: Ground-truth corrections ``[B, 3, D, H, W]``
        weight_map: Per-voxel weights ``[B, 1, D, H, W]``
            (broadcasts across correction channels).

    Returns:
        Scalar weighted MSE loss.
    """
    se = (pred - target) ** 2                          # [B, 3, D, H, W]
    num_channels = pred.shape[1]
    weight_sum = weight_map.sum().clamp(min=1.0)
    return (se * weight_map).sum() / (weight_sum * num_channels)


def radial_inplane_weight_map(
    shape: tuple[int, int, int],
    alpha: float,
    profile: str = "quadratic",
    sigma: float = 0.5,
    slice_axis: int = -1,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build an in-plane radial weight map (cylinder, constant along ``slice_axis``).

    Returns a ``[1, 1, *shape]`` tensor whose values are
    ``1`` at the in-plane edges/corners and ``1 + alpha`` at the centre,
    so multiplying an existing per-voxel weight by this map upweights the
    centre of every axial slice without changing the air/tissue balance.

    The weight depends only on the in-plane normalised radius ``r``, defined
    so that ``r = 0`` at the (geometric) centre and ``r = 1`` at the closest
    in-plane edge midpoint. Corners reach ``r = sqrt(2)``; for the linear /
    quadratic profiles we clamp at ``r = 1`` so the corner weight equals the
    edge-midpoint weight (``1``). The gaussian profile decays smoothly and
    is not clamped.

    Args:
        shape: ``(D, H, W)`` spatial shape of the volume.
        alpha: Peak extra weight at the centre. ``0`` disables the upweight.
        profile: One of ``"linear"``, ``"quadratic"``, ``"gaussian"``.
        sigma: Std of the gaussian profile, in units of normalised radius.
            Only used when ``profile == "gaussian"``.
        slice_axis: Which of the three spatial dims is the through-plane
            (slice / cylinder) axis. Defaults to the last dim (``W``), which
            matches the ``[B, C, D, H, W] -> 128x128x64`` convention used
            elsewhere in this project (W is the slice direction).
        dtype, device: Tensor dtype / device.

    Returns:
        Weight map of shape ``[1, 1, D, H, W]`` broadcastable across batch
        and channel dims.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if len(shape) != 3:
        raise ValueError(f"shape must be 3-D (D, H, W), got {shape}")

    spatial_axes = [0, 1, 2]
    slice_axis = slice_axis % 3
    in_plane_axes = [a for a in spatial_axes if a != slice_axis]
    a0, a1 = in_plane_axes
    n0, n1 = shape[a0], shape[a1]

    # Normalised in-plane coordinates in [-1, 1]; r=1 at the closest edge.
    c0 = torch.linspace(-1.0, 1.0, n0, dtype=dtype, device=device)
    c1 = torch.linspace(-1.0, 1.0, n1, dtype=dtype, device=device)
    g0, g1 = torch.meshgrid(c0, c1, indexing="ij")
    r = torch.sqrt(g0 ** 2 + g1 ** 2)  # [n0, n1]; up to sqrt(2) at corners

    if profile == "linear":
        w2d = 1.0 + alpha * (1.0 - r).clamp(min=0.0)
    elif profile == "quadratic":
        w2d = 1.0 + alpha * (1.0 - r ** 2).clamp(min=0.0)
    elif profile == "gaussian":
        w2d = 1.0 + alpha * torch.exp(-(r / sigma) ** 2)
    else:
        raise ValueError(
            f"profile must be 'linear', 'quadratic', or 'gaussian'; got {profile!r}"
        )

    # Insert the slice axis with size 1, then expand to full shape.
    insert_at = slice_axis
    w3d_small = w2d.unsqueeze(insert_at)  # length-1 along slice axis
    expand_shape = list(shape)
    w3d = w3d_small.expand(*expand_shape).contiguous()
    return w3d.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]


def tissue_masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss restricted to tissue voxels via a binary mask.

    Background phase error is a smooth field across the entire volume, but
    in air the velocity inputs are pure noise so the model has nothing
    useful to learn from.  This loss uses a precomputed binary tissue mask
    and computes MSE only over those voxels, treating all tissue equally
    regardless of signal intensity.

    Args:
        pred: Predicted corrections ``[B, 3, D, H, W]``
        target: Ground-truth corrections ``[B, 3, D, H, W]``
        mask: Binary tissue mask ``[B, 1, D, H, W]``.
            Broadcasts across the 3 correction channels.

    Returns:
        Scalar tissue-masked MSE loss.
    """
    tissue_mask = mask.float()                     # [B, 1, D, H, W]
    se = (pred - target) ** 2                      # [B, 3, D, H, W]
    num_channels = pred.shape[1]
    mask_sum = tissue_mask.sum().clamp(min=1.0)
    return (se * tissue_mask).sum() / (mask_sum * num_channels)


class SobelFilter3D(nn.Module):
    """Fixed (non-trainable) 3D Sobel gradient operator.

    Registers 3×3×3 Sobel kernels for the D, H, W axes as buffers and
    applies them via depthwise ``F.conv3d``.  Returns per-axis gradient
    maps or a gradient magnitude, depending on use.
    """

    def __init__(self) -> None:
        super().__init__()
        # 3D Sobel kernel for the W (x) axis; D and H kernels are transpositions.
        sobel_x = torch.tensor(
            [
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            ],
            dtype=torch.float32,
        )  # [3, 3, 3]

        # shape: [out_ch=1, in_ch=1, D, H, W]
        kx = sobel_x.unsqueeze(0).unsqueeze(0)
        ky = sobel_x.permute(0, 2, 1).unsqueeze(0).unsqueeze(0)
        kz = sobel_x.permute(2, 1, 0).unsqueeze(0).unsqueeze(0)

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("kz", kz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude of ``x`` (``[B, 1, D, H, W]``).

        Returns a ``[B, 1, D, H, W]`` tensor (slightly smaller due to valid
        convolution, but padded to preserve spatial size).
        """
        gx = F.conv3d(x, self.kx, padding=1)
        gy = F.conv3d(x, self.ky, padding=1)
        gz = F.conv3d(x, self.kz, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + 1e-8)


# Module-level singleton so kernels are allocated once and moved to the
# correct device lazily on first use.
_sobel_3d: SobelFilter3D | None = None


def _get_sobel(device: torch.device) -> SobelFilter3D:
    global _sobel_3d
    if _sobel_3d is None:
        _sobel_3d = SobelFilter3D()
    if next(_sobel_3d.buffers()).device != device:
        _sobel_3d = _sobel_3d.to(device)
    return _sobel_3d


def masked_sobel_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked L1 loss on 3D Sobel gradient magnitudes.

    Computes the Sobel gradient magnitude of both ``pred`` and ``target``,
    then returns the masked L1 distance between the two edge maps,
    normalised by the number of masked voxels.

    Args:
        pred: Predicted tensor ``[B, 1, D, H, W]``
        target: Ground-truth tensor ``[B, 1, D, H, W]``
        mask: Binary mask tensor ``[B, 1, D, H, W]``

    Returns:
        Scalar masked Sobel edge loss.
    """
    sobel = _get_sobel(pred.device)
    edge_pred = sobel(pred)
    edge_target = sobel(target)

    mask_sum = mask.sum().clamp(min=1.0)
    return (torch.abs(edge_pred - edge_target) * mask).sum() / mask_sum