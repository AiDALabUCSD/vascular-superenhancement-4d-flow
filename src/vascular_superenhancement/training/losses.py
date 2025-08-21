import torch
import torch.nn.functional as F
from monai.losses import SSIMLoss


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


def generator_l1_loss(fake_img: torch.Tensor, real_img: torch.Tensor, weight: float = 100.0) -> torch.Tensor:
    """
    L1 reconstruction loss between generated and real cine images.
    Typically weighted heavily (e.g., 100×) in Pix2Pix.
    """
    return weight * F.l1_loss(fake_img, real_img)


def generator_ssim_loss(fake_img: torch.Tensor, real_img: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """
    SSIM-based reconstruction loss using MONAI's 3D SSIM implementation.
    
    Args:
        fake_img: Generated image tensor [B, C, D, H, W]
        real_img: Target image tensor [B, C, D, H, W]
        weight: Weight factor for the loss
    
    Returns:
        SSIM loss value (1 - SSIM for minimization)
    """
    return weight * SSIMLoss(spatial_dims=3)(fake_img, real_img)