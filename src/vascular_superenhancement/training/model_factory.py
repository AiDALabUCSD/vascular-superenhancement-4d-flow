import torch
import torch.nn as nn
from monai.networks.nets import UNet


def build_generator(cfg) -> nn.Module:
    """
    Build the generator network (UNet) for 3D velocity to synthetic contrast image.
    """
    return UNet(
        spatial_dims=3,
        in_channels=cfg.model.generator.in_channels,     # e.g., 3 (vx, vy, vz)
        out_channels=cfg.model.generator.out_channels,   # e.g., 1 (cine prediction)
        channels=cfg.model.generator.channels,           # e.g., [32, 64, 128, 256]
        strides=cfg.model.generator.strides,             # e.g., [2, 2, 2]
        num_res_units=cfg.model.generator.num_res_units,
        act=cfg.model.generator.activation
    )


class PatchDiscriminator(nn.Module):
    """
    3D PatchGAN-style discriminator that outputs a patch-wise real/fake map.
    """
    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 1, kernel_size=4, padding=1)  # output: [B, 1, h, w, d]
        )

    def forward(self, x):
        return self.model(x)


def build_discriminator(cfg) -> nn.Module:
    """
    Build the PatchGAN-style discriminator.
    Expects concatenated input: [vx, vy, vz, cine_pred_or_gt]
    """
    return PatchDiscriminator(in_channels=cfg.model.discriminator.in_channels)  # e.g., 4
