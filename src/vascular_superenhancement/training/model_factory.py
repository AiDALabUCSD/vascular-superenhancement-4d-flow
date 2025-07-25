# import torch
import torch.nn as nn
from monai.networks.nets import UNet


def build_generator(cfg) -> nn.Module:
    """
    Build the generator network (UNet) for 3D velocity to synthetic contrast image.
    """
    return UNet(
        spatial_dims=3,
        in_channels=cfg.model.generator.in_channels,     # e.g., 2 (magnitude, speed)
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
    def __init__(self, in_channels: int = 3, model_variant: str = 'C64k4s2-C128k4s2-C256k4s2-C1k4s1'):
        super().__init__()
        
        # receptive field definition:
        # ri = ri-1 + (kernel_size - 1) * ji-1;
        # where,
        #   ri is the receptive field of the i-th layer,
        #   ji is the effective stride of the i-th layer
        #   defined as such:
        #   ji = si * ji-1
        #   where,
        #   si is the stride of the i-th layer
        
        if model_variant == 'C64k4s2-C128k4s2-C256k4s2-C1k4s1':
            # this 3D PatchGAN has a receptive field of 46x46x46
            self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1), # receptive field = 1 + (4 - 1) * 1 = 4, j1 = 2
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1), # receptive field = 4 + (4 - 1) * 2 = 10, j2 = 4
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1), # receptive field = 10 + (4-1) * 4 = 22, j3 = 8
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # nn.Conv3d(256, 512, kernel_size=4, stride=1, padding=1), # receptive field = 22 + (4-1) * 4 = 34, j4 = 4
            # nn.BatchNorm3d(512),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 1, kernel_size=4, padding=1)  # output: [B, 1, h, w, d], receptive field = 22 + (4-1) * 8 = 46
        )
    def forward(self, x):
        return self.model(x)


def build_discriminator(cfg) -> nn.Module:
    """
    Build the PatchGAN-style discriminator.
    Expects concatenated input: [mag, speed, cine_pred_or_gt]
    """
    
    return PatchDiscriminator(in_channels=cfg.model.discriminator.in_channels, model_variant=cfg.model.discriminator.model_variant)  # e.g., 3
