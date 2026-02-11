import torch
import torch.nn as nn
from monai.networks.nets import UNet


class DualHeadGenerator(nn.Module):
    """Dual-head generator: shared MONAI UNet backbone with separate
    cine enhancement and phase error correction output heads.

    The backbone produces a shared feature map, which is then fed into two
    independent heads:
      - **Cine head**: Conv3D layers with PReLU + final Sigmoid (output in [0, 1])
      - **Correction head**: Conv3D layers with Tanh + final Tanh (output in [-1, 1])

    Forward returns a single tensor ``[B, cine_out + correction_out, D, H, W]``
    with the cine channels first.
    """

    def __init__(
        self,
        backbone: nn.Module,
        shared_features: int,
        head_features: int,
        head_depth: int,
        cine_out_channels: int,
        correction_out_channels: int,
    ):
        super().__init__()
        self.backbone = backbone

        # --- Cine enhancement head (PReLU intermediates, Sigmoid output) ---
        cine_layers = []
        in_f = shared_features
        for _ in range(head_depth):
            cine_layers.append(nn.Conv3d(in_f, head_features, 3, padding=1))
            cine_layers.append(nn.PReLU())
            in_f = head_features
        cine_layers.append(nn.Conv3d(head_features, cine_out_channels, 1))
        cine_layers.append(nn.Sigmoid())
        self.cine_head = nn.Sequential(*cine_layers)

        # --- Phase correction head (Tanh intermediates, Tanh output) -------
        corr_layers = []
        in_f = shared_features
        for _ in range(head_depth):
            corr_layers.append(nn.Conv3d(in_f, head_features, 3, padding=1))
            corr_layers.append(nn.Tanh())
            in_f = head_features
        corr_layers.append(nn.Conv3d(head_features, correction_out_channels, 1))
        corr_layers.append(nn.Tanh())
        self.correction_head = nn.Sequential(*corr_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.backbone(x)                # [B, shared_features, D, H, W]
        cine = self.cine_head(shared)            # [B, cine_out, D, H, W]
        corr = self.correction_head(shared)      # [B, correction_out, D, H, W]
        return torch.cat([cine, corr], dim=1)    # [B, cine_out+correction_out, D, H, W]


def build_generator(cfg) -> nn.Module:
    """Build the generator network based on configuration.

    Supports two model types:
      - ``"dual_head"``: Uses :class:`DualHeadGenerator` with a shared MONAI
        UNet backbone and separate cine + correction heads.
      - ``"single_head"`` (default / backwards-compatible): Original
        ``nn.Sequential(UNet, Sigmoid)`` for single-output cine prediction.
    """
    model_type = cfg.model.generator.get("model_type", "single_head")

    if model_type == "dual_head":
        backbone = UNet(
            spatial_dims=3,
            in_channels=cfg.model.generator.in_channels,
            out_channels=cfg.model.generator.shared_features,
            channels=cfg.model.generator.channels,
            strides=cfg.model.generator.strides,
            num_res_units=cfg.model.generator.num_res_units,
            act=cfg.model.generator.activation,
        )
        return DualHeadGenerator(
            backbone=backbone,
            shared_features=cfg.model.generator.shared_features,
            head_features=cfg.model.generator.head_features,
            head_depth=cfg.model.generator.head_depth,
            cine_out_channels=cfg.model.generator.cine_out_channels,
            correction_out_channels=cfg.model.generator.correction_out_channels,
        )
    else:
        # Original single-head path (backwards compatible)
        unet = UNet(
            spatial_dims=3,
            in_channels=cfg.model.generator.in_channels,
            out_channels=cfg.model.generator.out_channels,
            channels=cfg.model.generator.channels,
            strides=cfg.model.generator.strides,
            num_res_units=cfg.model.generator.num_res_units,
            act=cfg.model.generator.activation,
        )
        return nn.Sequential(unet, nn.Sigmoid())


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
            
        elif model_variant == 'C64k4s2-C128k3s1-C256k3s1-C1k4s2':
            self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1), # receptive field = 1 + (4 - 1) * 1 = 4, j1 = 2
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1), # receptive field = 4 + (4 - 1) * 2 = 10, j2 = 4
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1), # receptive field = 10 + (4-1) * 4 = 22, j3 = 8
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(256, 1, kernel_size=4, stride=2, padding=1)  # output: [B, 1, h, w, d], receptive field = 22 + (4-1) * 8 = 46
        )

            
    def forward(self, x):
        return self.model(x)


def build_discriminator(cfg) -> nn.Module:
    """
    Build the PatchGAN-style discriminator.
    Expects concatenated input: [mag, speed, cine_pred_or_gt]
    """
    
    return PatchDiscriminator(in_channels=cfg.model.discriminator.in_channels, model_variant=cfg.model.discriminator.model_variant)  # e.g., 3