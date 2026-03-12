from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.nets import UNet


class UNetTrilinear(UNet):
    """MONAI UNet variant that replaces transpose convolutions in the decoder
    with trilinear interpolation followed by a regular convolution.

    This avoids the *checkerboard artifacts* that are common with strided
    transpose convolutions, while keeping the rest of the architecture
    (encoder, skip connections, residual units) identical to the base
    :class:`monai.networks.nets.UNet`.
    """

    def _get_up_layer(
        self,
        in_channels: int,
        out_channels: int,
        strides: int,
        is_top: bool,
    ) -> nn.Module:
        # Trilinear (or nearest) upsample → regular conv
        upsample = nn.Upsample(
            scale_factor=strides,
            mode="trilinear" if self.dimensions == 3 else "bilinear",
            align_corners=True,
        )
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=1,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=False,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            return nn.Sequential(upsample, conv, ru)

        return nn.Sequential(upsample, conv)


class CoarseCorrectionHead(nn.Module):
    """Phase error correction head that predicts a coarse smooth field.

    Architecture (no skip connections):
      1. Pre-processing conv block(s) at full resolution (PReLU activations)
      2. Learned stride-2 downsampling conv blocks (PReLU activations)
      3. 1x1x1 conv to ``out_channels`` at coarse resolution
      4. ``tanh`` activation to bound output in [-1, 1]
      5. Trilinear interpolation back to original spatial size

    The final 1x1x1 conv is zero-initialised so that the initial correction
    output is approximately zero everywhere.
    """

    def __init__(
        self,
        in_features: int,
        head_features: int,
        pre_depth: int,
        down_features: list[int],
        out_channels: int,
    ):
        super().__init__()

        # --- Pre-processing at full resolution ---
        pre_layers: list[nn.Module] = []
        in_f = in_features
        for _ in range(pre_depth):
            pre_layers.append(nn.Conv3d(in_f, head_features, 3, padding=1))
            pre_layers.append(nn.PReLU())
            in_f = head_features
        self.pre_block = nn.Sequential(*pre_layers) if pre_layers else nn.Identity()

        # --- Learned stride-2 downsampling ---
        down_layers: list[nn.Module] = []
        for df in down_features:
            down_layers.append(nn.Conv3d(in_f, df, 3, stride=2, padding=1))
            down_layers.append(nn.PReLU())
            in_f = df
        self.down_blocks = nn.Sequential(*down_layers)

        # --- Final 1x1x1 prediction (zero-initialised) ---
        self.final_conv = nn.Conv3d(in_f, out_channels, 1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        target_size = shared.shape[2:]  # (D, H, W)
        x = self.pre_block(shared)
        x = self.down_blocks(x)
        x = self.final_conv(x)
        x = torch.tanh(x)
        return F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)


class DualHeadGenerator(nn.Module):
    """Dual-head generator: shared MONAI UNet backbone with separate
    cine enhancement and phase error correction output heads.

    The backbone produces a shared feature map, which is then fed into two
    independent heads:
      - **Cine head**: Conv3D layers with PReLU + final Sigmoid (output in [0, 1])
      - **Correction head**: :class:`CoarseCorrectionHead` — predicts a smooth
        correction field at reduced resolution, then upsamples back to full
        resolution via trilinear interpolation (output in [-1, 1])

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
        correction_head_features: int | None = None,
        correction_pre_depth: int | None = None,
        correction_down_features: list[int] | None = None,
        correction_num_downsamples: int = 2,
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

        # --- Phase correction head (coarse prediction + trilinear upsample) ---
        _corr_features = correction_head_features if correction_head_features is not None else head_features
        _corr_pre_depth = correction_pre_depth if correction_pre_depth is not None else head_depth
        _corr_down_features = (
            list(correction_down_features) if correction_down_features is not None
            else [_corr_features] * correction_num_downsamples
        )
        self.correction_head = CoarseCorrectionHead(
            in_features=shared_features,
            head_features=_corr_features,
            pre_depth=_corr_pre_depth,
            down_features=_corr_down_features,
            out_channels=correction_out_channels,
        )

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
        upsample_mode = cfg.model.generator.get("upsample_mode", "deconv")
        BackboneClass = UNetTrilinear if upsample_mode == "trilinear" else UNet
        backbone = BackboneClass(
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
            correction_head_features=cfg.model.generator.get("correction_head_features", None),
            correction_pre_depth=cfg.model.generator.get("correction_pre_depth", None),
            correction_down_features=cfg.model.generator.get("correction_down_features", None),
            correction_num_downsamples=cfg.model.generator.get("correction_num_downsamples", 2),
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