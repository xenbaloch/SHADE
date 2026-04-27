"""SHADE network architecture.

A compact encoder-decoder that operates on packed Bayer (RGGB) RAW tensors
and produces a single enhanced sRGB image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    """Lightweight residual block with two 3×3 convolutions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SHADENet(nn.Module):
    """Compact RAW-to-sRGB network for backlit low-light enhancement.

    Args:
        in_channels:  Number of input channels (4 for packed RGGB Bayer).
        out_channels: Number of output channels (3 for sRGB).
        base_channels: Width multiplier for the hidden layers.
        num_res_blocks: Depth of the bottleneck residual stack.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 64,
        num_res_blocks: int = 8,
    ) -> None:
        super().__init__()

        # --- Encoder -------------------------------------------------------
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # --- Bottleneck ----------------------------------------------------
        self.res_blocks = nn.Sequential(
            *[_ResBlock(base_channels * 4) for _ in range(num_res_blocks)]
        )

        # --- Decoder -------------------------------------------------------
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Packed Bayer tensor of shape ``(B, 4, H, W)`` normalised to
               ``[0, 1]``.

        Returns:
            Enhanced sRGB tensor of shape ``(B, 3, H, W)`` in ``[0, 1]``.
        """
        x0 = self.head(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)

        x2 = self.res_blocks(x2)

        x = self.up2(x2) + x1
        x = self.up1(x) + x0
        return self.tail(x)
