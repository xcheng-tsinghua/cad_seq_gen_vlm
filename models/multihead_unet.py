from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class StructuredMultiHeadUNet(nn.Module):
    """Shared encoder-decoder + 4 output heads.

    Head order:
    0 prev_depth_map
    1 sketch_plane_mask
    2 reference_mask
    3 result_frame
    """

    def __init__(self, in_channels: int = 6, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c4, c4)

        self.up4 = UpBlock(c4, c4, c3)
        self.up3 = UpBlock(c3, c3, c2)
        self.up2 = UpBlock(c2, c2, c1)
        self.up1 = UpBlock(c1, c1, c1)

        self.head_prev_depth = nn.Conv2d(c1, 1, kernel_size=1)
        self.head_sketch_plane = nn.Conv2d(c1, 1, kernel_size=1)
        self.head_reference = nn.Conv2d(c1, 1, kernel_size=1)
        self.head_result_frame = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = torch.cat(
            [
                self.head_prev_depth(d1),
                self.head_sketch_plane(d1),
                self.head_reference(d1),
                self.head_result_frame(d1),
            ],
            dim=1,
        )
        return out

