"""Multi-View ControlNet with Cross-View Self-Attention.

Why we hack ``diffusers.ControlNetModel``
-----------------------------------------
``ControlNetModel`` contains a tiny CNN called ``controlnet_cond_embedding``
that consumes the raw control image and projects it into the UNet's
intermediate feature space. Everything *downstream* (down/mid/up blocks)
is a deep-copy of the frozen UNet and we do not want to touch those blocks
(they are huge, and rewriting them would lose pretrained weights).

The ``NUM_ROWS x NUM_VIEWS`` tiled grid (currently ``1 x 8`` -- one
``overlayed_all.png`` per camera angle) only makes geometric sense if
features in column ``v`` can attend to features in column ``v'``. So we
replace just the small conditioning CNN with a Conv-Attention-Conv stack
that:

    1) Convs the image normally (spatial inductive bias).
    2) Reshapes ``(B, C, NUM_ROWS*H, NUM_VIEWS*W) -> (B, NUM_VIEWS, C, NUM_ROWS*H, W)``
       and runs Self-Attn over the view axis to enforce 3D consistency.
    3) Reshapes back and continues with standard 2D convs.

The rest of ControlNet stays exactly as in ``diffusers``, so the residuals
plug into the SDXL UNet without further surgery.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.controlnets.controlnet import ControlNetModel
from einops import rearrange

from config import NUM_VIEWS


# ===========================================================================
# 1.  Cross-View Self-Attention block
# ===========================================================================
class CrossViewAttention(nn.Module):
    """Self-attention across the view axis of a (B, V, C, H, W) tensor.

    We treat each spatial location (h, w) as an independent token sequence
    of length ``V`` (= NUM_VIEWS = 8) and run multi-head self-attention
    over that axis. This is light (V is tiny) yet enough to enforce
    geometric consistency between the eight cameras.

    The forward signature is::

        x:  (B, V, C, H, W)  ->  (B, V, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_views: int = NUM_VIEWS,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})."
            )
        self.channels = channels
        self.num_heads = num_heads
        self.num_views = num_views

        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        # Single fused QKV projection (1x1 conv == linear over channel dim).
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=True)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        # Learnable per-view positional embedding so the attention layer can
        # distinguish the 8 cameras (it has no notion of "view index" otherwise).
        self.view_pos_embed = nn.Parameter(torch.zeros(1, num_views, channels, 1, 1))
        nn.init.normal_(self.view_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, C, H, W)
        b, v, c, h, w = x.shape
        if v != self.num_views:
            raise ValueError(f"Expected V={self.num_views}, got {v}.")

        residual = x
        x = x + self.view_pos_embed  # broadcast over H, W

        # GroupNorm wants (N, C, H, W); fold B and V together.
        # (B, V, C, H, W) -> (B*V, C, H, W)
        x_2d = rearrange(x, "b v c h w -> (b v) c h w")
        x_2d = self.norm(x_2d)

        # 1x1 conv produces (B*V, 3C, H, W). Split into Q, K, V.
        qkv = self.qkv(x_2d)                                   # (B*V, 3C, H, W)
        q, k, val = qkv.chunk(3, dim=1)                        # each: (B*V, C, H, W)

        # We want to attend over the V axis per spatial location, with
        # multi-head attention split along the C axis. Build a tensor
        # of shape (B*H*W, num_heads, V, head_dim).
        head_dim = c // self.num_heads

        def to_seq(t: torch.Tensor) -> torch.Tensor:
            # (B*V, C, H, W) -> (B, V, num_heads, head_dim, H, W)
            t = rearrange(
                t,
                "(b v) (heads d) h w -> b v heads d h w",
                b=b, v=v, heads=self.num_heads, d=head_dim,
            )
            # -> (B, H, W, num_heads, V, head_dim) so that the V axis is
            # the sequence dimension for attention.
            t = rearrange(t, "b v heads d h w -> b h w heads v d")
            # Merge spatial + batch into the leading "batch" dim of attn.
            # (B*H*W, num_heads, V, head_dim)
            t = rearrange(t, "b h w heads v d -> (b h w) heads v d")
            return t

        q_s = to_seq(q)
        k_s = to_seq(k)
        v_s = to_seq(val)

        # PyTorch >= 2.0 fused SDPA. Output: (B*H*W, num_heads, V, head_dim).
        attn_out = F.scaled_dot_product_attention(q_s, k_s, v_s)

        # Undo the reshape.
        # (B*H*W, num_heads, V, head_dim) -> (B, V, C, H, W)
        attn_out = rearrange(
            attn_out, "(b h w) heads v d -> (b v) (heads d) h w",
            b=b, h=h, w=w,
        )
        attn_out = self.proj_out(attn_out)                     # (B*V, C, H, W)
        attn_out = rearrange(attn_out, "(b v) c h w -> b v c h w", b=b, v=v)

        return residual + attn_out


# ===========================================================================
# 2.  Multi-View conditioning embedding
# ===========================================================================
class MultiViewConditioningEmbedding(nn.Module):
    """Replacement for ``ControlNetModel.controlnet_cond_embedding``.

    Layout follows the original diffusers module:

        Conv (3 -> c0) ─┐
        [Conv (c -> c) + SiLU] x n
        Conv (c -> conditioning_embedding_channels)

    We inject ``CrossViewAttention`` after every spatial downsample so the
    eight views can exchange information at multiple scales.

    The input is the tiled grid of shape ``(B, 3, NUM_ROWS*h, NUM_VIEWS*w)``;
    the output has the same downsampling factor as the original module
    (matches the latent resolution that ControlNet's first down-block expects).
    """

    def __init__(
        self,
        conditioning_channels: int = 3,
        conditioning_embedding_channels: int = 320,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        num_heads: int = 8,
        num_views: int = NUM_VIEWS,
    ) -> None:
        super().__init__()
        # We intentionally do NOT store ``num_rows``: the row axis stays
        # inside the per-view height (see forward), so the embedding doesn't
        # need to know how many rows there are. Keeping the parameter would
        # be unused state we'd have to keep in sync with config.
        self.num_views = num_views

        # Initial projection (no downsampling, matches diffusers).
        self.conv_in = nn.Conv2d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        # Body: blocks[i] expects in_ch=block_out_channels[i],
        # produces block_out_channels[i+1] with stride 2.
        self.blocks = nn.ModuleList()
        self.cross_view_attns = nn.ModuleList()
        for i in range(len(block_out_channels) - 1):
            ch_in = block_out_channels[i]
            ch_out = block_out_channels[i + 1]
            self.blocks.append(nn.Sequential(
                nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=2),
                nn.SiLU(),
            ))
            # Cross-view attention runs on the *downsampled* features.
            self.cross_view_attns.append(
                CrossViewAttention(channels=ch_out, num_heads=num_heads, num_views=num_views)
            )

        # Diffusers zero-inits the final conv so ControlNet starts as identity.
        self.conv_out = nn.Conv2d(
            block_out_channels[-1],
            conditioning_embedding_channels,
            kernel_size=3,
            padding=1,
        )
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        conditioning:
            ``(B, 3, NUM_ROWS*H, NUM_VIEWS*W)`` tiled grid in [-1, 1].

        Returns
        -------
        torch.Tensor
            ``(B, conditioning_embedding_channels, NUM_ROWS*H/8, NUM_VIEWS*W/8)``
            after 3 stride-2 downsamples (matches diffusers default).
        """
        # b c (num_rows h_tile) (num_views w_tile)
        x = self.conv_in(conditioning)
        x = F.silu(x)

        for block, cv_attn in zip(self.blocks, self.cross_view_attns):
            x = block(x)
            # Re-split the view axis to apply cross-view attention. Note we
            # keep the row axis *inside* the per-view height -- the attention
            # block sees ``(B, V, C, num_rows*h, w)`` per the project spec.
            #   b c (num_rows h) (num_views w) -> b num_views c (num_rows h) w
            if x.shape[-1] % self.num_views != 0:
                raise RuntimeError(
                    f"Expected feature width divisible by NUM_VIEWS={self.num_views}, "
                    f"got W={x.shape[-1]}."
                )
            x_views = rearrange(
                x,
                "b c h (num_views w) -> b num_views c h w",
                num_views=self.num_views,
            )
            x_views = cv_attn(x_views)
            # Merge back: b num_views c h w -> b c h (num_views w)
            x = rearrange(x_views, "b num_views c h w -> b c h (num_views w)")

        x = self.conv_out(x)
        return x


# ===========================================================================
# 3.  Hacked ControlNetModel
# ===========================================================================
class MultiViewControlNetModel(ControlNetModel):
    """``diffusers.ControlNetModel`` with a multi-view conditioning embedding.

    Usage::

        mv_cn = MultiViewControlNetModel.from_unet(
            unet,
            conditioning_channels=3,
            block_out_channels=(16, 32, 96, 256),
        )
        mv_cn.install_multiview_embedding(num_heads=8)

    After ``install_multiview_embedding`` the ``controlnet_cond_embedding``
    submodule is the new :class:`MultiViewConditioningEmbedding`. All other
    components (down/mid blocks, zero-convs, etc.) are unchanged.
    """

    def install_multiview_embedding(
        self,
        num_heads: int = 8,
        num_views: int = NUM_VIEWS,
        block_out_channels: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Swap the stock cond-embedding for the multi-view variant.

        We read the original module's hyper-parameters to make sure the
        replacement has the same output channels (the rest of the network
        depends on that).
        """
        old = self.controlnet_cond_embedding  # nn.Module from diffusers

        # The stock module exposes ``conv_in`` (input proj) and ``conv_out``
        # (output proj). Use them to recover the channel sizes.
        in_channels = old.conv_in.in_channels
        out_channels = old.conv_out.out_channels

        if block_out_channels is None:
            # Diffusers' default for ``conditioning_embedding_out_channels``.
            block_out_channels = (16, 32, 96, 256)

        new = MultiViewConditioningEmbedding(
            conditioning_channels=in_channels,
            conditioning_embedding_channels=out_channels,
            block_out_channels=tuple(block_out_channels),
            num_heads=num_heads,
            num_views=num_views,
        )
        # Place the new module on the same device / dtype as the old one.
        param = next(old.parameters(), None)
        if param is not None:
            new = new.to(device=param.device, dtype=param.dtype)
        self.controlnet_cond_embedding = new

    # Convenience: count trainable params of just the conditioning embed.
    def num_multiview_params(self) -> int:
        return sum(p.numel() for p in self.controlnet_cond_embedding.parameters())
