"""
BFF (Bidirectional Feature Fusion) Neck for SSD-DEIM.

Implements a bidirectional feature fusion neck (FPN + PAN) enhanced with:
    - BFE (Bidirectional Feature Enhancement): SE attention cross-weighting
      that replaces simple concatenation for more effective cross-scale fusion.
    - PCGU (Partial Convolution Gated Unit): Partial convolution + GLU gating
      that reduces FLOPs while adaptively suppressing background clutter.
    - C2f_PCGU: C2f-style block with PCGU replacing standard bottleneck conv.

Reference:
    SSD-DEIM paper - Bidirectional Feature Fusion (BFF) Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
#  Basic Conv block
# ---------------------------------------------------------------------------

class ConvBnSiLU(nn.Module):
    """Conv2d + BatchNorm2d + SiLU activation."""

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        padding: Optional[int] = None,
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride,
            padding=padding, groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chs)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
#  BFE (Bidirectional Feature Enhancement) Module
# ---------------------------------------------------------------------------

class BFEModule(nn.Module):
    """Bidirectional Feature Enhancement Module.

    Replaces simple concatenation in feature fusion with a bidirectional
    SE-attention mechanism that cross-weight channels from two feature maps
    before fusing them. This prevents destructive interference between
    features from different scales.

    Architecture:
        feat_a -> SE_gate_a ---+
                                +--> element-wise enhancement --> concat --> proj
        feat_b -> SE_gate_b ---+

    Args:
        channels: Number of channels in each input feature (both inputs must
            have the same channel count after adjustment).
        reduction: Channel reduction ratio for SE attention.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid_chs = max(channels // reduction, 8)

        # SE gate for feature A (modulated by B)
        self.gate_a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_chs, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_chs, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # SE gate for feature B (modulated by A)
        self.gate_b = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_chs, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_chs, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Output projection
        self.proj = ConvBnSiLU(channels * 2, channels, kernel_size=1)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """Forward pass of BFE module.

        Args:
            feat_a: First feature map [B, C, H, W].
            feat_b: Second feature map [B, C, H, W].

        Returns:
            Fused feature map [B, C, H, W].
        """
        # Bidirectional gating: each feature is enhanced by the other's gate
        gate_a = self.gate_a(feat_b)  # gate for A, computed from B
        gate_b = self.gate_b(feat_a)  # gate for B, computed from A

        feat_a_enhanced = feat_a * (1.0 + gate_a)
        feat_b_enhanced = feat_b * (1.0 + gate_b)

        # Concatenate and project
        fused = torch.cat([feat_a_enhanced, feat_b_enhanced], dim=1)
        out = self.proj(fused)
        return out


# ---------------------------------------------------------------------------
#  PCGU (Partial Convolution Gated Unit)
# ---------------------------------------------------------------------------

class PCGU(nn.Module):
    """Partial Convolution Gated Unit.

    A channel-efficient gating unit that restricts spatial filtering to an
    active channel partition (default 1/4 channels), reducing FLOPs to
    ~13/16 of full-channel convolution while adaptively suppressing
    background foliage clutter via element-wise soft gating (GLU-style).

    Architecture:
        x -> split -> active (C/4) -> PConv(3x3) -> gate path -> sigmoid gate --+
                  |                                                       |
                  +-> inactive (3C/4) -----------------------------------+--> concat --> proj

    Args:
        channels: Total input/output channels.
        partial_ratio: Ratio of active channels (default 0.25 = 1/4).
    """

    def __init__(self, channels: int, partial_ratio: float = 0.25):
        super().__init__()
        self.active_chs = int(channels * partial_ratio)
        self.inactive_chs = channels - self.active_chs

        # Partial 3x3 conv on active channels
        self.pconv = nn.Conv2d(
            self.active_chs, self.active_chs,
            kernel_size=3, padding=1, bias=False,
        )
        self.bn_pconv = nn.BatchNorm2d(self.active_chs)

        # Gate generation: pointwise conv + depthwise conv + activation
        self.gate_pw = nn.Conv2d(
            self.active_chs, self.active_chs, kernel_size=1, bias=False,
        )
        self.gate_dw = nn.Conv2d(
            self.active_chs, self.active_chs,
            kernel_size=3, padding=1, groups=self.active_chs, bias=False,
        )
        self.bn_gate = nn.BatchNorm2d(self.active_chs)
        self.gate_act = nn.SiLU(inplace=True)

        # Gate projection (produces the gate values)
        self.gate_proj = nn.Conv2d(
            self.active_chs, self.active_chs, kernel_size=1, bias=False,
        )

        # Skip projection
        self.skip_proj = nn.Conv2d(
            self.active_chs, self.active_chs, kernel_size=1, bias=False,
        )

        # Final projection after concatenation of active + inactive
        self.final_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.final_bn = nn.BatchNorm2d(channels)
        self.final_act = nn.SiLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into active and inactive partitions
        x_active = x[:, :self.active_chs, :, :]
        x_inactive = x[:, self.active_chs:, :, :]

        # Partial convolution on active channels
        x_pconv = self.bn_pconv(self.pconv(x_active))

        # Gate generation
        gate_feat = self.gate_pw(x_pconv)
        gate_feat = self.bn_gate(self.gate_dw(gate_feat))
        gate_feat = self.gate_act(gate_feat)

        # GLU-style gating
        gate = torch.sigmoid(self.gate_proj(gate_feat))
        gated_feat = x_pconv * gate

        # Skip connection
        skip_feat = self.skip_proj(x_pconv)
        out_active = gated_feat + skip_feat

        # Concatenate with inactive channels
        out = torch.cat([out_active, x_inactive], dim=1)

        # Final projection and activation
        out = self.final_act(self.final_bn(self.final_proj(out)))
        return out


# ---------------------------------------------------------------------------
#  C2f_PCGU Block
# ---------------------------------------------------------------------------

class C2fPCGU(nn.Module):
    """C2f block with PCGU replacing standard bottleneck convolutions.

    Inspired by YOLOv8's C2f structure, but uses PCGU (Partial Convolution
    Gated Unit) instead of standard Bottleneck blocks for greater efficiency
    and adaptive background suppression.

    Architecture:
        x -> Conv1x1 -> split -> [PCGU, PCGU, ...] -> concat -> Conv1x1
                           \_______________________/
                         residual connections between PCGUs

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of PCGU blocks.
        partial_ratio: Active channel ratio for PCGU.
        shortcut: Whether to use shortcut connection.
        g: Groups for conv (not used, kept for API consistency).
        e: Expansion ratio for hidden channels.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 2,
        partial_ratio: float = 0.25,
        shortcut: bool = False,
        e: float = 0.5,
    ):
        super().__init__()
        self.c_ = int(c2 * e)  # hidden channels
        self.n = n
        self.shortcut = shortcut and c1 == c2

        # Input projection: split into 2 paths
        self.cv1 = ConvBnSiLU(c1, 2 * self.c_, kernel_size=1)

        # PCGU blocks (operate on one of the split paths)
        self.m = nn.ModuleList(
            PCGU(self.c_, partial_ratio=partial_ratio) for _ in range(n)
        )

        # Output projection
        self.cv2 = ConvBnSiLU(
            (2 + n) * self.c_, c2, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection and split into two halves
        y = list(self.cv1(x).chunk(2, dim=1))

        # Pass through PCGU blocks, accumulating outputs
        y.extend(m(y[-1]) for m in self.m)

        # Concatenate all outputs and project
        out = self.cv2(torch.cat(y, dim=1))

        if self.shortcut:
            out = out + x

        return out


# ---------------------------------------------------------------------------
#  BFF Neck (Bidirectional Feature Fusion)
# ---------------------------------------------------------------------------

class BFFNeck(nn.Module):
    """Bidirectional Feature Fusion Neck.

    A PAN-style (Path Aggregation Network) neck with bidirectional feature
    fusion enhanced by BFE modules and C2f_PCGU blocks.

    Architecture:
        Top-down path (FPN): P5 -> P4 -> P3
            (high-level semantic flows to low-level)
        Bottom-up path (PAN): P3 -> P4 -> P5
            (low-level spatial details flow to high-level)

    Each fusion point uses BFE instead of simple concat, and feature
    processing uses C2f_PCGU instead of standard C2f.

    Args:
        in_channels: List of input channels [c3, c4, c5] from backbone.
        out_channels: List of output channels [o3, o4, o5].
        num_c2f_blocks: Number of PCGU blocks in each C2f_PCGU.
        partial_ratio: Active channel ratio for PCGU.
        use_bfe: Whether to use BFE module (default True).
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        num_c2f_blocks: int = 2,
        partial_ratio: float = 0.25,
        use_bfe: bool = True,
    ):
        super().__init__()
        assert len(in_channels) == 3, "BFFNeck expects 3 input feature levels"
        assert len(out_channels) == 3

        c3_in, c4_in, c5_in = in_channels
        c3_out, c4_out, c5_out = out_channels
        self.use_bfe = use_bfe

        # ---------------------------------------------------------------
        #  Top-down path (FPN): high -> low
        # ---------------------------------------------------------------

        # Reduce P5 channels before upsample to P4
        self.reduce_p5 = ConvBnSiLU(c5_in, c4_in, kernel_size=1)
        # C2f_PCGU after P4 + P5 fusion
        self.c2f_p4_td = C2fPCGU(
            c4_in + c4_in if use_bfe else c4_in,
            c4_out,
            n=num_c2f_blocks,
            partial_ratio=partial_ratio,
        )

        # Reduce P4 channels before upsample to P3
        self.reduce_p4 = ConvBnSiLU(c4_out, c3_in, kernel_size=1)
        # C2f_PCGU after P3 + P4 fusion
        self.c2f_p3 = C2fPCGU(
            c3_in + c3_in if use_bfe else c3_in,
            c3_out,
            n=num_c2f_blocks,
            partial_ratio=partial_ratio,
        )

        # BFE modules for top-down path
        if use_bfe:
            self.bfe_p4_td = BFEModule(c4_in)
            self.bfe_p3 = BFEModule(c3_in)

        # ---------------------------------------------------------------
        #  Bottom-up path (PAN): low -> high
        # ---------------------------------------------------------------

        # Downsample P3 to P4
        self.down_p3 = ConvBnSiLU(c3_out, c4_out, kernel_size=3, stride=2)
        # C2f_PCGU after P4 + P3_up fusion
        self.c2f_p4_bu = C2fPCGU(
            c4_out + c4_out if use_bfe else c4_out,
            c4_out,
            n=num_c2f_blocks,
            partial_ratio=partial_ratio,
        )

        # Downsample P4 to P5
        self.down_p4 = ConvBnSiLU(c4_out, c5_in, kernel_size=3, stride=2)
        # C2f_PCGU after P5 + P4_up fusion
        self.c2f_p5 = C2fPCGU(
            c5_in + c5_in if use_bfe else c5_in,
            c5_out,
            n=num_c2f_blocks,
            partial_ratio=partial_ratio,
        )

        # BFE modules for bottom-up path
        if use_bfe:
            self.bfe_p4_bu = BFEModule(c4_out)
            self.bfe_p5 = BFEModule(c5_in)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of BFF Neck.

        Args:
            feats: Tuple of (P3, P4, P5) feature maps from backbone.

        Returns:
            Tuple of enhanced (P3_out, P4_out, P5_out) feature maps.
        """
        p3, p4, p5 = feats

        # ============================================================
        #  Top-down path (FPN)
        # ============================================================

        # P5 -> reduce -> upsample -> fuse with P4
        p5_reduced = self.reduce_p5(p5)
        p5_up = F.interpolate(p5_reduced, size=p4.shape[2:], mode='nearest')

        if self.use_bfe:
            p4_fused_td = self.bfe_p4_td(p4, p5_up)
        else:
            p4_fused_td = torch.cat([p4, p5_up], dim=1)
        p4_td = self.c2f_p4_td(p4_fused_td)

        # P4 -> reduce -> upsample -> fuse with P3
        p4_reduced = self.reduce_p4(p4_td)
        p4_up = F.interpolate(p4_reduced, size=p3.shape[2:], mode='nearest')

        if self.use_bfe:
            p3_fused = self.bfe_p3(p3, p4_up)
        else:
            p3_fused = torch.cat([p3, p4_up], dim=1)
        p3_out = self.c2f_p3(p3_fused)

        # ============================================================
        #  Bottom-up path (PAN)
        # ============================================================

        # P3 -> downsample -> fuse with P4_td
        p3_down = self.down_p3(p3_out)

        if self.use_bfe:
            p4_fused_bu = self.bfe_p4_bu(p4_td, p3_down)
        else:
            p4_fused_bu = torch.cat([p4_td, p3_down], dim=1)
        p4_out = self.c2f_p4_bu(p4_fused_bu)

        # P4 -> downsample -> fuse with P5
        p4_down = self.down_p4(p4_out)

        if self.use_bfe:
            p5_fused = self.bfe_p5(p5, p4_down)
        else:
            p5_fused = torch.cat([p5, p4_down], dim=1)
        p5_out = self.c2f_p5(p5_fused)

        return (p3_out, p4_out, p5_out)


# ---------------------------------------------------------------------------
#  Factory function
# ---------------------------------------------------------------------------

def build_bff_neck(cfg: dict) -> BFFNeck:
    """Build BFF Neck from configuration.

    Args:
        cfg: Configuration dict with keys:
            - in_channels: Input channel list.
            - out_channels: Output channel list.
            - num_c2f_blocks: Number of PCGU blocks per C2f.
            - partial_ratio: Active channel ratio.
            - use_bfe: Whether to use BFE modules.

    Returns:
        BFFNeck instance.
    """
    return BFFNeck(
        in_channels=cfg['in_channels'],
        out_channels=cfg['out_channels'],
        num_c2f_blocks=cfg.get('num_c2f_blocks', 2),
        partial_ratio=cfg.get('partial_ratio', 0.25),
        use_bfe=cfg.get('use_bfe', True),
    )
