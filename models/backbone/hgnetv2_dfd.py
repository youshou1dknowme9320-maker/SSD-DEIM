"""
HGNetV2 Backbone with DFD (Duplex Feature Downsampling) for SSD-DEIM.

Implements the HGNetV2 backbone architecture with integrated Duplex Feature
Downsampling modules to preserve multi-scale fidelity for dense camouflaged
small object detection.

Architecture Overview:
    - StemBlock: Initial feature extraction (downsample x4)
    - DFD Module: Duplex Feature Downsampling (between stem and stage1)
    - HG_Stage 1-4: Hierarchical grouping stages with ESE attention
    - Output: 3 scale features (S3, S4, S5) at strides 8, 16, 32

Reference:
    - HGNetV2: PaddleDetection PPHGNetV2
    - DFD: SSD-DEIM paper - Duplex Feature Downsampling
    - DEIM (CVPR 2025) engine/backbone/hgnetv2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
#  Basic building blocks
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm + Activation.

    Standard convolution block with optional learnable affine transform.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        padding: Optional[int] = None,
        act: str = 'relu',
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride,
            padding=padding, groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chs)
        self.act = self._get_activation(act)

    @staticmethod
    def _get_activation(act: str) -> nn.Module:
        if act == 'relu':
            return nn.ReLU(inplace=True)
        elif act == 'silu' or act == 'swish':
            return nn.SiLU(inplace=True)
        elif act == 'gelu':
            return nn.GELU()
        elif act == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif act == 'none' or act is None:
            return nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LightConvBNAct(nn.Module):
    """Lightweight convolution: pointwise conv + depthwise conv.

    Reduces computation compared to standard conv while maintaining capacity.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int = 3,
        groups: int = 1,
        act: str = 'relu',
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs, out_chs, kernel_size=1, act='none'
        )
        self.conv2 = ConvBNAct(
            out_chs, out_chs, kernel_size=kernel_size,
            groups=out_chs, act=act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ---------------------------------------------------------------------------
#  Stem Block
# ---------------------------------------------------------------------------

class StemBlock(nn.Module):
    """HGNetV2 Stem Block.

    Initial feature extraction module that downsamples input by 4x.
    Uses a dual-path design with 2x2 conv and max pooling.

    Args:
        in_chs: Input channels (default 3 for RGB).
        mid_chs: Intermediate channels.
        out_chs: Output channels.
        act: Activation function.
    """

    def __init__(
        self,
        in_chs: int = 3,
        mid_chs: int = 16,
        out_chs: int = 16,
        act: str = 'relu',
    ):
        super().__init__()
        # Initial 3x3 conv (stride 2, downsample x2)
        self.stem1 = ConvBNAct(in_chs, mid_chs, kernel_size=3, stride=2, act=act)

        # Right branch: 2x2 conv path
        self.stem2a = ConvBNAct(mid_chs, mid_chs // 2, kernel_size=2, stride=1,
                                padding=0, act=act)
        self.stem2b = ConvBNAct(mid_chs // 2, mid_chs, kernel_size=2, stride=1,
                                padding=0, act=act)

        # Left branch: max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

        # Fusion and further downsampling
        self.stem3 = ConvBNAct(mid_chs * 2, mid_chs, kernel_size=3, stride=2, act=act)
        self.stem4 = ConvBNAct(mid_chs, out_chs, kernel_size=1, stride=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = self.stem1(x)

        # Right branch: 2x2 conv path
        x_pad = F.pad(x, (0, 1, 0, 1))  # right and bottom padding
        x2 = self.stem2a(x_pad)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)

        # Left branch: max pooling
        x1 = self.pool(x)

        # Concatenate
        x = torch.cat([x1, x2], dim=1)

        # Further processing
        x = self.stem3(x)
        x = self.stem4(x)
        return x


# ---------------------------------------------------------------------------
#  ESE (Effective Squeeze and Extraction) Module
# ---------------------------------------------------------------------------

class EseModule(nn.Module):
    """Effective Squeeze and Extraction attention module.

    A lightweight channel attention mechanism that uses a single 1x1 conv
    instead of the two-layer MLP in standard SE-Net.
    """

    def __init__(self, chs: int):
        super().__init__()
        self.conv = nn.Conv2d(chs, chs, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        # Squeeze: global average pooling
        x = x.mean((2, 3), keepdim=True)
        # Excitation: 1x1 conv + sigmoid
        x = self.conv(x)
        x = self.sigmoid(x)
        # Reweight
        return identity * x


# ---------------------------------------------------------------------------
#  HG Block (Hourglass Block)
# ---------------------------------------------------------------------------

class HG_Block(nn.Module):
    """Hierarchical Grouping Block.

    Core building block of HGNetV2. Features are extracted through multiple
    conv layers, then all intermediate features are concatenated and passed
    through an aggregation module with channel attention.

    Args:
        in_chs: Input channels.
        mid_chs: Intermediate channels in each conv layer.
        out_chs: Output channels.
        layer_num: Number of conv layers in the block.
        kernel_size: Conv kernel size.
        residual: Whether to use residual connection.
        light_block: Whether to use lightweight (depthwise) conv.
        agg: Aggregation type ('ese', 'se', 'none').
        drop_path: Drop path rate.
        act: Activation function.
    """

    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        layer_num: int,
        kernel_size: int = 3,
        residual: bool = False,
        light_block: bool = False,
        agg: str = 'ese',
        drop_path: float = 0.0,
        act: str = 'relu',
    ):
        super().__init__()
        self.residual = residual

        # Build conv layers
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            in_c = in_chs if i == 0 else mid_chs
            if light_block:
                self.layers.append(
                    LightConvBNAct(in_c, mid_chs, kernel_size=kernel_size, act=act)
                )
            else:
                self.layers.append(
                    ConvBNAct(in_c, mid_chs, kernel_size=kernel_size, stride=1, act=act)
                )

        # Aggregation: concat all layer outputs -> project -> attention
        total_chs = in_chs + layer_num * mid_chs

        if agg == 'ese':
            aggregation_conv = ConvBNAct(total_chs, out_chs, kernel_size=1, act=act)
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(aggregation_conv, att)
        elif agg == 'se':
            squeeze = ConvBNAct(total_chs, out_chs // 2, kernel_size=1, act=act)
            excitation = ConvBNAct(out_chs // 2, out_chs, kernel_size=1, act=act)
            self.aggregation = nn.Sequential(squeeze, excitation)
        elif agg == 'none':
            self.aggregation = ConvBNAct(total_chs, out_chs, kernel_size=1, act=act)
        else:
            raise ValueError(f"Unknown aggregation type: {agg}")

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        outputs = [x]

        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        # Concatenate all intermediate features
        x = torch.cat(outputs, dim=1)

        # Aggregation with attention
        x = self.aggregation(x)

        # Residual connection
        if self.residual:
            x = self.drop_path(x) + identity

        return x


# ---------------------------------------------------------------------------
#  HG Stage
# ---------------------------------------------------------------------------

class HG_Stage(nn.Module):
    """HG Stage: a sequence of HG_Blocks with optional downsampling.

    Args:
        in_chs: Input channels.
        mid_chs: Intermediate channels per HG_Block.
        out_chs: Output channels.
        block_num: Number of HG_Blocks in this stage.
        layer_num: Number of layers per HG_Block.
        downsample: Whether to downsample at the start of stage.
        light_block: Whether to use lightweight conv in blocks.
        kernel_size: Kernel size for blocks.
        agg: Aggregation type.
        drop_path: Drop path rate (float or list).
        act: Activation function.
    """

    def __init__(
        self,
        in_chs: int,
        mid_chs: int,
        out_chs: int,
        block_num: int,
        layer_num: int,
        downsample: bool = True,
        light_block: bool = False,
        kernel_size: int = 3,
        agg: str = 'ese',
        drop_path: float = 0.0,
        act: str = 'relu',
    ):
        super().__init__()

        # Downsample at the start (depthwise conv, stride 2)
        if downsample:
            self.downsample = ConvBNAct(
                in_chs, in_chs, kernel_size=3, stride=2,
                groups=in_chs, act='none',
            )
        else:
            self.downsample = nn.Identity()

        # Build HG blocks
        blocks = []
        for i in range(block_num):
            in_c = in_chs if i == 0 else out_chs
            dp = drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path
            blocks.append(
                HG_Block(
                    in_chs=in_c,
                    mid_chs=mid_chs,
                    out_chs=out_chs,
                    layer_num=layer_num,
                    kernel_size=kernel_size,
                    residual=False if i == 0 else True,
                    light_block=light_block,
                    agg=agg,
                    drop_path=dp,
                    act=act,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


# ---------------------------------------------------------------------------
#  DFD (Duplex Feature Downsampling) Module
# ---------------------------------------------------------------------------

class DFDModule(nn.Module):
    """Duplex Feature Downsampling Module.

    A two-branch complementary downsampling module that couples large-kernel
    spatial filtering with space-to-depth sampling and cross-branch channel
    recalibration (BFE - Bidirectional Feature Enhancement) to mitigate
    spatial aliasing and retain high-frequency boundary cues.

    Architecture:
        Branch 1 (Spatial Filtering): Conv7x7 -> Conv3x3(s=2) + MaxPool
        Branch 2 (Space Reordering):   DWConv(s=2) + 1x1 Projection
        Cross-Branch Modulation (BFE): Bidirectional SE attention gating
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = in_channels

        # Branch 1: Large-kernel spatial filtering path
        self.branch1_conv7 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=7,
            padding=3, groups=1, bias=False,
        )
        self.branch1_conv3 = nn.Conv2d(
            mid_channels, mid_channels // 2, kernel_size=3,
            stride=2, padding=1, bias=False,
        )
        self.branch1_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch1_proj = nn.Conv2d(
            mid_channels, mid_channels // 2, kernel_size=1, bias=False,
        )
        self.bn1_conv = nn.BatchNorm2d(mid_channels // 2)
        self.bn1_pool = nn.BatchNorm2d(mid_channels // 2)

        # Branch 2: Depthwise conv + space reordering path
        self.branch2_dwconv = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3,
            stride=2, padding=1, groups=mid_channels, bias=False,
        )
        self.branch2_proj = nn.Conv2d(
            mid_channels, mid_channels // 2, kernel_size=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels // 2)

        # BFE: Bidirectional Feature Enhancement (cross-branch SE gating)
        bfe_mid = max(mid_channels // 8, 8)
        self.bfe_gate_u = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels // 2, bfe_mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bfe_mid, mid_channels // 2, kernel_size=1),
            nn.Sigmoid(),
        )
        self.bfe_gate_v = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels // 2, bfe_mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bfe_mid, mid_channels // 2, kernel_size=1),
            nn.Sigmoid(),
        )

        # Final projection
        self.final_conv = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, bias=False,
        )
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.final_act = nn.SiLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _bfe(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Bidirectional Feature Enhancement.

        Adaptively recalibrates channel responses via bidirectional SE gating
        to prevent destructive interference between parallel representations.

        Args:
            u: Feature tensor from branch 1, [B, C, H, W]
            v: Feature tensor from branch 2, [B, C, H, W]

        Returns:
            Modulated and concatenated features, [B, 2C, H, W]
        """
        a_u = self.bfe_gate_u(u)  # [B, C, 1, 1]
        a_v = self.bfe_gate_v(v)  # [B, C, 1, 1]

        u_modulated = u * (1.0 + a_v)  # u enhanced by v's gate
        v_modulated = v * (1.0 + a_u)  # v enhanced by u's gate

        return torch.cat([u_modulated, v_modulated], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1: Spatial filtering (conv7 + complementary downsampling)
        x1 = self.branch1_conv7(x)
        x1_conv = self.bn1_conv(self.branch1_conv3(x1))
        x1_pool = self.bn1_pool(self.branch1_maxpool(self.branch1_proj(x1)))
        branch1 = x1_conv + x1_pool  # complementary downsampling

        # Branch 2: Space reordering (DWConv + 1x1 projection)
        x2 = self.branch2_dwconv(x)
        branch2 = self.bn2(self.branch2_proj(x2))

        # BFE: Bidirectional Feature Enhancement
        fused = self._bfe(branch1, branch2)

        # Final projection
        out = self.final_act(self.final_bn(self.final_conv(fused)))
        return out


# ---------------------------------------------------------------------------
#  HGNetV2 + DFD Backbone
# ---------------------------------------------------------------------------

class HGNetV2DFD(nn.Module):
    """HGNetV2 Backbone with DFD (Duplex Feature Downsampling).

    Complete HGNetV2 backbone with integrated DFD module between stem and
    the first stage for enhanced small-object feature preservation.

    Architecture:
        Input -> StemBlock (x4 down) -> DFD (x2 down) -> Stage1 -> Stage2 -> Stage3 -> Stage4
        Output features: S3 (stride 8), S4 (stride 16), S5 (stride 32)

    Args:
        name: Model size name ('n', 's', 'm', 'l', 'x').
        return_idx: Indices of stages to return (0-based, 0=stage1).
        act: Activation function.
        agg: Aggregation type for HG blocks.
        pretrained: Whether to load pretrained weights.
        use_dfd: Whether to use DFD module (default True).
    """

    # Architecture configurations for different model sizes
    arch_configs = {
        'n': {  # Nano
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_chs, mid_chs, out_chs, block_num, downsample, light_block, kernel_size, layer_num
                'stage1': [16, 16, 64, 1, False, False, 3, 3],
                'stage2': [64, 32, 256, 1, True, False, 3, 3],
                'stage3': [256, 64, 512, 2, True, True, 5, 3],
                'stage4': [512, 128, 1024, 1, True, True, 5, 3],
            },
        },
        's': {  # Small
            'stem_channels': [3, 24, 32],
            'stage_config': {
                'stage1': [32, 32, 64, 1, False, False, 3, 3],
                'stage2': [64, 48, 256, 1, True, False, 3, 3],
                'stage3': [256, 96, 512, 2, True, True, 5, 3],
                'stage4': [512, 192, 1024, 1, True, True, 5, 3],
            },
        },
        'm': {  # Medium
            'stem_channels': [3, 24, 32],
            'stage_config': {
                'stage1': [32, 32, 128, 1, False, False, 3, 5],
                'stage2': [128, 64, 512, 1, True, False, 3, 5],
                'stage3': [512, 128, 1024, 3, True, True, 5, 5],
                'stage4': [1024, 256, 2048, 1, True, True, 5, 5],
            },
        },
        'l': {  # Large
            'stem_channels': [3, 32, 48],
            'stage_config': {
                'stage1': [48, 48, 128, 1, False, False, 3, 6],
                'stage2': [128, 96, 512, 1, True, False, 3, 6],
                'stage3': [512, 192, 1024, 3, True, True, 5, 6],
                'stage4': [1024, 384, 2048, 1, True, True, 5, 6],
            },
        },
        'x': {  # X-Large
            'stem_channels': [3, 32, 64],
            'stage_config': {
                'stage1': [64, 64, 128, 1, False, False, 3, 6],
                'stage2': [128, 128, 512, 2, True, False, 3, 6],
                'stage3': [512, 256, 1024, 5, True, True, 5, 6],
                'stage4': [1024, 512, 2048, 2, True, True, 5, 6],
            },
        },
    }

    def __init__(
        self,
        name: str = 'n',
        return_idx: List[int] = None,
        act: str = 'relu',
        agg: str = 'ese',
        pretrained: bool = False,
        use_dfd: bool = True,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.name = name
        self.return_idx = return_idx or [1, 2, 3]  # default: S3, S4, S5
        self.use_dfd = use_dfd

        if name not in self.arch_configs:
            raise ValueError(
                f"Unknown HGNetV2 size '{name}'. "
                f"Available: {list(self.arch_configs.keys())}"
            )

        config = self.arch_configs[name]
        stem_channels = config['stem_channels']
        stage_config = config['stage_config']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # Stem block
        self.stem = StemBlock(
            in_chs=stem_channels[0],
            mid_chs=stem_channels[1],
            out_chs=stem_channels[2],
            act=act,
        )

        # DFD module (between stem and stage1)
        if use_dfd:
            dfd_out_channels = stage_config['stage1'][0]
            self.dfd = DFDModule(
                in_channels=stem_channels[2],
                out_channels=dfd_out_channels,
            )
        else:
            self.dfd = nn.Identity()

        # Compute drop path rates per block
        total_blocks = sum(
            stage_config[k][3] for k in stage_config
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Build stages
        self.stages = nn.ModuleList()
        block_idx = 0
        for stage_name in stage_config:
            (in_chs, mid_chs, out_chs, block_num, downsample,
             light_block, kernel_size, layer_num) = stage_config[stage_name]

            stage_dpr = dpr[block_idx:block_idx + block_num]
            block_idx += block_num

            self.stages.append(
                HG_Stage(
                    in_chs=in_chs,
                    mid_chs=mid_chs,
                    out_chs=out_chs,
                    block_num=block_num,
                    layer_num=layer_num,
                    downsample=downsample,
                    light_block=light_block,
                    kernel_size=kernel_size,
                    agg=agg,
                    drop_path=stage_dpr,
                    act=act,
                )
            )

        if pretrained:
            self._load_pretrained(name)

    def _load_pretrained(self, name: str):
        """Load pretrained HGNetV2 weights (skip DFD-specific layers)."""
        # Placeholder: in practice, load from checkpoint and filter
        pass

    @property
    def out_channels(self) -> List[int]:
        """Output channel counts for each returned stage."""
        return [self._out_channels[i] for i in self.return_idx]

    @property
    def out_strides(self) -> List[int]:
        """Output strides for each returned stage."""
        return [self._out_strides[i] for i in self.return_idx]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input image tensor [B, 3, H, W].

        Returns:
            List of feature maps from selected stages.
        """
        # Stem
        x = self.stem(x)

        # DFD module
        if self.use_dfd:
            x = self.dfd(x)

        # Stages
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs


# ---------------------------------------------------------------------------
#  Factory function
# ---------------------------------------------------------------------------

def build_hgnetv2_dfd(cfg: dict) -> HGNetV2DFD:
    """Build HGNetV2-DFD backbone from configuration.

    Args:
        cfg: Configuration dict with keys:
            - name: Model size ('n', 's', 'm', 'l', 'x').
            - return_idx: Stage indices to return.
            - act: Activation function.
            - agg: Aggregation type.
            - pretrained: Whether to use pretrained weights.
            - use_dfd: Whether to use DFD module.

    Returns:
        HGNetV2DFD model instance.
    """
    return HGNetV2DFD(
        name=cfg.get('name', 'n'),
        return_idx=cfg.get('return_idx', [1, 2, 3]),
        act=cfg.get('act', 'relu'),
        agg=cfg.get('agg', 'ese'),
        pretrained=cfg.get('pretrained', False),
        use_dfd=cfg.get('use_dfd', True),
        drop_path_rate=cfg.get('drop_path_rate', 0.0),
    )
