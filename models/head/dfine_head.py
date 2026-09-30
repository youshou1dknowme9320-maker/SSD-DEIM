"""
DFINE-style Detection Head for SSD-DEIM.

Implements a D-FINE style detection head with multi-scale deformable attention,
iterative box refinement, and dense one-to-one matching.

Key components:
    - MSDeformableAttention: Multi-scale deformable cross-attention
    - TransformerDecoderLayer: Decoder layer with self-attn + cross-attn + FFN
    - Gate: Gating mechanism for iterative refinement
    - Integral: Distribution-based box regression (non-uniform weighting)
    - LQE: Local Quality Estimation (classification-IoU joint learning)

Reference:
    D-FINE: Redefining Regression Targets in Object Detection (CVPR 2025)
    DEIM: DETR with Improved Matching for Fast Convergence (CVPR 2025)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import functools


# ---------------------------------------------------------------------------
#  Utility: Activation & MLP
# ---------------------------------------------------------------------------

def get_activation(name: str = 'relu') -> nn.Module:
    """Get activation function by name."""
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'silu' or name == 'swish':
        return nn.SiLU(inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(0.1, inplace=True)
    else:
        return nn.ReLU(inplace=True)


class MLP(nn.Module):
    """Multi-Layer Perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        act: str = 'relu',
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    """Initialize bias with prior probability for classification head."""
    return -math.log((1.0 - prior_prob) / prior_prob)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Inverse sigmoid (logit) function."""
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


# ---------------------------------------------------------------------------
#  MSDeformableAttention (simplified PyTorch implementation)
# ---------------------------------------------------------------------------

class MSDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention.

    Pure PyTorch implementation of MSDeformAttn used in D-FINE decoder.
    Each query attends to K reference points across L feature levels.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_levels: Number of feature levels.
        num_points: Number of sampling points per level per head.
        offset_scale: Scale factor for sampling offsets.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 3,
        num_points: int = 4,
        offset_scale: float = 0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.total_points = num_heads * num_levels * num_points

        # Sampling offsets: [bs, num_queries, total_points * 2]
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        # Attention weights: [bs, num_queries, total_points]
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize offsets in a grid pattern
        nn.init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile(
            [1, self.num_levels, self.num_points, 1]
        )
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32)
        scaling = scaling.reshape(1, 1, self.num_points, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data = grid_init.flatten()

        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)

        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: List[Tuple[int, int]],
        value_level_start_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: [bs, num_queries, embed_dim]
            reference_points: [bs, num_queries, num_levels, 2] or [bs, num_queries, num_levels, 4]
            value: [bs, num_value_tokens, embed_dim]
            value_spatial_shapes: List of (H, W) per level.
            value_level_start_index: Start index of each level in flattened value.

        Returns:
            output: [bs, num_queries, embed_dim]
        """
        bs, num_queries, _ = query.shape
        device = query.device

        # Value projection
        value = self.value_proj(value)

        # Compute level start indices
        if value_level_start_index is None:
            level_start_index = [0]
            for h, w in value_spatial_shapes:
                level_start_index.append(level_start_index[-1] + h * w)
            value_level_start_index = torch.tensor(
                level_start_index[:-1], dtype=torch.long, device=device
            )

        # Sampling offsets from query
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(
            bs, num_queries, self.num_heads, self.num_levels, self.num_points, 2
        )

        # Attention weights
        attention_weights = self.attention_weights(query).reshape(
            bs, num_queries, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.reshape(
            bs, num_queries, self.num_heads, self.num_levels, self.num_points
        )

        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            # reference_points: [bs, num_queries, num_levels, 2]
            offset_normalizer = torch.tensor(
                [[w, h] for h, w in value_spatial_shapes],
                dtype=torch.float32, device=device,
            )
            offset_normalizer = offset_normalizer.reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            # reference_points: [bs, num_queries, num_levels, 4] (cx, cy, w, h)
            ref_xy = reference_points[:, :, None, :, None, :2]
            ref_wh = reference_points[:, :, None, :, None, 2:]
            sampling_locations = ref_xy + sampling_offsets * ref_wh * self.offset_scale
        else:
            raise ValueError(
                f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}"
            )

        # Reshape value: [bs, num_heads, num_value_tokens, head_dim]
        value = value.reshape(bs, -1, self.num_heads, self.head_dim)
        value = value.permute(0, 2, 1, 3)  # [bs, num_heads, num_tokens, head_dim]

        # Sample and compute attention
        output = self._deformable_attn_core(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights,
        )

        # Output projection
        output = self.output_proj(output)
        return output

    def _deformable_attn_core(
        self,
        value: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Core deformable attention computation using bilinear sampling.

        Args:
            value: [bs, num_heads, num_tokens, head_dim]
            spatial_shapes: List of (H, W) per level.
            level_start_index: [num_levels] start indices.
            sampling_locations: [bs, num_queries, num_heads, num_levels, num_points, 2]
            attention_weights: [bs, num_queries, num_heads, num_levels, num_points]

        Returns:
            output: [bs, num_queries, embed_dim]
        """
        bs, num_heads, _, head_dim = value.shape
        num_queries = sampling_locations.shape[1]
        num_levels = self.num_levels
        num_points = self.num_points

        output = torch.zeros(
            bs, num_queries, num_heads, head_dim,
            device=value.device, dtype=value.dtype,
        )

        for level_idx in range(num_levels):
            h, w = spatial_shapes[level_idx]
            start = level_start_index[level_idx]
            end = start + h * w

            # Value for this level: [bs, num_heads, H*W, head_dim] -> [bs*num_heads, head_dim, H, W]
            v = value[:, :, start:end, :]
            v = v.reshape(bs * num_heads, h, w, head_dim)
            v = v.permute(0, 3, 1, 2)  # [bs*num_heads, head_dim, H, W]

            # Sampling locations for this level: [bs, num_queries, num_heads, num_points, 2]
            sampling_locs = sampling_locations[:, :, :, level_idx, :, :]
            # Normalize to [-1, 1] for grid_sample
            sampling_locs_2x = 2.0 * sampling_locs - 1.0
            # Reshape for grid_sample: [bs*num_heads, num_queries*num_points, 1, 2]
            sampling_locs_2x = sampling_locs_2x.permute(0, 2, 1, 3, 4)
            sampling_locs_2x = sampling_locs_2x.reshape(
                bs * num_heads, num_queries * num_points, 1, 2
            )

            # Bilinear sampling
            sampled = F.grid_sample(
                v, sampling_locs_2x,
                mode='bilinear', padding_mode='zeros', align_corners=False,
            )
            # [bs*num_heads, head_dim, num_queries*num_points, 1]
            sampled = sampled.squeeze(-1)
            sampled = sampled.reshape(bs, num_heads, head_dim, num_queries, num_points)
            sampled = sampled.permute(0, 3, 1, 2, 4)
            # [bs, num_queries, num_heads, head_dim, num_points]

            # Attention weights: [bs, num_queries, num_heads, num_points]
            attn = attention_weights[:, :, :, level_idx, :]
            attn = attn.unsqueeze(-2)  # [bs, num_queries, num_heads, 1, num_points]

            # Weighted sum
            output_level = (sampled * attn).sum(dim=-1)
            output = output + output_level

        output = output.reshape(bs, num_queries, num_heads * head_dim)
        return output


# ---------------------------------------------------------------------------
#  Transformer Decoder Layer
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer with self-attention + cross-attention + FFN.

    Args:
        d_model: Model dimension.
        n_head: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.
        activation: Activation function name.
        n_levels: Number of feature levels for cross-attention.
        n_points: Number of sampling points for deformable attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = 'relu',
        n_levels: int = 3,
        n_points: int = 4,
    ):
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention (multi-scale deformable)
        self.cross_attn = MSDeformableAttention(
            embed_dim=d_model,
            num_heads=n_head,
            num_levels=n_levels,
            num_points=n_points,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        reference_points: torch.Tensor,
        memory: torch.Tensor,
        memory_spatial_shapes: List[Tuple[int, int]],
        memory_level_start_index: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tgt: Decoder queries [bs, num_queries, d_model].
            reference_points: Reference points for cross-attn.
            memory: Encoder features (flattened multi-scale).
            memory_spatial_shapes: Spatial shapes per level.
            memory_level_start_index: Start indices per level.

        Returns:
            Updated tgt tensor.
        """
        # Self attention
        q = k = tgt
        tgt2 = self.self_attn(
            q, k, tgt, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention (deformable)
        tgt2 = self.cross_attn(
            query=tgt,
            reference_points=reference_points,
            value=memory,
            value_spatial_shapes=memory_spatial_shapes,
            value_level_start_index=memory_level_start_index,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


# ---------------------------------------------------------------------------
#  Gate Module (for iterative refinement)
# ---------------------------------------------------------------------------

class Gate(nn.Module):
    """Gate module for iterative box refinement.

    Controls how much new predictions replace old ones, enabling
    stable iterative refinement across decoder layers.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gate combination of old (x) and new (y) features.

        Args:
            x: Old features [bs, num_queries, dim].
            y: New features [bs, num_queries, dim].

        Returns:
            Gated combination.
        """
        g = self.gate(torch.cat([x, y], dim=-1))
        return g * x + (1.0 - g) * y


# ---------------------------------------------------------------------------
#  Integral Module (Distribution-based Box Regression)
# ---------------------------------------------------------------------------

class Integral(nn.Module):
    """Integral distribution-based box regression.

    Decodes predicted distribution logits into continuous box offsets using
    a non-uniform weighting function, enabling more precise regression
    especially for small objects.

    Args:
        reg_max: Number of distribution bins.
        reg_scale: Controls the curvature of the weighting function.
    """

    def __init__(self, reg_max: int = 16, reg_scale: float = 2.0):
        super().__init__()
        self.reg_max = reg_max
        self.reg_scale = abs(reg_scale)
        self._build_weighting_function()

    def _build_weighting_function(self):
        """Build non-uniform weighting function values."""
        reg_max = self.reg_max
        reg_scale = self.reg_scale

        upper_bound1 = reg_scale
        upper_bound2 = reg_scale * 2
        step = (upper_bound1 + 1) ** (2.0 / (reg_max - 2))

        left_values = [-(step ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step ** i) - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left_values + [0.0] + right_values + [upper_bound2]

        self.register_buffer(
            'proj', torch.tensor(values, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode distribution logits to continuous values.

        Args:
            x: Distribution logits [..., reg_max].

        Returns:
            Continuous regression values [..., 1].
        """
        weights = F.softmax(x, dim=-1)
        # Weighted sum of bin positions
        out = (weights * self.proj.to(x.dtype)).sum(dim=-1, keepdim=True)
        return out


# ---------------------------------------------------------------------------
#  LQE (Local Quality Estimation)
# ---------------------------------------------------------------------------

class LQE(nn.Module):
    """Local Quality Estimation.

    Predicts an IoU-aware quality score for each detection, used to
    modulate classification scores for better ranking.
    """

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.qscore = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict quality scores.

        Args:
            x: Decoder features [bs, num_queries, dim].

        Returns:
            Quality scores [bs, num_queries, 1].
        """
        return self.qscore(x)


# ---------------------------------------------------------------------------
#  DFINE Head
# ---------------------------------------------------------------------------

class DFINEHead(nn.Module):
    """DFINE-style Detection Head.

    Complete detection head with:
        - Multi-scale deformable attention decoder
        - Iterative box refinement with gating
        - Distribution-based integral regression
        - Dense one-to-one matching
        - LQE (Local Quality Estimation)

    Args:
        in_channels: List of input channel counts from neck.
        feat_channels: Feature dimension inside decoder.
        num_classes: Number of object classes.
        num_decoder_layers: Number of decoder layers.
        num_heads: Number of attention heads.
        num_points: Number of deformable attention points.
        reg_max: Maximum distribution bins for regression.
        reg_scale: Scale factor for weighting function.
        dim_feedforward: FFN dimension in decoder.
        dropout: Dropout rate.
        activation: Activation function.
    """

    def __init__(
        self,
        in_channels: List[int],
        feat_channels: int = 256,
        num_classes: int = 1,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_points: int = 4,
        reg_max: int = 16,
        reg_scale: float = 2.0,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = 'relu',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_decoder_layers = num_decoder_layers
        self.feat_channels = feat_channels
        self.num_levels = len(in_channels)
        self.reg_max = reg_max
        self.reg_scale = reg_scale

        # Input projection: project each level to feat_channels
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, feat_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, feat_channels),
            )
            for c in in_channels
        ])

        # Level embeddings (for multi-scale features)
        self.level_embed = nn.Parameter(
            torch.zeros(self.num_levels, feat_channels)
        )
        nn.init.normal_(self.level_embed, std=0.01)

        # Dense anchor points initialization (grid points per level)
        # Reference points are generated from feature grid positions

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=feat_channels,
                n_head=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                n_levels=self.num_levels,
                n_points=num_points,
            )
            for _ in range(num_decoder_layers)
        ])

        # Decoder layer norms
        self.decoder_norms = nn.ModuleList([
            nn.LayerNorm(feat_channels) for _ in range(num_decoder_layers)
        ])

        # Gate modules for iterative refinement
        self.gates = nn.ModuleList([
            Gate(feat_channels) for _ in range(num_decoder_layers)
        ])

        # Integral regression
        self.integral = Integral(reg_max=reg_max, reg_scale=reg_scale)

        # Classification head
        self.class_embed = nn.ModuleList([
            nn.Linear(feat_channels, num_classes)
            for _ in range(num_decoder_layers)
        ])

        # Regression head (predicts distribution over reg_max bins * 4 sides)
        self.bbox_embed = nn.ModuleList([
            MLP(feat_channels, feat_channels, 4 * reg_max, num_layers=3, act=activation)
            for _ in range(num_decoder_layers)
        ])

        # LQE head
        self.lqe = nn.ModuleList([
            LQE(feat_channels, num_classes)
            for _ in range(num_decoder_layers)
        ])

        # Initialize heads
        self._init_heads()

    def _init_heads(self):
        """Initialize classification and regression heads."""
        prior_prob = 0.01
        bias_value = bias_init_with_prob(prior_prob)
        for cls_embed in self.class_embed:
            nn.init.normal_(cls_embed.weight, std=0.01)
            nn.init.constant_(cls_embed.bias, bias_value)

        for bbox_embed in self.bbox_embed:
            for m in bbox_embed.layers:
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def _generate_reference_points(
        self,
        spatial_shapes: List[Tuple[int, int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Generate dense reference points (anchor grid) on multi-scale features.

        Args:
            spatial_shapes: List of (H, W) for each level.
            device: Target device.
            dtype: Target dtype.

        Returns:
            reference_points: [bs, num_queries, num_levels, 2] normalized coords.
            query_embeds: [bs, num_queries, feat_dim] (flattened level embeddings).
            num_queries_per_level: List of query counts per level.
        """
        all_points = []
        all_level_ids = []

        for level_idx, (h, w) in enumerate(spatial_shapes):
            # Grid centers in [0, 1]
            shift_x = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
            shift_y = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
            yy, xx = torch.meshgrid(shift_y, shift_x, indexing='ij')
            points = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [H*W, 2]
            all_points.append(points)
            all_level_ids.append(torch.full((h * w,), level_idx, device=device, dtype=torch.long))

        # Concatenate all levels
        reference_points = torch.cat(all_points, dim=0)  # [total_queries, 2]
        level_ids = torch.cat(all_level_ids, dim=0)

        num_queries_per_level = [h * w for h, w in spatial_shapes]
        total_queries = sum(num_queries_per_level)

        # Expand to [num_queries, num_levels, 2] - each query has reference for all levels
        ref_expanded = reference_points.unsqueeze(1).expand(-1, self.num_levels, -1)

        # Query embeddings from level embeddings
        query_embeds = self.level_embed[level_ids]  # [total_queries, feat_dim]

        return ref_expanded.unsqueeze(0), query_embeds.unsqueeze(0), num_queries_per_level

    def _prepare_memory(
        self,
        feats: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]], torch.Tensor]:
        """Prepare multi-scale features for deformable attention.

        Args:
            feats: Tuple of feature maps from neck.

        Returns:
            memory: Flattened features [bs, total_tokens, feat_dim].
            spatial_shapes: List of (H, W) per level.
            level_start_index: Start index of each level.
        """
        bs = feats[0].shape[0]
        device = feats[0].device

        projected = []
        spatial_shapes = []

        for i, feat in enumerate(feats):
            # Project to feat_channels
            proj = self.input_proj[i](feat)  # [bs, feat_dim, H, W]
            h, w = proj.shape[2:]
            spatial_shapes.append((h, w))
            # Flatten: [bs, feat_dim, H*W] -> [bs, H*W, feat_dim]
            proj_flat = proj.flatten(2).transpose(1, 2)
            # Add level embedding
            proj_flat = proj_flat + self.level_embed[i].to(proj_flat.dtype).unsqueeze(0).unsqueeze(0)
            projected.append(proj_flat)

        memory = torch.cat(projected, dim=1)

        # Level start indices
        level_start = [0]
        for h, w in spatial_shapes:
            level_start.append(level_start[-1] + h * w)
        level_start_index = torch.tensor(
            level_start[:-1], dtype=torch.long, device=device
        )

        return memory, spatial_shapes, level_start_index

    def forward(
        self,
        feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: Optional[List[Dict]] = None,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            feats: Tuple of (P3, P4, P5) feature maps from neck.
            targets: List of target dicts (for training).
            return_loss: Whether to compute losses.

        Returns:
            Dict with predictions and/or losses.
        """
        bs = feats[0].shape[0]
        device = feats[0].device

        # Prepare memory and reference points
        memory, spatial_shapes, level_start_index = self._prepare_memory(feats)
        reference_points, query_embeds, num_queries_per_level = (
            self._generate_reference_points(spatial_shapes, device, memory.dtype)
        )

        # Initial queries: learnable embeddings + reference point info
        tgt = query_embeds.expand(bs, -1, -1)

        # Decode initial boxes from reference points
        init_boxes = self._ref_to_init_boxes(reference_points[:, :, 0, :], spatial_shapes)

        # Iterative refinement
        all_cls_scores = []
        all_bbox_preds = []
        all_lqe_scores = []

        ref_points = reference_points  # [bs, num_queries, num_levels, 2]

        for layer_idx in range(self.num_decoder_layers):
            # Store old features for gating
            tgt_old = tgt.clone()

            # Decoder layer
            tgt = self.decoder_layers[layer_idx](
                tgt=tgt,
                reference_points=ref_points,
                memory=memory,
                memory_spatial_shapes=spatial_shapes,
                memory_level_start_index=level_start_index,
            )
            tgt = self.decoder_norms[layer_idx](tgt)

            # Gating
            tgt = self.gates[layer_idx](tgt_old, tgt)

            # Classification
            cls_score = self.class_embed[layer_idx](tgt)
            all_cls_scores.append(cls_score)

            # Bbox regression (distribution)
            bbox_dist = self.bbox_embed[layer_idx](tgt)
            # Decode distribution to offsets
            bbox_dist = bbox_dist.reshape(bs, -1, 4, self.reg_max)
            bbox_offsets = self.integral(bbox_dist).squeeze(-1)  # [bs, num_queries, 4]

            # Decode boxes from offsets (relative to current reference)
            bbox_pred = self._decode_boxes(
                init_boxes, bbox_offsets, self.reg_scale
            )
            all_bbox_preds.append(bbox_pred)

            # LQE
            lqe_score = self.lqe[layer_idx](tgt)
            all_lqe_scores.append(lqe_score)

            # Update reference points for next layer (from current bbox)
            if layer_idx < self.num_decoder_layers - 1:
                ref_points = self._box_to_refpoints(bbox_pred, self.num_levels)

        # Stack predictions from all layers
        outputs_class = torch.stack(all_cls_scores, dim=0)  # [num_layers, bs, num_queries, num_classes]
        outputs_coord = torch.stack(all_bbox_preds, dim=0)  # [num_layers, bs, num_queries, 4]
        outputs_lqe = torch.stack(all_lqe_scores, dim=0)    # [num_layers, bs, num_queries, 1]

        result = {
            'pred_logits': outputs_class[-1],   # [bs, num_queries, num_classes]
            'pred_boxes': outputs_coord[-1],    # [bs, num_queries, 4] (cxcywh normalized)
            'pred_lqe': outputs_lqe[-1],        # [bs, num_queries, 1]
            'aux_outputs': [
                {
                    'pred_logits': outputs_class[i],
                    'pred_boxes': outputs_coord[i],
                    'pred_lqe': outputs_lqe[i],
                }
                for i in range(self.num_decoder_layers - 1)
            ],
            'num_queries_per_level': num_queries_per_level,
            'spatial_shapes': spatial_shapes,
        }

        return result

    @staticmethod
    def _ref_to_init_boxes(
        ref_points: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """Convert reference points to initial anchor boxes (cxcywh format).

        Args:
            ref_points: [bs, num_queries, 2] normalized centers.
            spatial_shapes: List of (H, W) per level.

        Returns:
            Initial boxes [bs, num_queries, 4] in cxcywh normalized format.
        """
        bs, num_queries = ref_points.shape[:2]
        device = ref_points.device

        # Compute initial size based on feature level stride
        # Use a simple heuristic: initial size proportional to stride
        boxes = []
        start = 0
        for level_idx, (h, w) in enumerate(spatial_shapes):
            n = h * w
            stride_h = 1.0 / h
            stride_w = 1.0 / w
            # Initial size: scale relative to stride
            size_scale = 4.0
            w_init = torch.full((n,), stride_w * size_scale, device=device, dtype=ref_points.dtype)
            h_init = torch.full((n,), stride_h * size_scale, device=device, dtype=ref_points.dtype)

            level_ref = ref_points[:, start:start + n, :]  # [bs, n, 2]
            cx, cy = level_ref[..., 0], level_ref[..., 1]

            level_boxes = torch.stack([cx, cy, w_init, h_init], dim=-1)
            boxes.append(level_boxes)
            start += n

        return torch.cat(boxes, dim=1)

    @staticmethod
    def _decode_boxes(
        anchor_boxes: torch.Tensor,
        offsets: torch.Tensor,
        reg_scale: float = 2.0,
    ) -> torch.Tensor:
        """Decode predicted offsets to box coordinates (cxcywh normalized).

        Args:
            anchor_boxes: [bs, num_queries, 4] anchor boxes in cxcywh.
            offsets: [bs, num_queries, 4] predicted distances (l, t, r, b).
            reg_scale: Regression scale factor.

        Returns:
            Decoded boxes [bs, num_queries, 4] in cxcywh normalized.
        """
        reg_scale = abs(reg_scale)
        cx, cy, w, h = anchor_boxes.unbind(-1)
        dl, dt, dr, db = offsets.unbind(-1)

        # Distance to boundaries
        x1 = cx - (0.5 * reg_scale + dl) * (w / reg_scale)
        y1 = cy - (0.5 * reg_scale + dt) * (h / reg_scale)
        x2 = cx + (0.5 * reg_scale + dr) * (w / reg_scale)
        y2 = cy + (0.5 * reg_scale + db) * (h / reg_scale)

        # Convert to cxcywh
        new_cx = (x1 + x2) / 2.0
        new_cy = (y1 + y2) / 2.0
        new_w = (x2 - x1).clamp(min=0.0)
        new_h = (y2 - y1).clamp(min=0.0)

        return torch.stack([new_cx, new_cy, new_w, new_h], dim=-1)

    @staticmethod
    def _box_to_refpoints(
        boxes: torch.Tensor,
        num_levels: int,
    ) -> torch.Tensor:
        """Convert boxes to reference points for deformable attention.

        Args:
            boxes: [bs, num_queries, 4] in cxcywh format.
            num_levels: Number of feature levels.

        Returns:
            Reference points [bs, num_queries, num_levels, 4] (cx, cy, w, h).
        """
        # Expand boxes to all levels
        ref = boxes.unsqueeze(2).expand(-1, -1, num_levels, -1)
        return ref


# ---------------------------------------------------------------------------
#  Factory function
# ---------------------------------------------------------------------------

def build_dfine_head(cfg: dict) -> DFINEHead:
    """Build DFINE detection head from configuration.

    Args:
        cfg: Configuration dict.

    Returns:
        DFINEHead instance.
    """
    return DFINEHead(
        in_channels=cfg['in_channels'],
        feat_channels=cfg.get('feat_channels', 256),
        num_classes=cfg.get('num_classes', 1),
        num_decoder_layers=cfg.get('num_decoder_layers', 6),
        num_heads=cfg.get('num_heads', 8),
        num_points=cfg.get('num_points', 4),
        reg_max=cfg.get('reg_max', 16),
        reg_scale=cfg.get('reg_scale', 2.0),
        dim_feedforward=cfg.get('dim_feedforward', 1024),
        dropout=cfg.get('dropout', 0.0),
        activation=cfg.get('activation', 'relu'),
    )
