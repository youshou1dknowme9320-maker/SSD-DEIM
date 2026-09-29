# SSD-DEIM Model Implementation
# Paper: SSD-DEIM: Preserving Multi-Scale Fidelity and Geometric Priors
#         for Dense Camouflaged Small Object Detection (ICASSP 2027)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class DuplexFeatureDownsampling(nn.Module):
    """
    Duplex Feature Downsampling (DFD) Module.
    
    A two-stage complementary downsampling module that couples large-kernel
    spatial filtering with space-to-depth sampling and cross-branch channel
    recalibration to mitigate spatial aliasing and retain high-frequency
    boundary cues.
    
    Architecture:
        Branch 1 (Spatial Filtering): Conv7x7 -> Conv3x3(s=2) + MaxPool
        Branch 2 (Space Reordering):   DWConv(s=2) + Space-to-Depth Projection
        Cross-Branch Modulation:       Bidirectional gating via channel attention
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = in_channels
        
        # Branch 1: Large-kernel spatial filtering path
        self.branch1_conv7 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=7,
            padding=3, groups=1, bias=False
        )
        self.branch1_conv3 = nn.Conv2d(
            mid_channels, mid_channels // 2, kernel_size=3,
            stride=2, padding=1, bias=False
        )
        self.branch1_maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        )
        self.branch1_proj = nn.Conv2d(
            mid_channels // 2, mid_channels // 2,
            kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels // 2)
        self.bn1_pool = nn.BatchNorm2d(mid_channels // 2)
        
        # Branch 2: Depthwise conv + space-to-depth path
        self.branch2_dwconv = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3,
            stride=2, padding=1, groups=mid_channels, bias=False
        )
        self.branch2_proj = nn.Conv2d(
            mid_channels, mid_channels // 2,
            kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels // 2)
        
        # Cross-branch modulation (bidirectional gating)
        self.cross_mod_mlp_u = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels // 2, mid_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 8, mid_channels // 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.cross_mod_mlp_v = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels // 2, mid_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 8, mid_channels // 2, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final projection
        self.final_conv = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, bias=False
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
    
    def cross_branch_modulation(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Cross-Branch Modulation operator B(u, v).
        
        Adaptively recalibrates channel responses via bidirectional gating
        to prevent destructive interference between parallel representations.
        
        Args:
            u: Feature tensor from branch 1, shape [B, C, H, W]
            v: Feature tensor from branch 2, shape [B, C, H, W]
        
        Returns:
            Modulated and concatenated features, shape [B, 2C, H, W]
        """
        a_u = self.cross_mod_mlp_u(u)  # [B, C, 1, 1]
        a_v = self.cross_mod_mlp_v(v)  # [B, C, 1, 1]
        
        u_modulated = u * (1.0 + a_v)  # u enhanced by v's gate
        v_modulated = v * (1.0 + a_u)  # v enhanced by u's gate
        
        return torch.cat([u_modulated, v_modulated], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1: Spatial filtering path
        x1 = self.branch1_conv7(x)
        x1_conv = self.bn1(self.branch1_conv3(x1))
        x1_pool = self.bn1_pool(self.branch1_maxpool(self.branch1_proj(x1)))
        branch1 = x1_conv + x1_pool  # complementary downsampling
        
        # Branch 2: Space reorder path
        x2 = self.branch2_dwconv(x)
        branch2 = self.bn2(self.branch2_proj(x2))
        
        # Cross-branch modulation
        fused = self.cross_branch_modulation(branch1, branch2)
        
        # Final projection
        out = self.final_act(self.final_bn(self.final_conv(fused)))
        
        return out


class PartialConvGatedUnit(nn.Module):
    """
    Partial Convolution Gated Unit (PCGU).
    
    A channel-efficient gating unit that restricts spatial filtering to
    an active channel partition (1/4 channels), reducing FLOPs to ~13/16
    of full-channel convolution while adaptively suppressing background
    foliage clutter via element-wise soft gating.
    """
    
    def __init__(self, channels: int, partial_ratio: float = 0.25):
        super().__init__()
        self.active_channels = int(channels * partial_ratio)
        self.inactive_channels = channels - self.active_channels
        
        # Partial 3x3 conv on active channels only
        self.pconv = nn.Conv2d(
            self.active_channels, self.active_channels,
            kernel_size=3, padding=1, bias=False
        )
        
        # Gate generation path
        self.gate_conv1 = nn.Conv2d(
            self.active_channels, self.active_channels,
            kernel_size=1, bias=False
        )
        self.gate_dwconv = nn.Conv2d(
            self.active_channels, self.active_channels,
            kernel_size=3, padding=1,
            groups=self.active_channels, bias=False
        )
        self.gate_activation = nn.SiLU(inplace=True)
        
        # Linear projections for gate computation
        self.proj_gate = nn.Conv2d(
            self.active_channels, self.active_channels,
            kernel_size=1, bias=False
        )
        self.proj_skip = nn.Conv2d(
            self.active_channels, self.active_channels,
            kernel_size=1, bias=False
        )
        
        # Final projection after concatenation
        self.final_proj = nn.Conv2d(
            channels, channels, kernel_size=1, bias=False
        )
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
        # Split into active and invariant partitions
        x_active = x[:, :self.active_channels, :, :]
        x_inactive = x[:, self.active_channels:, :, :]
        
        # Partial convolution on active channels
        x_pconv = self.pconv(x_active)
        
        # Gate generation
        gate_feat = self.gate_conv1(x_pconv)
        gate_spatial = self.gate_dwconv(gate_feat)
        gate_spatial = self.gate_activation(gate_spatial)
        
        # Element-wise soft gating
        gate = torch.sigmoid(self.proj_gate(gate_spatial))
        gated_feat = x_pconv * gate
        
        # Skip connection
        skip_feat = self.proj_skip(x_pconv)
        out_active = gated_feat + skip_feat
        
        # Concatenate with invariant channels
        out = torch.cat([out_active, x_inactive], dim=1)
        
        # Final projection
        out = self.final_act(self.final_bn(self.final_proj(out)))
        
        return out


class GatedBidirectionalFeatureFusion(nn.Module):
    """
    Gated Bidirectional Feature Fusion (BFF) Module.
    
    A cross-scale feature fusion unit combining Partial Convolution
    Gated Units (PCGU) with bidirectional enhancement, reducing redundant
    channel operations while adaptively attenuating background foliage clutter.
    """
    
    def __init__(self, in_channels: List[int], out_channels: List[int]):
        super().__init__()
        assert len(in_channels) == 3, "BFF expects 3 input feature levels"
        assert len(out_channels) == 3
        
        c3, c4, c5 = in_channels
        o3, o4, o5 = out_channels
        
        # Top-down path (high-level to low-level)
        self.top_down_pcgu5 = PartialConvGatedUnit(c5)
        self.top_down_pcgu4 = PartialConvGatedUnit(c4)
        self.top_down_pcgu3 = PartialConvGatedUnit(c3)
        
        self.reduce5 = nn.Conv2d(c5, c4, kernel_size=1, bias=False)
        self.reduce4 = nn.Conv2d(c4, c3, kernel_size=1, bias=False)
        
        # Bottom-up path (low-level to high-level)
        self.bottom_up_pcgu3 = PartialConvGatedUnit(c3)
        self.bottom_up_pcgu4 = PartialConvGatedUnit(c4)
        self.bottom_up_pcgu5 = PartialConvGatedUnit(c5)
        
        self.enhance3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)
        self.enhance4 = nn.Conv2d(c4, c5, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Output projections
        self.out_conv3 = nn.Conv2d(c3, o3, kernel_size=1, bias=False)
        self.out_conv4 = nn.Conv2d(c4, o4, kernel_size=1, bias=False)
        self.out_conv5 = nn.Conv2d(c5, o5, kernel_size=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(o3)
        self.bn4 = nn.BatchNorm2d(o4)
        self.bn5 = nn.BatchNorm2d(o5)
        self.act = nn.SiLU(inplace=True)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple:
        """
        Args:
            feats: Tuple of (P3, P4, P5) feature maps from backbone
        
        Returns:
            Tuple of enhanced (P3, P4, P5) feature maps
        """
        p3, p4, p5 = feats
        
        # ===== Top-down path =====
        # P5 -> P4
        p5_gated = self.top_down_pcgu5(p5)
        p5_up = F.interpolate(self.reduce5(p5_gated), size=p4.shape[2:], mode='nearest')
        p4_td = p4 + p5_up
        p4_td = self.top_down_pcgu4(p4_td)
        
        # P4 -> P3
        p4_up = F.interpolate(self.reduce4(p4_td), size=p3.shape[2:], mode='nearest')
        p3_td = p3 + p4_up
        p3_out = self.top_down_pcgu3(p3_td)
        
        # ===== Bottom-up path =====
        # P3 -> P4
        p3_gated = self.bottom_up_pcgu3(p3_out)
        p3_down = self.enhance3(p3_gated)
        p4_bu = p4_td + p3_down
        p4_out = self.bottom_up_pcgu4(p4_bu)
        
        # P4 -> P5
        p4_down = self.enhance4(p4_out)
        p5_bu = p5_gated + p4_down
        p5_out = self.bottom_up_pcgu5(p5_bu)
        
        # Output projections
        out3 = self.act(self.bn3(self.out_conv3(p3_out)))
        out4 = self.act(self.bn4(self.out_conv4(p4_out)))
        out5 = self.act(self.bn5(self.out_conv5(p5_out)))
        
        return (out3, out4, out5)


class MorphologyAwareGeometricLoss(nn.Module):
    """
    Morphology-Aware Geometric Loss (L_GPL).
    
    An integrated bounding box regression objective that incorporates
    adaptive hard sample reweighting alongside an aspect ratio
    regularization term to mitigate localization drift in overlapping clusters.
    
    Loss = 1 - IoU_f + D_shape + 0.5 * Omega_shape
    
    where:
        IoU_f: Difficulty-aware reweighted IoU (emphasizes borderline samples)
        D_shape: Normalized center distance loss
        Omega_shape: Shape regularizer toward quasi-spherical geometry
    """
    
    def __init__(
        self,
        iou_mode: str = 'ciou',
        d: float = 0.00,
        u: float = 0.95,
        shape_weight: float = 0.5,
        use_aspect_ratio_reg: bool = True,
        loss_weight: float = 2.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.iou_mode = iou_mode
        self.d = d
        self.u = u
        self.shape_weight = shape_weight
        self.use_aspect_ratio_reg = use_aspect_ratio_reg
        self.loss_weight = loss_weight
        self.eps = eps
    
    def _iou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute CIoU loss between predicted and target boxes (xywh format)."""
        # Convert to xyxy
        pred_x1 = pred[..., 0] - pred[..., 2] / 2
        pred_y1 = pred[..., 1] - pred[..., 3] / 2
        pred_x2 = pred[..., 0] + pred[..., 2] / 2
        pred_y2 = pred[..., 1] + pred[..., 3] / 2
        
        target_x1 = target[..., 0] - target[..., 2] / 2
        target_y1 = target[..., 1] - target[..., 3] / 2
        target_x2 = target[..., 0] + target[..., 2] / 2
        target_y2 = target[..., 1] + target[..., 3] / 2
        
        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter = inter_w * inter_h
        
        # Union
        pred_area = pred[..., 2] * pred[..., 3]
        target_area = target[..., 2] * target[..., 3]
        union = pred_area + target_area - inter + self.eps
        iou = inter / union
        
        # Diagonal of smallest enclosing box
        cw = torch.max(pred_x2, target_x2) - torch.min(pred_x1, target_x1)
        ch = torch.max(pred_y2, target_y2) - torch.min(pred_y1, target_y1)
        c2 = cw ** 2 + ch ** 2 + self.eps
        
        # Center distance
        rho2 = ((pred[..., 0] - target[..., 0]) ** 2 +
                (pred[..., 1] - target[..., 1]) ** 2)
        
        # Aspect ratio penalty
        v = (4 / (3.14159 ** 2)) * torch.pow(
            torch.atan(target[..., 2] / (target[..., 3] + self.eps)) -
            torch.atan(pred[..., 2] / (pred[..., 3] + self.eps)), 2
        )
        alpha = v / (1 - iou + v + self.eps)
        
        ciou = iou - rho2 / c2 - alpha * v
        return 1 - ciou, iou
    
    def _shape_regularizer(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute D_shape and Omega_shape terms.
        
        D_shape: Normalized center distance (scale-dependent)
        Omega_shape: Aspect ratio regularizer toward quasi-spherical geometry
        """
        # Center distance normalized by diagonal
        c = torch.sqrt(pred[..., 2] ** 2 + pred[..., 3] ** 2) + self.eps
        dx = (pred[..., 0] - target[..., 0]) ** 2
        dy = (pred[..., 1] - target[..., 1]) ** 2
        d_shape = (pred[..., 3:4] * dx + pred[..., 2:3] * dy) / (c ** 2)
        d_shape = d_shape.mean(dim=-1) if d_shape.dim() > 1 else d_shape
        
        # Aspect ratio shape regularizer
        if self.use_aspect_ratio_reg:
            w_pred, h_pred = pred[..., 2], pred[..., 3]
            w_gt, h_gt = target[..., 2], target[..., 3]
            
            # Scale-dependent coefficients
            w_hat = h_pred / (torch.max(w_pred, w_gt) + self.eps)
            h_hat = w_pred / (torch.max(h_pred, h_gt) + self.eps)
            
            omega_w = w_hat * torch.abs(w_pred - w_gt) / (torch.max(w_pred, w_gt) + self.eps)
            omega_h = h_hat * torch.abs(h_pred - h_gt) / (torch.max(h_pred, h_gt) + self.eps)
            
            omega_shape = (
                (1 - torch.exp(-omega_w)) ** 4 +
                (1 - torch.exp(-omega_h)) ** 4
            )
        else:
            omega_shape = torch.zeros_like(d_shape)
        
        return d_shape, omega_shape
    
    def _reweight_iou(self, iou: torch.Tensor) -> torch.Tensor:
        """
        Difficulty-aware IoU reweighting.
        
        Allocates larger gradient magnitudes to borderline samples
        within the range [d, u].
        """
        iou_f = torch.clamp((iou - self.d) / (self.u - self.d + self.eps), 0, 1)
        return iou_f
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted boxes in xywh format, shape [..., 4]
            target: Target boxes in xywh format, shape [..., 4]
            weight: Optional per-sample weights
        
        Returns:
            Scalar loss value
        """
        # CIoU loss with reweighting
        iou_loss_raw, iou = self._iou_loss(pred, target)
        iou_f = self._reweight_iou(iou)
        iou_loss = 1.0 - iou_f
        
        # Shape terms
        d_shape, omega_shape = self._shape_regularizer(pred, target)
        
        # Combined loss
        loss = iou_loss + d_shape + self.shape_weight * omega_shape
        
        if weight is not None:
            loss = loss * weight
        
        return loss.mean() * self.loss_weight


class SSDDEIM(nn.Module):
    """
    SSD-DEIM: End-to-end detector for dense camouflaged small object detection.
    
    Integrates:
    - DFD (Duplex Feature Downsampling) in backbone stem
    - BFF (Gated Bidirectional Feature Fusion) in neck
    - L_GPL (Morphology-Aware Geometric Loss) in head
    
    Built on the DEIM (CVPR 2025) dense O2O matching baseline.
    """
    
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
    
    def forward(self, x, img_metas=None, return_loss=True, **kwargs):
        # Backbone feature extraction (with DFD modules)
        feats = self.backbone(x)
        
        # Bidirectional feature fusion (BFF)
        fused_feats = self.neck(feats)
        
        # Detection head with GPL loss
        if return_loss:
            losses = self.head.forward_train(fused_feats, img_metas=img_metas, **kwargs)
            return losses
        else:
            results = self.head.forward_test(fused_feats, img_metas=img_metas, **kwargs)
            return results


def build_ssd_deim(cfg):
    """Build SSD-DEIM model from configuration."""
    from .backbone import build_backbone
    from .neck import build_neck
    from .head import build_head
    
    backbone = build_backbone(cfg.model.backbone)
    neck = build_neck(cfg.model.neck)
    head = build_head(cfg.model.head)
    
    model = SSDDEIM(backbone, neck, head)
    return model
