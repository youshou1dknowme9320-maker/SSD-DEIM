"""
GPL Loss: Focaler-ShapeIoU Loss for SSD-DEIM.

Implements the Geometry-aware Perceptual Loss (GPL) with Focaler hard sample
focusing and ShapeIoU shape-aware bounding box regression.

Loss Formula:
    L_GPL = L_focaler_shapeiou
           = focaler_weight * (1 - ShapeIoU)

    ShapeIoU = IoU - D_shape - Omega_shape_cost

    where:
        - IoU: Intersection over Union with scale-dependent weighting
        - D_shape: Shape-aware normalized center distance
        - Omega_shape_cost: Shape cost term (aspect ratio regularization)
        - focaler_weight: Difficulty-aware adaptive weight focusing on hard samples

Reference:
    SSD-DEIM: Preserving Multi-Scale Fidelity and Geometric Priors
    for Dense Camouflaged Small Object Detection (ICASSP 2027)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
#  ShapeIoU Computation
# ---------------------------------------------------------------------------

def shape_iou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    scale: float = 0.0,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ShapeIoU loss between aligned box pairs.

    ShapeIoU incorporates scale-aware shape distance and shape cost to
    better handle objects of varying aspect ratios, especially small
    and elongated objects.

    Args:
        pred_boxes: Predicted boxes [N, 4] in xyxy format.
        target_boxes: Target boxes [N, 4] in xyxy format.
        scale: Scale parameter for shape-aware weighting (0 = standard SIoU).
        eps: Small constant for numerical stability.

    Returns:
        loss: ShapeIoU loss = 1 - ShapeIoU, shape [N].
        iou: Standard IoU values, shape [N].
    """
    # Unpack coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target_boxes.unbind(-1)

    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)

    # Intersection and IoU
    inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0)
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Enclosing box
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps

    # ---------------------------------------------------------------
    #  Shape-aware center distance (D_shape)
    # ---------------------------------------------------------------
    # Scale-dependent weighting: ww and hh balance x/y distance
    # based on the target box's aspect ratio
    if scale != 0.0:
        w2_s = torch.pow(w2, scale)
        h2_s = torch.pow(h2, scale)
        denom_s = w2_s + h2_s + eps
        ww = 2.0 * w2_s / denom_s
        hh = 2.0 * h2_s / denom_s
    else:
        ww = hh = torch.ones_like(w2)

    # Center distance
    center_dx = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4.0
    center_dy = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4.0
    center_distance = hh * center_dx + ww * center_dy
    d_shape = center_distance / c2

    # ---------------------------------------------------------------
    #  Shape cost term (Omega_shape)
    # ---------------------------------------------------------------
    omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2).clamp(min=eps)
    omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2).clamp(min=eps)
    shape_cost = (
        torch.pow(1 - torch.exp(-omiga_w), 4) +
        torch.pow(1 - torch.exp(-omiga_h), 4)
    )

    # ---------------------------------------------------------------
    #  ShapeIoU = IoU - D_shape - 0.5 * shape_cost
    # ---------------------------------------------------------------
    shape_iou = iou - d_shape - 0.5 * shape_cost

    # Loss = 1 - ShapeIoU
    loss = 1.0 - shape_iou

    return loss, iou


# ---------------------------------------------------------------------------
#  Focaler: Hard Sample Focusing Mechanism
# ---------------------------------------------------------------------------

class Focaler(nn.Module):
    """Focaler: Difficulty-aware hard sample focusing mechanism.

    An enhanced version of focal loss for box regression that adaptively
    reweights samples based on their IoU difficulty. Samples with IoU
    within a critical range [d, u] (borderline/hard samples) receive
    higher weight, while very easy (high IoU) and very hard (low IoU)
    samples receive lower weight.

    This focuses training attention on the most informative borderline
    samples, which is particularly important for dense small object
    detection where many predictions cluster near decision boundaries.

    Args:
        d: Lower bound of the critical IoU range.
        u: Upper bound of the critical IoU range.
        gamma: Focusing strength (higher = more aggressive focusing).
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        d: float = 0.0,
        u: float = 0.95,
        gamma: float = 2.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.d = d
        self.u = u
        self.gamma = gamma
        self.eps = eps

    def forward(self, iou: torch.Tensor) -> torch.Tensor:
        """Compute focaler weights from IoU values.

        Weight function:
            - For IoU <= d: weight = 0 (samples too poor to learn from)
            - For d < IoU < u: weight peaks at the midpoint, creating
              a bell-shaped focusing curve that emphasizes borderline samples
            - For IoU >= u: weight decays (samples are already good)

        Args:
            iou: IoU values [N].

        Returns:
            Focaler weights [N].
        """
        # Normalize IoU to [0, 1] within the critical range [d, u]
        iou_norm = torch.clamp(
            (iou - self.d) / (self.u - self.d + self.eps), 0.0, 1.0
        )

        # Focaler weighting: bell-shaped curve centered at midpoint
        # Weight = (4 * iou_norm * (1 - iou_norm))^gamma
        # This peaks at 1.0 when iou_norm = 0.5, and is 0 at the boundaries
        focal_term = 4.0 * iou_norm * (1.0 - iou_norm)
        weight = torch.pow(focal_term.clamp(min=0.0), self.gamma)

        # Add a small baseline weight to avoid zero gradients
        weight = weight + 1e-4

        return weight


# ---------------------------------------------------------------------------
#  Classification Losses
# ---------------------------------------------------------------------------

class VarifocalLoss(nn.Module):
    """Varifocal Loss for dense detection.

    Asymmetric focal loss that trains with IoU-aware quality targets.
    Positive samples are weighted by their IoU (quality), while negative
    samples are down-weighted using a focal mechanism.

    Reference:
        VarifocalNet: An IoU-aware Dense Object Detector (CVPR 2021)
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        iou_weighted: bool = True,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        iou_target: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute varifocal loss.

        Args:
            pred: Predicted logits [N, num_classes].
            target: Target class labels [N] (long) or [N, num_classes] (float).
            iou_target: Optional IoU targets for quality weighting [N].
            weight: Optional per-sample weights [N].

        Returns:
            Scalar loss value.
        """
        pred_sigmoid = pred.sigmoid()

        # Prepare target
        if target.dtype == torch.long:
            # Class indices -> one-hot
            target_onehot = torch.zeros_like(pred_sigmoid)
            target_onehot.scatter_(1, target.unsqueeze(-1), 1.0)
        else:
            target_onehot = target

        # IoU-weighted targets
        if self.iou_weighted and iou_target is not None:
            # Scale positive targets by IoU (quality score)
            iou_target = iou_target.unsqueeze(-1).clamp(min=0.0, max=1.0)
            target_weighted = target_onehot * iou_target
        else:
            target_weighted = target_onehot

        # Focal weight for negative samples
        neg_weight = torch.pow(pred_sigmoid, self.gamma) * (1 - target_onehot)
        # Positive weight
        pos_weight = target_onehot

        # Combined weight
        focal_weight = (
            self.alpha * pos_weight + (1 - self.alpha) * neg_weight
        )

        # BCE loss with focal weighting
        loss = F.binary_cross_entropy_with_logits(
            pred, target_weighted, reduction='none'
        ) * focal_weight

        if weight is not None:
            loss = loss * weight.unsqueeze(-1)

        # Average by number of positive samples
        num_pos = target_onehot.sum().clamp(min=1.0)
        return loss.sum() / num_pos * self.loss_weight


# ---------------------------------------------------------------------------
#  GPL Loss (Complete)
# ---------------------------------------------------------------------------

class GPLLoss(nn.Module):
    """Geometry-aware Perceptual Loss (GPL) - Focaler-ShapeIoU.

    Complete bounding box regression loss that combines:
        1. ShapeIoU: Scale-aware shape distance + shape cost for better
           geometric alignment, especially for small/elongated objects.
        2. Focaler: Difficulty-aware hard sample focusing that emphasizes
           borderline samples for faster convergence and better precision.

    Loss = Focaler(IoU) * ShapeIoU_Loss * loss_weight

    Args:
        iou_mode: IoU type ('shapeiou', 'ciou', 'siou', 'diou', 'giou', 'iou').
        scale: ShapeIoU scale parameter (shape distance weighting).
        d: Focaler lower IoU bound.
        u: Focaler upper IoU bound.
        gamma: Focaler focusing strength.
        loss_weight: Overall loss weight.
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        iou_mode: str = 'shapeiou',
        scale: float = 0.0,
        d: float = 0.0,
        u: float = 0.95,
        gamma: float = 2.0,
        loss_weight: float = 2.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.iou_mode = iou_mode
        self.scale = scale
        self.loss_weight = loss_weight
        self.eps = eps

        # Focaler mechanism
        self.focaler = Focaler(d=d, u=u, gamma=gamma, eps=eps)

    def _compute_iou_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU loss and IoU values.

        Args:
            pred: Predicted boxes [N, 4] in xyxy format.
            target: Target boxes [N, 4] in xyxy format.

        Returns:
            loss: IoU-based loss [N].
            iou: Standard IoU [N].
        """
        mode = self.iou_mode.lower()

        if mode == 'shapeiou':
            return shape_iou_loss(pred, target, scale=self.scale, eps=self.eps)

        # Standard IoU variants
        b1_x1, b1_y1, b1_x2, b1_y2 = pred.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = target.unbind(-1)
        w1 = (b1_x2 - b1_x1).clamp(min=self.eps)
        h1 = (b1_y2 - b1_y1).clamp(min=self.eps)
        w2 = (b2_x2 - b2_x1).clamp(min=self.eps)
        h2 = (b2_y2 - b2_y1).clamp(min=self.eps)

        inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0)
        inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0)
        inter = inter_w * inter_h
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union

        if mode == 'iou':
            return 1.0 - iou, iou

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + self.eps
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4.0

        if mode == 'giou':
            c_area = cw * ch + self.eps
            giou = iou - (c_area - union) / c_area
            return 1.0 - giou, iou

        if mode == 'diou':
            diou = iou - rho2 / c2
            return 1.0 - diou, iou

        if mode == 'ciou':
            v = (4.0 / math.pi ** 2) * (
                torch.atan(w2 / h2) - torch.atan(w1 / h1)
            ).pow(2)
            with torch.no_grad():
                alpha = v / (v - iou + 1.0 + self.eps)
            ciou = iou - (rho2 / c2 + v * alpha)
            return 1.0 - ciou, iou

        if mode == 'siou':
            # SIoU with angle cost
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + self.eps
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + self.eps
            sigma = torch.sqrt(s_cw ** 2 + s_ch ** 2)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = math.sqrt(2) / 2.0
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2.0)

            rho_x = (s_cw / (cw + self.eps)) ** 2
            rho_y = (s_ch / (ch + self.eps)) ** 2
            gamma = angle_cost - 2.0
            distance_cost = 2.0 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2).clamp(min=self.eps)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2).clamp(min=self.eps)
            shape_cost = (
                torch.pow(1 - torch.exp(-omiga_w), 4) +
                torch.pow(1 - torch.exp(-omiga_h), 4)
            )
            siou = iou - 0.5 * (distance_cost + shape_cost)
            return 1.0 - siou, iou

        raise ValueError(f"Unknown iou_mode: {mode}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass of GPL loss.

        Args:
            pred: Predicted boxes [N, 4] in xyxy format.
            target: Target boxes [N, 4] in xyxy format.
            weight: Optional per-sample weights [N].
            avg_factor: Average factor for normalization. If None, uses N.

        Returns:
            loss: Scalar loss value.
            loss_dict: Dict with individual loss components for logging.
        """
        assert pred.shape == target.shape, (
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        )

        if pred.numel() == 0:
            zero_loss = pred.sum()
            return zero_loss, {
                'loss_bbox': zero_loss,
                'loss_iou_term': zero_loss,
                'loss_focaler': zero_loss,
                'avg_iou': zero_loss,
            }

        # Compute IoU loss and IoU values
        iou_loss, iou = self._compute_iou_loss(pred, target)

        # Focaler weighting
        focaler_weight = self.focaler(iou)

        # Apply focaler to the loss
        weighted_loss = iou_loss * focaler_weight

        if weight is not None:
            weighted_loss = weighted_loss * weight

        # Average
        if avg_factor is None:
            avg_factor = max(pred.shape[0], 1)
        loss = weighted_loss.sum() / avg_factor * self.loss_weight

        loss_dict = {
            'loss_bbox': loss,
            'loss_iou_term': (iou_loss.sum() / avg_factor * self.loss_weight).detach(),
            'loss_focaler': (focaler_weight.mean()).detach(),
            'avg_iou': iou.mean().detach(),
        }

        return loss, loss_dict


# ---------------------------------------------------------------------------
#  SetCriterion: Complete loss computation for SSD-DEIM
# ---------------------------------------------------------------------------

class SetCriterion(nn.Module):
    """Complete loss criterion for SSD-DEIM.

    Combines classification loss (Varifocal) and bbox regression loss
    (GPL / Focaler-ShapeIoU) with auxiliary supervision from all
    decoder layers.

    Args:
        num_classes: Number of object classes.
        matcher: Matching module (Hungarian/DenseOD matcher).
        loss_cls: Classification loss module.
        loss_bbox: Bounding box regression loss module (GPLLoss).
        loss_cls_weight: Classification loss weight.
        loss_bbox_weight: Bbox loss weight.
        num_decoder_layers: Number of decoder layers (for aux loss).
        aux_loss_weight: Auxiliary loss weight (decays per layer).
    """

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        loss_cls: nn.Module,
        loss_bbox: nn.Module,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 2.0,
        num_decoder_layers: int = 6,
        aux_loss_weight: float = 0.8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.num_decoder_layers = num_decoder_layers
        self.aux_loss_weight = aux_loss_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses.

        Args:
            outputs: Dict with predictions from the detection head.
            targets: List of target dicts, each with 'boxes' and 'labels'.
            epoch: Current training epoch.

        Returns:
            Dict of loss tensors.
        """
        losses = {}

        # Outputs from the last decoder layer
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != 'aux_outputs'
        }

        # Match predictions to targets
        match_result = self.matcher(outputs_without_aux, targets, epoch=epoch)
        indices = match_result['indices']

        # Compute losses for the last layer
        layer_losses = self._compute_losses(
            outputs_without_aux, targets, indices,
        )
        losses.update(layer_losses)

        # Auxiliary losses from intermediate decoder layers
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_match = self.matcher(aux_outputs, targets, epoch=epoch)
                aux_indices = aux_match['indices']

                aux_losses = self._compute_losses(
                    aux_outputs, targets, aux_indices,
                )

                # Apply decaying weight to aux losses
                weight = self.aux_loss_weight ** (
                    self.num_decoder_layers - 2 - i
                )
                for k, v in aux_losses.items():
                    losses[f'{k}_aux_{i}'] = v * weight

        return losses

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute classification and bbox losses for one decoder layer.

        Args:
            outputs: Dict with pred_logits, pred_boxes, pred_lqe.
            targets: List of target dicts.
            indices: List of (pred_idx, tgt_idx) tuples per image.

        Returns:
            Dict of loss values.
        """
        device = outputs['pred_logits'].device
        bs, num_queries, _ = outputs['pred_logits'].shape

        # Collect matched predictions and targets
        idx = self._get_src_permutation_idx(indices)

        # Classification targets
        target_classes_o = torch.cat(
            [t['labels'][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            (bs, num_queries), self.num_classes, dtype=torch.long, device=device
        )
        target_classes[idx] = target_classes_o

        # Bbox targets
        target_boxes = torch.cat(
            [t['boxes'][J] for t, (_, J) in zip(targets, indices)], dim=0
        )
        src_boxes = outputs['pred_boxes'][idx]

        # Number of positive samples
        num_pos = max(len(target_boxes), 1)

        # -----------------------------------------------------------
        #  Classification loss (Varifocal)
        # -----------------------------------------------------------
        pred_logits = outputs['pred_logits']
        target_onehot = torch.zeros_like(pred_logits)
        valid_mask = target_classes < self.num_classes
        target_onehot[valid_mask, target_classes[valid_mask]] = 1.0

        # Compute IoU for quality weighting
        if isinstance(self.loss_cls, VarifocalLoss) and self.loss_cls.iou_weighted:
            # Compute IoU between matched pairs
            from ..utils.box_ops import box_cxcywh_to_xyxy, bbox_iou_aligned
            src_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tgt_xyxy = box_cxcywh_to_xyxy(target_boxes)
            iou_matched = bbox_iou_aligned(src_xyxy, tgt_xyxy, iou_type='iou')

            # Prepare IoU targets
            iou_target = torch.zeros(bs * num_queries, device=device)
            flat_idx = idx[0] * num_queries + idx[1]
            iou_target[flat_idx] = iou_matched
            iou_target = iou_target.reshape(bs, num_queries)

            loss_cls = self.loss_cls(
                pred_logits.reshape(-1, self.num_classes),
                target_onehot.reshape(-1, self.num_classes),
                iou_target=iou_target.reshape(-1),
            )
        else:
            loss_cls = self.loss_cls(
                pred_logits.reshape(-1, self.num_classes),
                target_onehot.reshape(-1, self.num_classes),
            )

        # -----------------------------------------------------------
        #  Bbox regression loss (GPL / Focaler-ShapeIoU)
        # -----------------------------------------------------------
        from ..utils.box_ops import box_cxcywh_to_xyxy
        src_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(target_boxes)

        loss_bbox, bbox_loss_dict = self.loss_bbox(
            src_xyxy, tgt_xyxy, avg_factor=num_pos,
        )

        # LQE loss (quality estimation)
        loss_lqe = torch.tensor(0.0, device=device)
        if 'pred_lqe' in outputs:
            pred_lqe = outputs['pred_lqe'][idx].squeeze(-1)
            if len(pred_lqe) > 0:
                # LQE target = IoU between pred and gt
                from ..utils.box_ops import bbox_iou_aligned
                lqe_target = bbox_iou_aligned(src_xyxy, tgt_xyxy, iou_type='iou')
                loss_lqe = F.mse_loss(pred_lqe.sigmoid(), lqe_target)
                loss_lqe = loss_lqe * 0.1  # small weight for LQE

        total_loss = (
            self.loss_cls_weight * loss_cls
            + self.loss_bbox_weight * loss_bbox
            + loss_lqe
        )

        result = {
            'loss_total': total_loss,
            'loss_cls': loss_cls * self.loss_cls_weight,
            'loss_bbox': loss_bbox * self.loss_bbox_weight,
            'loss_lqe': loss_lqe,
            'avg_iou': bbox_loss_dict.get('avg_iou', torch.tensor(0.0)),
        }

        return result

    @staticmethod
    def _get_src_permutation_idx(
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get flat indices for matched predictions.

        Args:
            indices: List of (pred_idx, tgt_idx) per image.

        Returns:
            (batch_idx, src_idx) tensors for indexing flattened predictions.
        """
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


# ---------------------------------------------------------------------------
#  Factory function
# ---------------------------------------------------------------------------

def build_gpl_loss(cfg: dict) -> GPLLoss:
    """Build GPL loss from configuration.

    Args:
        cfg: Configuration dict.

    Returns:
        GPLLoss instance.
    """
    return GPLLoss(
        iou_mode=cfg.get('iou_mode', 'shapeiou'),
        scale=cfg.get('scale', 0.0),
        d=cfg.get('d', 0.0),
        u=cfg.get('u', 0.95),
        gamma=cfg.get('gamma', 2.0),
        loss_weight=cfg.get('loss_weight', 2.0),
        eps=cfg.get('eps', 1e-7),
    )


def build_criterion(cfg: dict, matcher: nn.Module) -> SetCriterion:
    """Build complete loss criterion.

    Args:
        cfg: Configuration dict.
        matcher: Matching module.

    Returns:
        SetCriterion instance.
    """
    # Classification loss
    loss_cls_cfg = cfg.get('loss_cls', {})
    if loss_cls_cfg.get('type', 'VarifocalLoss') == 'VarifocalLoss':
        loss_cls = VarifocalLoss(
            alpha=loss_cls_cfg.get('alpha', 0.75),
            gamma=loss_cls_cfg.get('gamma', 2.0),
            iou_weighted=loss_cls_cfg.get('iou_weighted', True),
            loss_weight=loss_cls_cfg.get('loss_weight', 1.0),
        )
    else:
        loss_cls = VarifocalLoss()

    # Bbox loss (GPL / Focaler-ShapeIoU)
    loss_bbox_cfg = cfg.get('loss_bbox', {})
    loss_bbox = build_gpl_loss(loss_bbox_cfg)

    num_classes = cfg.get('num_classes', 1)
    num_decoder_layers = cfg.get('num_decoder_layers', 6)

    return SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        loss_cls=loss_cls,
        loss_bbox=loss_bbox,
        loss_cls_weight=1.0,
        loss_bbox_weight=loss_bbox_cfg.get('loss_weight', 2.0),
        num_decoder_layers=num_decoder_layers,
        aux_loss_weight=cfg.get('aux_loss_weight', 0.8),
    )
