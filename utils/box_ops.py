"""
Bounding box operations for SSD-DEIM.

Includes coordinate format conversion, IoU computation, NMS, and
various box utilities used throughout the detection pipeline.

Reference:
- https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
- DEIM (CVPR 2025) engine/deim/box_ops.py
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import nms as tv_nms


# ---------------------------------------------------------------------------
#  Coordinate format conversions
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    """Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.

    Args:
        x: Tensor of shape (..., 4) in cxcywh format.

    Returns:
        Tensor of same shape in xyxy format.
    """
    x_c, y_c, w, h = x.unbind(-1)
    w = w.clamp(min=0.0)
    h = h.clamp(min=0.0)
    b = [x_c - 0.5 * w, y_c - 0.5 * h,
         x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    """Convert bounding boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format.

    Args:
        x: Tensor of shape (..., 4) in xyxy format.

    Returns:
        Tensor of same shape in cxcywh format.
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2.0, (y0 + y1) / 2.0,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x: Tensor) -> Tensor:
    """Convert xyxy to xywh (top-left corner + width/height)."""
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([x0, y0, x1 - x0, y1 - y0], dim=-1)


def box_xywh_to_xyxy(x: Tensor) -> Tensor:
    """Convert xywh to xyxy."""
    x, y, w, h = x.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


# ---------------------------------------------------------------------------
#  Pairwise IoU (matrix form)
# ---------------------------------------------------------------------------

def box_area(boxes: Tensor) -> Tensor:
    """Compute area of boxes in xyxy format.

    Args:
        boxes: Tensor of shape (N, 4) in xyxy format.

    Returns:
        Tensor of shape (N,) with box areas.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """Compute pairwise IoU between two sets of boxes (xyxy format).

    Args:
        boxes1: Tensor of shape (N, 4).
        boxes2: Tensor of shape (M, 4).

    Returns:
        iou: Tensor of shape (N, M) with IoU values.
        union: Tensor of shape (N, M) with union areas.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # [N, M, 2]

    wh = (rb - lt).clamp(min=0)                          # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]                    # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-7)
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Generalized IoU (GIoU) between two sets of boxes.

    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format.
        boxes2: Tensor of shape (M, 4) in xyxy format.

    Returns:
        Tensor of shape (N, M) with GIoU values.
    """
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area.clamp(min=1e-7)


# ---------------------------------------------------------------------------
#  Aligned IoU variants (element-wise, for matched pairs)
# ---------------------------------------------------------------------------

IOU_TYPES = ('iou', 'giou', 'diou', 'ciou', 'eiou', 'siou', 'shapeiou')


def bbox_iou_aligned(
    boxes1: Tensor,
    boxes2: Tensor,
    iou_type: str = 'giou',
    eps: float = 1e-7,
    scale: float = 0.0,
) -> Tensor:
    """Compute IoU variants between aligned box pairs in xyxy format.

    ``boxes1`` and ``boxes2`` must have the same shape [N, 4].
    The returned tensor has shape [N].

    Args:
        boxes1: Predicted boxes [N, 4] in xyxy format.
        boxes2: Target boxes [N, 4] in xyxy format.
        iou_type: Type of IoU to compute. One of IOU_TYPES.
        eps: Small constant for numerical stability.
        scale: Scale parameter for ShapeIoU.

    Returns:
        Tensor of shape [N] with IoU values.
    """
    assert boxes1.shape == boxes2.shape and boxes1.shape[-1] == 4
    iou_type = iou_type.lower()
    assert iou_type in IOU_TYPES, f"Unsupported iou_type: {iou_type}"

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(-1)
    w1 = b1_x2 - b1_x1
    h1 = b1_y2 - b1_y1
    w2 = b2_x2 - b2_x1
    h2 = b2_y2 - b2_y1

    # Intersection
    inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0)
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if iou_type == 'iou':
        return iou

    # Enclosing box
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

    if iou_type == 'giou':
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area

    # Center distance squared
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4.0

    if iou_type == 'diou':
        return iou - rho2 / c2

    if iou_type == 'ciou':
        v = (4.0 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) -
                                     torch.atan(w1 / (h1 + eps))).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + 1.0 + eps)
        return iou - (rho2 / c2 + v * alpha)

    if iou_type == 'eiou':
        rho_w2 = (w2 - w1) ** 2
        rho_h2 = (h2 - h1) ** 2
        cw2 = cw ** 2 + eps
        ch2 = ch ** 2 + eps
        return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)

    if iou_type == 'siou':
        # Angle cost
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
        sigma = torch.sqrt(s_cw ** 2 + s_ch ** 2)
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = math.sqrt(2) / 2.0
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2.0)

        # Distance cost
        rho_x = (s_cw / (cw + eps)) ** 2
        rho_y = (s_ch / (ch + eps)) ** 2
        gamma = angle_cost - 2.0
        distance_cost = 2.0 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

        # Shape cost
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2).clamp(min=eps)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2).clamp(min=eps)
        shape_cost = (torch.pow(1 - torch.exp(-omiga_w), 4) +
                      torch.pow(1 - torch.exp(-omiga_h), 4))

        return iou - 0.5 * (distance_cost + shape_cost) + eps

    if iou_type == 'shapeiou':
        # ShapeIoU: scale-aware shape distance + shape cost
        w2_s = torch.pow(w2.clamp(min=eps), scale)
        h2_s = torch.pow(h2.clamp(min=eps), scale)
        denom_s = w2_s + h2_s + eps
        ww = 2.0 * w2_s / denom_s
        hh = 2.0 * h2_s / denom_s

        c2 = cw ** 2 + ch ** 2 + eps
        center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4.0
        center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4.0
        center_distance = hh * center_distance_x + ww * center_distance_y
        distance = center_distance / c2

        omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2).clamp(min=eps)
        omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2).clamp(min=eps)
        shape_cost = (torch.pow(1 - torch.exp(-omiga_w), 4) +
                      torch.pow(1 - torch.exp(-omiga_h), 4))
        return iou - distance - 0.5 * shape_cost

    raise ValueError(f"Unknown iou_type: {iou_type}")


# ---------------------------------------------------------------------------
#  Non-Maximum Suppression
# ---------------------------------------------------------------------------

def multiclass_nms(
    boxes: Tensor,
    scores: Tensor,
    score_thr: float = 0.05,
    nms_thr: float = 0.5,
    max_per_img: int = 300,
) -> tuple[Tensor, Tensor, Tensor]:
    """Multi-class NMS.

    Args:
        boxes: Tensor of shape (N, 4) in xyxy format.
        scores: Tensor of shape (N, num_classes) with class scores.
        score_thr: Score threshold to filter low-confidence detections.
        nms_thr: IoU threshold for NMS.
        max_per_img: Maximum number of detections per image.

    Returns:
        det_boxes: (M, 4) filtered boxes.
        det_scores: (M,) filtered scores.
        det_labels: (M,) class labels.
    """
    num_classes = scores.shape[1]
    all_boxes = []
    all_scores = []
    all_labels = []

    for cls_id in range(num_classes):
        cls_scores = scores[:, cls_id]
        valid_mask = cls_scores > score_thr
        if not valid_mask.any():
            continue

        cls_boxes = boxes[valid_mask]
        cls_scores_valid = cls_scores[valid_mask]

        keep = tv_nms(cls_boxes, cls_scores_valid, nms_thr)
        all_boxes.append(cls_boxes[keep])
        all_scores.append(cls_scores_valid[keep])
        all_labels.append(torch.full_like(keep, cls_id, dtype=torch.long))

    if len(all_boxes) == 0:
        return (torch.zeros(0, 4, device=boxes.device),
                torch.zeros(0, device=boxes.device),
                torch.zeros(0, dtype=torch.long, device=boxes.device))

    det_boxes = torch.cat(all_boxes, dim=0)
    det_scores = torch.cat(all_scores, dim=0)
    det_labels = torch.cat(all_labels, dim=0)

    # Sort by score and keep top-k
    if det_scores.numel() > max_per_img:
        _, topk_idx = torch.topk(det_scores, max_per_img)
        det_boxes = det_boxes[topk_idx]
        det_scores = det_scores[topk_idx]
        det_labels = det_labels[topk_idx]

    return det_boxes, det_scores, det_labels


# ---------------------------------------------------------------------------
#  Distance-to-bbox conversion (for DFINE-style regression)
# ---------------------------------------------------------------------------

def distance2bbox(points: Tensor, distance: Tensor, reg_scale: float = 2.0) -> Tensor:
    """Decode edge-distances into bounding box coordinates (cxcywh format).

    Args:
        points: (B, N, 4) or (N, 4) in [cx, cy, w, h] format, where (cx, cy)
            is the anchor center and (w, h) are anchor width/height.
        distance: (B, N, 4) or (N, 4) with distances to left, top, right, bottom.
        reg_scale: Controls the curvature of the weighting function.

    Returns:
        Bounding boxes in (B, N, 4) or (N, 4) cxcywh format.
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)

    bboxes_xyxy = torch.stack([x1, y1, x2, y2], -1)
    return box_xyxy_to_cxcywh(bboxes_xyxy)


# ---------------------------------------------------------------------------
#  Masks to boxes
# ---------------------------------------------------------------------------

def masks_to_boxes(masks: Tensor) -> Tensor:
    """Compute bounding boxes around binary masks.

    Args:
        masks: Tensor of shape (N, H, W) with binary masks.

    Returns:
        Tensor of shape (N, 4) in xyxy format.
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x, indexing='ij')

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~masks.bool(), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~masks.bool(), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


# ---------------------------------------------------------------------------
#  Batch losses (pairwise, for matcher cost computation)
# ---------------------------------------------------------------------------

def batch_sigmoid_ce_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Pairwise BCE-with-logits cost between logits and binary targets.

    Args:
        inputs: (N, D) logits.
        targets: (M, D) binary targets.

    Returns:
        (N, M) pairwise BCE loss matrix.
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    num_points = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, 1 - targets
    )
    return loss / num_points


def batch_dice_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Pairwise Dice loss between mask logits and binary target masks.

    Args:
        inputs: (N, D) logits.
        targets: (M, D) binary targets.

    Returns:
        (N, M) pairwise Dice loss matrix.
    """
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


# ---------------------------------------------------------------------------
#  Scale / coordinate helpers
# ---------------------------------------------------------------------------

def scale_boxes(boxes: Tensor, scale_factor: tuple[float, float]) -> Tensor:
    """Scale boxes by (scale_x, scale_y).

    Args:
        boxes: Tensor of shape (..., 4) in xyxy format.
        scale_factor: (sx, sy) scaling factors.

    Returns:
        Scaled boxes of same shape.
    """
    sx, sy = scale_factor
    boxes = boxes.clone()
    boxes[..., 0::2] *= sx
    boxes[..., 1::2] *= sy
    return boxes


def clip_boxes(boxes: Tensor, img_shape: tuple[int, int]) -> Tensor:
    """Clip boxes to image boundaries.

    Args:
        boxes: Tensor of shape (..., 4) in xyxy format.
        img_shape: (H, W) image dimensions.

    Returns:
        Clipped boxes.
    """
    h, w = img_shape
    boxes = boxes.clone()
    boxes[..., 0::2] = boxes[..., 0::2].clamp(min=0, max=w)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(min=0, max=h)
    return boxes


def normalize_boxes(boxes: Tensor, img_shape: tuple[int, int]) -> Tensor:
    """Normalize boxes to [0, 1] range (cxcywh format).

    Args:
        boxes: Tensor of shape (..., 4) in xyxy pixel coordinates.
        img_shape: (H, W) image dimensions.

    Returns:
        Normalized boxes in cxcywh format.
    """
    h, w = img_shape
    boxes_cxcywh = box_xyxy_to_cxcywh(boxes)
    boxes_cxcywh[..., 0::2] /= w
    boxes_cxcywh[..., 1::2] /= h
    return boxes_cxcywh


def denormalize_boxes(boxes: Tensor, img_shape: tuple[int, int]) -> Tensor:
    """Denormalize boxes from [0, 1] to pixel coordinates (xyxy format).

    Args:
        boxes: Tensor of shape (..., 4) in cxcywh normalized coordinates.
        img_shape: (H, W) image dimensions.

    Returns:
        Pixel boxes in xyxy format.
    """
    h, w = img_shape
    boxes = boxes.clone()
    boxes[..., 0::2] *= w
    boxes[..., 1::2] *= h
    return box_cxcywh_to_xyxy(boxes)
