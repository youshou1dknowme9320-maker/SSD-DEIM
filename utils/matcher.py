"""
Dense One-to-One Matcher for SSD-DEIM.

Implements the Hungarian matching algorithm for dense one-to-one (O2O)
assignment between predicted queries and ground truth boxes.

Reference:
- DEIM: DETR with Improved Matching for Fast Convergence (CVPR 2025)
- https://github.com/facebookresearch/detr/blob/main/models/matcher.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional

from .box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
    box_iou,
    batch_sigmoid_ce_loss,
)


class HungarianMatcher(nn.Module):
    """Hungarian Matcher for dense one-to-one assignment.

    Computes an optimal assignment between predictions and targets using
    the Hungarian algorithm. The cost matrix combines classification cost,
    L1 box distance, and GIoU loss.

    Attributes:
        cost_class: Weight of classification cost.
        cost_bbox: Weight of L1 box distance cost.
        cost_giou: Weight of GIoU cost.
        use_focal_loss: Whether to use focal loss style class cost.
        alpha: Focal loss alpha parameter.
        gamma: Focal loss gamma parameter.
        use_iou_order: Whether to use IoU-ordered matching (DEIM-style).
        iou_order_alpha: Alpha parameter for IoU-ordered matching.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        use_focal_loss: bool = True,
        alpha: float = 0.25,
        gamma: float = 2.0,
        use_iou_order: bool = False,
        iou_order_alpha: float = 1.0,
    ):
        """Initialize the Hungarian matcher.

        Args:
            cost_class: Classification cost weight.
            cost_bbox: L1 box distance cost weight.
            cost_giou: GIoU cost weight.
            use_focal_loss: Use focal loss for class cost computation.
            alpha: Focal loss alpha (positive class weight).
            gamma: Focal loss gamma (focusing parameter).
            use_iou_order: Use DEIM-style IoU-ordered matching.
            iou_order_alpha: Power factor for IoU in ordered matching.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.use_iou_order = use_iou_order
        self.iou_order_alpha = iou_order_alpha

        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "All costs cannot be zero"

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        return_topk: int = 0,
        epoch: int = 0,
    ) -> Dict[str, List[tuple[torch.Tensor, torch.Tensor]]]:
        """Perform the matching.

        Args:
            outputs: Dict with at least:
                - "pred_logits": [batch_size, num_queries, num_classes]
                - "pred_boxes": [batch_size, num_queries, 4] in cxcywh format
            targets: List of target dicts, each with:
                - "labels": [num_target_boxes] class labels
                - "boxes": [num_target_boxes, 4] in cxcywh format
            return_topk: If > 0, also return top-k matches for O2M supervision.
            epoch: Current training epoch (for adaptive matching).

        Returns:
            Dict containing:
                - 'indices': List of (pred_idx, tgt_idx) tuples per image.
                - 'indices_o2m' (optional): Top-k matches if return_topk > 0.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten for batch computation
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [bs*num_queries, 4]

        # Concatenate targets
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.use_iou_order and epoch >= 0:
            # DEIM-style: class_score * iou^alpha
            class_score = out_prob[:, tgt_ids]
            bbox_iou, _ = box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox),
            )
            C = -1.0 * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))
        else:
            # Classification cost
            if self.use_focal_loss:
                out_prob_sel = out_prob[:, tgt_ids]
                neg_cost_class = (
                    (1 - self.alpha)
                    * (out_prob_sel ** self.gamma)
                    * (-(1 - out_prob_sel + 1e-8).log())
                )
                pos_cost_class = (
                    self.alpha
                    * ((1 - out_prob_sel) ** self.gamma)
                    * (-(out_prob_sel + 1e-8).log())
                )
                cost_class = pos_cost_class - neg_cost_class
            else:
                out_prob_sel = out_prob[:, tgt_ids]
                neg_cost_class = out_prob_sel * (-(1 - out_prob_sel + 1e-8).log())
                pos_cost_class = (1 - out_prob_sel) * (-(out_prob_sel + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class

            # L1 box cost
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # GIoU cost
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox),
            )

            # Combined cost matrix
            C = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )

        # Reshape to [bs, num_queries, total_targets] and move to CPU
        C = C.view(bs, num_queries, -1).cpu()
        C = torch.nan_to_num(C, nan=1.0)

        # Split by image and solve assignment
        sizes = [len(v["boxes"]) for v in targets]
        indices_pre = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        indices = [
            (torch.as_tensor(i, dtype=torch.int64),
             torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]

        result = {"indices": indices}

        # Top-k matches for O2M supervision
        if return_topk > 0:
            result["indices_o2m"] = self._get_top_k_matches(
                C, sizes=sizes, k=return_topk, initial_indices=indices_pre
            )

        return result

    @staticmethod
    def _get_top_k_matches(
        C: torch.Tensor,
        sizes: List[int],
        k: int = 1,
        initial_indices: Optional[list] = None,
    ) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """Get top-k matches by iteratively removing matched pairs.

        Args:
            C: Cost matrix of shape [bs, num_queries, total_targets].
            sizes: List of target counts per image.
            k: Number of top matches.
            initial_indices: Pre-computed first match indices.

        Returns:
            List of (pred_idx, tgt_idx) tuples with top-k matches merged.
        """
        import numpy as np

        indices_list = []
        for i in range(k):
            if i > 0 or initial_indices is None:
                indices_k = [
                    linear_sum_assignment(c[i])
                    for i, c in enumerate(C.split(sizes, -1))
                ]
            else:
                indices_k = initial_indices

            indices_list.append([
                (torch.as_tensor(idx_i, dtype=torch.int64),
                 torch.as_tensor(idx_j, dtype=torch.int64))
                for idx_i, idx_j in indices_k
            ])

            # Mask out matched predictions for next round
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k_arr = np.stack(idx_k)
                c[:, idx_k_arr[0]] = 1e6

        # Concatenate top-k matches
        num_imgs = len(sizes)
        merged = []
        for j in range(num_imgs):
            all_pred = torch.cat([indices_list[i][j][0] for i in range(k)], dim=0)
            all_tgt = torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)
            merged.append((all_pred, all_tgt))

        return merged


class DenseODMatcher(nn.Module):
    """Dense One-to-One (O2O) Matcher for DFINE-style dense detectors.

    This matcher operates on dense feature grid points rather than learned
    queries. It performs Hungarian matching on the dense set of anchor points.

    Reference:
        D-FINE: Redefining Regression Targets in Object Detection (CVPR 2025)
    """

    def __init__(
        self,
        topk: int = 13,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        use_focal_loss: bool = True,
        alpha: float = 0.25,
        gamma: float = 2.0,
        use_iou_order: bool = True,
        iou_order_alpha: float = 1.0,
    ):
        """Initialize the dense O2O matcher.

        Args:
            topk: Number of top-k candidate queries per GT for matching.
            cost_class: Classification cost weight.
            cost_bbox: L1 box distance cost weight.
            cost_giou: GIoU cost weight.
            use_focal_loss: Use focal loss for class cost.
            alpha: Focal loss alpha.
            gamma: Focal loss gamma.
            use_iou_order: Use IoU-ordered matching cost.
            iou_order_alpha: Power factor for IoU in ordered matching.
        """
        super().__init__()
        self.topk = topk
        self.hungarian_matcher = HungarianMatcher(
            cost_class=cost_class,
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
            use_focal_loss=use_focal_loss,
            alpha=alpha,
            gamma=gamma,
            use_iou_order=use_iou_order,
            iou_order_alpha=iou_order_alpha,
        )

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        num_queries_per_level: Optional[List[int]] = None,
        epoch: int = 0,
    ) -> Dict[str, List[tuple[torch.Tensor, torch.Tensor]]]:
        """Perform dense one-to-one matching.

        Args:
            outputs: Dict with pred_logits, pred_boxes, and optionally
                reference_points.
            targets: List of target dicts with labels and boxes.
            num_queries_per_level: Number of queries per feature level.
            epoch: Current training epoch.

        Returns:
            Dict with 'indices' list of (pred_idx, tgt_idx) tuples.
        """
        # Directly use Hungarian matcher on all queries
        # For efficiency, one could first filter to top-k candidates per GT
        result = self.hungarian_matcher(
            outputs, targets, return_topk=0, epoch=epoch
        )
        return result


def build_matcher(cfg: dict) -> nn.Module:
    """Build a matcher from configuration.

    Args:
        cfg: Configuration dict with matcher parameters.

    Returns:
        Matcher module instance.
    """
    matcher_type = cfg.get('type', 'HungarianMatcher')

    if matcher_type == 'DenseODMatcher':
        return DenseODMatcher(
            topk=cfg.get('topk', 13),
            cost_class=cfg.get('cost_class', 2.0),
            cost_bbox=cfg.get('cost_bbox', 5.0),
            cost_giou=cfg.get('cost_giou', 2.0),
            use_focal_loss=cfg.get('use_focal_loss', True),
            alpha=cfg.get('alpha', 0.25),
            gamma=cfg.get('gamma', 2.0),
            use_iou_order=cfg.get('use_iou_order', True),
            iou_order_alpha=cfg.get('iou_order_alpha', 1.0),
        )
    elif matcher_type == 'HungarianMatcher':
        return HungarianMatcher(
            cost_class=cfg.get('cost_class', 2.0),
            cost_bbox=cfg.get('cost_bbox', 5.0),
            cost_giou=cfg.get('cost_giou', 2.0),
            use_focal_loss=cfg.get('use_focal_loss', True),
            alpha=cfg.get('alpha', 0.25),
            gamma=cfg.get('gamma', 2.0),
        )
    else:
        raise ValueError(f"Unknown matcher type: {matcher_type}")
