"""
SSD-DEIM: End-to-End Detector for Dense Camouflaged Small Object Detection.

Integrates the complete SSD-DEIM architecture:
    - Backbone: HGNetV2 + DFD (Duplex Feature Downsampling)
    - Neck: BFF (Bidirectional Feature Fusion) with PCGU and BFE
    - Head: DFINE-style detector with MSDeformAttn + iterative refinement
    - Loss: GPL (Focaler-ShapeIoU) + Varifocal classification loss

Paper:
    SSD-DEIM: Preserving Multi-Scale Fidelity and Geometric Priors
    for Dense Camouflaged Small Object Detection (ICASSP 2027)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

from .backbone.hgnetv2_dfd import build_hgnetv2_dfd
from .neck.bff_neck import build_bff_neck
from .head.dfine_head import build_dfine_head
from .losses.gpl_loss import build_criterion


class SSDDEIM(nn.Module):
    """SSD-DEIM: Complete detection model.

    End-to-end detector for dense camouflaged small object detection that
    preserves multi-scale feature fidelity and incorporates geometric
    priors through shape-aware loss.

    Architecture flow:
        Image -> Backbone(HGNetV2+DFD) -> Neck(BFF) -> Head(DFINE) -> Detections

    Args:
        backbone_cfg: Backbone configuration dict.
        neck_cfg: Neck configuration dict.
        head_cfg: Head configuration dict.
        loss_cfg: Loss configuration dict.
        matcher_cfg: Matcher configuration dict.
        num_classes: Number of object classes.
    """

    def __init__(
        self,
        backbone_cfg: dict,
        neck_cfg: dict,
        head_cfg: dict,
        loss_cfg: dict,
        matcher_cfg: dict,
        num_classes: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Build backbone
        self.backbone = build_hgnetv2_dfd(backbone_cfg)
        backbone_out_channels = self.backbone.out_channels

        # Update neck in_channels from backbone
        neck_cfg = dict(neck_cfg)
        neck_cfg['in_channels'] = backbone_out_channels

        # Build neck
        self.neck = build_bff_neck(neck_cfg)
        neck_out_channels = neck_cfg['out_channels']

        # Update head in_channels from neck
        head_cfg = dict(head_cfg)
        head_cfg['in_channels'] = neck_out_channels
        head_cfg['num_classes'] = num_classes

        # Build head
        self.head = build_dfine_head(head_cfg)

        # Build matcher
        from ..utils.matcher import build_matcher
        self.matcher = build_matcher(matcher_cfg)

        # Build loss criterion
        loss_cfg = dict(loss_cfg)
        loss_cfg['num_classes'] = num_classes
        loss_cfg['num_decoder_layers'] = head_cfg.get('num_decoder_layers', 6)
        self.criterion = build_criterion(loss_cfg, self.matcher)

        # Test configuration
        self.test_cfg = {
            'max_per_img': 300,
            'score_thr': 0.01,
            'nms_thr': 0.5,
        }

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        return_loss: bool = True,
        epoch: int = 0,
    ):
        """Forward pass.

        Args:
            x: Input images [B, 3, H, W].
            targets: List of target dicts for training.
            return_loss: Whether to compute losses (training mode).
            epoch: Current training epoch.

        Returns:
            If return_loss: Dict of loss tensors.
            Else: List of detection results per image.
        """
        # Backbone feature extraction (with DFD modules)
        backbone_feats = self.backbone(x)

        # Bidirectional feature fusion (BFF Neck)
        neck_feats = self.neck(tuple(backbone_feats))

        # Detection head
        head_outputs = self.head(neck_feats)

        if return_loss and targets is not None:
            # Training: compute losses
            losses = self.criterion(head_outputs, targets, epoch=epoch)
            return losses
        else:
            # Inference: decode detections
            results = self._postprocess(head_outputs, x.shape[-2:])
            return results

    def _postprocess(
        self,
        outputs: Dict[str, torch.Tensor],
        img_shape: Tuple[int, int],
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process head outputs to final detections.

        Args:
            outputs: Head output dict with pred_logits, pred_boxes, pred_lqe.
            img_shape: (H, W) of input images.

        Returns:
            List of dicts per image with 'boxes', 'scores', 'labels'.
        """
        from ..utils.box_ops import box_cxcywh_to_xyxy, multiclass_nms

        pred_logits = outputs['pred_logits']   # [bs, num_queries, num_classes]
        pred_boxes = outputs['pred_boxes']     # [bs, num_queries, 4] (cxcywh normalized)
        pred_lqe = outputs.get('pred_lqe')     # [bs, num_queries, 1]

        bs = pred_logits.shape[0]
        device = pred_logits.device
        h, w = img_shape

        results = []
        for i in range(bs):
            # Sigmoid scores
            scores = pred_logits[i].sigmoid()  # [num_queries, num_classes]

            # Quality score: cls_score * IoU_estimate (from LQE)
            if pred_lqe is not None:
                quality = pred_lqe[i].sigmoid()  # [num_queries, 1]
                scores = scores * quality

            # Convert boxes from normalized cxcywh to pixel xyxy
            boxes_cxcywh = pred_boxes[i].clone()  # [num_queries, 4]
            boxes_cxcywh[..., 0::2] *= w
            boxes_cxcywh[..., 1::2] *= h
            boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)

            # Multi-class NMS
            det_boxes, det_scores, det_labels = multiclass_nms(
                boxes_xyxy,
                scores,
                score_thr=self.test_cfg['score_thr'],
                nms_thr=self.test_cfg['nms_thr'],
                max_per_img=self.test_cfg['max_per_img'],
            )

            results.append({
                'boxes': det_boxes,
                'scores': det_scores,
                'labels': det_labels,
            })

        return results


# ---------------------------------------------------------------------------
#  Factory function
# ---------------------------------------------------------------------------

def build_ssd_deim(cfg) -> SSDDEIM:
    """Build SSD-DEIM model from configuration.

    Args:
        cfg: Configuration object (module or dict) with model settings.

    Returns:
        SSDDEIM model instance.
    """
    # Extract model config
    if hasattr(cfg, 'model'):
        model_cfg = cfg.model
    else:
        model_cfg = cfg['model']

    backbone_cfg = model_cfg['backbone']
    neck_cfg = model_cfg['neck']
    head_cfg = model_cfg['head']['head_module']
    loss_cfg = {
        'loss_cls': model_cfg['head']['loss_cls'],
        'loss_bbox': model_cfg['head']['loss_bbox'],
        'aux_loss_weight': 0.8,
    }
    matcher_cfg = model_cfg['head']['train_cfg']['assigner']

    num_classes = head_cfg.get('num_classes', 1)

    model = SSDDEIM(
        backbone_cfg=backbone_cfg,
        neck_cfg=neck_cfg,
        head_cfg=head_cfg,
        loss_cfg=loss_cfg,
        matcher_cfg=matcher_cfg,
        num_classes=num_classes,
    )

    return model
