"""Loss modules for SSD-DEIM."""

from .gpl_loss import (
    GPLLoss,
    Focaler,
    VarifocalLoss,
    SetCriterion,
    shape_iou_loss,
    build_gpl_loss,
    build_criterion,
)

__all__ = [
    'GPLLoss',
    'Focaler',
    'VarifocalLoss',
    'SetCriterion',
    'shape_iou_loss',
    'build_gpl_loss',
    'build_criterion',
]
