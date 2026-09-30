"""Detection head modules for SSD-DEIM."""

from .dfine_head import (
    DFINEHead,
    MSDeformableAttention,
    TransformerDecoderLayer,
    Integral,
    Gate,
    LQE,
    build_dfine_head,
)

__all__ = [
    'DFINEHead',
    'MSDeformableAttention',
    'TransformerDecoderLayer',
    'Integral',
    'Gate',
    'LQE',
    'build_dfine_head',
]
