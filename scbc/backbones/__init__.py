"""Benchmark-specific backbones: token-level subset fusion (CMU-MOSEI) and
speaker-aware dialogue encoders with a masked fusion head (IEMOCAP-4)."""
from .dialogue import DialogueEncoder, ResidualFusionHead, branch_inputs
from .subset_fusion import MaskedEncoderLayer, ModalityTokenizer, SubsetFusion

__all__ = ["ModalityTokenizer", "MaskedEncoderLayer", "SubsetFusion",
           "DialogueEncoder", "ResidualFusionHead", "branch_inputs"]
