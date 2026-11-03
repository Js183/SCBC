"""SCBC on IEMOCAP-4: dialogue encoders + masked fusion head + debt book / corrector / objective."""
import torch
from torch import nn

from ..backbones import DialogueEncoder, ResidualFusionHead, branch_inputs
from ..backbones.dialogue import NUM_CLASSES
from ..constants import MODALITIES
from ..core import DebtBook, DebtCorrector, SCBCObjective

# SCBC settings for IEMOCAP-4 (lambda_d and the correction schedule are per-subset config keys).
LAMBDA_BASE = 0.6


class IemocapSCBC(nn.Module):
    """Per-modality dialogue encoders, one masked fusion head, and the debt book /
    corrector / objective chain built on top of them (each part is constructed from
    the previous one). The masked view uses the subset mask, the reference view runs
    the same encoders and fusion head with the full mask under stop-gradient, during
    training only."""

    def __init__(self, feature_shapes, student_modalities, class_weights, cfg):
        super().__init__()
        self.student_modalities = student_modalities
        hidden = cfg["encoder_hidden_dim"]
        self.encoders = nn.ModuleDict({
            m: DialogueEncoder(feature_shapes[m], hidden, cfg["speaker_dim"], cfg["encoder_layers"], cfg["encoder_dropout"])
            for m in MODALITIES
        })
        per_modality = 2 * NUM_CLASSES + 2 + hidden
        self.fusion_head = ResidualFusionHead(per_modality * len(MODALITIES), cfg["fusion_hidden_dim"],
                                              cfg["fusion_dropout"], len(MODALITIES))
        student_mask = [1.0 if m in student_modalities else 0.0 for m in MODALITIES]
        self.register_buffer("student_mask", torch.tensor(student_mask), persistent=False)
        self.register_buffer("full_mask", torch.ones(len(MODALITIES)), persistent=False)
        self.debt_book = DebtBook(student_mask, self.fusion_head.mlp[-1].out_features)
        self.corrector = DebtCorrector(self.debt_book, self.fusion_head.input_norm.normalized_shape[0],
                                       cfg["correction_scale"], cfg["correction_warmup_epochs"],
                                       cfg["correction_ramp_epochs"])
        self.objective = SCBCObjective(self.corrector, LAMBDA_BASE, class_weights)

    def forward(self, dialogue):
        return self._branches(dialogue, reference=False)[1]

    def loss(self, dialogue):
        """SCBC objective for one dialogue; the dialogue is also written into the debt book."""
        base_logits, final_logits, reference_logits, debts = self._branches(dialogue)
        return self.objective(base_logits, final_logits, debts, reference_logits, dialogue["labels"])

    def _branches(self, dialogue, reference=True):
        outputs = {m: enc(dialogue["features"][m], dialogue["speakers"]) for m, enc in self.encoders.items()}
        student_features, student_stack = branch_inputs(outputs, self.student_mask)
        base_logits = self.fusion_head(student_features, student_stack, self.student_mask)
        reference_logits = None
        if reference:
            with torch.no_grad():
                reference_features, reference_stack = branch_inputs(outputs, self.full_mask)
                reference_logits = self.fusion_head(reference_features, reference_stack, self.full_mask)
        final_logits, debts = self.corrector(student_features, base_logits)
        return base_logits, final_logits, reference_logits, debts
