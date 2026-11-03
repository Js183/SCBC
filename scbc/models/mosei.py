"""SCBC on CMU-MOSEI: subset fusion backbone + debt book / corrector / objective."""
import torch
from torch import nn

from ..backbones import SubsetFusion
from ..constants import MODALITIES
from ..core import DebtBook, DebtCorrector, SCBCObjective

# SCBC settings for CMU-MOSEI.
DEBT_PROFILE_SCALE = 24.0
CORRECTION_WARMUP_EPOCHS = 1
CORRECTION_RAMP_EPOCHS = 5
LAMBDA_BASE = 0.5


class MoseiSCBC(nn.Module):
    """Subset-aware backbone and the debt book / corrector / objective chain built on
    top of it (each part is constructed from the previous one). The masked view runs
    the backbone with the subset mask, the reference view runs the same backbone with
    the full mask under stop-gradient."""

    def __init__(self, input_dims, num_classes, student_mask, cfg):
        super().__init__()
        self.register_buffer("student_mask", torch.tensor(student_mask, dtype=torch.float32), persistent=False)
        self.register_buffer("full_mask", torch.ones(len(MODALITIES)), persistent=False)
        self.student = SubsetFusion(input_dims, num_classes, cfg)
        self.debt_book = DebtBook(student_mask, self.student.classifier.out_features)
        self.corrector = DebtCorrector(self.debt_book, self.student.h_proj.out_features, DEBT_PROFILE_SCALE,
                                       CORRECTION_WARMUP_EPOCHS, CORRECTION_RAMP_EPOCHS)
        self.objective = SCBCObjective(self.corrector, LAMBDA_BASE)

    def _student(self, inputs):
        h, base_logits = self.student(inputs, self.student_mask)
        final_logits, debts = self.corrector(h, base_logits)
        return base_logits, final_logits, debts

    def forward(self, inputs):
        return self._student(inputs)[1]

    def loss(self, inputs, labels):
        """SCBC objective for one batch; the batch is also written into the debt book."""
        base_logits, final_logits, debts = self._student(inputs)
        with torch.no_grad():
            reference_logits = self.student(inputs, self.full_mask)[1]
        return self.objective(base_logits, final_logits, debts, reference_logits, labels)
