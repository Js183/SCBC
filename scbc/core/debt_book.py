"""Modality-factorized violation memory (Sec. 3.2)."""
import torch
import torch.nn.functional as F
from torch import nn

from ..constants import EPS, MU, MU_N, NU, TAU_M


class DebtBook(nn.Module):
    """Modality-factorized violation memory D_m (Sec. 3.2).

    The masked view sees a fixed modality subset S; the reference view is the same
    model on the full modality set. The excess violation of the masked view over the
    reference view is split evenly across the modalities missing from S.
    """

    def __init__(self, student_mask, num_classes):
        super().__init__()
        num_modalities = len(student_mask)
        self.num_classes = num_classes
        self.register_buffer("missing", 1.0 - torch.tensor(student_mask, dtype=torch.float32))
        self.register_buffer("offdiag", ~torch.eye(num_classes, dtype=torch.bool))
        self.register_buffer("V", torch.zeros(num_modalities, num_classes, num_classes))
        self.register_buffer("N", torch.zeros(num_modalities, num_classes))
        self.register_buffer("D", torch.zeros(num_modalities, num_classes, num_classes))

    def debts(self, batch_size):
        """D_S = sum_{m not in S} D_m / (|M \\ S| + eps)  (Eq. 6)."""
        weights = self.missing / (self.missing.sum() + EPS)
        return torch.einsum("m,mcd->cd", weights, self.D).unsqueeze(0).expand(batch_size, -1, -1)

    def _pair_signal(self, logits, labels):
        """phi_ab(z) (Eq. 7)."""
        true_mask = F.one_hot(labels, num_classes=self.num_classes).bool()
        competitor = torch.softmax(logits.masked_fill(true_mask, float("-inf")) / TAU_M, dim=-1)
        competitor = torch.nan_to_num(competitor, nan=0.0, posinf=0.0, neginf=0.0).masked_fill(true_mask, 0.0)
        margins = logits.gather(1, labels.unsqueeze(1)) - logits
        violations = torch.sigmoid((TAU_M - margins) / NU).masked_fill(true_mask, 0.0)
        return competitor * violations

    @torch.no_grad()
    def update(self, student_logits, reference_logits, labels):
        with torch.autocast(device_type=student_logits.device.type, enabled=False):
            self._update(student_logits.float(), reference_logits.float(), labels)

    def _update(self, student_logits, reference_logits, labels):
        missing = self.missing.view(1, -1).expand(labels.size(0), -1)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(dtype=student_logits.dtype)

        # Excess violation (Eq. 8), split evenly across the missing modalities.
        excess = torch.relu(self._pair_signal(student_logits, labels) - self._pair_signal(reference_logits, labels))
        write = missing / missing.sum(dim=-1, keepdim=True).clamp_min(1.0)
        group_count = torch.einsum("bm,bc->mc", missing, one_hot)
        groups = group_count > 0.0
        hat_e = torch.einsum("bm,bc,bk->mck", write, one_hot, excess) / group_count.clamp_min(1.0).unsqueeze(-1)
        hat_n = group_count / missing.sum(dim=0).unsqueeze(-1).clamp_min(1.0)

        # Write-after-read EMA with James-Stein-style shrinkage (Eq. 9).
        if groups.any():
            self.V[groups] = MU * self.V[groups] + (1.0 - MU) * hat_e[groups]
            self.N[groups] = MU_N * self.N[groups] + (1.0 - MU_N) * hat_n[groups]
        self.D.copy_(self.V * (self.N / (self.N + NU)).unsqueeze(-1) * self.offdiag.unsqueeze(0))
