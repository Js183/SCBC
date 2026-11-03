"""SCBC training objective (Eqs. 18-20)."""
import torch
import torch.nn.functional as F
from torch import nn

from ..constants import EPS_W, LAMBDA_PAIR, TAU_M


def pairwise_margin_loss(logits, labels, debts):
    """L_pair (Eq. 19) with Laplace-smoothed debt weights (Eq. 18)."""
    num_classes = logits.size(1)
    per_pair = F.softplus(TAU_M - (logits.gather(1, labels.unsqueeze(1)) - logits))
    rows = torch.arange(logits.size(0), device=logits.device)
    competitor = (~torch.eye(num_classes, dtype=torch.bool, device=logits.device))[labels].float()
    row = debts[rows, labels] * competitor
    weights = (row + EPS_W / (num_classes - 1)) * competitor / (row.sum(dim=-1, keepdim=True) + EPS_W)
    return (per_pair * weights).sum(dim=-1).mean()


class SCBCObjective(nn.Module):
    """Training objective (Eq. 20) built around a corrector and its debt book: writes the
    masked/reference excess violation of the batch into the book (after it was read), then
    returns CE on the corrected logits + weighted CE on the base logits + L_pair."""

    def __init__(self, corrector, lambda_base, class_weights=None):
        super().__init__()
        self.corrector = corrector
        self.lambda_base = lambda_base
        self.register_buffer("class_weights", class_weights, persistent=False)

    def forward(self, base_logits, final_logits, debts, reference_logits, labels):
        self.corrector.book.update(base_logits.detach(), reference_logits, labels)
        return (F.cross_entropy(final_logits, labels, weight=self.class_weights)
                + self.lambda_base * F.cross_entropy(base_logits, labels, weight=self.class_weights)
                + LAMBDA_PAIR * pairwise_margin_loss(final_logits, labels, debts))
