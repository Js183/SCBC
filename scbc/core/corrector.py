"""Subset-adaptive displacement head (Sec. 3.3)."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..constants import BETA_MAX, CORRECTOR_DROPOUT, CORRECTOR_HIDDEN_DIM, EPS, OMEGA


def zero_sum(x):
    """Projection Pi_0 onto {r : 1^T r = 0}."""
    return x - x.mean(dim=-1, keepdim=True)


class DebtCorrector(nn.Module):
    """Subset-adaptive displacement head (Sec. 3.3).

    Built around a debt book: class and modality counts come from the book, the
    debts D_S are read from it and the subset indicator s is the complement of the
    book's missing-modality mask. The correction is off for the first
    `warmup_epochs` epochs and then ramps up linearly over `ramp_epochs` epochs."""

    def __init__(self, book, feature_dim, scale, warmup_epochs, ramp_epochs):
        super().__init__()
        self.book = book
        num_classes, num_modalities = book.num_classes, book.missing.numel()
        self.scale = scale
        self.warmup_epochs, self.ramp_epochs = warmup_epochs, ramp_epochs
        input_dim = feature_dim + num_modalities + 2 * num_classes + 1
        hidden = max(feature_dim, num_classes * 4, CORRECTOR_HIDDEN_DIM)
        self.step_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(CORRECTOR_DROPOUT),
            nn.Linear(hidden, num_classes),
        )
        nn.init.normal_(self.step_mlp[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.step_mlp[-1].bias)
        self.register_buffer("runtime_scale", torch.tensor(1.0))

    @torch.no_grad()
    def set_epoch(self, epoch):
        if epoch < self.warmup_epochs:
            value = 0.0
        else:
            value = min(1.0, float(epoch - self.warmup_epochs + 1) / float(self.ramp_epochs))
        self.runtime_scale.fill_(value)

    def forward(self, features, base_logits):
        """Returns z_final = z_base + r_S (Eq. 17) and the debts D_S it was corrected with."""
        batch = base_logits.size(0)
        debts = self.book.debts(batch)
        subset_masks = (1.0 - self.book.missing).view(1, -1).expand(batch, -1)
        p = torch.softmax(base_logits, dim=-1)
        # v_S = Diag(p) D_S 1 - D_S^T p ;  q_S = Pi_0(v_S)   (Eq. 10)
        v = p * debts.sum(dim=2) - torch.einsum("bc,bcd->bd", p, debts)
        q = zero_sum(v)
        v_abs = v.abs()
        v_l1 = v_abs.sum(dim=-1, keepdim=True)

        # chi_S = (1 - H(u)/log C)^omega,  u = |v| / (||v||_1 + eps)   (Eq. 14)
        u = v_abs / (v_l1 + EPS)
        entropy = -(u * u.clamp_min(EPS).log()).sum(dim=-1, keepdim=True)
        chi = (1.0 - entropy / math.log(self.book.num_classes)).clamp(0.0, 1.0) ** OMEGA

        # c_S = [h; s; q; |v|; log(||v||_1 + eps)],  beta = beta_max * tanh(softplus(MLP(c)))   (Eqs. 15-16)
        context = torch.cat([features, subset_masks, q, v_abs, torch.log(v_l1 + EPS)], dim=-1)
        beta = BETA_MAX * torch.tanh(F.softplus(self.step_mlp(context)))

        r = zero_sum(self.scale * float(self.runtime_scale.item()) * chi * (beta * q))
        return base_logits + r, debts
