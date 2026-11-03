"""Speaker-aware dialogue encoders and the masked residual fusion head for IEMOCAP-4."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..constants import MODALITIES

NUM_CLASSES = 4


class DialogueEncoder(nn.Module):
    """Per-modality utterance encoder + speaker-aware bidirectional GRU over the dialogue."""

    def __init__(self, feature_shape, hidden_dim, speaker_dim, num_layers, dropout):
        super().__init__()
        self.sequence_input = len(feature_shape) == 2
        self.speaker_embedding = nn.Embedding(2, speaker_dim)
        if self.sequence_input:
            token_dim = feature_shape[-1]
            self.token_input_proj = nn.Sequential(
                nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
            self.token_encoder = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
            self.token_attention = nn.Linear(hidden_dim, 1)
            self.token_fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        else:
            self.input_proj = nn.Sequential(
                nn.LayerNorm(feature_shape[0]), nn.Linear(feature_shape[0], hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.encoder = nn.GRU(hidden_dim + speaker_dim, hidden_dim // 2, num_layers=num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_CLASSES))

    def forward(self, x, speakers):
        """x: [T, D] or [T, P, D] for one dialogue. Returns hidden [T, H] and logits [T, C]."""
        if self.sequence_input:
            tokens, _ = self.token_encoder(self.token_input_proj(x))
            weights = torch.softmax(self.token_attention(tokens), dim=1)
            pooled = (weights * tokens).sum(dim=1)
            center = tokens[:, tokens.size(1) // 2, :]
            encoded = self.token_fusion(torch.cat([center, pooled], dim=-1))
        else:
            encoded = self.input_proj(x)
        hidden, _ = self.encoder(torch.cat([encoded, self.speaker_embedding(speakers)], dim=-1).unsqueeze(0))
        hidden = hidden.squeeze(0)
        return hidden, self.classifier(hidden)


class ResidualFusionHead(nn.Module):
    """Temperature-calibrated weighted sum of per-modality logits plus an MLP correction."""

    def __init__(self, input_dim, hidden_dim, dropout, num_modalities):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_CLASSES))
        self.modality_weight_logits = nn.Parameter(torch.zeros(num_modalities))
        self.log_temperature = nn.Parameter(torch.zeros(num_modalities))

    def forward(self, features, logits_stack, mask):
        """mask: [M] float, 1 for an observed modality; unobserved modalities get zero weight."""
        calibrated = logits_stack / torch.exp(self.log_temperature).clamp(0.5, 4.0).view(1, -1, 1)
        weights = torch.softmax(self.modality_weight_logits.masked_fill(mask <= 0.0, float("-inf")), dim=0)
        return (weights.view(1, -1, 1) * calibrated).sum(dim=1) + self.mlp(self.input_norm(features))


def branch_inputs(outputs, mask):
    """Per-modality evidence blocks; blocks of unobserved modalities are zeroed."""
    blocks, logits_stack = [], []
    for m, keep in zip(MODALITIES, mask.tolist()):
        hidden, logits = outputs[m]
        if keep > 0.5:
            probs = torch.softmax(logits, dim=-1).clamp_min(1.0e-8)
            confidence = torch.cat([probs.max(dim=-1, keepdim=True).values,
                                    -(probs * probs.log()).sum(dim=-1, keepdim=True) / math.log(NUM_CLASSES)], dim=-1)
            blocks.extend([F.layer_norm(logits, (logits.size(-1),)), torch.softmax(logits, dim=-1), confidence,
                           F.layer_norm(hidden, (hidden.size(-1),))])
        else:
            logits = torch.zeros_like(logits)
            blocks.extend([logits, logits, logits.new_zeros(logits.size(0), 2), torch.zeros_like(hidden)])
        logits_stack.append(logits)
    return torch.cat(blocks, dim=-1), torch.stack(logits_stack, dim=1)
