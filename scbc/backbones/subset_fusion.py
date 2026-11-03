"""Subset-aware token-level fusion encoder for CMU-MOSEI."""
import torch
from torch import nn

from ..constants import MODALITIES


class ModalityTokenizer(nn.Module):
    """Expands one pre-extracted feature vector into K tokens."""

    def __init__(self, input_dim, hidden_dim, num_tokens, dropout):
        super().__init__()
        self.hidden_dim, self.num_tokens = hidden_dim, num_tokens
        self.input_norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * num_tokens),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * num_tokens, hidden_dim * num_tokens),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.output_norm(self.net(self.input_norm(x)).view(x.size(0), self.num_tokens, self.hidden_dim))


class MaskedEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, hidden):
        h = self.norm1(x)
        x = x + self.dropout1(self.attn(h, h, h, key_padding_mask=hidden, need_weights=False)[0])
        x = x.masked_fill(hidden.unsqueeze(-1), 0.0)
        x = x + self.ffn(self.norm2(x))
        return x.masked_fill(hidden.unsqueeze(-1), 0.0)


class SubsetFusion(nn.Module):
    """Token-level cross-modal encoder over the modalities in the subset mask."""

    def __init__(self, input_dims, num_classes, cfg):
        super().__init__()
        d, k = cfg["hidden_dim"], cfg["tokens_per_modality"]
        self.hidden_dim, self.tokens = d, k
        self.tokenizers = nn.ModuleDict(
            {m: ModalityTokenizer(input_dims[m], d, k, cfg["encoder_dropout"]) for m in MODALITIES}
        )
        n = len(MODALITIES)
        self.modality_embeddings = nn.Embedding(n, d)
        self.present_embeddings = nn.Embedding(n, d)
        self.missing_embeddings = nn.Embedding(n, d)
        self.local_position_embeddings = nn.ParameterDict({m: nn.Parameter(torch.zeros(k, d)) for m in MODALITIES})
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.encoder_layers = nn.ModuleList(
            [MaskedEncoderLayer(d, cfg["attention_heads"], cfg["ffn_dim"], cfg["encoder_dropout"])
             for _ in range(cfg["encoder_layers"])]
        )
        self.h_proj = nn.Linear(2 * d, d)
        self.classifier = nn.Linear(d, num_classes)
        self.register_buffer("modality_indices", torch.arange(n), persistent=False)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        for m in MODALITIES:
            nn.init.normal_(self.local_position_embeddings[m], mean=0.0, std=0.02)

    def forward(self, inputs, subset_mask):
        """subset_mask: [M] float, 1 for an observed modality. Returns (h, logits)."""
        batch = inputs[0].size(0)
        blocks = []
        for i, (m, x) in enumerate(zip(MODALITIES, inputs)):
            if subset_mask[i] > 0.5:
                block = self.tokenizers[m](x)
            else:
                block = subset_mask.new_zeros(batch, self.tokens, self.hidden_dim)
            block = block + self.modality_embeddings(self.modality_indices[i]).view(1, 1, -1) \
                + self.local_position_embeddings[m].view(1, self.tokens, -1)
            blocks.append(block * subset_mask[i])
        masks = subset_mask.view(1, -1).expand(batch, -1)
        subset_token = (masks.unsqueeze(-1) * self.present_embeddings(self.modality_indices).unsqueeze(0)
                        + (1.0 - masks).unsqueeze(-1) * self.missing_embeddings(self.modality_indices).unsqueeze(0)).sum(dim=1)
        x = torch.cat([self.cls_token.expand(batch, -1, -1), subset_token.unsqueeze(1), *blocks], dim=1)
        visible = torch.cat([torch.ones(batch, 2, device=x.device), masks.repeat_interleave(self.tokens, dim=1)], dim=1) > 0
        for layer in self.encoder_layers:
            x = layer(x, ~visible)
        h = self.h_proj(torch.cat([x[:, 0], x[:, 1]], dim=-1))
        return h, self.classifier(h)
