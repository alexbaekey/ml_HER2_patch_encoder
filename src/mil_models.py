from __future__ import annotations
import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
    """
    ABMIL: attention pooling over instance features -> bag representation -> classifier.
    """
    def __init__(self, in_dim: int, attn_dim: int = 256, n_classes: int = 3, dropout: float = 0.25):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.classifier = nn.Linear(attn_dim, n_classes)

    def forward(self, x: torch.Tensor):
        """
        x: (N, D)
        returns logits: (C,), attn_weights: (N,)
        """
        h = self.embed(x)                # (N, A)
        a = self.attn(h).squeeze(-1)     # (N,)
        w = torch.softmax(a, dim=0)      # (N,)
        z = (w.unsqueeze(-1) * h).sum(dim=0)  # (A,)
        logits = self.classifier(z)      # (C,)
        return logits, w
