from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import timm


@dataclass(frozen=True)
class EncoderSpec:
    name: str = "resnet50"
    pretrained: bool = True
    img_size: int = 224
    pooling: str = "avg"


def build_frozen_encoder(spec: EncoderSpec, device: torch.device) -> Tuple[nn.Module, int]:
    """
    Build a frozen feature extractor that returns (B, D).
    """
    model = timm.create_model(
        spec.name,
        pretrained=spec.pretrained,
        num_classes=0,
        global_pool=spec.pooling
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, spec.img_size, spec.img_size, device=device)
        out = model(dummy)
        feat_dim = int(out.shape[-1])
    return model, feat_dim
