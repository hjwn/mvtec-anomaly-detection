from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Protocol
import torch

@dataclass
class MethodOutput:
    # image-level anomaly score (B,)
    scores: torch.Tensor
    # pixel-level heatmap (B,1,H,W) or None
    heatmaps: Optional[torch.Tensor] = None

class AnomalyMethod(Protocol):
    def fit(self, train_loader) -> Dict[str, float]:
        ...

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> MethodOutput:
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> None:
        ...
