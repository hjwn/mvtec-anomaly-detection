from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import MethodOutput

class ConvAE(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(base_ch, base_ch*2, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(base_ch*4, base_ch*8, 4, 2, 1), nn.ReLU(True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_ch*8, base_ch*4, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(base_ch*2, base_ch,   4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(base_ch,   in_ch,     4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))

@dataclass
class AEConfig:
    device: str = "cpu"
    lr: float = 1e-3
    epochs: int = 5
    base_ch: int = 64
    loss: str = "l2"  # l1 or l2

class AEMethod:
    def __init__(self, cfg: AEConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = ConvAE(base_ch=cfg.base_ch).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def _loss(self, x, recon):
        if self.cfg.loss == "l1":
            return F.l1_loss(recon, x)
        return F.mse_loss(recon, x)

    def fit(self, train_loader) -> Dict[str, float]:
        self.model.train()
        last = 0.0
        for ep in range(1, self.cfg.epochs + 1):
            total, n = 0.0, 0
            for x, _, _, _ in train_loader:
                x = x.to(self.device)
                self.opt.zero_grad()
                recon = self.model(x)
                loss = self._loss(x, recon)
                loss.backward()
                self.opt.step()
                total += loss.item() * x.size(0)
                n += x.size(0)
            last = total / max(n, 1)
            print(f"[AE] epoch {ep}/{self.cfg.epochs} loss={last:.6f}")
        return {"train_loss": float(last)}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> MethodOutput:
        self.model.eval()
        x = x.to(self.device)
        recon = self.model(x)
        heat = (x - recon).pow(2).mean(dim=1, keepdim=True)   # (B,1,H,W)
        scores = heat.amax(dim=(1,2,3))                       # (B,)
        return MethodOutput(scores=scores.cpu(), heatmaps=heat.cpu())

    def save(self, path: str) -> None:
        torch.save({"cfg": self.cfg.__dict__, "state": self.model.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["state"])
        self.model.to(self.device)
