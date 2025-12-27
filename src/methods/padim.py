from __future__ import annotations
from dataclasses import dataclass
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.methods.base import MethodOutput

@dataclass
class PaDiMConfig:
    device: str = "cpu"
    image_size: int = 224
    eps: float = 1e-6

class PaDiMMethod:
    def __init__(self, cfg: PaDiMConfig, backbone):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.backbone = backbone
        self.mean = None
        self.inv_cov = None

    def fit(self, loader: DataLoader):
        feats = []

        for x, _, _, _ in loader:
            with torch.no_grad():
                f = self.backbone(x)

                # layer2, layer3
                f2 = f["layer2"]
                f3 = torch.nn.functional.interpolate(
                    f["layer3"],
                    size=f2.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                # concat -> (B, C, H, W)
                f_cat = torch.cat([f2, f3], dim=1)
                f_cat = F.normalize(f_cat, p=2, dim=1)
                B, C, H, W = f_cat.shape

                # (B*H*W, C)
                f_cat = f_cat.permute(0, 2, 3, 1).reshape(-1, C)
                feats.append(f_cat.cpu())

        feats = torch.cat(feats, dim=0)          # (N, C)
        feats = feats.view(-1, H * W, C)          # (num_imgs, HW, C)
        feats = feats.permute(1, 0, 2)            # (HW, num_imgs, C)

        mean = []
        inv_cov = []

        for i in range(feats.shape[0]):
            f_i = feats[i]                        # (num_imgs, C)
            mu = f_i.mean(dim=0)

            cov = torch.cov(f_i.T)
            cov += self.cfg.eps * torch.eye(C)    # 안정화
            inv = torch.linalg.inv(cov)

            mean.append(mu)
            inv_cov.append(inv)

        self.mean = torch.stack(mean).to(self.device)       # (HW, C)
        self.inv_cov = torch.stack(inv_cov).to(self.device) # (HW, C, C)

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            f = self.backbone(x)

            f2 = f["layer2"]
            f3 = torch.nn.functional.interpolate(
                f["layer3"],
                size=f2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            f_cat = torch.cat([f2, f3], dim=1)  # (B, C, H, W)
            f_cat = F.normalize(f_cat, p=2, dim=1)
            B, C, H, W = f_cat.shape

            f_cat = f_cat.permute(0, 2, 3, 1).reshape(B, H * W, C)

            maps = []
            for b in range(B):
                diff = f_cat[b] - self.mean               # (HW, C)
                md = torch.einsum("nc,ncd,nd->n", diff, self.inv_cov, diff)
                maps.append(md.view(H, W))

            maps = torch.stack(maps)                      # (B, H, W)
            img_scores = maps.view(B, -1).max(dim=1).values

        return MethodOutput(
            scores=img_scores.detach().cpu(),
            heatmaps=maps.detach().cpu()[:, None, :, :]
        )
            
    def save(self, path: str):
        state = {
            "mean": self.mean.cpu(),
            "inv_cov": self.inv_cov.cpu(),
        }
        torch.save(state, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.mean = state["mean"].to(self.device)
        self.inv_cov = state["inv_cov"].to(self.device)
