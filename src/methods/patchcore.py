from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.methods.base import MethodOutput

@dataclass
class PatchCoreConfig:
    device: str = "cpu"
    image_size: int = 224
    coreset_ratio: float = 0.1  # 10% 저장
    k: int = 1                  # kNN

class PatchCoreMethod:
    def __init__(self, cfg: PatchCoreConfig, backbone):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.backbone = backbone
        self.memory = None  # (M, C)

    def fit(self, loader: DataLoader):
        feats = []

        for x, _, _, _ in loader:
            with torch.no_grad():
                f = self.backbone(x)

                f2 = f["layer2"]
                f3 = torch.nn.functional.interpolate(
                    f["layer3"],
                    size=f2.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                f_cat = torch.cat([f2, f3], dim=1)  # (B,C,H,W)
                f_cat = F.normalize(f_cat, p=2, dim=1)
                B, C, H, W = f_cat.shape
                f_cat = f_cat.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
                feats.append(f_cat.cpu())

        feats = torch.cat(feats, dim=0)  # (N, C)

        # --- random coreset sampling ---
        N = feats.shape[0]
        m = max(1, int(N * self.cfg.coreset_ratio))
        idx = torch.randperm(N)[:m]
        self.memory = feats[idx].to(self.device)

        print(f"[PatchCore] memory bank: {self.memory.shape} (ratio={self.cfg.coreset_ratio})")

    def predict(self, x: torch.Tensor):
        if self.memory is None:
            raise RuntimeError("memory bank is empty. call fit() or load() first.")

        with torch.no_grad():
            f = self.backbone(x)

            f2 = f["layer2"]
            f3 = torch.nn.functional.interpolate(
                f["layer3"],
                size=f2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            f_cat = torch.cat([f2, f3], dim=1)  # (B,C,H,W)
            f_cat = F.normalize(f_cat, p=2, dim=1)
            B, C, H, W = f_cat.shape
            f_cat = f_cat.permute(0, 2, 3, 1).reshape(B, H * W, C)

            maps = []
            for b in range(B):
                # (HW, C) vs (M, C)
                d = torch.cdist(f_cat[b].to(self.device), self.memory)  # (HW, M)
                knn = d.topk(self.cfg.k, largest=False).values          # (HW, k)
                patch_score = knn.mean(dim=1)                           # (HW,)
                maps.append(patch_score.view(H, W))

            maps = torch.stack(maps)                # (B,H,W)
            img_scores = maps.view(B, -1).max(dim=1).values
        return MethodOutput(
            scores=img_scores.detach().cpu(),            # (B,)
            heatmaps=maps.detach().cpu()[:, None, :, :]  # (B,1,H,W)
        )


    def save(self, path: str):
        if self.memory is None:
            raise RuntimeError("nothing to save: memory is None")
        torch.save({"memory": self.memory.cpu(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.memory = state["memory"].to(self.device)
