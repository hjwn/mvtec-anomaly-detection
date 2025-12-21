from __future__ import annotations
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.mvtec import MVTecAD
from src.metrics.auroc import auroc

# NOTE: feature extractor is needed for PaDiM/PatchCore
# i'll plug ResNetFeatureExtractor later.
# from src.models.feature_extractor import ResNetFeatureExtractor

#from src.methods.ae import AEMethod, AEConfig
#from src.methods.padim import PaDiMMethod, PaDiMConfig
#from src.methods.patchcore import PatchCoreMethod, PatchCoreConfig

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, choices=["ae", "padim", "patchcore"], required=True)
    p.add_argument("--category", type=str, default="bottle")
    p.add_argument("--root", type=str, default="data/mvtec_ad")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=5)  # AE only
    return p.parse_args()

def main():
    args = parse_args()

    # ---- dataset: AE는 none, 나머지는 imagenet ----
    norm = "none" if args.method == "ae" else "imagenet"

    train_ds = MVTecAD(args.root, args.category, mode="train", image_size=args.image_size, normalize=norm)
    test_ds  = MVTecAD(args.root, args.category, mode="test",  image_size=args.image_size, normalize=norm)


    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- method (lazy import) ----
    if args.method == "ae":
        from src.methods.ae import AEMethod, AEConfig
        method = AEMethod(AEConfig(device=args.device, epochs=args.epochs))

    elif args.method == "padim":
        from src.methods.padim import PaDiMMethod, PaDiMConfig
        from src.models.feature_extractor import ResNetFeatureExtractor

        backbone = ResNetFeatureExtractor(device=args.device)
        method = PaDiMMethod(PaDiMConfig(device=args.device, image_size=args.image_size), backbone)


    elif args.method == "patchcore":
        from src.methods.patchcore import PatchCoreMethod, PatchCoreConfig
        from src.models.feature_extractor import ResNetFeatureExtractor

        backbone = ResNetFeatureExtractor(device=args.device, layers=("layer2", "layer3"))
        method = PatchCoreMethod(
            PatchCoreConfig(device=args.device, image_size=args.image_size, coreset_ratio=0.1, k=1),
            backbone
        )


    # ---- fit ----
    t0 = time.perf_counter()
    method.fit(train_loader)
    prep_time = time.perf_counter() - t0

    # ---- eval ----
    y_true, y_score = [], []
    t1 = time.perf_counter()
    for x, y, mask, meta in test_loader:
        out = method.predict(x)
        y_true.extend(y.numpy().tolist())
        y_score.extend(out.scores.numpy().tolist())
    infer_time = time.perf_counter() - t1

    score = auroc(y_true, y_score)
    print(f"\n[{args.method}] category={args.category}")
    print(f"prep_time(s): {prep_time:.3f}")
    print(f"infer_time(s): {infer_time:.3f}  |  per_img(ms): {infer_time/len(test_ds)*1000:.3f}")
    print(f"Image AUROC  : {score}")


if __name__ == "__main__":
    main()
