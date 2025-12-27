from __future__ import annotations
import argparse, time
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import functional as TF

from src.datasets.mvtec import MVTecAD
from src.metrics.auroc import auroc
from sklearn.metrics import roc_auc_score

from src.models.feature_extractor import ResNetFeatureExtractor
from src.methods.ae import AEMethod, AEConfig
from src.methods.padim import PaDiMMethod, PaDiMConfig
from src.methods.patchcore import PatchCoreMethod, PatchCoreConfig
import src.methods.padim as padim_mod
import src.methods.patchcore as patchcore_mod
print("[DEBUG] padim module file:", padim_mod.__file__)
print("[DEBUG] patchcore module file:", patchcore_mod.__file__)

def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    # x: (3,H,W) in [0,1] expected
    x = x.detach().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def map_to_pil(m: torch.Tensor) -> Image.Image:
    # m: (H,W) float -> normalize to 0..255 grayscale
    m = m.detach().cpu().numpy()
    m = m - m.min()
    if m.max() > 1e-8:
        m = m / m.max()
    arr = (m * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")

def overlay_heatmap(img: Image.Image, heat: Image.Image, alpha=0.45) -> Image.Image:
    if heat.size != img.size:
        heat = heat.resize(img.size, resample=Image.BILINEAR)
    heat_rgb = Image.merge("RGB", (heat, Image.new("L", heat.size), Image.new("L", heat.size)))
    return Image.blend(img.convert("RGB"), heat_rgb, alpha)

def pixel_auroc(all_masks, all_maps) -> float:
    y_true = np.concatenate([m.reshape(-1) for m in all_masks]).astype(np.uint8)
    y_score = np.concatenate([s.reshape(-1) for s in all_maps]).astype(np.float32)
    # 만약 전체가 0이면 AUROC 정의 불가 -> None 처리
    if y_true.max() == 0:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def run_one(method_name: str, method, train_loader, test_loader, out_dir: Path, save_n: int = 10):
    # ---- fit ----
    t0 = time.perf_counter()
    method.fit(train_loader)
    prep = time.perf_counter() - t0

    # ---- eval ----
    y_true_img, y_score_img = [], []
    all_masks, all_maps = [], []

    t1 = time.perf_counter()
    saved = 0

    for x, y, mask, meta in test_loader:
        out = method.predict(x)  # must have .scores (B,) and .heatmaps (B,1,H,W) or (B,H,W)

        y_true_img.extend(y.numpy().tolist())
        all_masks.append(mask.numpy())  # (B,1,H,W)

        # pixel-level collect
        hm = out.heatmaps
        if hm is None:
            raise RuntimeError(f"{method_name}: heatmaps is None (pixel-level eval requires heatmaps)")

        # (B,H,W) -> (B,1,H,W)
        if hm.ndim == 3:
            hm = hm[:, None, :, :]
        elif hm.ndim == 4:
            pass
        else:
            raise RuntimeError(f"{method_name}: unexpected heatmaps shape: {hm.shape}")

        # align heatmap spatial size to mask size for pixel metrics/visuals
        if hm.shape[-2:] != mask.shape[-2:]:
            hm = F.interpolate(hm, size=mask.shape[-2:], mode="bilinear", align_corners=False)

        # gaussian smoothing for more stable image-level scoring
        hm = TF.gaussian_blur(hm, kernel_size=7, sigma=2.0)

        # recompute image scores from (smoothed) heatmaps
        scores = hm.amax(dim=(1, 2, 3)).detach().cpu().numpy().tolist()
        y_score_img.extend(scores)

        all_maps.append(hm.cpu().numpy())  # (B,1,H,W)

        # save some visuals
        if saved < save_n:
            B = x.shape[0]
            for i in range(B):
                if saved >= save_n:
                    break
                img = tensor_to_pil(x[i])
                gt  = map_to_pil(mask[i,0])
                hm_img  = map_to_pil(hm[i, 0])
                ov      = overlay_heatmap(img, hm_img)

                dt = meta["defect_type"]
                defect = dt[i] if isinstance(dt, (list, tuple)) else dt
                stem = f"{saved:03d}_{defect}"
                img.save(out_dir / f"{stem}_img.png")
                gt.save(out_dir  / f"{stem}_gt.png")
                hm_img.save(out_dir  / f"{stem}_map.png")
                ov.save(out_dir  / f"{stem}_overlay.png")
                saved += 1

    infer = time.perf_counter() - t1

    img_auc = float(auroc(y_true_img, y_score_img))
    # concat masks/maps
    masks_np = np.concatenate(all_masks, axis=0)  # (N,1,H,W)
    maps_np  = np.concatenate(all_maps, axis=0)   # (N,1,H,W)
    px_auc   = pixel_auroc(masks_np, maps_np)

    per_img_ms = infer / len(test_loader.dataset) * 1000.0

    return {
        "method": method_name,
        "img_auc": img_auc,
        "px_auc": px_auc,
        "prep_s": prep,
        "infer_s": infer,
        "ms_img": per_img_ms,
    }

def write_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = ["category", "method", "img_auc", "px_auc", "prep_s", "infer_s", "ms_img"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def get_categories(root: Path, category_arg: str):
    if category_arg != "all":
        return [category_arg]
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def build_loaders(root: str, category: str, image_size: int, batch_size: int, normalize: str):
    train_ds = MVTecAD(root, category, mode="train", image_size=image_size, normalize=normalize)
    test_ds  = MVTecAD(root, category, mode="test",  image_size=image_size, normalize=normalize)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/mvtec_ad")
    ap.add_argument("--category", type=str, default="bottle")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--epochs", type=int, default=5)          # AE
    ap.add_argument("--coreset_ratio", type=float, default=0.1)  # PatchCore
    ap.add_argument("--out", type=str, default="outputs/bottle")
    ap.add_argument("--save_n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    categories = get_categories(Path(args.root), args.category)
    all_rows = []

    for category in categories:
        cat_out_root = out_root if args.category != "all" else out_root / category
        cat_out_root.mkdir(parents=True, exist_ok=True)

        # shared backbone for padim/patchcore
        backbone = ResNetFeatureExtractor(device=args.device, layers=("layer2", "layer3"))

        results = []

        # AE (no normalization)
        train_loader, test_loader = build_loaders(
            args.root, category, args.image_size, args.batch_size, normalize="none"
        )
        ae_dir = cat_out_root / "ae"
        ae_dir.mkdir(exist_ok=True)
        ae = AEMethod(AEConfig(device=args.device, epochs=args.epochs))
        results.append(run_one("ae", ae, train_loader, test_loader, ae_dir, save_n=args.save_n))

        # PaDiM (imagenet normalization)
        train_loader, test_loader = build_loaders(
            args.root, category, args.image_size, args.batch_size, normalize="imagenet"
        )
        padim_dir = cat_out_root / "padim"
        padim_dir.mkdir(exist_ok=True)
        padim = PaDiMMethod(PaDiMConfig(device=args.device, image_size=args.image_size), backbone)
        results.append(run_one("padim", padim, train_loader, test_loader, padim_dir, save_n=args.save_n))

        # PatchCore (imagenet normalization)
        train_loader, test_loader = build_loaders(
            args.root, category, args.image_size, args.batch_size, normalize="imagenet"
        )
        pc_dir = cat_out_root / "patchcore"
        pc_dir.mkdir(exist_ok=True)
        pc = PatchCoreMethod(PatchCoreConfig(device=args.device, image_size=args.image_size,
                                             coreset_ratio=args.coreset_ratio, k=1), backbone)
        results.append(run_one("patchcore", pc, train_loader, test_loader, pc_dir, save_n=args.save_n))

        # print table per category
        print(f"\n[benchmark] category={category} device={args.device}")
        print(f"| {'method':<9} | {'image AUROC':>11} | {'pixel AUROC':>11} | {'prep(s)':>7} | {'infer(s)':>7} | {'ms/img':>7} |")
        print(f"|:{'-'*9}-|{'-'*12}:|{'-'*12}:|{'-'*8}:|{'-'*8}:|{'-'*8}:|")
        for r in results:
            px = float("nan") if np.isnan(r["px_auc"]) else r["px_auc"]
            print(
                f"| {r['method']:<9} | {r['img_auc']:>11.4f} | {px:>11.4f} |"
                f" {r['prep_s']:>7.3f} | {r['infer_s']:>7.3f} | {r['ms_img']:>7.3f} |"
            )
            all_rows.append({"category": category, **r})

    # write summary csv
    csv_path = out_root / "summary.csv"
    write_csv(csv_path, all_rows)
    
    # average across categories if needed
    if args.category == "all" and all_rows:
        print("\n[benchmark] macro average by method")
        print("| method | image AUROC | pixel AUROC |")
        print("|---|---:|---:|")
        for method in sorted({r["method"] for r in all_rows}):
            rows = [r for r in all_rows if r["method"] == method]
            img_vals = [r["img_auc"] for r in rows if not np.isnan(r["img_auc"])]
            px_vals = [r["px_auc"] for r in rows if not np.isnan(r["px_auc"])]
            img_avg = float(np.mean(img_vals)) if img_vals else float("nan")
            px_avg = float(np.mean(px_vals)) if px_vals else float("nan")
            img_s = "nan" if np.isnan(img_avg) else f"{img_avg:.4f}"
            px_s = "nan" if np.isnan(px_avg) else f"{px_avg:.4f}"
            print(f"| {method} | {img_s} | {px_s} |")

    print(f"\nSaved outputs to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
