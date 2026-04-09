from __future__ import annotations

import argparse
from datetime import datetime
import io
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL import ImageDraw
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from src.datasets.mvtec import MVTecAD
from src.metrics.auroc import auroc
from src.methods.ae import AEConfig, AEMethod
from src.methods.padim import PaDiMConfig, PaDiMMethod
from src.methods.patchcore import PatchCoreConfig, PatchCoreMethod
from src.models.feature_extractor import ResNetFeatureExtractor
from src.utils.scoring import (
    GROUPED_2X2_COLUMNS,
    GROUPED_ALIGNMENT_COLUMNS,
    GROUPED_DEFECT_COLUMNS,
    LONG_COLUMNS,
    RAW_METRIC_COLUMNS,
    SCENARIO_COLUMNS,
    SCENARIO_SUMMARY_COLUMNS,
    SCENARIO_WINNERS_BY_CATEGORY_COLUMNS,
    SCENARIO_WINNERS_OVERALL_COLUMNS,
    SEED_AGGREGATED_COLUMNS,
    SEED_RUN_COLUMNS,
    SENSITIVITY_COLUMNS,
    SENSITIVITY_SUMMARY_COLUMNS,
    aggregate_seed_raw_rows,
    build_grouped_2x2_rows,
    build_grouped_alignment_rows,
    build_grouped_defect_rows,
    build_long_rows,
    build_scenario_rows,
    build_seed_aggregated_summary,
    build_seed_runs_rows,
    build_sensitivity_rows,
    build_sensitivity_summary,
    build_summary_rows,
    build_winners_by_category,
    build_winners_overall,
    convert_raw_rows,
    write_csv,
    write_json,
)


AE_LOSS_PATTERN = re.compile(r"^\[AE\] epoch (\d+)/(\d+) loss=([0-9eE+\-.]+)$")


class TeeWriter(io.TextIOBase):
    def __init__(self, *writers):
        self.writers = writers

    def write(self, text):
        for writer in self.writers:
            writer.write(text)
        return len(text)

    def flush(self):
        for writer in self.writers:
            writer.flush()


def _unnormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    return x * std + mean


def tensor_to_pil(x: torch.Tensor, normalize: str) -> Image.Image:
    x = x.detach().cpu()
    if normalize == "imagenet":
        x = _unnormalize_imagenet(x)
    x = x.clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def map_to_pil(m: torch.Tensor) -> Image.Image:
    m = m.detach().cpu().numpy()
    m = m - m.min()
    if m.max() > 1e-8:
        m = m / m.max()
    arr = (m * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def overlay_heatmap(img: Image.Image, heat: Image.Image, alpha: float = 0.45) -> Image.Image:
    if heat.size != img.size:
        heat = heat.resize(img.size, resample=Image.BILINEAR)
    heat_rgb = Image.merge("RGB", (heat, Image.new("L", heat.size), Image.new("L", heat.size)))
    return Image.blend(img.convert("RGB"), heat_rgb, alpha)


def pixel_auroc(all_masks, all_maps) -> float:
    y_true = np.concatenate([m.reshape(-1) for m in all_masks]).astype(np.uint8)
    y_score = np.concatenate([s.reshape(-1) for s in all_maps]).astype(np.float32)
    if y_true.max() == 0:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def parse_ae_loss_history(log_text: str) -> list[dict]:
    history = []
    for line in log_text.splitlines():
        match = AE_LOSS_PATTERN.match(line.strip())
        if not match:
            continue
        history.append(
            {
                "epoch": int(match.group(1)),
                "total_epochs": int(match.group(2)),
                "loss": float(match.group(3)),
            }
        )
    return history


def save_ae_loss_artifacts(base_dir: Path, history: list[dict]) -> None:
    if not history:
        return
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / "ae_loss_curve.csv"
    with csv_path.open("w", newline="") as f:
        f.write("epoch,total_epochs,loss\n")
        for row in history:
            f.write(f"{row['epoch']},{row['total_epochs']},{row['loss']}\n")

    width, height = 800, 480
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 40, 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="black", width=2)
    draw.text((plot_left, 10), "AE Loss Curve", fill="black")
    draw.text((width // 2 - 20, height - 35), "Epoch", fill="black")
    draw.text((10, plot_top - 5), "Loss", fill="black")

    epochs = [row["epoch"] for row in history]
    losses = [row["loss"] for row in history]
    min_epoch = min(epochs)
    max_epoch = max(epochs)
    min_loss = min(losses)
    max_loss = max(losses)
    if min_epoch == max_epoch:
        max_epoch += 1
    if np.isclose(min_loss, max_loss):
        max_loss = min_loss + 1e-6

    points = []
    for epoch, loss in zip(epochs, losses):
        x = plot_left + (epoch - min_epoch) / (max_epoch - min_epoch) * (plot_right - plot_left)
        y = plot_bottom - (loss - min_loss) / (max_loss - min_loss) * (plot_bottom - plot_top)
        points.append((x, y))

    for tick in range(min_epoch, max_epoch + 1):
        x = plot_left + (tick - min_epoch) / (max_epoch - min_epoch) * (plot_right - plot_left)
        draw.line((x, plot_bottom, x, plot_bottom + 5), fill="black", width=1)
        draw.text((x - 8, plot_bottom + 10), str(tick), fill="black")

    for ratio in (0.0, 0.5, 1.0):
        loss_tick = min_loss + ratio * (max_loss - min_loss)
        y = plot_bottom - ratio * (plot_bottom - plot_top)
        draw.line((plot_left - 5, y, plot_left, y), fill="black", width=1)
        draw.text((5, y - 8), f"{loss_tick:.4f}", fill="black")

    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="blue")
    else:
        draw.line(points, fill="blue", width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="blue")

    image.save(base_dir / "ae_loss_curve.png")


def run_one(
    method_name: str,
    method,
    train_loader,
    test_loader,
    out_dir: Path,
    normalize: str,
    save_n: int = 10,
    save_all: bool = False,
    ae_history_dir: Path | None = None,
):
    t0 = time.perf_counter()
    if method_name == "ae" and ae_history_dir is not None:
        log_buffer = io.StringIO()
        tee = TeeWriter(sys.stdout, log_buffer)
        previous_stdout = sys.stdout
        try:
            sys.stdout = tee
            method.fit(train_loader)
        finally:
            sys.stdout = previous_stdout
        save_ae_loss_artifacts(ae_history_dir, parse_ae_loss_history(log_buffer.getvalue()))
    else:
        method.fit(train_loader)
    prep = time.perf_counter() - t0

    y_true_img, y_score_img = [], []
    all_masks, all_maps = [], []
    t1 = time.perf_counter()
    saved = 0

    for x, y, mask, meta in test_loader:
        out = method.predict(x)
        y_true_img.extend(y.numpy().tolist())
        all_masks.append(mask.numpy())

        hm = out.heatmaps
        if hm is None:
            raise RuntimeError(f"{method_name}: heatmaps is None (pixel-level eval requires heatmaps)")
        if hm.ndim == 3:
            hm = hm[:, None, :, :]
        elif hm.ndim != 4:
            raise RuntimeError(f"{method_name}: unexpected heatmaps shape: {hm.shape}")

        if hm.shape[-2:] != mask.shape[-2:]:
            hm = F.interpolate(hm, size=mask.shape[-2:], mode="bilinear", align_corners=False)

        hm = TF.gaussian_blur(hm, kernel_size=7, sigma=2.0)
        y_score_img.extend(hm.amax(dim=(1, 2, 3)).detach().cpu().numpy().tolist())
        all_maps.append(hm.cpu().numpy())

        if save_all or saved < save_n:
            batch_size = x.shape[0]
            for idx in range(batch_size):
                if (not save_all) and saved >= save_n:
                    break
                img = tensor_to_pil(x[idx], normalize=normalize)
                gt = map_to_pil(mask[idx, 0])
                hm_img = map_to_pil(hm[idx, 0])
                ov = overlay_heatmap(img, hm_img)
                defect_type = meta["defect_type"]
                defect = defect_type[idx] if isinstance(defect_type, (list, tuple)) else defect_type
                defect_dir = out_dir / defect
                defect_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{saved:03d}_{defect}"
                img.save(defect_dir / f"{stem}_img.png")
                gt.save(defect_dir / f"{stem}_gt.png")
                hm_img.save(defect_dir / f"{stem}_map.png")
                ov.save(defect_dir / f"{stem}_overlay.png")
                saved += 1

    infer = time.perf_counter() - t1
    masks_np = np.concatenate(all_masks, axis=0)
    maps_np = np.concatenate(all_maps, axis=0)

    return {
        "method": method_name,
        "img_auc": float(auroc(y_true_img, y_score_img)),
        "px_auc": pixel_auroc(masks_np, maps_np),
        "prep_s": prep,
        "infer_s": infer,
        "ms_img": infer / len(test_loader.dataset) * 1000.0,
    }


def get_categories(root: Path, category_arg: str) -> list[str]:
    if category_arg == "all":
        return sorted([p.name for p in root.iterdir() if p.is_dir()])
    return [part.strip() for part in category_arg.split(",") if part.strip()]


def build_loaders(root: str, category: str, image_size: int, batch_size: int, normalize: str):
    train_ds = MVTecAD(root, category, mode="train", image_size=image_size, normalize=normalize)
    test_ds = MVTecAD(root, category, mode="test", image_size=image_size, normalize=normalize)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def create_run_dir(base_out: str) -> Path:
    run_root = Path(base_out) / "results"
    timestamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    run_dir = run_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_seeds(args) -> list[int]:
    if args.seeds:
        return [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    return [int(args.seed)]


def print_category_results(category: str, seed: int, rows: list[dict]) -> None:
    print(f"\n[benchmark] seed={seed} category={category}")
    print(f"{'method':<10} {'image_auroc':>12} {'pixel_auroc':>12} {'prep_s':>10} {'infer_s':>10} {'ms_per_img':>12}")
    for row in rows:
        px = "nan" if np.isnan(row["px_auc"]) else f"{row['px_auc']:.4f}"
        print(
            f"{row['method']:<10} {row['img_auc']:>12.4f} {px:>12} "
            f"{row['prep_s']:>10.3f} {row['infer_s']:>10.3f} {row['ms_img']:>12.3f}"
        )


def print_macro_average(raw_rows: list[dict]) -> None:
    print("\n[benchmark] macro average by method")
    print(f"{'method':<10} {'image_auroc':>12} {'pixel_auroc':>12}")
    for method in sorted({row["method"] for row in raw_rows}):
        method_rows = [row for row in raw_rows if row["method"] == method]
        img_avg = float(np.mean([row["img_auc"] for row in method_rows]))
        px_avg = float(np.mean([row["px_auc"] for row in method_rows]))
        print(f"{method:<10} {img_avg:>12.4f} {px_avg:>12.4f}")


def print_scenario_summary(rows: list[dict]) -> None:
    print("\n[benchmark] scenario score summary")
    print(f"{'model':<10} {'S_perf':>10} {'S_bal':>10} {'S_eff':>10}")
    for method in sorted({row["model"] for row in rows}):
        method_rows = [row for row in rows if row["model"] == method]
        print(
            f"{method:<10} "
            f"{float(np.mean([row['S_perf'] for row in method_rows])):>10.4f} "
            f"{float(np.mean([row['S_bal'] for row in method_rows])):>10.4f} "
            f"{float(np.mean([row['S_eff'] for row in method_rows])):>10.4f}"
        )


def run_seed(args, seed: int, categories: list[str], visuals_root: Path) -> list[dict]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_rows = []
    for category in categories:
        category_visual_root = visuals_root / f"seed_{seed}" / category
        category_visual_root.mkdir(parents=True, exist_ok=True)
        ae_history_root = visuals_root.parent / "AE_epoch_test" / f"seed_{seed}" / category
        backbone = ResNetFeatureExtractor(device=args.device, layers=("layer2", "layer3"))
        category_results = []

        train_loader, test_loader = build_loaders(args.root, category, args.image_size, args.batch_size, normalize="none")
        ae = AEMethod(AEConfig(device=args.device, epochs=args.epochs))
        category_results.append(
            run_one(
                "ae",
                ae,
                train_loader,
                test_loader,
                category_visual_root / "ae",
                normalize="none",
                save_n=args.save_n,
                save_all=args.save_all,
                ae_history_dir=ae_history_root,
            )
        )

        train_loader, test_loader = build_loaders(args.root, category, args.image_size, args.batch_size, normalize="imagenet")
        padim = PaDiMMethod(PaDiMConfig(device=args.device, image_size=args.image_size), backbone)
        category_results.append(
            run_one(
                "padim",
                padim,
                train_loader,
                test_loader,
                category_visual_root / "padim",
                normalize="imagenet",
                save_n=args.save_n,
                save_all=args.save_all,
            )
        )

        train_loader, test_loader = build_loaders(args.root, category, args.image_size, args.batch_size, normalize="imagenet")
        patchcore = PatchCoreMethod(
            PatchCoreConfig(
                device=args.device,
                image_size=args.image_size,
                coreset_ratio=args.coreset_ratio,
                pre_sample_ratio=0.12,
                k=5,
            ),
            backbone,
        )
        category_results.append(
            run_one(
                "patchcore",
                patchcore,
                train_loader,
                test_loader,
                category_visual_root / "patchcore",
                normalize="imagenet",
                save_n=args.save_n,
                save_all=args.save_all,
            )
        )

        print_category_results(category, seed, category_results)
        for row in category_results:
            all_rows.append({"seed": seed, "category": category, **row})

    return all_rows


def export_analysis(run_dir: Path, raw_rows: list[dict], seed_rows: list[dict]) -> None:
    normalized_metadata = {}
    raw_metric_rows = convert_raw_rows(raw_rows, n_seeds=int(raw_rows[0].get("n_seeds", 1)) if raw_rows else 1)
    scenario_rows = build_scenario_rows(raw_rows, normalization_metadata=normalized_metadata)
    scenario_summary_rows = build_summary_rows(scenario_rows)
    winners_by_category = build_winners_by_category(scenario_rows)
    winners_overall = build_winners_overall(scenario_rows)
    long_rows = build_long_rows(scenario_rows)
    grouped_alignment_rows = build_grouped_alignment_rows(scenario_rows)
    grouped_defect_rows = build_grouped_defect_rows(scenario_rows)
    grouped_2x2_rows = build_grouped_2x2_rows(scenario_rows)
    sensitivity_rows = build_sensitivity_rows(scenario_rows)
    sensitivity_summary_rows = build_sensitivity_summary(sensitivity_rows)

    write_csv(run_dir / "raw_metrics.csv", raw_metric_rows, RAW_METRIC_COLUMNS)
    write_csv(run_dir / "scenario_scores.csv", scenario_rows, SCENARIO_COLUMNS)
    write_csv(run_dir / "scenario_scores_summary.csv", scenario_summary_rows, SCENARIO_SUMMARY_COLUMNS)
    write_csv(run_dir / "scenario_winners_by_category.csv", winners_by_category, SCENARIO_WINNERS_BY_CATEGORY_COLUMNS)
    write_csv(run_dir / "scenario_winners_overall.csv", winners_overall, SCENARIO_WINNERS_OVERALL_COLUMNS)
    write_csv(run_dir / "scenario_scores_long.csv", long_rows, LONG_COLUMNS)
    write_csv(run_dir / "grouped_scores_by_alignment.csv", grouped_alignment_rows, GROUPED_ALIGNMENT_COLUMNS)
    write_csv(run_dir / "grouped_scores_by_defect_extent.csv", grouped_defect_rows, GROUPED_DEFECT_COLUMNS)
    write_csv(run_dir / "grouped_scores_2x2.csv", grouped_2x2_rows, GROUPED_2X2_COLUMNS)
    write_csv(run_dir / "sensitivity_scores.csv", sensitivity_rows, SENSITIVITY_COLUMNS)
    write_csv(run_dir / "sensitivity_summary.csv", sensitivity_summary_rows, SENSITIVITY_SUMMARY_COLUMNS)
    write_json(run_dir / "normalization_metadata.json", normalized_metadata)

    if seed_rows:
        seed_metadata = {}
        seed_raw_base_rows = [
            {
                "category": row["category"],
                "method": row["method"],
                "img_auc": row["img_auc"],
                "px_auc": row["px_auc"],
                "prep_s": row["prep_s"],
                "infer_s": row["infer_s"],
                "ms_img": row["ms_img"],
                "seed": row["seed"],
            }
            for row in seed_rows
        ]
        seed_scenario_rows = build_scenario_rows(seed_raw_base_rows, normalization_metadata=seed_metadata)
        seed_runs_rows = build_seed_runs_rows(seed_scenario_rows, seed_rows)
        seed_aggregated_summary = build_seed_aggregated_summary(seed_runs_rows)
        write_csv(run_dir / "seed_runs.csv", seed_runs_rows, SEED_RUN_COLUMNS)
        write_csv(run_dir / "seed_aggregated_summary.csv", seed_aggregated_summary, SEED_AGGREGATED_COLUMNS)

    print_macro_average(raw_rows)
    print_scenario_summary(scenario_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/mvtec_ad")
    parser.add_argument("--category", type=str, default="all")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--coreset_ratio", type=float, default=0.1)
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--save_n", type=int, default=10)
    parser.add_argument("--save_all", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="")
    args = parser.parse_args()

    categories = get_categories(Path(args.root), args.category)
    run_dir = create_run_dir(args.out)
    visuals_root = run_dir / "visuals"
    seeds = parse_seeds(args)

    seed_rows = []
    for seed in seeds:
        seed_rows.extend(run_seed(args, seed, categories, visuals_root))

    raw_rows = aggregate_seed_raw_rows(seed_rows)
    export_analysis(run_dir, raw_rows, seed_rows)
    print(f"\nSaved run outputs to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
