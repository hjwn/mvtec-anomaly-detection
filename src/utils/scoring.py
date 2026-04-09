from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


CATEGORY_METADATA = {
    "capsule": {"alignment_type": "aligned", "defect_extent": "localized"},
    "bottle": {"alignment_type": "aligned", "defect_extent": "global"},
    "screw": {"alignment_type": "unaligned", "defect_extent": "localized"},
    "cable": {"alignment_type": "unaligned", "defect_extent": "global"},
}

SCENARIO_SPECS = {
    "S_perf": 0.7,
    "S_bal": 0.5,
    "S_eff": 0.3,
}
SENSITIVITY_WEIGHTS = (0.5, 0.7, 0.9)
SCENARIO_KEYS = list(SCENARIO_SPECS.keys())
SCORE_VALUE_COLUMNS = ["A_fixed", "E_fixed", "S_perf", "S_bal", "S_eff"]
GROUP_SCORE_COLUMNS = ["A_fixed", "E_fixed", "S_perf", "S_bal", "S_eff"]

RAW_METRIC_COLUMNS = [
    "category",
    "model",
    "image_auroc",
    "pixel_auroc",
    "prep_time",
    "infer_time",
    "ms_per_img",
    "alignment_type",
    "defect_extent",
    "n_seeds",
]

SCENARIO_COLUMNS = [
    "category",
    "model",
    "image_auroc",
    "pixel_auroc",
    "prep_time",
    "ms_per_img",
    "A_fixed",
    "T_cost",
    "R_cost",
    "E_fixed",
    "S_perf",
    "S_bal",
    "S_eff",
    "rank_perf",
    "rank_bal",
    "rank_eff",
    "alignment_type",
    "defect_extent",
    "n_seeds",
]

SCENARIO_SUMMARY_COLUMNS = [
    "summary_type",
    "scenario",
    "group",
    "model",
    "score",
    "rank",
    "n_items",
]

SCENARIO_WINNERS_BY_CATEGORY_COLUMNS = [
    "category",
    "scenario",
    "winner_model",
    "winner_score",
    "alignment_type",
    "defect_extent",
]

SCENARIO_WINNERS_OVERALL_COLUMNS = [
    "scenario",
    "winner_model",
    "winner_score",
]

LONG_COLUMNS = [
    "category",
    "model",
    "scenario",
    "score_type",
    "value",
]

GROUPED_ALIGNMENT_COLUMNS = [
    "alignment_type",
    "model",
    "A_fixed_mean",
    "E_fixed_mean",
    "S_perf_mean",
    "S_bal_mean",
    "S_eff_mean",
    "n_categories",
]

GROUPED_DEFECT_COLUMNS = [
    "defect_extent",
    "model",
    "A_fixed_mean",
    "E_fixed_mean",
    "S_perf_mean",
    "S_bal_mean",
    "S_eff_mean",
    "n_categories",
]

GROUPED_2X2_COLUMNS = [
    "alignment_type",
    "defect_extent",
    "model",
    "A_fixed_mean",
    "E_fixed_mean",
    "S_perf_mean",
    "S_bal_mean",
    "S_eff_mean",
    "n_categories",
]

SENSITIVITY_COLUMNS = [
    "category",
    "model",
    "alignment_type",
    "defect_extent",
    "w1",
    "w2",
    "scenario",
    "w3",
    "A",
    "T_cost",
    "R_cost",
    "E",
    "S",
    "rank",
]

SENSITIVITY_SUMMARY_COLUMNS = [
    "summary_type",
    "w1",
    "w2",
    "scenario",
    "w3",
    "model",
    "value",
]

SEED_RUN_COLUMNS = [
    "seed",
    "category",
    "model",
    "image_auroc",
    "pixel_auroc",
    "prep_time",
    "infer_time",
    "ms_per_img",
    "A_fixed",
    "E_fixed",
    "S_perf",
    "S_bal",
    "S_eff",
    "rank_perf",
    "rank_bal",
    "rank_eff",
    "alignment_type",
    "defect_extent",
]

SEED_AGGREGATED_COLUMNS = [
    "summary_type",
    "category",
    "model",
    "n_seeds",
    "image_auroc_mean",
    "image_auroc_std",
    "pixel_auroc_mean",
    "pixel_auroc_std",
    "prep_time_mean",
    "prep_time_std",
    "ms_per_img_mean",
    "ms_per_img_std",
    "A_fixed_mean",
    "A_fixed_std",
    "E_fixed_mean",
    "E_fixed_std",
    "S_perf_mean",
    "S_perf_std",
    "S_bal_mean",
    "S_bal_std",
    "S_eff_mean",
    "S_eff_std",
]


def _safe_float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _float_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def _float_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return 0.0
    return float(np.nanstd(arr, ddof=0))


def _rank_rows(rows: list[dict], value_key: str, rank_key: str) -> None:
    sorted_rows = sorted(rows, key=lambda row: (-_safe_float(row[value_key]), row["model"]))
    for rank, row in enumerate(sorted_rows, start=1):
        row[rank_key] = rank


def _normalize_costs(values: Iterable[float], label: str) -> tuple[list[float], dict]:
    raw_values = np.asarray([max(_safe_float(v), 0.0) for v in values], dtype=np.float64)
    log_values = np.log1p(raw_values)
    finite_mask = np.isfinite(log_values)
    if not finite_mask.any():
        raise ValueError(f"No finite values available for {label} cost normalization.")

    finite_log = log_values[finite_mask]
    log_min = float(finite_log.min())
    log_max = float(finite_log.max())

    if math.isclose(log_min, log_max):
        costs = [0.0 if np.isfinite(v) else float("nan") for v in log_values]
    else:
        denom = log_max - log_min
        costs = [float((v - log_min) / denom) if np.isfinite(v) else float("nan") for v in log_values]

    metadata = {
        "raw_min": float(raw_values.min()) if raw_values.size else float("nan"),
        "raw_max": float(raw_values.max()) if raw_values.size else float("nan"),
        "log_min": log_min,
        "log_max": log_max,
    }
    return costs, metadata


def _validate_raw_rows(rows: list[dict]) -> None:
    required = ["category", "method", "img_auc", "px_auc", "prep_s", "infer_s", "ms_img"]
    missing_fields = []
    for idx, row in enumerate(rows):
        row_missing = [field for field in required if field not in row]
        if row_missing:
            missing_fields.append(f"row {idx}: {', '.join(row_missing)}")
    if missing_fields:
        raise ValueError(f"Raw benchmark rows are missing required fields: {'; '.join(missing_fields)}")

    expected_methods = sorted({row["method"] for row in rows})
    for category in sorted({row["category"] for row in rows}):
        category_methods = sorted({row["method"] for row in rows if row["category"] == category})
        if category_methods != expected_methods:
            raise ValueError(
                "Missing benchmark results for category "
                f"'{category}'. Expected methods={expected_methods}, got={category_methods}."
            )


def _with_metadata(row: dict) -> dict:
    enriched = dict(row)
    enriched.update(CATEGORY_METADATA.get(row["category"], {}))
    return enriched


def convert_raw_rows(rows: list[dict], n_seeds: int = 1) -> list[dict]:
    _validate_raw_rows(rows)
    converted = []
    for row in rows:
        converted.append(
            {
                "category": row["category"],
                "model": row["method"],
                "image_auroc": _safe_float(row["img_auc"]),
                "pixel_auroc": _safe_float(row["px_auc"]),
                "prep_time": _safe_float(row["prep_s"]),
                "infer_time": _safe_float(row["infer_s"]),
                "ms_per_img": _safe_float(row["ms_img"]),
                "n_seeds": int(n_seeds),
                **CATEGORY_METADATA.get(row["category"], {}),
            }
        )
    return converted


def build_normalization_metadata(rows: list[dict], t_meta: dict, r_meta: dict) -> dict:
    categories = sorted({row["category"] for row in rows})
    models = sorted({row["method"] for row in rows})
    return {
        "time_cost": {
            "raw_min": t_meta["raw_min"],
            "raw_max": t_meta["raw_max"],
            "log_min": t_meta["log_min"],
            "log_max": t_meta["log_max"],
        },
        "prep_cost": {
            "raw_min": r_meta["raw_min"],
            "raw_max": r_meta["raw_max"],
            "log_min": r_meta["log_min"],
            "log_max": r_meta["log_max"],
        },
        "log1p_used": True,
        "global_normalization": True,
        "categories": categories,
        "models": models,
    }


def build_scenario_rows(
    rows: list[dict],
    w1: float = 0.7,
    w2: float = 0.7,
    normalization_metadata: dict | None = None,
) -> list[dict]:
    if not rows:
        return []

    _validate_raw_rows(rows)
    t_costs, t_meta = _normalize_costs((row["ms_img"] for row in rows), label="T")
    r_costs, r_meta = _normalize_costs((row["prep_s"] for row in rows), label="R")
    if normalization_metadata is not None:
        normalization_metadata.update(build_normalization_metadata(rows, t_meta, r_meta))

    scenario_rows = []
    for row, t_cost, r_cost in zip(rows, t_costs, r_costs):
        image_auroc = _safe_float(row["img_auc"])
        pixel_auroc = _safe_float(row["px_auc"])
        prep_time = _safe_float(row["prep_s"])
        ms_per_img = _safe_float(row["ms_img"])
        a_score = float(w1 * image_auroc + (1.0 - w1) * pixel_auroc)
        e_score = float(1.0 - (w2 * t_cost + (1.0 - w2) * r_cost))
        scenario_row = _with_metadata(
            {
                "category": row["category"],
                "model": row["method"],
                "image_auroc": image_auroc,
                "pixel_auroc": pixel_auroc,
                "prep_time": prep_time,
                "ms_per_img": ms_per_img,
                "A_fixed": a_score,
                "T_cost": float(t_cost),
                "R_cost": float(r_cost),
                "E_fixed": e_score,
                "S_perf": float(SCENARIO_SPECS["S_perf"] * a_score + (1.0 - SCENARIO_SPECS["S_perf"]) * e_score),
                "S_bal": float(SCENARIO_SPECS["S_bal"] * a_score + (1.0 - SCENARIO_SPECS["S_bal"]) * e_score),
                "S_eff": float(SCENARIO_SPECS["S_eff"] * a_score + (1.0 - SCENARIO_SPECS["S_eff"]) * e_score),
                "rank_perf": 0,
                "rank_bal": 0,
                "rank_eff": 0,
                "n_seeds": int(row.get("n_seeds", 1)),
            }
        )
        if "seed" in row:
            scenario_row["seed"] = int(row["seed"])
        scenario_rows.append(scenario_row)

    group_keys = defaultdict(list)
    for row in scenario_rows:
        rank_group = (row["category"], row.get("seed"))
        group_keys[rank_group].append(row)
    for grouped_rows in group_keys.values():
        _rank_rows(grouped_rows, "S_perf", "rank_perf")
        _rank_rows(grouped_rows, "S_bal", "rank_bal")
        _rank_rows(grouped_rows, "S_eff", "rank_eff")

    return scenario_rows


def build_summary_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    summary_rows = []
    models = sorted({row["model"] for row in rows})
    for scenario in SCENARIO_KEYS:
        sorted_models = sorted(
            (
                {
                    "model": model,
                    "score": _float_mean(row[scenario] for row in rows if row["model"] == model),
                }
                for model in models
            ),
            key=lambda item: (-item["score"], item["model"]),
        )
        for rank, item in enumerate(sorted_models, start=1):
            summary_rows.append(
                {
                    "summary_type": "mean_by_model",
                    "scenario": scenario,
                    "group": "ALL",
                    "model": item["model"],
                    "score": float(item["score"]),
                    "rank": rank,
                    "n_items": len([row for row in rows if row["model"] == item["model"]]),
                }
            )
    return summary_rows


def build_winners_by_category(rows: list[dict]) -> list[dict]:
    winners = []
    for category in sorted({row["category"] for row in rows}):
        category_rows = [row for row in rows if row["category"] == category]
        meta = CATEGORY_METADATA.get(category, {})
        for scenario in SCENARIO_KEYS:
            winner = max(category_rows, key=lambda row: (_safe_float(row[scenario]), row["model"]))
            winners.append(
                {
                    "category": category,
                    "scenario": scenario,
                    "winner_model": winner["model"],
                    "winner_score": float(winner[scenario]),
                    "alignment_type": meta.get("alignment_type", ""),
                    "defect_extent": meta.get("defect_extent", ""),
                }
            )
    return winners


def build_winners_overall(rows: list[dict]) -> list[dict]:
    winners = []
    for scenario in SCENARIO_KEYS:
        model_scores = {
            model: _float_mean(row[scenario] for row in rows if row["model"] == model)
            for model in sorted({row["model"] for row in rows})
        }
        winner_model = max(model_scores, key=lambda model: (model_scores[model], model))
        winners.append(
            {
                "scenario": scenario,
                "winner_model": winner_model,
                "winner_score": float(model_scores[winner_model]),
            }
        )
    return winners


def build_long_rows(rows: list[dict]) -> list[dict]:
    long_rows = []
    for row in rows:
        for score_type in SCORE_VALUE_COLUMNS:
            if score_type.startswith("S_"):
                scenario = score_type
            else:
                scenario = "fixed"
            long_rows.append(
                {
                    "category": row["category"],
                    "model": row["model"],
                    "scenario": scenario,
                    "score_type": score_type,
                    "value": float(row[score_type]),
                }
            )
    return long_rows


def _build_grouped_rows(rows: list[dict], key_names: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in key_names) + (row["model"],)].append(row)

    out_rows = []
    for group_key in sorted(grouped):
        group_rows = grouped[group_key]
        record = {key: value for key, value in zip(key_names + ["model"], group_key)}
        for score_key in GROUP_SCORE_COLUMNS:
            record[f"{score_key}_mean"] = _float_mean(row[score_key] for row in group_rows)
        record["n_categories"] = len({row["category"] for row in group_rows})
        out_rows.append(record)
    return out_rows


def build_grouped_alignment_rows(rows: list[dict]) -> list[dict]:
    return _build_grouped_rows(rows, ["alignment_type"])


def build_grouped_defect_rows(rows: list[dict]) -> list[dict]:
    return _build_grouped_rows(rows, ["defect_extent"])


def build_grouped_2x2_rows(rows: list[dict]) -> list[dict]:
    return _build_grouped_rows(rows, ["alignment_type", "defect_extent"])


def build_sensitivity_rows(rows: list[dict]) -> list[dict]:
    sensitivity_rows = []
    for w1 in SENSITIVITY_WEIGHTS:
        for w2 in SENSITIVITY_WEIGHTS:
            combo_rows = []
            for row in rows:
                a_score = float(w1 * row["image_auroc"] + (1.0 - w1) * row["pixel_auroc"])
                e_score = float(1.0 - (w2 * row["T_cost"] + (1.0 - w2) * row["R_cost"]))
                for scenario, w3 in SCENARIO_SPECS.items():
                    combo_rows.append(
                        {
                            "category": row["category"],
                            "model": row["model"],
                            "alignment_type": row.get("alignment_type", ""),
                            "defect_extent": row.get("defect_extent", ""),
                            "w1": float(w1),
                            "w2": float(w2),
                            "scenario": scenario,
                            "w3": float(w3),
                            "A": a_score,
                            "T_cost": float(row["T_cost"]),
                            "R_cost": float(row["R_cost"]),
                            "E": e_score,
                            "S": float(w3 * a_score + (1.0 - w3) * e_score),
                            "rank": 0,
                        }
                    )
            rank_groups = defaultdict(list)
            for combo_row in combo_rows:
                rank_groups[(combo_row["category"], combo_row["w1"], combo_row["w2"], combo_row["scenario"])].append(combo_row)
            for ranked_rows in rank_groups.values():
                _rank_rows(ranked_rows, "S", "rank")
            sensitivity_rows.extend(combo_rows)
    return sensitivity_rows


def build_sensitivity_summary(rows: list[dict]) -> list[dict]:
    summary_rows = []
    combo_keys = sorted({(row["w1"], row["w2"], row["scenario"], row["w3"]) for row in rows})
    models = sorted({row["model"] for row in rows})

    for w1, w2, scenario, w3 in combo_keys:
        combo_rows = [row for row in rows if (row["w1"], row["w2"], row["scenario"], row["w3"]) == (w1, w2, scenario, w3)]
        mean_scores = {
            model: _float_mean(row["S"] for row in combo_rows if row["model"] == model)
            for model in models
        }
        winner_model = max(mean_scores, key=lambda model: (mean_scores[model], model))
        summary_rows.append(
            {
                "summary_type": "combo_winner",
                "w1": w1,
                "w2": w2,
                "scenario": scenario,
                "w3": w3,
                "model": winner_model,
                "value": float(mean_scores[winner_model]),
            }
        )

    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        summary_rows.append(
            {
                "summary_type": "mean_rank",
                "w1": "",
                "w2": "",
                "scenario": "ALL",
                "w3": "",
                "model": model,
                "value": _float_mean(row["rank"] for row in model_rows),
            }
        )
        summary_rows.append(
            {
                "summary_type": "win_count",
                "w1": "",
                "w2": "",
                "scenario": "ALL",
                "w3": "",
                "model": model,
                "value": float(sum(1 for row in model_rows if int(row["rank"]) == 1)),
            }
        )

    return summary_rows


def aggregate_seed_raw_rows(seed_rows: list[dict]) -> list[dict]:
    if not seed_rows:
        return []

    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["category"], row["method"])].append(row)

    aggregated = []
    for (category, method), rows in sorted(grouped.items()):
        aggregated.append(
            {
                "category": category,
                "method": method,
                "img_auc": _float_mean(row["img_auc"] for row in rows),
                "px_auc": _float_mean(row["px_auc"] for row in rows),
                "prep_s": _float_mean(row["prep_s"] for row in rows),
                "infer_s": _float_mean(row["infer_s"] for row in rows),
                "ms_img": _float_mean(row["ms_img"] for row in rows),
                "n_seeds": len(rows),
            }
        )
    return aggregated


def build_seed_runs_rows(seed_scenario_rows: list[dict], seed_raw_rows: list[dict]) -> list[dict]:
    raw_lookup = {
        (int(row["seed"]), row["category"], row["method"]): row
        for row in seed_raw_rows
    }
    output_rows = []
    for row in seed_scenario_rows:
        raw = raw_lookup[(int(row["seed"]), row["category"], row["model"])]
        output_rows.append(
            {
                "seed": int(row["seed"]),
                "category": row["category"],
                "model": row["model"],
                "image_auroc": float(raw["img_auc"]),
                "pixel_auroc": float(raw["px_auc"]),
                "prep_time": float(raw["prep_s"]),
                "infer_time": float(raw["infer_s"]),
                "ms_per_img": float(raw["ms_img"]),
                "A_fixed": float(row["A_fixed"]),
                "E_fixed": float(row["E_fixed"]),
                "S_perf": float(row["S_perf"]),
                "S_bal": float(row["S_bal"]),
                "S_eff": float(row["S_eff"]),
                "rank_perf": int(row["rank_perf"]),
                "rank_bal": int(row["rank_bal"]),
                "rank_eff": int(row["rank_eff"]),
                "alignment_type": row.get("alignment_type", ""),
                "defect_extent": row.get("defect_extent", ""),
            }
        )
    return output_rows


def build_seed_aggregated_summary(seed_runs_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in seed_runs_rows:
        grouped[("category_model", row["category"], row["model"])].append(row)
        grouped[("overall_model", "ALL", row["model"])].append(row)

    summary_rows = []
    metrics = [
        "image_auroc",
        "pixel_auroc",
        "prep_time",
        "ms_per_img",
        "A_fixed",
        "E_fixed",
        "S_perf",
        "S_bal",
        "S_eff",
    ]
    for (summary_type, category, model), rows in sorted(grouped.items()):
        summary_row = {
            "summary_type": summary_type,
            "category": category,
            "model": model,
            "n_seeds": len({row["seed"] for row in rows}),
        }
        for metric in metrics:
            summary_row[f"{metric}_mean"] = _float_mean(row[metric] for row in rows)
            summary_row[f"{metric}_std"] = _float_std(row[metric] for row in rows)
        summary_rows.append(summary_row)
    return summary_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
