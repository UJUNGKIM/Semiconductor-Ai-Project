"""Run post-selection Grad-CAM sanity checks on the 27 qualitative WM-811K samples."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from diagnose_wm811k import load_checkpoint, predict_wafer_batch, predict_with_gradcam, resize_nearest
from select_demo_samples import LEGACY_CASES


MASK_FRACTION = 0.10
RANDOM_REPEATS = 10
RANDOM_SEED = 42


def attribution_mask(wafer: np.ndarray, heatmap: np.ndarray, fraction: float, *, largest: bool = True) -> np.ndarray:
    """Return a copy with a fixed fraction of active die removed by attribution rank."""
    if not 0 < fraction < 1:
        raise ValueError("fraction은 0과 1 사이여야 합니다.")
    wafer = resize_nearest(wafer)
    heatmap = np.asarray(heatmap, dtype=float)
    if heatmap.shape != wafer.shape or not np.isfinite(heatmap).all():
        raise ValueError("heatmap 형상 또는 값이 잘못되었습니다.")
    active = np.flatnonzero((wafer > 0).ravel())
    if not len(active):
        raise ValueError("활성 die가 없습니다.")
    count = max(1, int(round(len(active) * fraction)))
    order = np.argsort(heatmap.ravel()[active], kind="stable")
    selected = active[order[-count:] if largest else order[:count]]
    masked = wafer.copy().ravel()
    masked[selected] = 0
    return masked.reshape(wafer.shape)


def random_active_mask(wafer: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    wafer = resize_nearest(wafer)
    active = np.flatnonzero((wafer > 0).ravel())
    count = max(1, int(round(len(active) * fraction)))
    selected = rng.choice(active, size=count, replace=False)
    masked = wafer.copy().ravel()
    masked[selected] = 0
    return masked.reshape(wafer.shape)


def _correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    x, y = first[mask].astype(float), second[mask].astype(float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def evaluate(checkpoint_path: Path, manifest_path: Path, sample_dir: Path, output_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The audit keeps its original population: the 27 legacy cases, not the
    # typical/boundary examples added later for the dashboard gallery.
    manifest = {**manifest, "samples": [item for item in manifest["samples"] if item["selection_case"] in LEGACY_CASES]}
    if manifest.get("used_for_model_selection") is not False or len(manifest["samples"]) != 27:
        raise ValueError("모델 선택 후 고정한 기존 27개 정성 예시가 아닙니다.")
    model, class_names, _ = load_checkpoint(checkpoint_path)
    temperature = float(manifest["temperature"])
    wafers = [np.load(sample_dir / item["npy_file"], allow_pickle=False) for item in manifest["samples"]]
    rng = np.random.default_rng(RANDOM_SEED)

    original = []
    top_masked = []
    bottom_masked = []
    random_masked = []
    random_owner = []
    for sample_index, wafer in enumerate(wafers):
        result = predict_with_gradcam(model, class_names, wafer, temperature=temperature)
        original.append(result)
        top_masked.append(attribution_mask(wafer, result["heatmap"], MASK_FRACTION, largest=True))
        bottom_masked.append(attribution_mask(wafer, result["heatmap"], MASK_FRACTION, largest=False))
        for _ in range(RANDOM_REPEATS):
            random_masked.append(random_active_mask(wafer, MASK_FRACTION, rng))
            random_owner.append(sample_index)

    top_predictions = predict_wafer_batch(model, class_names, top_masked, temperature=temperature)
    bottom_predictions = predict_wafer_batch(model, class_names, bottom_masked, temperature=temperature)
    random_predictions = predict_wafer_batch(model, class_names, random_masked, temperature=temperature)

    torch.manual_seed(RANDOM_SEED)
    randomized_model = copy.deepcopy(model)
    randomized_model.classifier[-1].reset_parameters()
    randomized_model.eval()
    randomized_heatmaps = [
        predict_with_gradcam(randomized_model, class_names, wafer, target_class=int(base["predicted_class"]), temperature=temperature)["heatmap"]
        for wafer, base in zip(wafers, original)
    ]

    rows = []
    for index, (item, wafer, base, top, bottom, randomized_heatmap) in enumerate(zip(manifest["samples"], wafers, original, top_predictions, bottom_predictions, randomized_heatmaps)):
        target = int(base["predicted_class"])
        baseline_confidence = float(base["probabilities"][target])
        random_values = [float(prediction["probabilities"][target]) for owner, prediction in zip(random_owner, random_predictions) if owner == index]
        random_drop = baseline_confidence - float(np.mean(random_values))
        top_drop = baseline_confidence - float(top["probabilities"][target])
        bottom_drop = baseline_confidence - float(bottom["probabilities"][target])
        resized = resize_nearest(wafer)
        active = resized > 0
        defects = resized == 2
        heatmap = np.asarray(base["heatmap"], dtype=float)
        total_mass = float(heatmap.sum())
        active_mass = float(heatmap[active].sum() / total_mass) if total_mass > 0 else 0.0
        active_fraction = float(active.mean())
        within_active_mass = float(heatmap[active].sum())
        defect_mass = float(heatmap[defects].sum() / within_active_mass) if within_active_mass > 0 else 0.0
        defect_fraction = float(defects.sum() / max(active.sum(), 1))
        rows.append({"sample_id": item["sample_id"], "true_class": item["true_class"], "predicted_class": base["predicted_label"], "selection_case": item["selection_case"], "baseline_confidence": baseline_confidence, "active_attribution_mass": active_mass, "active_area_fraction": active_fraction, "active_attribution_lift": active_mass / active_fraction if active_fraction else np.nan, "defect_attribution_mass_within_active": defect_mass, "defect_area_fraction_within_active": defect_fraction, "defect_attribution_lift": defect_mass / defect_fraction if defect_fraction else np.nan, "top10_confidence_drop": top_drop, "random10_confidence_drop_mean": random_drop, "bottom10_confidence_drop": bottom_drop, "top_vs_random_advantage": top_drop - random_drop, "top_vs_bottom_advantage": top_drop - bottom_drop, "top10_prediction_changed": int(top["predicted_class"]) != target, "randomized_head_heatmap_correlation": _correlation(heatmap, np.asarray(randomized_heatmap), active), "heatmap_nonzero": total_mass > 0})

    records = pd.DataFrame(rows)
    finite_correlations = records["randomized_head_heatmap_correlation"].dropna()
    summary = {
        "schema_version": 1,
        "purpose": "post_selection_qualitative_gradcam_sanity_check",
        "used_for_model_selection": False,
        "thresholds_retuned": False,
        "representative_population_estimate": False,
        "parameters_declared_before_execution": True,
        "sample_count": len(records),
        "mask_fraction": MASK_FRACTION,
        "random_repeats": RANDOM_REPEATS,
        "random_seed": RANDOM_SEED,
        "mean_active_attribution_mass": float(records["active_attribution_mass"].mean()),
        "mean_active_attribution_lift": float(records["active_attribution_lift"].mean()),
        "mean_top10_confidence_drop": float(records["top10_confidence_drop"].mean()),
        "mean_random10_confidence_drop": float(records["random10_confidence_drop_mean"].mean()),
        "mean_bottom10_confidence_drop": float(records["bottom10_confidence_drop"].mean()),
        "mean_top_vs_random_advantage": float(records["top_vs_random_advantage"].mean()),
        "mean_top_vs_bottom_advantage": float(records["top_vs_bottom_advantage"].mean()),
        "top_drop_exceeds_random_rate": float((records["top_vs_random_advantage"] > 0).mean()),
        "top_drop_exceeds_bottom_rate": float((records["top_vs_bottom_advantage"] > 0).mean()),
        "top10_prediction_change_rate": float(records["top10_prediction_changed"].mean()),
        "randomized_head_heatmap_correlation_mean": float(finite_correlations.mean()),
        "randomized_head_heatmap_correlation_rows": int(len(finite_correlations)),
        "limitations": ["클래스별로 의도적으로 고른 27개 정성 예시이며 전체 test의 대표 표본이 아닙니다.", "die 제거는 실제 공정 개입이 아닌 off-manifold 민감도 점검입니다.", "분류기 head만 무작위화한 상관 점검은 완전한 모델 무작위화 검정이 아닙니다.", "Grad-CAM은 상관적 위치 단서이며 물리적 원인이나 인과관계를 증명하지 않습니다."],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_dir / "gradcam_faithfulness_records.csv", index=False)
    (output_dir / "gradcam_faithfulness_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    by_case = records.groupby("selection_case")[["top10_confidence_drop", "random10_confidence_drop_mean", "bottom10_confidence_drop"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    by_case.plot(kind="bar", ax=axes[0], color=["#e15759", "#4c78a8", "#59a14f"])
    axes[0].set(title="Confidence drop after masking 10% active die", xlabel="Demo selection case", ylabel="Mean confidence drop")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(["Top attribution", "Random", "Bottom attribution"], fontsize=8)
    axes[1].scatter(records["active_area_fraction"], records["active_attribution_mass"], c=records["top_vs_random_advantage"], cmap="coolwarm", edgecolor="black", linewidth=0.3)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set(title="Attribution mass inside active wafer", xlabel="Active die area fraction", ylabel="Active attribution mass", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(output_dir / "gradcam_faithfulness_dashboard.png", dpi=160)
    plt.close(fig)

    lines = ["# WM-811K Grad-CAM 충실도·sanity 점검", "", f"- 활성 die 영역 attribution mass: {summary['mean_active_attribution_mass']:.2%}", f"- 상위 10% 제거 신뢰도 변화: {summary['mean_top10_confidence_drop']:+.4f}", f"- 무작위 10% 제거 신뢰도 변화: {summary['mean_random10_confidence_drop']:+.4f}", f"- 하위 10% 제거 신뢰도 변화: {summary['mean_bottom10_confidence_drop']:+.4f}", f"- 상위 제거가 무작위보다 더 크게 낮춘 예시: {summary['top_drop_exceeds_random_rate']:.2%}", f"- 상위 제거가 하위보다 더 크게 낮춘 예시: {summary['top_drop_exceeds_bottom_rate']:.2%}", f"- 분류기 head 무작위화 후 heatmap 상관 평균: {summary['randomized_head_heatmap_correlation_mean']:.4f}", "", "상위 영역은 하위 영역보다 민감했지만 무작위 제거보다 일관되게 중요하지 않았습니다. 현재 결과는 강한 충실도 근거를 제공하지 않으므로 Grad-CAM은 위치 참고용 시각화로만 사용합니다.", "", "27개는 시연을 위해 클래스별로 의도적으로 고른 정성 예시이며 전체 test 대표 표본이 아닙니다. die 제거는 실제 공정 개입이 아닌 off-manifold 점검이고, Grad-CAM은 물리적 원인이나 인과관계를 증명하지 않습니다."]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.manifest, args.sample_dir, args.output_dir), ensure_ascii=False, indent=2))
