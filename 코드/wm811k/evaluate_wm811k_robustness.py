"""Evaluate WM-811K inference robustness on held-out qualitative demo wafers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnose_wm811k import (
    assess_wafer_ood_batch,
    load_checkpoint,
    load_ood_reference,
    predict_wafer_batch,
    resize_nearest,
    sha256_file,
)
from select_demo_samples import LEGACY_CASES


SCENARIOS = (
    {"scenario": "active_die_dropout_1pct", "label": "활성 die 1% 누락", "kind": "active_dropout", "amount": 0.01, "severity": "mild", "guardrail_metric": "prediction_stability", "guardrail_minimum": 0.90},
    {"scenario": "active_die_dropout_5pct", "label": "활성 die 5% 누락", "kind": "active_dropout", "amount": 0.05, "severity": "severe", "guardrail_metric": "changed_prediction_capture", "guardrail_minimum": 0.80},
    {"scenario": "state_flip_1pct", "label": "die 상태 1% 반전", "kind": "state_flip", "amount": 0.01, "severity": "mild", "guardrail_metric": "prediction_stability", "guardrail_minimum": 0.85},
    {"scenario": "state_flip_5pct", "label": "die 상태 5% 반전", "kind": "state_flip", "amount": 0.05, "severity": "severe", "guardrail_metric": "changed_prediction_capture", "guardrail_minimum": 0.80},
    {"scenario": "block_sensor_loss_12px", "label": "12×12 블록 손실", "kind": "block_loss", "amount": 12, "severity": "severe", "guardrail_metric": "changed_prediction_capture", "guardrail_minimum": 0.80},
    {"scenario": "translation_shift_4px", "label": "4px 위치 이동", "kind": "translation", "amount": 4, "severity": "severe", "guardrail_metric": "changed_prediction_capture", "guardrail_minimum": 0.80},
)


def perturb_wafer(wafer: np.ndarray, *, kind: str, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Return a deterministic categorical corruption without mutating the input."""
    corrupted = resize_nearest(wafer).copy()
    active = np.argwhere(corrupted > 0)
    if kind in {"active_dropout", "state_flip"}:
        count = min(len(active), max(1, int(round(len(active) * float(amount)))))
        selected = active[rng.choice(len(active), size=count, replace=False)]
        rows, columns = selected[:, 0], selected[:, 1]
        if kind == "active_dropout":
            corrupted[rows, columns] = 0
        else:
            corrupted[rows, columns] = 3 - corrupted[rows, columns]
    elif kind == "block_loss":
        side = int(amount)
        if not 1 <= side <= min(corrupted.shape):
            raise ValueError("블록 손실 크기가 웨이퍼 범위를 벗어났습니다.")
        top = int(rng.integers(0, corrupted.shape[0] - side + 1))
        left = int(rng.integers(0, corrupted.shape[1] - side + 1))
        corrupted[top : top + side, left : left + side] = 0
    elif kind == "translation":
        pixels = int(amount)
        if pixels <= 0 or pixels >= min(corrupted.shape):
            raise ValueError("위치 이동 크기가 웨이퍼 범위를 벗어났습니다.")
        dy = pixels if bool(rng.integers(0, 2)) else -pixels
        dx = pixels if bool(rng.integers(0, 2)) else -pixels
        shifted = np.zeros_like(corrupted)
        source_y = slice(max(0, -dy), min(corrupted.shape[0], corrupted.shape[0] - dy))
        source_x = slice(max(0, -dx), min(corrupted.shape[1], corrupted.shape[1] - dx))
        target_y = slice(max(0, dy), min(corrupted.shape[0], corrupted.shape[0] + dy))
        target_x = slice(max(0, dx), min(corrupted.shape[1], corrupted.shape[1] + dx))
        shifted[target_y, target_x] = corrupted[source_y, source_x]
        corrupted = shifted
    else:
        raise ValueError(f"지원하지 않는 교란 유형입니다: {kind}")
    return corrupted


def load_demo_wafers(demo_dir: Path) -> tuple[dict, list[np.ndarray]]:
    """Load the 27 legacy demo cases this stress test was defined on."""
    manifest = json.loads((demo_dir / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest["samples"]) != int(manifest["sample_count"]):
        raise ValueError("데모 manifest 샘플 수가 기록과 다릅니다.")
    legacy = [item for item in manifest["samples"] if item["selection_case"] in LEGACY_CASES]
    if len(legacy) != 27:
        raise ValueError("기존 27개 정성 예시(고신뢰·검토 대상 정답, 대표 오분류)가 아닙니다.")
    wafers = [np.load(demo_dir / item["npy_file"], allow_pickle=False) for item in legacy]
    return {**manifest, "samples": legacy, "sample_count": len(legacy)}, wafers


def evaluate_scenario(*, scenario: dict, wafers: list[np.ndarray], manifest: dict, baseline_predictions: list[dict], baseline_reviews: list[bool], model, class_names: list[str], ood_reference: dict, temperature: float, review_threshold: float, repeats: int, seed: int) -> tuple[dict, pd.DataFrame]:
    corrupted_wafers: list[np.ndarray] = []
    metadata: list[tuple[int, int]] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat * 10_000)
        for sample_index, wafer in enumerate(wafers):
            corrupted_wafers.append(perturb_wafer(wafer, kind=scenario["kind"], amount=scenario["amount"], rng=rng))
            metadata.append((repeat, sample_index))
    predictions = predict_wafer_batch(model, class_names, corrupted_wafers, temperature=temperature, batch_size=128)
    ood_results = assess_wafer_ood_batch(model, class_names, corrupted_wafers, ood_reference, batch_size=128)
    rows: list[dict] = []
    for (repeat, sample_index), prediction, ood in zip(metadata, predictions, ood_results):
        baseline = baseline_predictions[sample_index]
        changed = int(prediction["predicted_class"]) != int(baseline["predicted_class"])
        review = float(prediction["confidence"]) < review_threshold or ood["ood_status"] != "in_distribution"
        sample = manifest["samples"][sample_index]
        rows.append({
            "scenario": scenario["scenario"], "scenario_label": scenario["label"], "repeat": repeat,
            "sample_id": sample["sample_id"], "true_class": sample["true_class"], "selection_case": sample["selection_case"],
            "baseline_prediction": baseline["predicted_label"], "corrupted_prediction": prediction["predicted_label"],
            "prediction_changed": changed, "baseline_confidence": float(baseline["confidence"]),
            "baseline_review_or_hold": baseline_reviews[sample_index],
            "corrupted_confidence": float(prediction["confidence"]),
            "confidence_change": float(prediction["confidence"]) - float(baseline["confidence"]),
            "ood_score": float(ood["ood_score"]), "ood_status": ood["ood_status"], "review_or_hold": bool(review),
            "changed_prediction_captured": bool(changed and review),
        })
    records = pd.DataFrame(rows)
    changed_count = int(records["prediction_changed"].sum())
    automatic_origin_changed = records["prediction_changed"] & ~records["baseline_review_or_hold"]
    capture = float(records.loc[records["prediction_changed"], "review_or_hold"].mean()) if changed_count else 1.0
    metrics = {
        "scenario": scenario["scenario"], "scenario_label": scenario["label"], "severity": scenario["severity"],
        "rows": int(len(records)), "prediction_stability": float(1.0 - records["prediction_changed"].mean()),
        "changed_prediction_count": changed_count, "changed_prediction_capture": capture,
        "unreviewed_changed_count": int((records["prediction_changed"] & ~records["review_or_hold"]).sum()),
        "baseline_automatic_changed_count": int(automatic_origin_changed.sum()),
        "baseline_automatic_changed_capture": float(records.loc[automatic_origin_changed, "review_or_hold"].mean()) if automatic_origin_changed.any() else None,
        "review_or_hold_rate": float(records["review_or_hold"].mean()),
        "ood_hold_rate": float((records["ood_status"] == "out_of_distribution").mean()),
        "mean_confidence_change": float(records["confidence_change"].mean()),
        "guardrail_metric": scenario["guardrail_metric"], "guardrail_minimum": float(scenario["guardrail_minimum"]),
    }
    metrics["guardrail_value"] = float(metrics[scenario["guardrail_metric"]])
    metrics["guardrail_pass"] = bool(metrics["guardrail_value"] >= metrics["guardrail_minimum"])
    return metrics, records


def save_dashboard(summary: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    labels = summary["scenario_label"].tolist()
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].bar(positions, summary["prediction_stability"], color=["#2E75B6" if value else "#C00000" for value in summary["guardrail_pass"]])
    axes[0].set_title("교란 후 예측 유지율"); axes[0].set_ylim(0, 1.05); axes[0].set_xticks(positions, labels, rotation=28, ha="right"); axes[0].set_ylabel("비율"); axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(positions, summary["changed_prediction_capture"], color="#70AD47")
    axes[1].set_title("변경된 예측의 검토·보류 포착률"); axes[1].set_ylim(0, 1.05); axes[1].set_xticks(positions, labels, rotation=28, ha="right"); axes[1].set_ylabel("비율"); axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("WM-811K 입력 오류 강건성 스트레스 테스트"); figure.tight_layout(); figure.savefig(output_path, dpi=160, bbox_inches="tight"); plt.close(figure)


def run_evaluation(project_dir: Path, output_dir: Path, repeats: int, seed: int) -> dict:
    selected_dir = project_dir / "결과물" / "wm811k" / "selected_model_results"
    demo_dir = project_dir / "결과물" / "wm811k" / "demo_samples"
    ood_dir = project_dir / "결과물" / "wm811k" / "ood_results"
    manifest, wafers = load_demo_wafers(demo_dir)
    model, class_names, _ = load_checkpoint(selected_dir / "best_model.pt")
    ood_reference = load_ood_reference(ood_dir / "ood_reference.json", ood_dir / "ood_reference.npz")
    policy = json.loads((selected_dir / "uncertainty_policy.json").read_text(encoding="utf-8"))
    temperature, review_threshold = float(policy["temperature"]), float(policy["review_threshold"])
    baseline_predictions = predict_wafer_batch(model, class_names, wafers, temperature=temperature)
    baseline_ood = assess_wafer_ood_batch(model, class_names, wafers, ood_reference)
    baseline_reviews = [
        float(prediction["confidence"]) < review_threshold or ood["ood_status"] != "in_distribution"
        for prediction, ood in zip(baseline_predictions, baseline_ood)
    ]
    summaries, record_frames = [], []
    for scenario_index, scenario in enumerate(SCENARIOS):
        metrics, records = evaluate_scenario(scenario=scenario, wafers=wafers, manifest=manifest, baseline_predictions=baseline_predictions, baseline_reviews=baseline_reviews, model=model, class_names=class_names, ood_reference=ood_reference, temperature=temperature, review_threshold=review_threshold, repeats=repeats, seed=seed + scenario_index * 1_000_000)
        summaries.append(metrics); record_frames.append(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_table, detail_table = pd.DataFrame(summaries), pd.concat(record_frames, ignore_index=True)
    summary_table.to_csv(output_dir / "stress_test_summary.csv", index=False)
    detail_table.to_csv(output_dir / "stress_test_records.csv", index=False)
    save_dashboard(summary_table, output_dir / "robustness_dashboard.png")
    result = {
        "schema_version": 1, "purpose": "post_selection_input_error_stress_test", "used_for_model_selection": False,
        "thresholds_retuned": False, "source_split": "test_qualitative_demo_subset", "representative_population_estimate": False,
        "sample_count": len(wafers), "repeats": repeats, "evaluated_rows": int(len(detail_table)), "seed": seed,
        "baseline_review_count": sum(baseline_reviews),
        "guardrails_declared_before_evaluation": True,
        "artifact_sha256": {
            "checkpoint": sha256_file(selected_dir / "best_model.pt"),
            "demo_manifest": sha256_file(demo_dir / "manifest.json"),
            "ood_reference_json": sha256_file(ood_dir / "ood_reference.json"),
            "ood_reference_npz": sha256_file(ood_dir / "ood_reference.npz"),
            "uncertainty_policy": sha256_file(selected_dir / "uncertainty_policy.json"),
        },
        "scenario_count": len(summaries), "guardrail_pass_count": int(summary_table["guardrail_pass"].sum()),
        "guardrail_fail_count": int((~summary_table["guardrail_pass"]).sum()),
        "failed_scenarios": summary_table.loc[~summary_table["guardrail_pass"], "scenario"].tolist(), "scenarios": summaries,
        "limitations": ["27개 test 정성 예시는 전체 WM-811K 분포의 대표 표본이 아닙니다.", "합성 입력 오류는 실제 장비·센서 고장 분포를 대체하지 않습니다.", "이 결과는 모델 선택이나 임계값 조정에 사용하지 않았습니다."],
    }
    (output_dir / "robustness_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text("# WM-811K 입력 오류 강건성 결과\n\n선택이 끝난 모델에 실제 test 정성 예시 27개의 합성 교란을 주입했습니다. 결과는 모델 선택과 임계값 조정에 사용하지 않습니다.\n\n- `stress_test_summary.csv`: 교란 조건별 요약\n- `stress_test_records.csv`: 반복·웨이퍼별 상세 기록\n- `robustness_dashboard.png`: 예측 유지율과 안전장치 포착률\n\n이 실험은 실제 외부 장비·센서 고장 검증을 대체하지 않습니다.\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats는 1 이상이어야 합니다.")
    output_dir = args.output_dir or args.project_dir / "결과물" / "wm811k" / "robustness_results"
    print(json.dumps(run_evaluation(args.project_dir.resolve(), output_dir.resolve(), args.repeats, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
