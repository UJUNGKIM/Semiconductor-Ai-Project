"""Calibrate and validate a train-only SECOM batch safety circuit breaker."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from batch_safety_gate import evaluate_gate_from_rates
from safety_policy import model_alert_mask
from train_compare_models import RANDOM_STATE, TEST_SIZE, load_data


PROFILE = "balanced_f2"
MINIMUM_BATCH_ROWS = 20
SIGNAL_FLOORS = {
    "ood_any_rate": {"warning_floor": 0.20, "stop_floor": 0.35},
    "ood_severe_rate": {"warning_floor": 0.05, "stop_floor": 0.10},
    "model_disagreement_rate": {"warning_floor": 0.50, "stop_floor": 0.65},
}


def leave_one_out_ood_rates(
    raw_train: pd.DataFrame, reference: dict
) -> dict[str, float]:
    features = reference["input_features"]
    values = raw_train[features].to_numpy(dtype=float, copy=True)
    observed = np.isfinite(values)
    lower = np.asarray(reference["q01"], dtype=float)
    upper = np.asarray(reference["q99"], dtype=float)
    outside = observed & ((values < lower) | (values > upper))
    range_rate = outside.sum(axis=1) / np.maximum(observed.sum(axis=1), 1)
    missing_rate = raw_train[reference["raw_features"]].isna().mean(axis=1).to_numpy()
    embedding = np.asarray(reference["reference_embedding"], dtype=float)
    distance = NearestNeighbors(n_neighbors=2).fit(embedding).kneighbors(
        embedding
    )[0][:, 1]
    limits = reference["ood_thresholds"]
    severe = (
        (distance > limits["distance_outlier"])
        | (missing_rate > limits["missing_outlier"])
        | (range_rate > limits["range_outlier"])
    )
    review = severe | (
        (distance > limits["distance_review"])
        | (missing_rate > limits["missing_review"])
        | (range_rate > limits["range_review"])
    )
    return {
        "ood_any_rate": float(review.mean()),
        "ood_severe_rate": float(severe.mean()),
    }


def plot_validation(validation: pd.DataFrame, output_path: Path) -> None:
    status_order = ["PASS", "CAUTION", "STOP", "ROW_LEVEL_ONLY"]
    table = (
        validation.groupby(["scenario", "status"]).size().unstack(fill_value=0)
    )
    for status in status_order:
        if status not in table:
            table[status] = 0
    table = table[status_order]
    proportions = table.div(table.sum(axis=1), axis=0)
    colors = {
        "PASS": "#2e8b57",
        "CAUTION": "#f0ad4e",
        "STOP": "#d1495b",
        "ROW_LEVEL_ONLY": "#2878b5",
    }
    axis = proportions.plot(
        kind="bar",
        stacked=True,
        figsize=(12.5, 6.2),
        color=[colors[column] for column in proportions.columns],
    )
    axis.set_title("SECOM batch circuit-breaker decisions under sensor corruption")
    axis.set_xlabel("Stress scenario")
    axis.set_ylabel("Decision share")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=55, labelsize=8)
    axis.legend(title="Gate decision", loc="upper left", bbox_to_anchor=(1.01, 1))
    axis.grid(axis="y", alpha=0.25)
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def english_scenario(kind: str, severity: float) -> str:
    if kind == "baseline":
        return "Baseline"
    if kind == "random_missing":
        return f"Missing {severity:.0%}"
    if kind == "gaussian_noise":
        return f"Noise {severity:.2f} IQR"
    if kind == "sensor_spike":
        return f"Spike {severity:.0%}"
    return f"Column loss {severity:.0%}"


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    model_dir = project_dir / "결과물" / "secom" / "dual_model_results"
    advanced_dir = project_dir / "결과물" / "secom" / "advanced_diagnostics"
    safety_dir = project_dir / "결과물" / "secom" / "safety_policy_results"
    robustness_dir = project_dir / "결과물" / "secom" / "robustness_results"
    output_dir = project_dir / "결과물" / "secom" / "batch_safety_gate"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] fixed train에서 leave-one-out OOD 기준율을 계산합니다...")
    X_raw, y, _ = load_data(project_dir)
    X_train, _, _, _ = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    reference = joblib.load(advanced_dir / "advanced_reference.joblib")
    baseline_rates = leave_one_out_ood_rates(X_train, reference)

    print("[2/5] OOF 모델 불일치 기준율을 계산합니다...")
    cat_bundle = joblib.load(model_dir / "catboost_model.joblib")
    xgb_bundle = joblib.load(model_dir / "xgboost_model.joblib")
    cat_threshold = float(cat_bundle["operating_thresholds"][PROFILE])
    xgb_threshold = float(xgb_bundle["operating_thresholds"][PROFILE])
    oof = pd.read_csv(model_dir / "oof_predictions.csv")
    cat_alert = model_alert_mask(oof["catboost_probability"], cat_threshold)
    xgb_alert = model_alert_mask(oof["xgboost_probability"], xgb_threshold)
    baseline_rates["model_disagreement_rate"] = float(
        (cat_alert != xgb_alert).mean()
    )
    signals = {
        name: {"baseline_rate": baseline_rates[name], **floors}
        for name, floors in SIGNAL_FLOORS.items()
    }
    gate_reference = {
        "version": 1,
        "profile": PROFILE,
        "fit_scope": "fixed_train_and_repeated_oof_only",
        "test_used_for_threshold_selection": False,
        "minimum_batch_rows": MINIMUM_BATCH_ROWS,
        "adaptive_limit_method": (
            "baseline rate + one-sided normal limit; z=2.326 warning, z=3.090 stop"
        ),
        "signals": signals,
        "row_level_severe_ood_always_held": True,
        "automatic_decision_on_stop": False,
    }
    (output_dir / "batch_gate_reference.json").write_text(
        json.dumps(gate_reference, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[3/5] 깨끗한 고정 test 배치가 오작동 차단되지 않는지 확인합니다...")
    test = pd.read_csv(safety_dir / "review_queue_test.csv")
    clean_rates = {
        "ood_any_rate": float((test["ood_status"] != "in_distribution").mean()),
        "ood_severe_rate": float(
            (test["ood_status"] == "out_of_distribution").mean()
        ),
        "model_disagreement_rate": float(
            (test["catboost_alert"] != test["xgboost_alert"]).mean()
        ),
    }
    clean_result = evaluate_gate_from_rates(clean_rates, len(test), gate_reference)

    print("[4/5] 센서 스트레스 50개 배치에 차단기를 적용합니다...")
    stress = pd.read_csv(robustness_dir / "stress_test_replicates.csv")
    validation_rows = []
    for _, row in stress.iterrows():
        rates = {
            "ood_any_rate": float(row["ood_review_rate"]),
            "ood_severe_rate": float(row["ood_severe_rate"]),
            "model_disagreement_rate": float(row["model_disagreement_rate"]),
        }
        result = evaluate_gate_from_rates(rates, len(test), gate_reference)
        validation_rows.append(
            {
                "scenario": english_scenario(str(row["kind"]), float(row["severity"])),
                "scenario_korean": row["scenario"],
                "kind": row["kind"],
                "severity": row["severity"],
                "replicate": row["replicate"],
                **rates,
                "status": result["status"],
                "automatic_decision_allowed": result["automatic_decision_allowed"],
                "reasons": "; ".join(result["reasons"]),
            }
        )
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(output_dir / "gate_stress_validation.csv", index=False)
    plot_validation(validation, output_dir / "gate_stress_validation.png")

    stressed = validation.loc[validation["kind"] != "baseline"]
    condition_summary = (
        stressed.groupby(["scenario", "scenario_korean", "kind", "severity"])
        .agg(
            replicates=("status", "size"),
            stop_rate=("status", lambda values: float((values == "STOP").mean())),
            caution_rate=(
                "status", lambda values: float((values == "CAUTION").mean())
            ),
            pass_rate=("status", lambda values: float((values == "PASS").mean())),
        )
        .reset_index()
    )
    condition_summary.to_csv(output_dir / "gate_condition_summary.csv", index=False)
    stopped_conditions = condition_summary.loc[
        condition_summary["stop_rate"] > 0, "scenario_korean"
    ].tolist()
    summary = {
        "version": 1,
        "test_used_for_threshold_selection": False,
        "reference_fit_scope": gate_reference["fit_scope"],
        "clean_test_decision": clean_result,
        "stress_batches": int(len(stressed)),
        "stress_stop_count": int((stressed["status"] == "STOP").sum()),
        "stress_caution_count": int((stressed["status"] == "CAUTION").sum()),
        "stress_pass_count": int((stressed["status"] == "PASS").sum()),
        "conditions_with_stop": stopped_conditions,
        "limitation": (
            "배치 차단은 자동판정을 보류하는 안전장치이지 센서 고장 원인을 진단하는 "
            "기능이 아닙니다. 실제 공정에서 경보율과 대응 시간을 전향 검증해야 합니다."
        ),
    }
    (output_dir / "batch_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = f"""# SECOM 배치 안전 차단기

- 기준 산출: fixed train leave-one-out OOD + 반복 OOF 모델 불일치
- test 기반 기준 선택: 사용하지 않음
- 최소 배치 크기: {MINIMUM_BATCH_ROWS}행
- 깨끗한 고정 test 결정: {clean_result['status']}
- 센서 교란 배치: {len(stressed)}개
- STOP: {summary['stress_stop_count']}개
- CAUTION: {summary['stress_caution_count']}개
- PASS: {summary['stress_pass_count']}개

STOP이면 배치 전체 자동판정을 금지하고 전문가 검토·재측정을 요구합니다. 20행 미만은
배치 비율이 불안정하므로 행 단위 OOD 보류만 적용합니다. 이 기능은 고장 원인 진단이
아니며 실제 공정에서 경보율과 대응 시간을 전향 검증해야 합니다.
"""
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")

    print("[5/5] 완료")
    print(
        f"  clean test={clean_result['status']}, stress STOP="
        f"{summary['stress_stop_count']}/{len(stressed)}"
    )
    print(f"  결과: {output_dir}")


if __name__ == "__main__":
    main()
