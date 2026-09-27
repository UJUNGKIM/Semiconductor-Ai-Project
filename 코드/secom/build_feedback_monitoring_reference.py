"""Build train-OOF reference and a non-operational feedback-monitoring replay."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feedback_monitoring import analyze_feedback_history, build_candidate_manifest
from monitoring_log import REQUIRED, validate_monitoring_log

PROJECT = Path(__file__).resolve().parents[2]
DUAL_DIR = PROJECT / "결과물" / "secom" / "dual_model_results"
SAFETY_DIR = PROJECT / "결과물" / "secom" / "safety_policy_results"
ADVANCED_DIR = PROJECT / "결과물" / "secom" / "advanced_diagnostics"
OUTPUT_DIR = PROJECT / "결과물" / "secom" / "feedback_monitoring"


def metrics(labels: np.ndarray, alerts: np.ndarray) -> dict:
    tp = int(np.sum((labels == 1) & alerts))
    fp = int(np.sum((labels == 0) & alerts))
    fn = int(np.sum((labels == 1) & ~alerts))
    tn = int(np.sum((labels == 0) & ~alerts))
    return {
        "rows": len(labels),
        "defects": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": float(tp / (tp + fp)),
        "recall": float(tp / (tp + fn)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(DUAL_DIR / "oof_predictions.csv")
    policy = json.loads(
        (SAFETY_DIR / "safety_policy_summary.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (ADVANCED_DIR / "metadata.json").read_text(encoding="utf-8")
    )
    thresholds = policy["thresholds"]
    cat_alert = oof["catboost_probability"].to_numpy() >= thresholds["CatBoost"]
    xgb_alert = oof["xgboost_probability"].to_numpy() >= thresholds["XGBoost"]
    alerts = cat_alert | xgb_alert
    labels = oof["label"].to_numpy(dtype=int)
    reference = {
        "version": 1,
        "created_from": "repeated_5x3_oof_train_only",
        "test_labels_used_for_guardrails": False,
        "alert_rule": "CatBoost OR XGBoost alert at OOF-selected thresholds",
        "profile": "balanced_f2",
        "thresholds": thresholds,
        "oof_reference": metrics(labels, alerts),
        "minimum_evidence": {"batches": 3, "rows": 200, "defects": 30},
        "guardrails": {
            "recall_drop_margin": 0.15,
            "persistent_batches": 3,
            "prevalence_ratio_warning": 2.0,
        },
        "model_hashes": metadata["model_hashes"],
        "automatic_retraining_allowed": False,
    }
    (OUTPUT_DIR / "feedback_reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = []
    chunks = np.array_split(np.arange(len(oof)), 5)
    for number, indices in enumerate(chunks, start=1):
        chunk_labels = labels[indices]
        chunk_cat = cat_alert[indices]
        chunk_xgb = xgb_alert[indices]
        chunk_alert = alerts[indices]
        both = int((chunk_cat & chunk_xgb).sum())
        one = int((chunk_cat ^ chunk_xgb).sum())
        actual = int(chunk_labels.sum())
        true_positive = int(((chunk_labels == 1) & chunk_alert).sum())
        row_count = len(indices)
        rows.append(
            {
                "log_schema_version": 2,
                "recorded_at": pd.Timestamp("2026-01-01", tz="Asia/Seoul")
                + pd.Timedelta(days=number - 1),
                "batch_id": f"oof-replay-{number:02d}",
                "source_name": "train_oof_replay",
                "profile": "balanced_f2",
                "row_count": row_count,
                "both_models_defect": both,
                "one_model_defect": one,
                "both_models_normal": row_count - both - one,
                "defect_alert_rate": (both + one) / row_count,
                "model_disagreement_rate": one / row_count,
                "ood_any_rate": 0.0,
                "ood_severe_rate": 0.0,
                "gate_status": "PASS",
                "automatic_decision_allowed": True,
                "review_target_count": both + one,
                "input_digest": f"oof-replay-{number:052d}",
                "catboost_model_hash": metadata["model_hashes"]["CatBoost"],
                "xgboost_model_hash": metadata["model_hashes"]["XGBoost"],
                "outcome_confirmed": True,
                "confirmed_defects": actual,
                "operator_id": "OOF_REPLAY",
                "note": "train OOF 집계 재생이며 실제 운영 이력이 아님",
                "confirmed_alerted_defects": true_positive,
                "feedback_evidence": "oof_replay",
            }
        )
    replay = validate_monitoring_log(pd.DataFrame(rows, columns=REQUIRED))
    replay.to_csv(OUTPUT_DIR / "oof_feedback_replay.csv", index=False, encoding="utf-8-sig")
    summary = analyze_feedback_history(replay, reference)
    (OUTPUT_DIR / "feedback_demo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "retraining_candidate_manifest_demo.json").write_text(
        json.dumps(build_candidate_manifest(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    chart = replay.copy()
    chart["batch_recall"] = chart["confirmed_alerted_defects"] / chart[
        "confirmed_defects"
    ]
    chart["actual_defect_rate"] = chart["confirmed_defects"] / chart["row_count"]
    ax = chart.plot(
        x="batch_id", y=["batch_recall", "actual_defect_rate"], marker="o", figsize=(9, 4)
    )
    ax.axhline(summary["recall_guardrail"], color="red", linestyle="--", label="recall guardrail")
    ax.set_ylim(0, 1); ax.grid(alpha=0.25); ax.legend(); plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "feedback_replay.png", dpi=160); plt.close()
    (OUTPUT_DIR / "summary.md").write_text(
        "# SECOM 검수 피드백 모니터링\n\n"
        "기준은 고정 test가 아니라 train 반복 OOF 예측으로 사전 생성했습니다. "
        "데모는 OOF 집계 재생이며 실제 운영 성능이 아닙니다. 자동 재학습은 금지됩니다.\n\n"
        f"- OOF 경보 재현율: {reference['oof_reference']['recall']:.4f}\n"
        f"- 감시 하한: {summary['recall_guardrail']:.4f}\n"
        f"- 최소 근거: 3배치, 200행, 실제 불량 30건\n"
        f"- 재생 상태: {summary['status']}\n",
        encoding="utf-8",
    )
    print(f"검수 피드백 기준 생성 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
