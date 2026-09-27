"""Build a train-OOF-only champion/challenger governance demonstration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from champion_challenger import evaluate_champion_challenger

PROJECT = Path(__file__).resolve().parents[2]
DUAL_DIR = PROJECT / "결과물" / "secom" / "dual_model_results"
SAFETY_DIR = PROJECT / "결과물" / "secom" / "safety_policy_results"
OUTPUT_DIR = PROJECT / "결과물" / "secom" / "champion_challenger"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "version": 1,
        "champion": "CatBoost",
        "challenger": "XGBoost",
        "random_state": 42,
        "bootstrap_draws": 2000,
        "minimum_evidence": {
            "batches": 3,
            "rows": 200,
            "defects": 30,
            "discordant_predictions": 20,
        },
        "guardrails": {
            "recall_noninferiority_margin": 0.05,
            "maximum_fpr_increase": 0.05,
            "mcnemar_alpha": 0.05,
        },
        "automatic_model_promotion_allowed": False,
        "selection_data": "future fully labelled batches only",
        "demo_data": "repeated_5x3_train_oof_predictions",
        "fixed_test_labels_used_for_rules": False,
    }
    (OUTPUT_DIR / "promotion_policy.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    oof = pd.read_csv(DUAL_DIR / "oof_predictions.csv")
    safety = json.loads(
        (SAFETY_DIR / "safety_policy_summary.json").read_text(encoding="utf-8")
    )
    thresholds = safety["thresholds"]
    indices = np.arange(len(oof))
    batch_number = np.empty(len(oof), dtype=int)
    for number, chunk in enumerate(np.array_split(indices, 5), start=1):
        batch_number[chunk] = number
    feedback = pd.DataFrame(
        {
            "batch_id": [f"oof-replay-{number:02d}" for number in batch_number],
            "actual_label": oof["label"].astype(int),
            "catboost_alert": oof["catboost_probability"] >= thresholds["CatBoost"],
            "xgboost_alert": oof["xgboost_probability"] >= thresholds["XGBoost"],
        }
    )
    report = evaluate_champion_challenger(
        feedback, config, allow_promotion_review=False
    )
    (OUTPUT_DIR / "oof_comparison_demo.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metric_names = ["precision", "recall", "f2", "false_positive_rate"]
    chart = pd.DataFrame(
        {
            "CatBoost": [report["champion_metrics"][name] for name in metric_names],
            "XGBoost": [report["challenger_metrics"][name] for name in metric_names],
        },
        index=metric_names,
    )
    ax = chart.plot(kind="bar", figsize=(9, 4), rot=15)
    ax.set_ylim(0, 1); ax.set_ylabel("rate"); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "oof_comparison_demo.png", dpi=160); plt.close()
    recall_delta = report["paired_bootstrap_delta_challenger_minus_champion"]["recall"]
    fpr_delta = report["paired_bootstrap_delta_challenger_minus_champion"][
        "false_positive_rate"
    ]
    (OUTPUT_DIR / "summary.md").write_text(
        "# SECOM Champion–Challenger 승격 심사 데모\n\n"
        "반복 train OOF 예측을 5개 가상 배치로 나눈 통계 검증 데모이며 실제 승격 "
        "근거가 아닙니다. 고정 test 라벨은 규칙 설정에 사용하지 않았습니다.\n\n"
        f"- decision: {report['decision']}\n"
        f"- XGBoost−CatBoost 재현율 차이 95% CI: "
        f"[{recall_delta['ci_low']:.4f}, {recall_delta['ci_high']:.4f}]\n"
        f"- 정상 오탐률 차이 95% CI: "
        f"[{fpr_delta['ci_low']:.4f}, {fpr_delta['ci_high']:.4f}]\n"
        f"- McNemar exact p-value: {report['mcnemar_exact_p_value']:.6f}\n"
        f"- 재현율 비열등성: {report['recall_noninferiority_pass']}\n"
        f"- 정상 오탐률 guardrail: {report['false_positive_guardrail_pass']}\n"
        "- 자동 승격: 금지\n",
        encoding="utf-8",
    )
    print(f"Champion-challenger 데모 생성 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
