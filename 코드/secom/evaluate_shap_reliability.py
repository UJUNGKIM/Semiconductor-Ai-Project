"""Audit SECOM SHAP stability and deletion fidelity on the locked test split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from explain_dual_models import extract_shap_values
from shap_reliability import (
    build_masked_matrices,
    explanation_similarity,
    perturb_within_reference,
    summarize_model_metrics,
)
from train_compare_models import RANDOM_STATE, TEST_SIZE, load_data, structural_filter


MODELS = ("CatBoost", "XGBoost")
TOP_K = 10
STABILITY_REPEATS = 3
RANDOM_MASK_REPEATS = 5
NOISE_FRACTION = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    model_dir = project / "결과물" / "secom" / "dual_model_results"
    output_dir = project / "결과물" / "secom" / "shap_reliability"
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("[1/5] 고정 train/test와 배포 모델을 불러옵니다...")
    X_raw, y, _ = load_data(project)
    X, _, _ = structural_filter(X_raw)
    X_train, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    all_rows = []
    summaries = {}
    model_hashes = {}

    for model_index, model_name in enumerate(MODELS):
        print(f"[2/5] {model_name} SHAP 안정성과 마스킹 충실도를 계산합니다...")
        model_path = model_dir / f"{model_name.lower()}_model.joblib"
        model_hashes[model_name] = sha256_file(model_path)
        bundle = joblib.load(model_path)
        columns = list(bundle["input_features"])
        train_imputed = bundle["imputer"].transform(X_train[columns])
        test_imputed = bundle["imputer"].transform(X_test[columns])
        train_ready = np.asarray(bundle["selector"].transform(train_imputed), dtype=float)
        test_ready = np.asarray(bundle["selector"].transform(test_imputed), dtype=float)
        model = bundle["model"]
        threshold = float(bundle["primary_threshold"])
        clean_probability = model.predict_proba(test_ready)[:, 1]
        clean_shap, _ = extract_shap_values(model, test_ready)

        q01, q25, q50, q75, q99 = np.quantile(
            train_ready, [0.01, 0.25, 0.50, 0.75, 0.99], axis=0
        )
        iqr = q75 - q25
        top_masked, random_masked = build_masked_matrices(
            test_ready,
            clean_shap,
            q50,
            top_k=min(TOP_K, test_ready.shape[1]),
            random_repeats=RANDOM_MASK_REPEATS,
            rng=np.random.default_rng(RANDOM_STATE + model_index * 100),
        )
        top_probability = model.predict_proba(top_masked)[:, 1]
        top_effect = np.abs(top_probability - clean_probability)
        random_effect = np.mean(
            [
                np.abs(model.predict_proba(matrix)[:, 1] - clean_probability)
                for matrix in random_masked
            ],
            axis=0,
        )

        model_rows = []
        for repeat in range(STABILITY_REPEATS):
            rng = np.random.default_rng(
                RANDOM_STATE + model_index * 1000 + repeat + 1
            )
            perturbed = perturb_within_reference(
                test_ready,
                iqr,
                q01,
                q99,
                noise_fraction=NOISE_FRACTION,
                rng=rng,
            )
            perturbed_probability = model.predict_proba(perturbed)[:, 1]
            perturbed_shap, _ = extract_shap_values(model, perturbed)
            clean_prediction = clean_probability >= threshold
            perturbed_prediction = perturbed_probability >= threshold
            for row in range(len(test_ready)):
                similarity = explanation_similarity(
                    clean_shap[row],
                    perturbed_shap[row],
                    top_k=min(TOP_K, test_ready.shape[1]),
                )
                model_rows.append(
                    {
                        "model": model_name,
                        "test_position": row,
                        "original_row_index": int(X_test.index[row]),
                        "true_label": int(y_test.iloc[row]),
                        "repeat": repeat + 1,
                        **similarity,
                        "probability_change_under_noise": float(
                            abs(perturbed_probability[row] - clean_probability[row])
                        ),
                        "prediction_flipped_under_noise": bool(
                            perturbed_prediction[row] != clean_prediction[row]
                        ),
                        "top_shap_mask_effect": float(top_effect[row]),
                        "random_mask_effect": float(random_effect[row]),
                        "fidelity_advantage": float(
                            top_effect[row] - random_effect[row]
                        ),
                        "top_shap_mask_wins": bool(
                            top_effect[row] > random_effect[row]
                        ),
                    }
                )
        summaries[model_name] = summarize_model_metrics(model_rows)
        all_rows.extend(model_rows)

    print("[3/5] 표와 그림을 저장합니다...")
    per_sample = pd.DataFrame(all_rows)
    per_sample.to_csv(output_dir / "per_sample_reliability.csv", index=False)
    summary_frame = pd.DataFrame(
        [{"model": model, **{k: v for k, v in values.items() if k != "guardrails"}}
         for model, values in summaries.items()]
    )
    summary_frame.to_csv(output_dir / "model_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    names = list(MODELS)
    axes[0].bar(
        np.arange(len(names)) - 0.18,
        [summaries[name]["median_top10_jaccard"] for name in names],
        width=0.36,
        label="Top-10 Jaccard",
        color="#2878b5",
    )
    axes[0].bar(
        np.arange(len(names)) + 0.18,
        [summaries[name]["median_top10_sign_agreement"] for name in names],
        width=0.36,
        label="Sign agreement",
        color="#59a14f",
    )
    axes[0].set_xticks(range(len(names)), names)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Explanation stability under 5% IQR noise")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        np.arange(len(names)) - 0.18,
        [summaries[name]["median_top_shap_mask_effect"] for name in names],
        width=0.36,
        label="Top SHAP mask",
        color="#d1495b",
    )
    axes[1].bar(
        np.arange(len(names)) + 0.18,
        [summaries[name]["median_random_mask_effect"] for name in names],
        width=0.36,
        label="Random mask",
        color="#bab0ab",
    )
    axes[1].set_xticks(range(len(names)), names)
    axes[1].set_ylabel("Median absolute probability change")
    axes[1].set_title("Deletion fidelity: matched feature count")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "shap_reliability_dashboard.png", dpi=180)
    plt.close(fig)

    print("[4/5] 사전 고정 guardrail을 판정합니다...")
    overall_status = (
        "PASS" if all(item["status"] == "PASS" for item in summaries.values()) else "WARN"
    )
    result = {
        "version": 1,
        "status": overall_status,
        "evaluation_scope": "fixed_test_post_selection_audit_only",
        "test_rows": len(X_test),
        "test_defects": int(y_test.sum()),
        "random_state": RANDOM_STATE,
        "top_k": TOP_K,
        "stability_repeats": STABILITY_REPEATS,
        "noise_fraction_of_train_iqr": NOISE_FRACTION,
        "random_mask_repeats": RANDOM_MASK_REPEATS,
        "model_selection_uses_results": False,
        "threshold_selection_uses_results": False,
        "physical_causality_claimed": False,
        "model_hashes": model_hashes,
        "models": summaries,
        "limitations": [
            "고정 test를 사용한 사후 설명 감사이며 모델·임계값 선택에 재사용하지 않습니다.",
            "5% train IQR 독립 노이즈는 실제 장비 고장 분포를 대체하지 않습니다.",
            "변수 마스킹은 train 중앙값 대체 실험이며 센서 개입의 물리적 효과가 아닙니다.",
            "익명 변수의 SHAP 안정성은 물리적 원인이나 인과관계를 증명하지 않습니다.",
        ],
    }
    (output_dir / "shap_reliability_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[5/5] 해석 보고서를 작성합니다...")
    lines = [
        "# SECOM SHAP 설명 신뢰성 감사",
        "",
        f"- 판정: **{overall_status}**",
        f"- 범위: 모델 선택이 끝난 고정 test {len(X_test)}행(불량 {int(y_test.sum())}행)",
        f"- 안정성: train IQR의 {NOISE_FRACTION:.0%} 노이즈 × {STABILITY_REPEATS}회",
        f"- 충실도: 상위 SHAP {TOP_K}개 vs 동일 개수 무작위 변수 중앙값 마스킹",
        "",
    ]
    for model_name in MODELS:
        item = summaries[model_name]
        lines.extend(
            [
                f"## {model_name}",
                "",
                f"- 상위 10개 Jaccard 중앙값: {item['median_top10_jaccard']:.3f}",
                f"- 상위 10개 부호 일치율 중앙값: {item['median_top10_sign_agreement']:.3f}",
                f"- 노이즈 예측 반전율: {item['prediction_flip_rate_under_noise']:.3%}",
                f"- 상위 SHAP 마스킹 효과 중앙값: {item['median_top_shap_mask_effect']:.4f}",
                f"- 무작위 마스킹 효과 중앙값: {item['median_random_mask_effect']:.4f}",
                f"- 상위 SHAP 마스킹 승률: {item['top_shap_mask_win_rate']:.3%}",
                f"- guardrail 판정: **{item['status']}**",
                "",
            ]
        )
    lines.extend(
        [
            "## 해석 제한",
            "",
            *[f"- {text}" for text in result["limitations"]],
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"SHAP 신뢰성 감사 완료: status={overall_status}, output={output_dir}")


if __name__ == "__main__":
    main()
