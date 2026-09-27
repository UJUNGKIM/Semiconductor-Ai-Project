"""Generate reproducible SHAP explanations for the two SECOM finalists."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from train_compare_models import RANDOM_STATE, TEST_SIZE, load_data, structural_filter


MODELS = ("CatBoost", "XGBoost")
COLORS = {"CatBoost": "#d1495b", "XGBoost": "#2878b5"}
TOP_GLOBAL = 20
TOP_LOCAL = 8


def extract_shap_values(model, X_ready: np.ndarray) -> tuple[np.ndarray, float]:
    """Handle the common binary-class SHAP return formats."""
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(X_ready)
    if isinstance(raw_values, list):
        values = np.asarray(raw_values[-1])
    else:
        values = np.asarray(raw_values)
        if values.ndim == 3:
            values = values[:, :, -1]
    expected = np.asarray(explainer.expected_value).reshape(-1)
    base_value = float(expected[-1])
    if values.shape != X_ready.shape:
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; expected {X_ready.shape}"
        )
    return values, base_value


def choose_cases(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, int]:
    predictions = (probabilities >= threshold).astype(int)
    masks = {
        "TP": (labels == 1) & (predictions == 1),
        "FN": (labels == 1) & (predictions == 0),
        "FP": (labels == 0) & (predictions == 1),
        "TN": (labels == 0) & (predictions == 0),
    }
    selected = {}
    for case_type, mask in masks.items():
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            continue
        if case_type in {"TP", "FP"}:
            # Strongest positive prediction among detected/false-alarm examples.
            index = candidates[np.argmax(probabilities[candidates])]
        elif case_type == "FN":
            # Missed defect closest to the decision boundary.
            index = candidates[np.argmax(probabilities[candidates])]
        else:
            # Most confidently normal true negative.
            index = candidates[np.argmin(probabilities[candidates])]
        selected[case_type] = int(index)
    return selected


def save_global_comparison(
    output_dir: Path, importance_frames: dict[str, pd.DataFrame]
) -> None:
    merged = importance_frames["CatBoost"][["feature", "normalized_importance"]].merge(
        importance_frames["XGBoost"][["feature", "normalized_importance"]],
        on="feature",
        how="outer",
        suffixes=("_catboost", "_xgboost"),
    ).fillna(0.0)
    merged["mean_importance"] = merged[
        ["normalized_importance_catboost", "normalized_importance_xgboost"]
    ].mean(axis=1)
    top = merged.nlargest(TOP_GLOBAL, "mean_importance").sort_values(
        "mean_importance"
    )
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        y - 0.19,
        top["normalized_importance_catboost"],
        height=0.36,
        color=COLORS["CatBoost"],
        label="CatBoost",
    )
    ax.barh(
        y + 0.19,
        top["normalized_importance_xgboost"],
        height=0.36,
        color=COLORS["XGBoost"],
        label="XGBoost",
    )
    ax.set_yticks(y, top["feature"])
    ax.set_xlabel("Normalized mean |SHAP value|")
    ax.set_title("Global feature importance: finalist comparison")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "global_shap_comparison.png", dpi=190)
    plt.close(fig)


def save_local_plot(
    output_dir: Path,
    model_name: str,
    cases: dict[str, int],
    feature_names: list[str],
    values: np.ndarray,
    X_ready: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    labels: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    case_order = ("TP", "FN", "FP", "TN")
    for ax, case_type in zip(axes, case_order):
        if case_type not in cases:
            ax.axis("off")
            ax.set_title(f"{case_type}: no sample")
            continue
        row = cases[case_type]
        order = np.argsort(np.abs(values[row]))[-TOP_LOCAL:]
        signed_values = values[row, order]
        labels_for_plot = [
            f"{feature_names[i]} = {X_ready[row, i]:.3g}" for i in order
        ]
        colors = ["#d1495b" if value > 0 else "#2878b5" for value in signed_values]
        ax.barh(np.arange(len(order)), signed_values, color=colors)
        ax.set_yticks(np.arange(len(order)), labels_for_plot, fontsize=8)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.grid(axis="x", alpha=0.2)
        ax.set_title(
            f"{case_type}: y={labels[row]}, p={probabilities[row]:.3f}, "
            f"threshold={threshold:.3f}"
        )
        ax.set_xlabel("SHAP contribution to raw model score")
    fig.suptitle(f"{model_name}: representative local explanations", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name.lower()}_local_explanations.png", dpi=190)
    plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    model_dir = project_dir / "결과물" / "secom" / "dual_model_results"
    output_dir = model_dir / "xai"
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("[1/6] Loading data, fixed split, and both saved model bundles...")
    X_raw, y, timestamps = load_data(project_dir)
    X, _, _ = structural_filter(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    bundles = {
        model_name: joblib.load(model_dir / f"{model_name.lower()}_model.joblib")
        for model_name in MODELS
    }

    print("[2/6] Computing TreeSHAP values on the fixed test samples...")
    explanations = {}
    importance_frames = {}
    prediction_frame = pd.DataFrame(
        {
            "original_row_index": X_test.index,
            "timestamp": timestamps.loc[X_test.index].astype(str).to_numpy(),
            "label": y_test.to_numpy(),
        }
    )
    for model_name in MODELS:
        bundle = bundles[model_name]
        X_imputed = bundle["imputer"].transform(X_test[bundle["input_features"]])
        X_ready = bundle["selector"].transform(X_imputed)
        probabilities = bundle["model"].predict_proba(X_ready)[:, 1]
        values, base_value = extract_shap_values(bundle["model"], X_ready)
        feature_names = list(bundle["selected_features"])
        mean_absolute = np.mean(np.abs(values), axis=0)
        importance = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_absolute}
        ).sort_values("mean_abs_shap", ascending=False)
        importance["normalized_importance"] = (
            importance["mean_abs_shap"] / importance["mean_abs_shap"].sum()
        )
        importance["rank"] = np.arange(1, len(importance) + 1)
        importance.to_csv(
            output_dir / f"{model_name.lower()}_global_importance.csv", index=False
        )
        importance_frames[model_name] = importance
        threshold = float(bundle["primary_threshold"])
        prediction_frame[f"{model_name.lower()}_probability"] = probabilities
        prediction_frame[f"{model_name.lower()}_prediction"] = (
            probabilities >= threshold
        ).astype(int)
        explanations[model_name] = {
            "X_ready": X_ready,
            "probabilities": probabilities,
            "values": values,
            "base_value": base_value,
            "feature_names": feature_names,
            "threshold": threshold,
        }

        print(f"  {model_name}: {len(feature_names)} features, {len(X_test)} samples")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            values,
            X_ready,
            feature_names=feature_names,
            max_display=TOP_GLOBAL,
            show=False,
        )
        plt.title(f"{model_name} SHAP summary (fixed test)")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{model_name.lower()}_shap_beeswarm.png",
            dpi=190,
            bbox_inches="tight",
        )
        plt.close()

    print("[3/6] Measuring agreement between the two global explanations...")
    cat = importance_frames["CatBoost"]
    xgb = importance_frames["XGBoost"]
    agreement = {}
    for top_k in (10, 20, 50):
        cat_set = set(cat.head(top_k)["feature"])
        xgb_set = set(xgb.head(top_k)["feature"])
        intersection = sorted(cat_set & xgb_set)
        agreement[f"top_{top_k}"] = {
            "overlap_count": len(intersection),
            "jaccard": len(intersection) / len(cat_set | xgb_set),
            "shared_features": intersection,
        }
    merged = cat[["feature", "mean_abs_shap"]].merge(
        xgb[["feature", "mean_abs_shap"]],
        on="feature",
        how="outer",
        suffixes=("_catboost", "_xgboost"),
    ).fillna(0.0)
    rank_correlation = spearmanr(
        merged["mean_abs_shap_catboost"], merged["mean_abs_shap_xgboost"]
    )
    agreement["all_feature_spearman"] = {
        "statistic": float(rank_correlation.statistic),
        "pvalue": float(rank_correlation.pvalue),
        "note": "Unselected features are assigned zero importance.",
    }
    (output_dir / "global_explanation_agreement.json").write_text(
        json.dumps(agreement, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_global_comparison(output_dir, importance_frames)

    print("[4/6] Creating representative TP/FN/FP/TN local explanations...")
    local_rows = []
    selected_cases = {}
    for model_name in MODELS:
        info = explanations[model_name]
        cases = choose_cases(
            y_test.to_numpy(), info["probabilities"], info["threshold"]
        )
        selected_cases[model_name] = cases
        save_local_plot(
            output_dir,
            model_name,
            cases,
            info["feature_names"],
            info["values"],
            info["X_ready"],
            info["probabilities"],
            info["threshold"],
            y_test.to_numpy(),
        )
        for case_type, row in cases.items():
            order = np.argsort(np.abs(info["values"][row]))[::-1][:TOP_LOCAL]
            for rank, feature_index in enumerate(order, 1):
                shap_value = float(info["values"][row, feature_index])
                local_rows.append(
                    {
                        "model": model_name,
                        "case_type": case_type,
                        "test_position": row,
                        "original_row_index": int(X_test.index[row]),
                        "timestamp": str(timestamps.loc[X_test.index[row]]),
                        "true_label": int(y_test.iloc[row]),
                        "probability": float(info["probabilities"][row]),
                        "threshold": info["threshold"],
                        "rank": rank,
                        "feature": info["feature_names"][feature_index],
                        "feature_value_after_imputation": float(
                            info["X_ready"][row, feature_index]
                        ),
                        "shap_value_raw_score": shap_value,
                        "direction": (
                            "toward_defect" if shap_value > 0 else "toward_normal"
                        ),
                    }
                )
    pd.DataFrame(local_rows).to_csv(
        output_dir / "representative_local_explanations.csv", index=False
    )

    print("[5/6] Saving prediction agreement and disagreement cases...")
    prediction_frame["models_agree"] = (
        prediction_frame["catboost_prediction"]
        == prediction_frame["xgboost_prediction"]
    )
    prediction_frame["absolute_probability_gap"] = np.abs(
        prediction_frame["catboost_probability"]
        - prediction_frame["xgboost_probability"]
    )
    prediction_frame.to_csv(output_dir / "test_prediction_comparison.csv", index=False)
    disagreements = prediction_frame.loc[~prediction_frame["models_agree"]].sort_values(
        "absolute_probability_gap", ascending=False
    )
    disagreements.to_csv(output_dir / "model_disagreements.csv", index=False)

    print("[6/6] Writing interpretation report...")
    top_cat = cat.head(10)["feature"].tolist()
    top_xgb = xgb.head(10)["feature"].tolist()
    shared_top20 = agreement["top_20"]["shared_features"]
    agree_rate = float(prediction_frame["models_agree"].mean())
    report = f"""# CatBoost와 XGBoost 설명가능성 분석

- 설명 대상: 고정 test {len(X_test)}개 샘플
- 방법: TreeSHAP
- SHAP 부호가 양수이면 모델의 불량 raw score를 높이고, 음수이면 낮춤
- CatBoost 선택 변수: {len(explanations['CatBoost']['feature_names'])}개
- XGBoost 선택 변수: {len(explanations['XGBoost']['feature_names'])}개

## 전역 중요 변수

- CatBoost 상위 10개: {', '.join(top_cat)}
- XGBoost 상위 10개: {', '.join(top_xgb)}
- 상위 20개 중 공통 변수: {len(shared_top20)}개
- 공통 변수: {', '.join(shared_top20) if shared_top20 else '없음'}
- 전체 중요도 Spearman 상관: {agreement['all_feature_spearman']['statistic']:.4f}

## 예측 일치

- 균형형 임계값에서 두 모델 예측 일치율: {agree_rate:.4f}
- 불일치 샘플 수: {len(disagreements)}개

## 해석상 주의

SECOM 변수명은 센서 의미가 없는 익명 번호입니다. 따라서 SHAP은 어떤 feature 번호가
예측을 올리거나 내렸는지는 보여주지만, 그 자체로 온도·압력 같은 실제 공정 원인을
증명하지 않습니다. 센서 사전 및 공정 엔지니어 검토를 연결해야 '원인 진단'으로 확장할
수 있습니다. 또한 이 test는 앞선 실험에서 이미 관찰했으므로 성능 선택용이 아니라 설명
예시용으로만 사용했습니다.
"""
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    (output_dir / "representative_cases.json").write_text(
        json.dumps(selected_cases, indent=2), encoding="utf-8"
    )
    print(f"  results={output_dir}")


if __name__ == "__main__":
    main()
