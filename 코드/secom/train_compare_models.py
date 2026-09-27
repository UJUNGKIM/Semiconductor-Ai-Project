"""Compare imbalance strategies and tree models on the UCI SECOM dataset.

All learned preprocessing (median imputation, scaling, and SMOTE) is fitted only
on the training portion of each fold. The untouched test split is used once for
the final comparison.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
EXPECTED_RAW_FEATURES = 590
EXPECTED_REDUCED_FEATURES = 446
MODEL_NAMES = ("XGBoost", "CatBoost", "LightGBM")
STRATEGIES = ("SMOTE", "scale_pos_weight")


def locate_data(project_dir: Path) -> tuple[Path, Path]:
    """Find raw files either in the project root or its ``secom`` folder."""
    for data_dir in (
        project_dir / "데이터" / "SECOM 데이터셋" / "raw",
        project_dir,
        project_dir / "secom",
    ):
        data_path = data_dir / "secom.data"
        labels_path = data_dir / "secom_labels.data"
        if data_path.is_file() and labels_path.is_file():
            return data_path, labels_path
    raise FileNotFoundError(
        "Could not find secom.data and secom_labels.data in the project root "
        "or project/secom directory."
    )


def load_data(project_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load sensor features, convert labels from -1/1 to 0/1, parse timestamps."""
    data_path, labels_path = locate_data(project_dir)
    X = pd.read_csv(data_path, sep=r"\s+", header=None)
    labels = pd.read_csv(labels_path, sep=r"\s+", header=None)

    if X.shape[0] != labels.shape[0]:
        raise ValueError(f"Row mismatch: X={X.shape[0]}, labels={labels.shape[0]}")
    if X.shape[1] != EXPECTED_RAW_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_RAW_FEATURES} sensor columns, got {X.shape[1]}"
        )
    if labels.shape[1] != 2:
        raise ValueError(f"Expected 2 label columns, got {labels.shape[1]}")

    raw_y = labels.iloc[:, 0].astype(int)
    if set(raw_y.unique()) != {-1, 1}:
        raise ValueError(f"Unexpected labels: {sorted(raw_y.unique().tolist())}")

    X.columns = [f"feature_{i}" for i in range(X.shape[1])]
    y = raw_y.map({-1: 0, 1: 1}).astype("int8").rename("defect")
    timestamps = pd.to_datetime(
        labels.iloc[:, 1], format="%d/%m/%Y %H:%M:%S", errors="raise"
    ).rename("timestamp")
    return X, y, timestamps


def structural_filter(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Remove >=50%-missing columns and constants without using target labels."""
    high_missing = X.columns[X.isna().mean() >= 0.5].tolist()
    reduced = X.drop(columns=high_missing)
    constants = reduced.columns[reduced.nunique(dropna=True) <= 1].tolist()
    reduced = reduced.drop(columns=constants)
    if reduced.shape[1] != EXPECTED_REDUCED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_REDUCED_FEATURES} retained columns, "
            f"got {reduced.shape[1]}"
        )
    return reduced, high_missing, constants


def build_model(name: str, strategy: str, ratio: float, seed: int):
    common = {"random_state": seed}
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            scale_pos_weight=ratio if strategy == "scale_pos_weight" else 1.0,
            **common,
        )
    if name == "CatBoost":
        params = dict(
            iterations=400,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            random_seed=seed,
        )
        if strategy == "scale_pos_weight":
            params["scale_pos_weight"] = ratio
        return CatBoostClassifier(**params)
    if name == "LightGBM":
        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary",
            verbosity=-1,
            n_jobs=-1,
            scale_pos_weight=ratio if strategy == "scale_pos_weight" else 1.0,
            **common,
        )
    raise ValueError(f"Unknown model: {name}")


def optimal_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the precision-recall-curve threshold that maximizes positive F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "accuracy": accuracy_score(y_true, predictions),
        "mcc": matthews_corrcoef(y_true, predictions),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def prepare_fold(
    X_fit: pd.DataFrame,
    X_eval: pd.DataFrame,
    y_fit: pd.Series,
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer, StandardScaler | None]:
    """Fit learned preprocessing on one training fold only."""
    imputer = SimpleImputer(strategy="median")
    X_fit_ready = imputer.fit_transform(X_fit)
    X_eval_ready = imputer.transform(X_eval)
    scaler = None

    if strategy == "SMOTE":
        # Scaling matters for SMOTE because it uses nearest-neighbour distances.
        scaler = StandardScaler()
        X_fit_ready = scaler.fit_transform(X_fit_ready)
        X_eval_ready = scaler.transform(X_eval_ready)
        smote = SMOTE(random_state=seed)
        X_fit_ready, y_resampled = smote.fit_resample(X_fit_ready, y_fit)
    else:
        y_resampled = y_fit.to_numpy()

    return X_fit_ready, X_eval_ready, np.asarray(y_resampled), imputer, scaler


def cross_validate_configuration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    strategy: str,
) -> list[dict[str, float | int | str]]:
    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    rows: list[dict[str, float | int | str]] = []

    for fold, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_fit, X_val = X_train.iloc[fit_idx], X_train.iloc[val_idx]
        y_fit, y_val = y_train.iloc[fit_idx], y_train.iloc[val_idx]
        seed = RANDOM_STATE + fold
        ratio = float((y_fit == 0).sum() / (y_fit == 1).sum())
        X_fit_ready, X_val_ready, y_fit_ready, _, _ = prepare_fold(
            X_fit, X_val, y_fit, strategy, seed
        )
        model = build_model(model_name, strategy, ratio, seed)
        started = time.perf_counter()
        model.fit(X_fit_ready, y_fit_ready)
        elapsed = time.perf_counter() - started
        probabilities = model.predict_proba(X_val_ready)[:, 1]
        threshold = optimal_f1_threshold(y_val.to_numpy(), probabilities)
        metrics = classification_metrics(y_val.to_numpy(), probabilities, threshold)
        rows.append(
            {
                "model": model_name,
                "strategy": strategy,
                "fold": fold,
                "fit_seconds": elapsed,
                "train_rows_before": len(y_fit),
                "train_rows_after": len(y_fit_ready),
                "train_positives_before": int(y_fit.sum()),
                "train_positives_after": int(y_fit_ready.sum()),
                **metrics,
            }
        )
        print(
            f"  fold {fold}/{N_SPLITS}: PR-AUC={metrics['pr_auc']:.4f}, "
            f"F1={metrics['f1']:.4f}, threshold={threshold:.4f}, "
            f"rows={len(y_fit)}->{len(y_fit_ready)}"
        )
    return rows


def fit_final_configuration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    strategy: str,
    threshold: float,
) -> tuple[dict, dict[str, float | int | str], np.ndarray]:
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    X_fit_ready, X_test_ready, y_fit_ready, imputer, scaler = prepare_fold(
        X_train, X_test, y_train, strategy, RANDOM_STATE
    )
    model = build_model(model_name, strategy, ratio, RANDOM_STATE)
    started = time.perf_counter()
    model.fit(X_fit_ready, y_fit_ready)
    elapsed = time.perf_counter() - started
    probabilities = model.predict_proba(X_test_ready)[:, 1]
    metrics = {
        "model": model_name,
        "strategy": strategy,
        "fit_seconds": elapsed,
        "train_rows_before": len(y_train),
        "train_rows_after": len(y_fit_ready),
        "train_positives_before": int(y_train.sum()),
        "train_positives_after": int(y_fit_ready.sum()),
        **classification_metrics(y_test.to_numpy(), probabilities, threshold),
    }
    bundle = {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "threshold": threshold,
        "model_name": model_name,
        "strategy": strategy,
        "feature_names": X_train.columns.tolist(),
        "label_mapping": {-1: 0, 1: 1},
        "random_state": RANDOM_STATE,
    }
    return bundle, metrics, probabilities


def summarize_cv(fold_results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "mcc",
        "threshold",
        "fit_seconds",
    ]
    summary = fold_results.groupby(["model", "strategy"])[metric_columns].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().sort_values(
        ["pr_auc_mean", "f1_mean"], ascending=False
    )


def save_plots(
    output_dir: Path,
    cv_summary: pd.DataFrame,
    test_results: pd.DataFrame,
    y_test: pd.Series,
    test_probabilities: dict[tuple[str, str], np.ndarray],
) -> None:
    labels = (cv_summary["model"] + "\n" + cv_summary["strategy"]).tolist()
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        positions,
        cv_summary["pr_auc_mean"],
        yerr=cv_summary["pr_auc_std"],
        capsize=4,
        color="#3677a9",
    )
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_ylabel("5-fold PR-AUC (mean +/- SD)")
    ax.set_title("SECOM model and imbalance-strategy comparison")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "cv_pr_auc_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for _, row in test_results.iterrows():
        key = (str(row["model"]), str(row["strategy"]))
        probabilities = test_probabilities[key]
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        ax.plot(
            recall,
            precision,
            linewidth=1.7,
            label=f"{key[0]} / {key[1]} (AP={row['pr_auc']:.3f})",
        )
    ax.axhline(float(y_test.mean()), color="grey", linestyle="--", label="prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Untouched test-set precision-recall curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "test_precision_recall_curves.png", dpi=180)
    plt.close(fig)

    plot_frame = test_results.copy()
    plot_frame["configuration"] = plot_frame["model"] + "\n" + plot_frame["strategy"]
    plot_frame = plot_frame.set_index("configuration")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_frame[["precision", "recall", "f1", "balanced_accuracy"]].plot(
        kind="bar", ax=ax, width=0.78
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Final test metrics at CV-derived thresholds")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "test_metric_comparison.png", dpi=180)
    plt.close(fig)


def write_markdown_summary(
    output_dir: Path,
    dataset_info: dict,
    cv_summary: pd.DataFrame,
    test_results: pd.DataFrame,
    best_overall: pd.Series,
    best_smote: pd.Series,
) -> None:
    display_columns = [
        "model",
        "strategy",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "threshold",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    table = test_results[display_columns].copy()
    numeric_columns = table.select_dtypes(include="number").columns
    table[numeric_columns] = table[numeric_columns].round(4)
    cv_table = cv_summary[
        [
            "model",
            "strategy",
            "pr_auc_mean",
            "pr_auc_std",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "threshold_mean",
        ]
    ].copy()
    cv_numeric = cv_table.select_dtypes(include="number").columns
    cv_table[cv_numeric] = cv_table[cv_numeric].round(4)

    text = f"""# SECOM 모델 비교 결과

## 데이터와 전처리

- 원본: {dataset_info['rows']}행 × {dataset_info['raw_features']}개 센서 변수
- 레이블 변환: -1(정상)→0, 1(불량)→1
- 전체 클래스: 정상 {dataset_info['normal_rows']}건, 불량 {dataset_info['defect_rows']}건
- 결측률 50% 이상 제거: {dataset_info['high_missing_removed']}개
- 상수 변수 제거: {dataset_info['constant_removed']}개
- 최종 입력 변수: {dataset_info['retained_features']}개
- 고정 분할: train {dataset_info['train_rows']}건 / test {dataset_info['test_rows']}건
- median, StandardScaler(SMOTE 전용), SMOTE는 fold 학습 데이터에만 적합

## 5-fold 교차검증

{cv_table.to_markdown(index=False)}

## 미사용 테스트 세트 결과

{table.to_markdown(index=False)}

## 선택 결과

- CV PR-AUC 기준 전체 1위: **{best_overall['model']} / {best_overall['strategy']}**
- 전체 1위 test: PR-AUC={best_overall['pr_auc']:.4f}, precision={best_overall['precision']:.4f}, recall={best_overall['recall']:.4f}, F1={best_overall['f1']:.4f}
- SMOTE 후보 1위: **{best_smote['model']} / SMOTE**
- SMOTE 1위 test: PR-AUC={best_smote['pr_auc']:.4f}, precision={best_smote['precision']:.4f}, recall={best_smote['recall']:.4f}, F1={best_smote['f1']:.4f}

## 증강 해석

SMOTE는 full train을 {dataset_info['train_rows']}건에서 {dataset_info['smote_train_rows']}건으로 늘렸습니다.
이는 2,000~3,000건 범위에 들어가지만 합성 표본은 새로운 실측 공정 정보가 아닙니다.
따라서 성능 판단은 증강하지 않은 test {dataset_info['test_rows']}건에서만 수행했습니다.
"""
    (output_dir / "summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "결과물" / "secom" / "model_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("[1/8] Loading data and converting labels...")
    X_raw, y, timestamps = load_data(project_dir)
    print(
        f"  X={X_raw.shape}, labels={y.value_counts().sort_index().to_dict()}, "
        f"time={timestamps.min()}..{timestamps.max()}"
    )

    print("[2/8] Removing >=50%-missing and constant features...")
    X, high_missing, constants = structural_filter(X_raw)
    print(
        f"  {X_raw.shape[1]} - {len(high_missing)} - {len(constants)} "
        f"= {X.shape[1]} features"
    )

    print("[3-4/8] Creating untouched stratified test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(
        f"  train={len(y_train)} {y_train.value_counts().sort_index().to_dict()}, "
        f"test={len(y_test)} {y_test.value_counts().sort_index().to_dict()}"
    )
    print("  Median imputation will be fitted inside each training fold.")

    print("[5-7/8] Running 5-fold CV for 6 configurations...")
    all_fold_rows: list[dict[str, float | int | str]] = []
    for model_name in MODEL_NAMES:
        for strategy in STRATEGIES:
            print(f"\n{model_name} / {strategy}")
            all_fold_rows.extend(
                cross_validate_configuration(X_train, y_train, model_name, strategy)
            )

    fold_results = pd.DataFrame(all_fold_rows)
    cv_summary = summarize_cv(fold_results)
    fold_results.to_csv(output_dir / "cv_fold_results.csv", index=False)
    cv_summary.to_csv(output_dir / "cv_summary.csv", index=False)

    print("\n[7/8] Refitting each configuration and evaluating untouched test...")
    test_rows = []
    bundles: dict[tuple[str, str], dict] = {}
    test_probabilities: dict[tuple[str, str], np.ndarray] = {}
    for _, cv_row in cv_summary.iterrows():
        model_name = str(cv_row["model"])
        strategy = str(cv_row["strategy"])
        fold_thresholds = fold_results.loc[
            (fold_results["model"] == model_name)
            & (fold_results["strategy"] == strategy),
            "threshold",
        ]
        final_threshold = float(fold_thresholds.median())
        bundle, metrics, probabilities = fit_final_configuration(
            X_train,
            y_train,
            X_test,
            y_test,
            model_name,
            strategy,
            final_threshold,
        )
        key = (model_name, strategy)
        bundles[key] = bundle
        test_probabilities[key] = probabilities
        test_rows.append(metrics)
        print(
            f"  {model_name}/{strategy}: PR-AUC={metrics['pr_auc']:.4f}, "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, threshold={final_threshold:.4f}"
        )

    test_results = pd.DataFrame(test_rows)
    ranking = cv_summary[["model", "strategy", "pr_auc_mean", "f1_mean"]].merge(
        test_results, on=["model", "strategy"], how="left"
    )
    ranking = ranking.sort_values(["pr_auc_mean", "f1_mean"], ascending=False)
    test_results = test_results.merge(
        cv_summary[["model", "strategy", "pr_auc_mean"]],
        on=["model", "strategy"],
        how="left",
    ).sort_values("pr_auc_mean", ascending=False)
    test_results.to_csv(output_dir / "test_results.csv", index=False)

    best_overall = ranking.iloc[0]
    best_smote = ranking[ranking["strategy"] == "SMOTE"].iloc[0]
    joblib.dump(
        bundles[(str(best_overall["model"]), str(best_overall["strategy"]))],
        output_dir / "best_overall_model.joblib",
    )
    joblib.dump(
        bundles[(str(best_smote["model"]), "SMOTE")],
        output_dir / "best_smote_model.joblib",
    )

    smote_rows = int(
        test_results.loc[test_results["strategy"] == "SMOTE", "train_rows_after"].iloc[0]
    )
    dataset_info = {
        "rows": len(y),
        "raw_features": X_raw.shape[1],
        "normal_rows": int((y == 0).sum()),
        "defect_rows": int((y == 1).sum()),
        "high_missing_removed": len(high_missing),
        "constant_removed": len(constants),
        "retained_features": X.shape[1],
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "smote_train_rows": smote_rows,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_splits": N_SPLITS,
        "threshold_rule": "maximize F1 on each validation fold; final=median",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plots(output_dir, cv_summary, test_results, y_test, test_probabilities)
    write_markdown_summary(
        output_dir, dataset_info, cv_summary, test_results, best_overall, best_smote
    )

    print("\n[8/8] Complete")
    print(
        f"  Overall CV winner: {best_overall['model']} / {best_overall['strategy']}"
    )
    print(f"  Best SMOTE model: {best_smote['model']} / SMOTE")
    print(f"  Results: {output_dir}")


if __name__ == "__main__":
    main()
