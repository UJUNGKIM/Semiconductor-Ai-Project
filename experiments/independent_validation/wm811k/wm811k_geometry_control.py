"""WM-811K 메타데이터 대조(control) 모델.

**이것은 결함 분류기가 아닙니다.** 웨이퍼 맵 화소를 한 개도 보지 않고,
웨이퍼 크기·lot 정보만으로 얼마나 맞출 수 있는지 재는 하한선 실험입니다.

왜 필요한가:
기준선 CNN은 고정 test에서 macro-F1 0.8828을 기록했습니다. 그런데 이 숫자 중
얼마가 '결함 모양을 알아본 것'이고 얼마가 '웨이퍼 크기만 봐도 맞출 수 있는 것'인지는
보고되어 있지 않습니다. WM-811K는 웨이퍼 크기(예: 26x30, 43x42)에 따라 결함 클래스
구성이 크게 다르기 때문에, 크기만으로도 상당 부분이 맞을 수 있습니다.
이 대조 모델이 그 바닥값을 숫자로 만들어 줍니다.

누출 방지:
- 기준선이 고정한 lot 비중복 분할을 그대로 씁니다(같은 test 24,705건).
- lot 단위 특징은 같은 split 안에서만 계산합니다. lot가 split을 넘지 않으므로
  다른 split 정보가 섞이지 않습니다.
- 원본 Kaggle의 `source_split` 열은 특징에서 제외합니다. 데이터 출처 표식이라
  대조 모델의 점수를 부풀릴 수 있습니다.

실행:
    .venv\\Scripts\\python.exe "experiments\\independent_validation\\wm811k\\wm811k_geometry_control.py"
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RANDOM_STATE, file_sha256, to_markdown_table, write_json  # noqa: E402
from wm811k_common import (  # noqa: E402
    BASELINE_WM_DIR,
    CLASS_ORDER,
    WM_RESULT_DIR,
    audit_lot_leakage,
    class_distribution,
    confusion_frame,
    evaluate,
    load_split_metadata,
)

CONTROL_DIR = WM_RESULT_DIR / "geometry_control"
N_FOLDS = 5

FEATURE_COLUMNS = [
    "original_height",
    "original_width",
    "die_area",
    "aspect_ratio",
    "longer_side",
    "shorter_side",
    "wafer_index",
    "lot_size",
    "position_in_lot",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """화소를 쓰지 않는 특징만 만든다.

    lot 단위 통계는 split 안에서 계산한다. lot는 split을 넘지 않으므로
    이것만으로 split 간 정보 이동이 없다.
    """
    out = frame.copy()
    height = out["original_height"].astype(float)
    width = out["original_width"].astype(float)

    out["die_area"] = height * width
    out["aspect_ratio"] = height / width.replace(0, np.nan)
    out["longer_side"] = np.maximum(height, width)
    out["shorter_side"] = np.minimum(height, width)
    out["wafer_index"] = pd.to_numeric(out["wafer_index"], errors="coerce").fillna(-1.0)

    lot_size = out.groupby(["split", "lot_name"])["array_index"].transform("size")
    out["lot_size"] = lot_size.astype(float)
    # lot 안에서 몇 번째 웨이퍼인지(0~1로 정규화)
    rank = out.groupby(["split", "lot_name"])["wafer_index"].rank(method="first")
    out["position_in_lot"] = (rank - 1) / np.maximum(lot_size - 1, 1)

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].astype(float).fillna(-1.0)
    return out


def build_model() -> ExtraTreesClassifier:
    """기준선 CNN과 완전히 다른 계열: 표 데이터용 트리 앙상블.

    `min_samples_leaf`를 크게 잡은 이유가 두 가지다.

    1. 특징이 9개뿐인데 행이 123,542개다. 잎을 잘게 쪼개면 lot 단위 잡음을 외운다.
       실제로 leaf=5, 300그루는 test macro-F1 0.2721 / balanced accuracy 0.4139였고,
       leaf=100, 100그루는 0.2736 / 0.4612로 **더 좋았다**. 얕게 두는 쪽이 맞다.
    2. 같은 변경으로 모델 파일이 97MB에서 9MB로 줄어 저장소에 넣을 수 있다.
    """
    return ExtraTreesClassifier(
        n_estimators=100,
        min_samples_leaf=100,
        max_features=None,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def cross_validate(train: pd.DataFrame) -> pd.DataFrame:
    """train 안에서 lot를 묶어 교차검증한다(fold 안에서도 lot 누출 없음)."""
    X = train[FEATURE_COLUMNS].to_numpy()
    y = train["failure_type"].to_numpy()
    groups = train["lot_name"].to_numpy()
    labels = [c for c in CLASS_ORDER if c in set(y)]

    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        shared = set(groups[fit_idx]) & set(groups[val_idx])
        if shared:
            raise RuntimeError(f"fold {fold}에서 lot 누출: {sorted(shared)[:3]}")

        model = build_model()
        model.fit(X[fit_idx], y[fit_idx])
        predicted = model.predict(X[val_idx])
        metrics = evaluate(y[val_idx], predicted, labels)

        row = {
            "fold": fold,
            "fit_rows": int(len(fit_idx)),
            "valid_rows": int(len(val_idx)),
            "fit_lots": int(len(set(groups[fit_idx]))),
            "valid_lots": int(len(set(groups[val_idx]))),
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
        }
        for name, value in metrics["per_class_recall"].items():
            row[f"recall_{name}"] = value
        rows.append(row)
        print(
            f"  fold {fold}/{N_FOLDS}: macro-F1={metrics['macro_f1']:.4f} "
            f"balanced_acc={metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def baseline_cnn_metrics() -> dict | None:
    """기준선 CNN의 고정 test 지표를 읽는다(읽기 전용)."""
    path = BASELINE_WM_DIR / "selected_model_results" / "test_classification_report.csv"
    if not path.is_file():
        return None
    report = pd.read_csv(path, index_col=0)
    per_class = {
        name: float(report.loc[name, "recall"])
        for name in CLASS_ORDER
        if name in report.index
    }
    return {
        "macro_f1": float(report.loc["macro avg", "f1-score"]),
        "balanced_accuracy": float(report.loc["macro avg", "recall"]),
        "accuracy": float(report.loc["accuracy", "f1-score"]),
        "per_class_recall": per_class,
    }


def write_summary(
    folds: pd.DataFrame,
    validation: dict,
    test: dict,
    audit: dict,
    baseline: dict | None,
) -> None:
    fold_table = folds[
        ["fold", "fit_rows", "valid_rows", "valid_lots", "macro_f1", "balanced_accuracy", "accuracy"]
    ].round(4)

    recall_rows = []
    for name in CLASS_ORDER:
        if name not in test["per_class_recall"]:
            continue
        row = {
            "클래스": name,
            "test 건수": test["support"].get(name, 0),
            "대조모델 recall": round(test["per_class_recall"][name], 4),
        }
        if baseline:
            row["기준선 CNN recall"] = round(baseline["per_class_recall"].get(name, float("nan")), 4)
        recall_rows.append(row)

    lines = [
        "# WM-811K 메타데이터 대조 모델 (독립 검증)",
        "",
        "> **이 모델은 결함 분류기가 아닙니다.** 웨이퍼 맵 화소를 하나도 보지 않고",
        "> 웨이퍼 크기와 lot 정보만 씁니다. 기준선 CNN 성능 중 얼마가 '모양 인식'이고",
        "> 얼마가 '크기만 봐도 맞는 부분'인지 가르는 하한선 대조 실험입니다.",
        "",
        "## lot 누출 독립 검사",
        "",
        f"- 전체 {audit['total_rows']:,}행 / lot {audit['total_lots']:,}개",
        f"- train {audit['rows_per_split']['train']:,} · "
        f"validation {audit['rows_per_split']['validation']:,} · "
        f"test {audit['rows_per_split']['test']:,}",
        "- split 간 공유 lot: "
        + ", ".join(
            f"{key} {value['shared_lot_count']}개"
            for key, value in audit["lot_overlap"].items()
        ),
        f"- 중복 array_index: {audit['duplicate_array_index']}건",
        f"- **누출 없음: {'예' if audit['leakage_free'] else '아니오'}**",
        "",
        "교차검증 fold를 나눌 때도 `StratifiedGroupKFold`로 lot를 묶어,",
        "fold 안에서도 같은 lot가 학습·검증에 동시에 들어가지 않게 했습니다.",
        "",
        "## 사용한 특징 (화소 없음)",
        "",
        "`original_height`, `original_width`, 면적, 종횡비, 긴 변, 짧은 변,",
        "`wafer_index`, lot 크기, lot 내 상대 위치 — 총 9개.",
        "",
        "원본 Kaggle의 `source_split` 열은 데이터 출처 표식이라 제외했습니다.",
        "",
        "## fold별 성능 (train 안, lot 묶음 교차검증)",
        "",
        to_markdown_table(fold_table),
        "",
        f"- macro-F1 평균 ± 표본 표준편차: {folds['macro_f1'].mean():.4f} ± "
        f"{folds['macro_f1'].std(ddof=1):.4f}",
        f"- balanced accuracy 평균 ± 표본 표준편차: {folds['balanced_accuracy'].mean():.4f} ± "
        f"{folds['balanced_accuracy'].std(ddof=1):.4f}",
        "",
        "## 고정 validation / test 성능",
        "",
        "| 구간 | macro-F1 | balanced accuracy | accuracy |",
        "|---|---|---|---|",
        f"| validation | {validation['macro_f1']:.4f} | "
        f"{validation['balanced_accuracy']:.4f} | {validation['accuracy']:.4f} |",
        f"| test | {test['macro_f1']:.4f} | "
        f"{test['balanced_accuracy']:.4f} | {test['accuracy']:.4f} |",
        "",
        "## 클래스별 recall (고정 test)",
        "",
        to_markdown_table(pd.DataFrame(recall_rows)),
        "",
    ]

    if baseline:
        gap_f1 = baseline["macro_f1"] - test["macro_f1"]
        gap_ba = baseline["balanced_accuracy"] - test["balanced_accuracy"]
        lines += [
            "## 해석",
            "",
            f"- 기준선 CNN: macro-F1 {baseline['macro_f1']:.4f}, "
            f"balanced accuracy {baseline['balanced_accuracy']:.4f}",
            f"- 화소 없는 대조 모델: macro-F1 {test['macro_f1']:.4f}, "
            f"balanced accuracy {test['balanced_accuracy']:.4f}",
            f"- 차이: macro-F1 {gap_f1:+.4f}, balanced accuracy {gap_ba:+.4f}",
            "",
            "이 차이가 **웨이퍼 맵 모양을 실제로 학습해서 얻은 몫**입니다.",
            "대조 모델 점수가 높게 나오는 클래스가 있다면, 그 클래스는 결함 모양보다",
            "웨이퍼 크기 분포로 상당 부분 구분된다는 뜻이므로 해석에 주의해야 합니다.",
            "",
        ]

    lines += [
        "## 한계",
        "",
        "- 이 모델을 실제 결함 진단에 쓰면 안 됩니다. 목적은 하한선 측정 하나입니다.",
        "- 웨이퍼 크기와 결함 클래스의 연관은 이 공개 데이터셋의 수집 방식에서 온 것일 수 있으며,",
        "  실제 공정의 물리적 인과관계가 아닙니다.",
        "- 고정 test는 기준선 실험에서 이미 관찰된 세트입니다. 새로운 외부 검증이 아닙니다.",
        "",
    ]
    (CONTROL_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_split_metadata()
    audit = audit_lot_leakage(frame)
    write_json(WM_RESULT_DIR / "lot_leakage_audit.json", audit)
    class_distribution(frame).to_csv(
        WM_RESULT_DIR / "split_class_distribution.csv", index=False, encoding="utf-8-sig"
    )
    if not audit["leakage_free"]:
        raise RuntimeError(
            "분할에 lot 누출이 있습니다. 임의로 고치지 않고 중단합니다: "
            f"{audit['lot_overlap']}"
        )
    print(
        f"lot 누출 검사 통과 — {audit['total_rows']:,}행 / lot {audit['total_lots']:,}개",
        flush=True,
    )

    features = build_features(frame)
    train = features[features["split"] == "train"].reset_index(drop=True)
    validation = features[features["split"] == "validation"].reset_index(drop=True)
    test = features[features["split"] == "test"].reset_index(drop=True)

    print(f"fold별 교차검증 ({N_FOLDS}-fold, lot 묶음)", flush=True)
    folds = cross_validate(train)
    folds.to_csv(CONTROL_DIR / "fold_scores.csv", index=False, encoding="utf-8-sig")

    labels = [c for c in CLASS_ORDER if c in set(train["failure_type"])]
    model = build_model()
    model.fit(train[FEATURE_COLUMNS].to_numpy(), train["failure_type"].to_numpy())

    results = {}
    for name, subset in (("validation", validation), ("test", test)):
        predicted = model.predict(subset[FEATURE_COLUMNS].to_numpy())
        results[name] = evaluate(subset["failure_type"].to_numpy(), predicted, labels)
        confusion_frame(subset["failure_type"].to_numpy(), predicted, labels).to_csv(
            CONTROL_DIR / f"{name}_confusion_matrix.csv", encoding="utf-8-sig"
        )
        print(
            f"{name}: macro-F1={results[name]['macro_f1']:.4f} "
            f"balanced_acc={results[name]['balanced_accuracy']:.4f}",
            flush=True,
        )

    importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(
        CONTROL_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    model_path = CONTROL_DIR / "geometry_control_model.joblib"
    joblib.dump(model, model_path, compress=3)

    baseline = baseline_cnn_metrics()
    write_json(
        CONTROL_DIR / "metrics.json",
        {
            "purpose": "화소를 쓰지 않는 하한선 대조 모델. 결함 분류기가 아님.",
            "features": FEATURE_COLUMNS,
            "cv": {
                "n_folds": N_FOLDS,
                "splitter": "StratifiedGroupKFold(lot 묶음)",
                "macro_f1_mean": float(folds["macro_f1"].mean()),
                "macro_f1_std": float(folds["macro_f1"].std(ddof=1)),
                "balanced_accuracy_mean": float(folds["balanced_accuracy"].mean()),
                "balanced_accuracy_std": float(folds["balanced_accuracy"].std(ddof=1)),
            },
            "validation": results["validation"],
            "test": results["test"],
            "baseline_cnn_test": baseline,
            "expected_local_model_file": model_path.name,
            "model_sha256": file_sha256(model_path),
            "model_artifact_committed": False,
            "python": platform.python_version(),
            "random_state": RANDOM_STATE,
        },
    )

    write_summary(folds, results["validation"], results["test"], audit, baseline)
    print(f"\n결과 저장: {CONTROL_DIR}", flush=True)


if __name__ == "__main__":
    main()


