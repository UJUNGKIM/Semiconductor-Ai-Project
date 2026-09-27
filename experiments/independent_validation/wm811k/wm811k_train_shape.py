"""WM-811K 형태 서술자 기반 독립 분류기 학습·평가.

기준선(2채널 CNN + class-balanced focal loss + GPU)과 다른 구성:

| 항목 | 기준선 | 독립 검증 |
|---|---|---|
| 표현 | CNN이 화소에서 학습 | 52개 기하 서술자를 직접 정의 |
| 분류기 | CNN head | HistGradientBoosting (표 데이터용 부스팅) |
| 불균형 | class-balanced focal loss, 제곱근 균형 샘플링 | 클래스 역빈도 sample_weight |
| 증강 | 회전·반전 | 없음 (서술자 대부분이 회전 불변) |
| 설명 | Grad-CAM, Integrated Gradients | 이름이 붙은 서술자의 순열 중요도 |
| 연산 | T4 GPU | CPU |

분할과 지표는 비교를 위해 기준선과 동일하게 맞춥니다.
lot 비중복 고정 분할(train 123,542 / validation 24,703 / test 24,705),
macro-F1 · balanced accuracy · 클래스별 recall.
교차검증도 `StratifiedGroupKFold`로 lot를 묶어 fold 안에서도 누출이 없게 합니다.

실행:
    # 전체 학습 (wafer_maps_64.npy 필요)
    .venv\\Scripts\\python.exe "experiments\\independent_validation\\wm811k\\wm811k_train_shape.py"

    # 원본 없이 파이프라인만 점검 (저장소의 정성 예시 사용)
    .venv\\Scripts\\python.exe "experiments\\independent_validation\\wm811k\\wm811k_train_shape.py" --demo
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PROJECT_DIR,
    RANDOM_STATE,
    file_sha256,
    to_markdown_table,
    write_json,
)
import wm811k_shape_features as shape_features  # noqa: E402
from wm811k_common import (  # noqa: E402
    CLASS_ORDER,
    WM_RESULT_DIR,
    MissingPixelData,
    audit_lot_leakage,
    confusion_frame,
    evaluate,
    find_pixel_array,
    load_demo_samples,
    load_split_metadata,
    load_pixel_arrays,
)

SHAPE_DIR = WM_RESULT_DIR / "shape_model"
# 추출한 특징은 용량이 커서 Git에 넣지 않는다(.gitignore 처리).
CACHE_DIR = WM_RESULT_DIR / "cache"
N_FOLDS = 5
# test 표본이 이보다 적은 클래스는 recall 한 건의 변동이 커서 따로 표시한다.
SMALL_SAMPLE_THRESHOLD = 200


def build_model() -> HistGradientBoostingClassifier:
    """표 데이터용 부스팅. SECOM 쪽 ExtraTrees와도 다른 계열을 골랐다."""
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def class_weights(y: np.ndarray) -> np.ndarray:
    """클래스 역빈도 가중치. none이 전체의 86%라 그대로 두면 다 none이라 한다."""
    classes, counts = np.unique(y, return_counts=True)
    weight = {name: len(y) / (len(classes) * count) for name, count in zip(classes, counts)}
    return np.array([weight[value] for value in y], dtype=float)


def load_or_extract_features(frame: pd.DataFrame) -> np.ndarray:
    """형태 특징을 추출한다. 한 번 만들면 캐시에서 다시 읽는다."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"shape_features_{len(frame)}.npy"
    if cache_path.is_file():
        cached = np.load(cache_path)
        if cached.shape == (len(frame), len(shape_features.feature_names())):
            print(f"캐시에서 특징 로드: {cache_path.name}", flush=True)
            return cached
        print("캐시 형상이 달라 다시 추출합니다.", flush=True)

    array = load_pixel_arrays(frame)
    indices = frame["array_index"].to_numpy()
    print(f"형태 특징 추출 시작: {len(indices):,}장", flush=True)

    started = time.perf_counter()
    rows = np.empty((len(indices), len(shape_features.feature_names())), dtype=float)
    for position, array_index in enumerate(indices):
        rows[position] = shape_features.extract(array[array_index])
        if (position + 1) % 10_000 == 0:
            elapsed = time.perf_counter() - started
            rate = (position + 1) / elapsed
            remaining = (len(indices) - position - 1) / rate
            print(
                f"  {position + 1:,}/{len(indices):,} "
                f"({rate:.0f}장/초, 남은 시간 약 {remaining / 60:.1f}분)",
                flush=True,
            )
    np.save(cache_path, rows)
    print(f"특징 캐시 저장: {cache_path}", flush=True)
    return rows


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    labels = [c for c in CLASS_ORDER if c in set(y)]
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        shared = set(groups[fit_idx]) & set(groups[val_idx])
        if shared:
            raise RuntimeError(f"fold {fold}에서 lot 누출: {sorted(shared)[:3]}")

        model = build_model()
        model.fit(X[fit_idx], y[fit_idx], sample_weight=class_weights(y[fit_idx]))
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


def descriptor_importance(model, X: np.ndarray, y: np.ndarray, sample_size: int = 4000) -> pd.DataFrame:
    """이름이 붙은 서술자의 순열 중요도.

    Grad-CAM은 '어느 화소를 봤는가'를 보여 주지만, 이 표는
    '어떤 모양 성질이 판단을 좌우했는가'를 바로 말해 준다.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    if len(X) > sample_size:
        pick = rng.choice(len(X), size=sample_size, replace=False)
        X, y = X[pick], y[pick]

    result = permutation_importance(
        model,
        X,
        y,
        scoring="f1_macro",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "descriptor": shape_features.feature_names(),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def run_demo() -> None:
    """원본 없이 코드 경로만 점검한다. 성능 수치로 쓰면 안 된다."""
    maps, names = load_demo_samples()
    labels = np.array([name.split("_")[0] for name in names])
    X = shape_features.extract_many(maps)

    print(f"정성 예시 {len(maps)}장에서 특징 {X.shape[1]}개 추출 완료")
    print(f"결측·무한대 없음: {not np.isnan(X).any() and not np.isinf(X).any()}")

    # 본 학습용 파라미터(min_samples_leaf=40)는 정성 예시 수보다 커서 분기가 생기지 않는다.
    # 점검 목적이므로 작은 데이터용 값으로 바꿔 코드 경로가 도는지만 본다.
    model = HistGradientBoostingClassifier(
        max_iter=50,
        min_samples_leaf=1,
        max_leaf_nodes=8,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    model.fit(X, labels, sample_weight=class_weights(labels))
    predicted = model.predict(X)
    print(f"클래스 {len(set(labels))}종, 학습 데이터 재예측 일치율: {(predicted == labels).mean():.3f}")
    print(
        f"\n이것은 파이프라인 점검일 뿐입니다. {len(maps)}장으로 학습하고 같은 {len(maps)}장을 예측했으므로\n"
        "성능 지표로 해석하면 안 됩니다. 실제 평가는 wafer_maps_64.npy가 있어야 합니다."
    )


def write_summary(
    folds: pd.DataFrame,
    validation: dict,
    test: dict,
    importance: pd.DataFrame,
    audit: dict,
    baseline: dict | None,
) -> None:
    recall_rows = []
    for name in CLASS_ORDER:
        if name not in test["per_class_recall"]:
            continue
        support = test["support"].get(name, 0)
        row = {
            "클래스": name,
            "test 건수": support,
            # test 표본이 적으면 recall 한 건이 크게 흔들린다. 표에서 바로 보이게 둔다.
            "표본 적음": "주의" if support < SMALL_SAMPLE_THRESHOLD else "",
            "독립 검증 recall": round(test["per_class_recall"][name], 4),
            "독립 검증 F1": round(test["per_class_f1"].get(name, 0.0), 4),
        }
        if baseline:
            row["기준선 CNN recall"] = round(
                baseline["per_class_recall"].get(name, float("nan")), 4
            )
        recall_rows.append(row)

    lines = [
        "# WM-811K 형태 서술자 모델 (독립 검증)",
        "",
        "기준선 CNN과 같은 lot 비중복 분할·같은 지표로 평가한 독립 구현입니다.",
        "화소에서 표현을 학습하는 대신 52개 기하 서술자를 직접 정의해 부스팅에 넣었습니다.",
        "",
        "## lot 누출 검사",
        "",
        f"- split 간 공유 lot: "
        + ", ".join(
            f"{key} {value['shared_lot_count']}개"
            for key, value in audit["lot_overlap"].items()
        ),
        f"- 교차검증도 lot 묶음(`StratifiedGroupKFold`)이라 fold 안에서도 누출이 없습니다.",
        "",
        "## fold별 성능 (train 안, lot 묶음 교차검증)",
        "",
        to_markdown_table(
            folds[
                ["fold", "fit_rows", "valid_rows", "valid_lots", "macro_f1", "balanced_accuracy", "accuracy"]
            ].round(4)
        ),
        "",
        f"- macro-F1: {folds['macro_f1'].mean():.4f} ± {folds['macro_f1'].std(ddof=1):.4f}",
        f"- balanced accuracy: {folds['balanced_accuracy'].mean():.4f} ± "
        f"{folds['balanced_accuracy'].std(ddof=1):.4f}",
        "",
        "## 고정 validation / test",
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
        f"`표본 적음 = 주의`는 test 표본이 {SMALL_SAMPLE_THRESHOLD}장 미만이라는 뜻입니다. "
        + ", ".join(
            f"`{row['클래스']}` {row['test 건수']}장"
            for row in recall_rows
            if row["표본 적음"]
        )
        + "이 여기 해당합니다.",
        "",
        "특히 `Near-full`은 "
        + str(
            next(
                (r["test 건수"] for r in recall_rows if r["클래스"] == "Near-full"), 0
            )
        )
        + "장뿐이라 recall이 1.0이어도 한 장만 틀리면 값이 크게 바뀝니다. "
        "이 숫자를 '완벽하게 잡는다'로 읽으면 안 됩니다. 신뢰구간을 포함한 비교는 "
        "`결과/wm811k/final_comparison/summary.md`에 있습니다.",
        "",
        "## 어떤 모양 성질이 중요했는가 (순열 중요도 상위 15)",
        "",
        to_markdown_table(importance.head(15).round(5)),
        "",
        "Grad-CAM은 '어느 화소를 봤는지'를 그림으로 보여 줍니다.",
        "이 표는 '어떤 모양 성질이 판단을 좌우했는지'를 이름으로 말해 줍니다.",
        "두 방법은 서로를 대체하지 않으며, 같은 모델에 대한 설명도 아닙니다.",
        "",
        "## 한계",
        "",
        "- 정의하지 않은 패턴은 이 모델이 볼 수 없습니다. CNN이 스스로 찾는 표현과 다릅니다.",
        "- 64×64로 줄이는 과정에서 작은 결함이 뭉개집니다. 원본 크기가 제각각이라",
        "  같은 결함도 웨이퍼 크기에 따라 다르게 보일 수 있습니다.",
        "- 고정 test는 기준선 실험에서 이미 관찰된 세트입니다. 새로운 외부 검증이 아닙니다.",
        "- 웨이퍼 맵 분류는 물리적 고장 원인이나 공정 조정 지시를 증명하지 않습니다.",
        "",
    ]
    (SHAPE_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def rebuild_summary() -> None:
    """이미 저장된 산출물만 읽어 summary.md를 다시 쓴다(재학습 없음).

    문서 문구만 손봤을 때 3분짜리 특징 추출과 학습을 다시 돌리지 않으려고 둔다.
    지표는 metrics.json·fold_scores.csv에서 그대로 읽으므로 값이 바뀌지 않는다.
    """
    from common import read_json

    metrics_path = SHAPE_DIR / "metrics.json"
    if not metrics_path.is_file():
        raise SystemExit(
            "학습 산출물이 없습니다. --summary-only는 이미 학습한 뒤에만 쓸 수 있습니다."
        )

    metrics = read_json(metrics_path)
    folds = pd.read_csv(SHAPE_DIR / "fold_scores.csv")
    importance = pd.read_csv(SHAPE_DIR / "descriptor_importance.csv")
    audit = audit_lot_leakage(load_split_metadata())

    write_summary(
        folds,
        metrics["validation"],
        metrics["test"],
        importance,
        audit,
        metrics.get("baseline_cnn_test"),
    )
    print(f"summary.md 재생성 완료: {SHAPE_DIR / 'summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="WM-811K 형태 서술자 모델")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="원본 배열 없이 저장소의 정성 예시로 코드 경로만 점검",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="재학습 없이 저장된 산출물로 summary.md만 다시 생성",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if args.summary_only:
        rebuild_summary()
        return

    SHAPE_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_split_metadata()
    audit = audit_lot_leakage(frame)
    if not audit["leakage_free"]:
        raise RuntimeError(f"분할에 lot 누출이 있습니다: {audit['lot_overlap']}")

    try:
        pixel_path = find_pixel_array()
    except MissingPixelData as error:
        print(str(error), file=sys.stderr)
        print(
            "\n--demo 를 붙이면 원본 없이 코드 경로만 점검할 수 있습니다.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(f"웨이퍼 맵 배열: {pixel_path}", flush=True)
    X = load_or_extract_features(frame)
    y = frame["failure_type"].to_numpy()
    groups = frame["lot_name"].to_numpy()
    split = frame["split"].to_numpy()

    train_mask = split == "train"
    labels = [c for c in CLASS_ORDER if c in set(y[train_mask])]

    print(f"fold별 교차검증 ({N_FOLDS}-fold, lot 묶음)", flush=True)
    folds = cross_validate(X[train_mask], y[train_mask], groups[train_mask])
    folds.to_csv(SHAPE_DIR / "fold_scores.csv", index=False, encoding="utf-8-sig")

    model = build_model()
    model.fit(X[train_mask], y[train_mask], sample_weight=class_weights(y[train_mask]))

    results = {}
    for name in ("validation", "test"):
        mask = split == name
        predicted = model.predict(X[mask])
        results[name] = evaluate(y[mask], predicted, labels)
        confusion_frame(y[mask], predicted, labels).to_csv(
            SHAPE_DIR / f"{name}_confusion_matrix.csv", encoding="utf-8-sig"
        )
        pd.DataFrame(
            {
                "array_index": frame.loc[mask, "array_index"].to_numpy(),
                "lot_name": groups[mask],
                "actual": y[mask],
                "predicted": predicted,
            }
        ).to_csv(SHAPE_DIR / f"{name}_predictions.csv", index=False, encoding="utf-8-sig")
        print(
            f"{name}: macro-F1={results[name]['macro_f1']:.4f} "
            f"balanced_acc={results[name]['balanced_accuracy']:.4f}",
            flush=True,
        )

    print("서술자 순열 중요도 계산", flush=True)
    importance = descriptor_importance(model, X[split == "validation"], y[split == "validation"])
    importance.to_csv(
        SHAPE_DIR / "descriptor_importance.csv", index=False, encoding="utf-8-sig"
    )

    model_path = SHAPE_DIR / "shape_model.joblib"
    joblib.dump(model, model_path, compress=3)

    from wm811k_geometry_control import baseline_cnn_metrics

    baseline = baseline_cnn_metrics()
    write_json(
        SHAPE_DIR / "metrics.json",
        {
            "approach": "52개 기하 서술자 + HistGradientBoosting (CPU)",
            "descriptor_count": len(shape_features.feature_names()),
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
            "pixel_source": pixel_path.relative_to(PROJECT_DIR).as_posix(),
            "python": platform.python_version(),
            "random_state": RANDOM_STATE,
        },
    )
    write_summary(folds, results["validation"], results["test"], importance, audit, baseline)
    print(f"\n결과 저장: {SHAPE_DIR}", flush=True)


if __name__ == "__main__":
    main()


