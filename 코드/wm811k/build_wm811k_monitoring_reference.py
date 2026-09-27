"""Build a validation-only reference for WM-811K batch drift monitoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wm811k_monitoring import OOD_STATUSES, load_monitoring_reference


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_reference(project: Path) -> dict:
    wm_root = project / "결과물" / "wm811k"
    run_summary_path = wm_root / "selected_model_results" / "run_summary.json"
    records_path = wm_root / "ood_results" / "ood_score_records.csv"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    class_names = [str(value) for value in run_summary["class_names"]]
    records = pd.read_csv(records_path)
    required = {"split", "predicted_label_id", "ood_score", "ood_status"}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"OOD 기록 열 누락: {sorted(missing)}")
    validation = records.loc[records["split"].eq("validation")].copy()
    if validation.empty or set(validation["split"]) != {"validation"}:
        raise ValueError("validation OOD 기록이 없습니다.")
    predicted_ids = validation["predicted_label_id"].astype(int)
    if predicted_ids.min() < 0 or predicted_ids.max() >= len(class_names):
        raise ValueError("validation 예측 클래스 ID가 범위를 벗어납니다.")
    unknown_statuses = set(validation["ood_status"].astype(str)).difference(OOD_STATUSES)
    if unknown_statuses:
        raise ValueError(f"알 수 없는 OOD 상태: {sorted(unknown_statuses)}")
    scores = validation["ood_score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("validation OOD 점수가 유한하지 않습니다.")
    class_counts = predicted_ids.value_counts().reindex(range(len(class_names)), fill_value=0)
    status_counts = (
        validation["ood_status"].astype(str).value_counts().reindex(OOD_STATUSES, fill_value=0)
    )
    sample_count = len(validation)
    p95 = float(np.quantile(scores, 0.95))
    reference = {
        "version": 1,
        "source_split": "validation",
        "source_record_path": "결과물/wm811k/ood_results/ood_score_records.csv",
        "source_record_sha256": sha256_file(records_path),
        "model_checkpoint_sha256": sha256_file(
            wm_root / "selected_model_results" / "best_model.pt"
        ),
        "validation_sample_count": sample_count,
        "class_names": class_names,
        "class_probabilities": {
            name: float(class_counts[index] / sample_count)
            for index, name in enumerate(class_names)
        },
        "ood_status_probabilities": {
            status: float(status_counts[status] / sample_count) for status in OOD_STATUSES
        },
        "ood_score_quantiles": {
            "p50": float(np.quantile(scores, 0.50)),
            "p95": p95,
            "p99": float(np.quantile(scores, 0.99)),
        },
        "above_p95_rate": float(np.mean(scores > p95)),
        "monitoring_config": {
            "minimum_batch_size": 20,
            "bootstrap_quantile": 0.99,
            "bootstrap_repetitions": 4000,
            "seed": 42,
        },
        "test_split_used": False,
        "labels_required_during_monitoring": False,
        "automatic_retraining": False,
    }
    return reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    output_dir = args.output_dir or project / "결과물" / "wm811k" / "monitoring_reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / "monitoring_reference.json"
    reference = build_reference(project)
    reference_path.write_text(
        json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    load_monitoring_reference(reference_path)
    (output_dir / "README.md").write_text(
        "# WM-811K 배치 드리프트 감시 기준\n\n"
        f"- 기준 분할: validation {reference['validation_sample_count']:,}개\n"
        "- 감시 신호: 예측 클래스 구성, 검토·OOD 비율, severe OOD 비율, OOD 상위 점수 비율\n"
        "- 판정 경계: validation 분포에서 같은 크기의 배치를 4,000회 모의한 99% 상한\n"
        "- 최소 배치: 20개\n\n"
        "고정 test와 정답은 기준 생성에 사용하지 않습니다. HOLD는 성능 저하 확정이나 자동 재학습 "
        "명령이 아니라 배치 자동판정을 멈추고 입력·공정 변화를 검토하라는 안전 신호입니다.\n",
        encoding="utf-8",
    )
    print(f"WM811K monitoring reference: validation={reference['validation_sample_count']:,}")


if __name__ == "__main__":
    main()
