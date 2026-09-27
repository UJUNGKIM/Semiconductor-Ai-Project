"""독립 검증 결과와 기존 기준선 결과를 나란히 놓는 비교표를 만든다.

기준선 결과물(`결과물/secom/**`)은 **읽기만** 한다. 어떤 파일도 쓰거나 고치지 않는다.
비교 대상은 모두 같은 고정 test 314건(불량 21건) 위의 숫자다.

주의: 여기서 만든 표는 참고용 나란히 보기다. 최종 비교·선택은 별도 단계에서
두 사람이 함께 결정하기로 한 사항이며, 이 스크립트가 승패를 정하지 않는다.

실행:
    .venv\\Scripts\\python.exe "experiments\\independent_validation\\secom\\compare_with_baseline.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BASELINE_RESULT_DIR,
    RESULT_DIR,
    fbeta,
    read_json,
    to_markdown_table,
    write_json,
)

BASELINE_TUNED = BASELINE_RESULT_DIR / "tuning_results" / "test_confirmation.json"
BASELINE_MODELS = BASELINE_RESULT_DIR / "model_results" / "test_results.csv"


def collect_baseline_rows() -> list[dict]:
    """기준선의 고정 test 숫자를 모은다. 파일이 없으면 그 항목만 건너뛴다."""
    rows: list[dict] = []

    if BASELINE_TUNED.is_file():
        payload = read_json(BASELINE_TUNED)
        metrics = payload["test_metrics"]
        rows.append(
            {
                "출처": "기준선",
                "모델": f"{payload['model']} 튜닝 ({payload['config_id']})",
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f2": metrics["f2"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
                "threshold": metrics["threshold"],
            }
        )

    if BASELINE_MODELS.is_file():
        frame = pd.read_csv(BASELINE_MODELS)
        for _, row in frame.iterrows():
            rows.append(
                {
                    "출처": "기준선",
                    "모델": f"{row['model']} / {row['strategy']}",
                    "pr_auc": float(row["pr_auc"]),
                    "roc_auc": float(row["roc_auc"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f2": fbeta(float(row["precision"]), float(row["recall"])),
                    "tp": int(row["tp"]),
                    "fp": int(row["fp"]),
                    "fn": int(row["fn"]),
                    "tn": int(row["tn"]),
                    "threshold": float(row["threshold"]),
                }
            )
    return rows


def collect_friend_row() -> dict:
    payload = read_json(RESULT_DIR / "test_confirmation.json")
    metrics = payload["test_metrics"]
    return {
        "출처": "독립 검증",
        "모델": f"{payload['candidate']} (대안)",
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f2": metrics["f2"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "threshold": metrics["threshold"],
    }


def build_comparison() -> pd.DataFrame:
    rows = collect_baseline_rows()
    if not rows:
        raise FileNotFoundError(
            "기준선 결과 파일을 찾지 못했습니다. 저장소가 완전한지 확인하세요."
        )
    rows.append(collect_friend_row())
    frame = pd.DataFrame(rows)
    return frame.sort_values("f2", ascending=False).reset_index(drop=True)


def write_outputs(frame: pd.DataFrame) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        RESULT_DIR / "baseline_comparison.csv", index=False, encoding="utf-8-sig"
    )

    friend = frame[frame["출처"] == "독립 검증"].iloc[0]
    tuned = frame[frame["모델"].str.contains("튜닝", na=False)]
    display = frame[
        ["출처", "모델", "pr_auc", "roc_auc", "precision", "recall", "f2", "tp", "fp", "fn"]
    ].round(4)

    lines = [
        "# 고정 test 314건 나란히 보기 (기준선 vs 독립 검증)",
        "",
        "모든 행은 **같은 test 314건(불량 21건)** 위의 숫자입니다.",
        "기준선 수치는 `결과물/secom/`에서 읽기만 했고 어떤 파일도 바꾸지 않았습니다.",
        "F2는 불량을 놓치는 쪽을 더 무겁게 보는 지표입니다.",
        "",
        display.pipe(to_markdown_table),
        "",
        "## 읽는 법",
        "",
        "- 불량은 test 314건 중 21건뿐입니다. recall 0.05 차이는 웨이퍼 1건 차이입니다.",
        "  표의 순위를 성능 우열 결론으로 쓰면 안 됩니다.",
        "- 임계값 기준이 모델마다 다릅니다. 기준선 비교 실험은 fold별 F1 최적 임계값,",
        "  기준선 튜닝 모델과 독립 검증는 OOF에서 recall ≥ 0.60 조건의 F2 최적 임계값을 씁니다.",
        "- 이 test 분할은 양쪽 모두 이전 실험에서 이미 본 세트입니다. 새 외부 검증이 아닙니다.",
        "",
    ]

    if not tuned.empty:
        base = tuned.iloc[0]
        lines += [
            "## 기준선 튜닝 모델과의 차이",
            "",
            f"- 기준선: **{base['모델']}** — PR-AUC {base['pr_auc']:.4f}, "
            f"recall {base['recall']:.4f}, F2 {base['f2']:.4f}, "
            f"TP={int(base['tp'])} FP={int(base['fp'])} FN={int(base['fn'])}",
            f"- 독립 검증: **{friend['모델']}** — PR-AUC {friend['pr_auc']:.4f}, "
            f"recall {friend['recall']:.4f}, F2 {friend['f2']:.4f}, "
            f"TP={int(friend['tp'])} FP={int(friend['fp'])} FN={int(friend['fn'])}",
            f"- 차이(독립 검증 − 기준선): PR-AUC {friend['pr_auc'] - base['pr_auc']:+.4f}, "
            f"recall {friend['recall'] - base['recall']:+.4f}, "
            f"F2 {friend['f2'] - base['f2']:+.4f}, "
            f"놓친 불량 {int(friend['fn']) - int(base['fn']):+d}건, "
            f"헛경보 {int(friend['fp']) - int(base['fp']):+d}건",
            "",
            "두 모델의 차이는 불량 몇 건 수준이라 통계적으로 구분되지 않습니다.",
            "최종 선택은 놓친 불량 1건과 헛경보 1건 중 무엇이 더 비싼지에 대한",
            "운영 판단이 필요하며, 이 문서가 대신 결정하지 않습니다.",
            "",
        ]

    (RESULT_DIR / "baseline_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    write_json(
        RESULT_DIR / "baseline_comparison.json",
        {
            "test_rows": 314,
            "test_positives": 21,
            "note": "같은 고정 test 위의 참고용 나란히 보기. 최종 선택 근거가 아님.",
            "rows": frame.to_dict(orient="records"),
        },
    )


def main() -> None:
    frame = build_comparison()
    write_outputs(frame)
    print(frame[["출처", "모델", "pr_auc", "recall", "f2", "fn", "fp"]].to_string(index=False))
    print(f"\n저장: {RESULT_DIR / 'baseline_comparison.md'}")


if __name__ == "__main__":
    main()


