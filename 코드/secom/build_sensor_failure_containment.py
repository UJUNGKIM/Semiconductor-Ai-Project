"""Build cross-artifact evidence that failed sensor stresses stop automation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from sensor_failure_containment import (
    build_sensor_failure_containment,
    validate_sensor_failure_containment,
)


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT / "결과물" / "secom" / "sensor_failure_containment"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_sensor_failure_containment(
        PROJECT, datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    validation = validate_sensor_failure_containment(report, PROJECT)
    (OUTPUT_DIR / "containment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    details = "\n".join(
        f"- {item['scenario']}: STOP {item['stop_count']}/{item['expected_replicates']}, "
        f"자동판정 차단 {item['automatic_decision_blocked_count']}/"
        f"{item['expected_replicates']}"
        for item in report["failed_scenario_containment"]
    )
    (OUTPUT_DIR / "summary.md").write_text(
        "# SECOM 센서 실패 안전격리 교차검증\n\n"
        f"- 판정: **{report['status']}**\n"
        f"- 모델 강건성 판정: **{report['model_robustness_guardrail_status']}**\n"
        f"- 성능 실패 조건: {report['failed_scenario_count']}개\n"
        f"- 실패 조건 전체 안전격리: {report['all_failed_scenarios_contained']}\n"
        f"- 깨끗한 배치 판정: {report['clean_batch_status']}\n\n"
        f"{details}\n\n"
        "이 검증은 취약 조건의 자동판정을 중단하는 운영 안전장치 증거이며 모델의 "
        "재현율이 개선됐다는 뜻이 아닙니다. 실제 장비 고장은 별도 전향 검증이 필요합니다.\n",
        encoding="utf-8",
    )
    print(
        f"sensor containment: status={validation['status']} "
        f"failed={validation['failed_scenario_count']} "
        f"contained={validation['all_failed_scenarios_contained']}"
    )
    if report["status"] != "PASS":
        raise RuntimeError("센서 성능 실패 조건이 모두 안전하게 격리되지 않았습니다.")


if __name__ == "__main__":
    main()
