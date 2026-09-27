"""Cross-check sensor robustness failures against the batch circuit breaker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


INPUT_FILES = {
    "robustness_summary": "결과물/secom/robustness_results/robustness_summary.json",
    "stress_replicates": "결과물/secom/robustness_results/stress_test_replicates.csv",
    "batch_gate_summary": "결과물/secom/batch_safety_gate/batch_gate_summary.json",
    "gate_validation": "결과물/secom/batch_safety_gate/gate_stress_validation.csv",
}


def _text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def _load_inputs(project: Path) -> tuple[dict, pd.DataFrame, dict, pd.DataFrame]:
    paths = {name: project / relative for name, relative in INPUT_FILES.items()}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"안전격리 입력 증거 누락: {missing}")
    robustness = json.loads(paths["robustness_summary"].read_text(encoding="utf-8"))
    stress = pd.read_csv(paths["stress_replicates"])
    gate_summary = json.loads(paths["batch_gate_summary"].read_text(encoding="utf-8"))
    gate = pd.read_csv(paths["gate_validation"])
    return robustness, stress, gate_summary, gate


def _core_evidence(project: Path) -> dict:
    robustness, stress, gate_summary, gate = _load_inputs(project)
    keys = ["kind", "severity", "replicate"]
    stress_required = set(
        keys
        + ["scenario", "ood_review_rate", "ood_severe_rate", "model_disagreement_rate"]
    )
    gate_required = set(
        keys
        + [
            "scenario_korean",
            "ood_any_rate",
            "ood_severe_rate",
            "model_disagreement_rate",
            "status",
            "automatic_decision_allowed",
        ]
    )
    if not stress_required.issubset(stress.columns) or not gate_required.issubset(gate.columns):
        raise ValueError("강건성 또는 차단기 검증 열 계약이 올바르지 않습니다.")
    if stress.duplicated(keys).any() or gate.duplicated(keys).any():
        raise ValueError("강건성 또는 차단기 검증 키가 중복됐습니다.")
    joined = stress.merge(
        gate,
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_stress", "_gate"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("강건성 반복 실험과 차단기 검증 행이 일대일로 연결되지 않습니다.")
    rate_pairs = (
        ("ood_review_rate", "ood_any_rate"),
        ("ood_severe_rate_stress", "ood_severe_rate_gate"),
        ("model_disagreement_rate_stress", "model_disagreement_rate_gate"),
    )
    for left, right in rate_pairs:
        if not np.allclose(
            joined[left].to_numpy(dtype=float),
            joined[right].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"강건성·차단기 신호 불일치: {left} != {right}")

    failed = list(robustness.get("failed_scenarios", []))
    if int(robustness.get("guardrail_fail_count", -1)) != len(failed):
        raise ValueError("강건성 실패 조건 수와 목록이 다릅니다.")
    expected_replicates = int(robustness.get("replicates_per_scenario", 0))
    containment = []
    for scenario in failed:
        rows = joined.loc[joined["scenario_stress"] == scenario]
        auto_allowed = (
            rows["automatic_decision_allowed"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        )
        stop_count = int(rows["status"].eq("STOP").sum())
        blocked_count = int((~auto_allowed).sum())
        contained = (
            len(rows) == expected_replicates
            and stop_count == expected_replicates
            and blocked_count == expected_replicates
        )
        containment.append(
            {
                "scenario": scenario,
                "expected_replicates": expected_replicates,
                "matched_replicates": int(len(rows)),
                "stop_count": stop_count,
                "automatic_decision_blocked_count": blocked_count,
                "stop_rate": float(stop_count / expected_replicates)
                if expected_replicates
                else 0.0,
                "contained": contained,
            }
        )
    clean = gate_summary.get("clean_test_decision", {})
    clean_released = (
        clean.get("status") == "PASS"
        and clean.get("automatic_decision_allowed") is True
    )
    all_contained = bool(failed) and all(item["contained"] for item in containment)
    status = "PASS" if all_contained and clean_released else "BLOCK"
    input_hashes = {
        name: _text_sha256(project / relative) for name, relative in INPUT_FILES.items()
    }
    return {
        "report_version": 1,
        "status": status,
        "scope": "post_selection_fixed_test_sensor_stress_containment",
        "model_robustness_guardrail_status": "WARN" if failed else "PASS",
        "model_robustness_improved_by_this_audit": False,
        "failed_scenario_count": len(failed),
        "failed_scenarios": failed,
        "failed_scenario_containment": containment,
        "all_failed_scenarios_contained": all_contained,
        "clean_batch_status": clean.get("status"),
        "clean_batch_automatic_decision_allowed": clean.get(
            "automatic_decision_allowed"
        ),
        "stress_row_count": int(len(stress)),
        "gate_row_count": int(len(gate)),
        "cross_artifact_signal_match": True,
        "test_used_for_model_or_threshold_selection": False,
        "raw_sensor_values_persisted": False,
        "actual_hardware_faults_validated": False,
        "input_hashes": input_hashes,
        "limitations": [
            "고정 test 합성 교란의 사후 안전격리 검증이며 모델 성능 개선이 아닙니다.",
            "실제 장비 고장·신규 lot·운영 대응시간을 검증하지 않았습니다.",
            "배치 STOP은 원인 진단이 아니라 자동판정 보류와 재측정 요구입니다.",
        ],
    }


def build_sensor_failure_containment(project: Path, evaluated_at: str) -> dict:
    return {"evaluated_at": evaluated_at, **_core_evidence(project)}


def validate_sensor_failure_containment(report: Mapping[str, object], project: Path) -> dict:
    expected = _core_evidence(project)
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"안전격리 증거가 현재 입력과 다릅니다: {key}")
    if not str(report.get("evaluated_at", "")).endswith("Z"):
        raise ValueError("안전격리 검증 시각은 UTC Z 형식이어야 합니다.")
    return {
        "valid": True,
        "status": report["status"],
        "failed_scenario_count": report["failed_scenario_count"],
        "all_failed_scenarios_contained": report["all_failed_scenarios_contained"],
    }
