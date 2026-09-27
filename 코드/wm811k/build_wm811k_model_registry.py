"""Build and validate the WM-811K deployment champion decision record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_artifact(path: Path) -> str:
    """Hash text artifacts with canonical LF and binary artifacts byte-for-byte."""
    if path.suffix.lower() in {".json", ".csv", ".md", ".txt"}:
        text = path.read_text(encoding="utf-8-sig")
        canonical = text.replace("\r\n", "\n").replace("\r", "\n")
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return sha256_file(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_registry(project: Path) -> tuple[dict, list[dict[str, str]]]:
    wm_root = project / "결과물" / "wm811k"
    deployed = wm_root / "selected_model_results"
    checkpoint = deployed / "best_model.pt"
    selection = load_json(deployed / "selection_protocol.json")
    run = load_json(deployed / "run_summary.json")
    uncertainty = load_json(deployed / "uncertainty_policy.json")
    ood_path = wm_root / "ood_results" / "ood_reference.json"
    ood = load_json(ood_path)
    reproducibility = load_json(
        wm_root / "multiseed_results" / "reproducibility_summary.json"
    )
    reproducibility_validation = load_json(
        wm_root / "multiseed_results" / "validation_report.json"
    )
    weak_class = load_json(
        wm_root / "weak_class_validation_results" / "experiment_summary.json"
    )
    weak_margin = load_json(
        wm_root / "weak_margin_multiseed_results" / "reproducibility_summary.json"
    )
    weak_margin_validation = load_json(
        wm_root / "weak_margin_multiseed_results" / "bundle_validation.json"
    )
    locked_ensemble = load_json(
        wm_root / "robust_ensemble_locked_test_results"
        / "locked_test_summary.json"
    )

    checkpoint_sha = sha256_file(checkpoint)
    if selection != {
        "selected_strategy": "ce_sqrt_balanced",
        "primary_metric": "validation_macro_f1",
        "tie_breaker": "validation_balanced_accuracy",
        "selection_uses_test": False,
        "random_state": 42,
    }:
        raise ValueError("Unexpected deployed model selection contract")
    if run.get("seed") != 42 or run.get("loss") != "cross_entropy" or run.get("sampling") != "sqrt_balanced":
        raise ValueError("Unexpected deployed checkpoint training contract")
    if uncertainty.get("selection_uses_test") is not False:
        raise ValueError("Uncertainty policy used test for selection")
    if ood.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("OOD reference does not match deployed checkpoint")
    if reproducibility_validation.get("deployed_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Reproducibility evidence does not match deployed checkpoint")
    if reproducibility_validation.get("seed42_checkpoint_matches_deployment") is not True:
        raise ValueError("Seed 42 reproducibility checkpoint was not verified")
    if weak_class.get("selected_candidate") != "baseline" or weak_class.get("test_evaluated") is not False:
        raise ValueError("Weak-class experiment contract is inconsistent")
    if (
        weak_margin.get("selected_candidate") != "baseline"
        or weak_margin.get("checks", {}).get("eligible") is not False
        or weak_margin_validation.get("status") != "validated"
        or weak_margin_validation.get("test_evaluated") is not False
    ):
        raise ValueError("Weak-margin multi-seed rejection contract is inconsistent")
    if (
        locked_ensemble.get("configuration_locked_from_validation") is not True
        or locked_ensemble.get("test_evaluated") is not True
        or locked_ensemble.get("used_test_for_selection") is not False
        or locked_ensemble.get("selected_robust_weight") != 0.5
        or locked_ensemble.get("selected_none_logit_bias") != 0.1
        or locked_ensemble.get("checkpoint_sha256", {}).get("deployed_baseline")
        != checkpoint_sha
        or locked_ensemble.get("locked_test_gate_passed") is not False
        or locked_ensemble.get("checks", {}).get("none_recall_guardrail") is not False
    ):
        raise ValueError("Locked ensemble test rejection contract is inconsistent")

    decisions = [
        {
            "candidate": "ce_sqrt_balanced_seed42",
            "role": "champion",
            "status": "deployed",
            "selection_scope": "validation-selected; test evaluated after freeze",
            "reason": "validation winner, checkpoint-linked OOD policy, three-seed stability evidence",
        },
        {
            "candidate": "weak_class_retraining_family",
            "role": "challenger",
            "status": "rejected",
            "selection_scope": "validation only",
            "reason": "augmentation candidates failed macro-F1 guardrail; family baseline retained only inside experiment",
        },
        {
            "candidate": "weak_none_margin_005",
            "role": "challenger",
            "status": "rejected",
            "selection_scope": "paired validation across seeds 17, 42, 2026",
            "reason": "weak recall improved but per-seed macro-F1, none recall, and balanced-accuracy guardrails failed",
        },
        {
            "candidate": "baseline_robust_ensemble_50_50_bias_010",
            "role": "challenger",
            "status": "rejected",
            "selection_scope": "three-seed validation stress selection; one-time locked test",
            "reason": "locked test improved weak recall and macro-F1 but exceeded the pre-registered none-recall loss guardrail",
        },
    ]
    manifest = {
        "schema_version": 1,
        "status": "champion_frozen",
        "automatic_promotion_performed": False,
        "deployment": {
            "model_id": "wm811k_ce_sqrt_balanced_seed42",
            "strategy": selection["selected_strategy"],
            "seed": run["seed"],
            "checkpoint_path": "결과물/wm811k/selected_model_results/best_model.pt",
            "checkpoint_sha256": checkpoint_sha,
            "class_count": len(run["class_names"]),
            "validation_macro_f1": run["validation_metrics"]["macro_f1"],
            "frozen_test_macro_f1": run["test_metrics"]["macro_f1"],
            "selection_uses_test": False,
        },
        "safety_artifacts": {
            "uncertainty_policy_path": "결과물/wm811k/selected_model_results/uncertainty_policy.json",
            "uncertainty_policy_sha256": sha256_artifact(deployed / "uncertainty_policy.json"),
            "ood_reference_path": "결과물/wm811k/ood_results/ood_reference.json",
            "ood_reference_sha256": sha256_artifact(ood_path),
            "ood_checkpoint_matches_champion": True,
        },
        "reproducibility": {
            "seeds": reproducibility["seeds"],
            "strategy": reproducibility["strategy_id"],
            "validation_macro_f1_mean": reproducibility["metrics"]["validation_macro_f1"]["mean"],
            "validation_macro_f1_std": reproducibility["metrics"]["validation_macro_f1"]["std"],
            "test_macro_f1_mean": reproducibility["metrics"]["test_macro_f1"]["mean"],
            "test_macro_f1_std": reproducibility["metrics"]["test_macro_f1"]["std"],
            "seed42_checkpoint_matches_champion": True,
        },
        "latest_challenger_gate": {
            "candidate": "baseline_robust_ensemble_50_50_bias_010",
            "eligible": False,
            "validation_gate_passed": True,
            "validation_eligible_configuration_count": 442,
            "robust_weight": locked_ensemble["selected_robust_weight"],
            "none_logit_bias": locked_ensemble["selected_none_logit_bias"],
            "weak_recall_delta": locked_ensemble["candidate_minus_deployed"]["weak_recall"],
            "macro_f1_delta": locked_ensemble["candidate_minus_deployed"]["macro_f1"],
            "accuracy_delta": locked_ensemble["candidate_minus_deployed"]["accuracy"],
            "none_recall_delta": locked_ensemble["candidate_minus_deployed"]["none_recall"],
            "failed_guardrails": [
                key for key, passed in locked_ensemble["checks"].items()
                if not passed
            ],
            "test_sample_count": locked_ensemble["test_sample_count"],
            "test_evaluated": True,
            "used_test_for_selection": False,
        },
        "candidate_decisions": decisions,
        "decision": "keep_existing_champion_and_freeze_current_model_selection",
    }
    return manifest, decisions


def validate_registry(manifest: dict, project: Path) -> dict[str, object]:
    checkpoint = project / manifest["deployment"]["checkpoint_path"]
    uncertainty = project / manifest["safety_artifacts"]["uncertainty_policy_path"]
    ood = project / manifest["safety_artifacts"]["ood_reference_path"]
    checks = {
        "champion_checkpoint_hash_valid": sha256_file(checkpoint) == manifest["deployment"]["checkpoint_sha256"],
        "uncertainty_policy_hash_valid": sha256_artifact(uncertainty) == manifest["safety_artifacts"]["uncertainty_policy_sha256"],
        "ood_reference_hash_valid": sha256_artifact(ood) == manifest["safety_artifacts"]["ood_reference_sha256"],
        "test_not_used_for_selection": manifest["deployment"]["selection_uses_test"] is False,
        "challenger_not_promoted": manifest["latest_challenger_gate"]["eligible"] is False,
        "automatic_promotion_disabled": manifest["automatic_promotion_performed"] is False,
    }
    return {"status": "validated" if all(checks.values()) else "invalid", "checks": checks}


def write_registry(output_dir: Path, manifest: dict, decisions: list[dict[str, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_registry(manifest, output_dir.parents[2])
    if validation["status"] != "validated":
        raise ValueError(f"Registry validation failed: {validation['checks']}")
    (output_dir / "champion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "registry_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "candidate_decisions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=decisions[0].keys())
        writer.writeheader()
        writer.writerows(decisions)
    deployment = manifest["deployment"]
    challenger = manifest["latest_challenger_gate"]
    lines = [
        "# WM-811K 운영 모델 결정 기록",
        "",
        f"- 현재 champion: `{deployment['model_id']}`",
        f"- 체크포인트 SHA-256: `{deployment['checkpoint_sha256']}`",
        f"- Validation Macro-F1: {deployment['validation_macro_f1']:.4f}",
        f"- 선택 고정 후 Test Macro-F1: {deployment['frozen_test_macro_f1']:.4f}",
        "- Test를 모델 선택에 사용: 아니요",
        "- 자동 승격: 수행하지 않음",
        "",
        "## 최신 challenger 결정",
        "",
        f"`{challenger['candidate']}`는 validation 견고성 기준을 통과한 뒤 고정 test에서 "
        f"취약 클래스 재현율 {challenger['weak_recall_delta']:+.2%}p, Macro-F1 "
        f"{challenger['macro_f1_delta']:+.2%}p를 개선했습니다. 그러나 정상 재현율 차이 "
        f"{challenger['none_recall_delta']:+.2%}p가 사전 보호 기준을 넘어서 승격하지 "
        "않았습니다.",
        "",
        "Test 결과를 사용해 혼합 비율이나 bias를 다시 조정하지 않았습니다.",
        "",
        "따라서 실제 대시보드 추론은 계속 기존 champion 체크포인트와 이에 연결된 "
        "불확실성·OOD 정책을 사용합니다.",
    ]
    (output_dir / "결정기록.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    output_dir = args.output_dir or project / "결과물" / "wm811k" / "model_registry"
    manifest, decisions = build_registry(project)
    write_registry(output_dir, manifest, decisions)
    print(json.dumps(validate_registry(manifest, project), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
