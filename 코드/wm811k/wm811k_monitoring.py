"""Label-free batch drift guardrail for deployed WM-811K inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


OOD_STATUSES = ("in_distribution", "review", "out_of_distribution")


def load_monitoring_reference(path: Path) -> dict:
    """Load and validate a monitoring reference without deserializing code."""
    reference = json.loads(path.read_text(encoding="utf-8"))
    if int(reference.get("version", 0)) != 1:
        raise ValueError("지원하지 않는 WM811K 모니터링 기준 버전입니다.")
    if reference.get("source_split") != "validation":
        raise ValueError("모니터링 기준은 validation 분할에서만 만들어야 합니다.")
    class_names = reference.get("class_names")
    if not isinstance(class_names, list) or len(class_names) < 2:
        raise ValueError("모니터링 기준 클래스가 올바르지 않습니다.")
    if len(class_names) != len(set(class_names)):
        raise ValueError("모니터링 기준 클래스가 중복됩니다.")
    class_probabilities = reference.get("class_probabilities", {})
    if set(class_probabilities) != set(class_names):
        raise ValueError("모니터링 클래스 확률 키가 모델 클래스와 다릅니다.")
    class_total = sum(float(class_probabilities[name]) for name in class_names)
    if not np.isclose(class_total, 1.0, atol=1e-9):
        raise ValueError("모니터링 클래스 확률 합이 1이 아닙니다.")
    status_probabilities = reference.get("ood_status_probabilities", {})
    if set(status_probabilities) != set(OOD_STATUSES):
        raise ValueError("모니터링 OOD 상태 확률이 불완전합니다.")
    status_total = sum(float(status_probabilities[name]) for name in OOD_STATUSES)
    if not np.isclose(status_total, 1.0, atol=1e-9):
        raise ValueError("모니터링 OOD 상태 확률 합이 1이 아닙니다.")
    if int(reference.get("validation_sample_count", 0)) < 1:
        raise ValueError("모니터링 validation 표본 수가 없습니다.")
    return reference


def _jensen_shannon_divergence(observed: np.ndarray, expected: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    observed = observed / observed.sum()
    expected = expected / expected.sum()
    midpoint = (observed + expected) / 2.0

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return float((kl(observed, midpoint) + kl(expected, midpoint)) / 2.0)


def _upper_simulation_threshold(
    *,
    probability: float,
    sample_count: int,
    quantile: float,
    repetitions: int,
    random: np.random.Generator,
) -> float:
    simulated = random.binomial(sample_count, probability, size=repetitions) / sample_count
    return float(np.quantile(simulated, quantile, method="higher"))


def assess_batch_drift(
    *,
    predicted_labels: Iterable[str],
    ood_statuses: Iterable[str],
    ood_scores: Iterable[float],
    reference: Mapping,
) -> dict:
    """Compare one unlabeled production batch with the frozen validation reference."""
    class_names = [str(value) for value in reference["class_names"]]
    labels = [str(value) for value in predicted_labels]
    statuses = [str(value) for value in ood_statuses]
    scores = np.asarray(list(ood_scores), dtype=float)
    sample_count = len(labels)
    if sample_count == 0 or len(statuses) != sample_count or len(scores) != sample_count:
        raise ValueError("모니터링 입력 길이가 같고 1개 이상이어야 합니다.")
    unknown_labels = sorted(set(labels).difference(class_names))
    unknown_statuses = sorted(set(statuses).difference(OOD_STATUSES))
    if unknown_labels:
        raise ValueError(f"알 수 없는 예측 클래스: {unknown_labels}")
    if unknown_statuses:
        raise ValueError(f"알 수 없는 OOD 상태: {unknown_statuses}")
    if not np.isfinite(scores).all():
        raise ValueError("OOD 점수에 유한하지 않은 값이 있습니다.")

    config = reference["monitoring_config"]
    minimum_batch_size = int(config["minimum_batch_size"])
    quantile = float(config["bootstrap_quantile"])
    repetitions = int(config["bootstrap_repetitions"])
    if minimum_batch_size < 1 or repetitions < 100 or not 0.5 < quantile < 1.0:
        raise ValueError("모니터링 모의실험 설정이 올바르지 않습니다.")
    random = np.random.default_rng(int(config["seed"]) + sample_count * 1009)
    expected_class = np.asarray(
        [float(reference["class_probabilities"][name]) for name in class_names]
    )
    observed_counts = np.asarray([labels.count(name) for name in class_names])
    observed_class = observed_counts / sample_count
    class_js = _jensen_shannon_divergence(observed_class, expected_class)
    simulated_counts = random.multinomial(sample_count, expected_class, size=repetitions)
    simulated_js = np.asarray(
        [
            _jensen_shannon_divergence(row / sample_count, expected_class)
            for row in simulated_counts
        ]
    )
    class_js_threshold = float(np.quantile(simulated_js, quantile, method="higher"))

    expected_status = reference["ood_status_probabilities"]
    severe_rate = statuses.count("out_of_distribution") / sample_count
    review_or_ood_rate = (
        statuses.count("review") + statuses.count("out_of_distribution")
    ) / sample_count
    score_p95 = float(reference["ood_score_quantiles"]["p95"])
    high_score_rate = float(np.mean(scores > score_p95))
    severe_threshold = _upper_simulation_threshold(
        probability=float(expected_status["out_of_distribution"]),
        sample_count=sample_count,
        quantile=quantile,
        repetitions=repetitions,
        random=random,
    )
    review_threshold = _upper_simulation_threshold(
        probability=float(expected_status["review"])
        + float(expected_status["out_of_distribution"]),
        sample_count=sample_count,
        quantile=quantile,
        repetitions=repetitions,
        random=random,
    )
    high_score_threshold = _upper_simulation_threshold(
        probability=float(reference["above_p95_rate"]),
        sample_count=sample_count,
        quantile=quantile,
        repetitions=repetitions,
        random=random,
    )

    signals = {
        "predicted_class_mix": class_js > class_js_threshold,
        "review_or_ood_rate": review_or_ood_rate > review_threshold,
        "severe_ood_rate": severe_rate > severe_threshold,
        "high_ood_score_rate": high_score_rate > high_score_threshold,
    }
    if sample_count < minimum_batch_size:
        status = "INSUFFICIENT"
        reasons = [
            f"표본 {sample_count}개로 최소 {minimum_batch_size}개보다 작아 배치 드리프트를 판정하지 않음"
        ]
    else:
        failed = [name for name, triggered in signals.items() if triggered]
        status = "HOLD" if failed else "STABLE"
        reasons = failed or ["validation 기준의 99% 모의 변동 범위 안에 있음"]

    return {
        "version": 1,
        "status": status,
        "sample_count": sample_count,
        "minimum_batch_size": minimum_batch_size,
        "automatic_batch_decision_allowed": status == "STABLE",
        "signals": signals,
        "reasons": reasons,
        "metrics": {
            "predicted_class_js_divergence": class_js,
            "predicted_class_js_threshold": class_js_threshold,
            "review_or_ood_rate": review_or_ood_rate,
            "review_or_ood_rate_threshold": review_threshold,
            "severe_ood_rate": severe_rate,
            "severe_ood_rate_threshold": severe_threshold,
            "high_ood_score_rate": high_score_rate,
            "high_ood_score_rate_threshold": high_score_threshold,
            "mean_ood_score": float(scores.mean()),
        },
        "observed_class_counts": dict(zip(class_names, observed_counts.astype(int).tolist())),
        "limitations": [
            "정답 없이 입력·예측 분포 변화만 감시하며 실제 성능 저하를 직접 증명하지 않습니다.",
            "클래스 구성 변화는 생산 제품군 변화일 수 있으므로 자동 재학습 대신 전문가 검토로 연결합니다.",
            "고정 test는 기준이나 임계값 생성에 사용하지 않았습니다.",
        ],
    }
