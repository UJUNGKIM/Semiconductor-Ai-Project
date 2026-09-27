"""Pattern-based, non-causal inspection guidance for WM-811K predictions."""

from __future__ import annotations

from collections.abc import Mapping
import math


_CAUSE_LIBRARY = {
    "Center": (
        (
            "도포·회전 중심부 불균일",
            "중심부에 집중된 형태는 회전·도포 공정의 중심부 균일도 점검이 필요한 패턴입니다.",
            "회전 속도·도포량·노즐 정렬과 중심부 막 두께",
        ),
        (
            "중심부 온도·가스 분포 편차",
            "중앙 영역에 국한된 이상은 온도장이나 가스 흐름의 중심부 편차와 함께 확인할 수 있습니다.",
            "히터 존·온도 맵·가스 유량과 챔버 중심부 상태",
        ),
        (
            "중앙 영역 국부 오염·계측 이상",
            "공정 이상뿐 아니라 중앙부 파티클이나 검사 장비의 위치 편향도 같은 형태를 만들 수 있습니다.",
            "파티클 맵·재측정 결과·검사 장비 좌표 정합",
        ),
    ),
    "Donut": (
        (
            "반경 방향 공정 균일도 이상",
            "고리 형태는 반경에 따라 달라지는 도포·식각·증착 균일도를 우선 확인할 패턴입니다.",
            "반경별 막 두께·식각률·증착률 프로파일",
        ),
        (
            "온도·가스 흐름의 환형 편차",
            "특정 반경에 반복되는 이상은 히터 존이나 가스 분포의 환형 편차와 비교할 수 있습니다.",
            "히터 존별 온도·가스 유량·샤워헤드 상태",
        ),
        (
            "노광·초점 또는 계측의 반경 편향",
            "공정 결함과 별개로 초점·레벨링 또는 검사 민감도의 반경 편향 가능성도 남습니다.",
            "포커스·레벨링 로그와 다른 검사 장비의 재측정",
        ),
    ),
    "Edge-Loc": (
        (
            "가장자리 국부 오염",
            "가장자리 일부에 모인 패턴은 국부 파티클이나 챔버 오염 위치와 대조할 가치가 있습니다.",
            "해당 방향의 파티클 맵·챔버 벽·세정 이력",
        ),
        (
            "척·클램프·이송 접촉",
            "웨이퍼 가장자리의 국부 이상은 지지·고정·이송 과정의 접촉 위치와 겹칠 수 있습니다.",
            "척·클램프·로봇 암 접촉 위치와 카세트 슬롯",
        ),
        (
            "가장자리 가스·플라즈마 비대칭",
            "한쪽 가장자리 편향은 가스 흐름이나 플라즈마 분포의 방향성 이상 후보입니다.",
            "챔버 방향별 가스·플라즈마 지표와 장비 정렬",
        ),
    ),
    "Edge-Ring": (
        (
            "엣지 비드·가장자리 도포 이상",
            "둘레를 따라 이어진 패턴은 가장자리 도포와 edge-bead 제거 조건을 우선 확인할 형태입니다.",
            "edge-bead 제거·도포 조건과 가장자리 막 두께",
        ),
        (
            "가장자리 가스·플라즈마 균일도",
            "환형 가장자리 이상은 챔버 외곽의 가스 또는 플라즈마 균일도와 함께 점검할 수 있습니다.",
            "외곽 가스 유량·플라즈마 균일도·챔버 링 상태",
        ),
        (
            "척·edge exclusion 조건",
            "공정 이상 외에도 척 접촉부나 검사 edge-exclusion 설정이 패턴에 영향을 줄 수 있습니다.",
            "척 상태·edge exclusion 설정·재측정 결과",
        ),
    ),
    "Loc": (
        (
            "국부 파티클·오염",
            "한 영역에 모인 결함은 국부 오염이나 파티클 위치와 가장 먼저 비교할 수 있습니다.",
            "파티클 맵·세정 이력·동일 좌표 반복 여부",
        ),
        (
            "마스크·계측의 국부 이상",
            "좌표가 반복되는 국부 패턴은 마스크 또는 검사 장비의 위치성 이상 가능성이 있습니다.",
            "마스크 좌표·재측정·다른 검사 장비 결과",
        ),
        (
            "챔버 내 국부 비균일",
            "장비 방향에 고정된 위치라면 가스·온도·플라즈마의 국부 편차를 확인할 수 있습니다.",
            "웨이퍼 방향 정렬 후 장비·챔버별 위치 비교",
        ),
    ),
    "Near-full": (
        (
            "전역 레시피 이탈",
            "웨이퍼 전반의 결함은 단일 국부 원인보다 레시피 전체 조건의 이탈을 우선 의심할 패턴입니다.",
            "직전 정상 lot 대비 레시피·공정 시간·압력·온도",
        ),
        (
            "장비·챔버 상태의 전역 이상",
            "넓게 퍼진 형태는 챔버 상태나 장비 공통 조건의 급격한 변화와 함께 확인해야 합니다.",
            "장비 알람·챔버 세정·소모품·유지보수 이력",
        ),
        (
            "입력·검사 체계 이상",
            "값 매핑이나 검사 조건 오류도 거의 전면적인 결함처럼 보일 수 있어 자동판정 전에 배제해야 합니다.",
            "0·1·2 값 정의·검사 임계값·원본 맵 재수집",
        ),
    ),
    "Random": (
        (
            "파티클·불규칙 오염",
            "여러 위치에 흩어진 결함은 파티클 또는 불규칙 오염 이력과 비교할 수 있습니다.",
            "파티클 계측·세정·필터·챔버 오염 기록",
        ),
        (
            "장비 조건의 간헐적 불안정",
            "분산된 형태가 lot 내에서 증가하면 센서 변동이나 간헐 알람을 함께 확인해야 합니다.",
            "시계열 센서·알람·lot 내 발생 순서",
        ),
        (
            "검사 노이즈·데이터 품질",
            "실제 결함이 아닌 검사 민감도나 값 변환 오류도 무작위 형태로 나타날 수 있습니다.",
            "재측정·검사 장비 간 비교·파일 값 정의",
        ),
    ),
    "Scratch": (
        (
            "로봇·카세트·이송 접촉",
            "선형 결함은 웨이퍼 이송 과정의 기계적 접촉 경로와 우선 비교할 수 있습니다.",
            "로봇 암·카세트 슬롯·이송 경로와 접촉 흔적",
        ),
        (
            "세정·연마의 기계적 손상",
            "표면을 가로지르는 형태는 세정 브러시나 연마 공정의 손상 후보와도 연결될 수 있습니다.",
            "세정 브러시·CMP 패드·압력·교체 이력",
        ),
        (
            "검사 스테이지·스캔 아티팩트",
            "동일 방향의 선형 패턴이 반복되면 검사 장비의 스캔 또는 스테이지 이상을 배제해야 합니다.",
            "다른 검사 장비 재측정·스캔 방향·스테이지 상태",
        ),
    ),
}

SUPPORTED_LABELS = (*_CAUSE_LIBRARY.keys(), "none")


def _finite_feature(features: Mapping[str, object], name: str) -> float:
    try:
        value = float(features[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"형상 지표가 없거나 숫자가 아닙니다: {name}") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"형상 지표가 유효하지 않습니다: {name}")
    return value


def _shape_observations(features: Mapping[str, object]) -> list[str]:
    defect_ratio = _finite_feature(features, "defect_ratio")
    component_count = int(_finite_feature(features, "component_count_8"))
    largest = _finite_feature(features, "largest_component_fraction")
    center = _finite_feature(features, "center_defect_fraction")
    edge = _finite_feature(features, "edge_defect_fraction")
    boundary = _finite_feature(features, "boundary_defect_coverage")
    spread = _finite_feature(features, "defect_bbox_fraction")
    observations = [
        f"결함 die {defect_ratio:.1%}",
        f"8-이웃 연결 영역 {component_count:,}개",
    ]
    if center >= 0.5:
        observations.append(f"결함의 {center:.1%}가 중심부")
    if edge >= 0.5:
        observations.append(f"결함의 {edge:.1%}가 가장자리")
    if boundary >= 0.5:
        observations.append(f"웨이퍼 경계의 {boundary:.1%}에 결함")
    if largest >= 0.6 and defect_ratio > 0:
        observations.append(f"최대 연결 영역이 결함의 {largest:.1%}")
    if spread >= 0.6:
        observations.append(f"결함 분포 범위 {spread:.1%}")
    return observations


def build_wafer_cause_guidance(
    predicted_label: str,
    shape_features: Mapping[str, object],
    *,
    ood_status: str = "in_distribution",
) -> dict[str, object]:
    """Return inspection hypotheses without claiming physical causality."""
    label = str(predicted_label)
    if label not in SUPPORTED_LABELS:
        raise ValueError(f"지원하지 않는 WM-811K 클래스입니다: {label}")
    if ood_status not in {"in_distribution", "review", "out_of_distribution"}:
        raise ValueError(f"지원하지 않는 OOD 상태입니다: {ood_status}")
    observations = _shape_observations(shape_features)
    if label == "none":
        return {
            "predicted_label": label,
            "interpretation": "정상 패턴 · 원인 후보 없음",
            "evidence_level": "모니터링",
            "observations": observations,
            "candidates": [],
            "disclaimer": (
                "현재 맵에서 학습된 불량 패턴이 선택되지 않았습니다. 정상 판정은 공정 "
                "무결성 보증이 아니므로 lot 추세와 OOD 상태를 계속 확인하세요."
            ),
        }
    candidates = [
        {
            "priority": index,
            "cause": cause,
            "rationale": rationale,
            "check": check,
        }
        for index, (cause, rationale, check) in enumerate(
            _CAUSE_LIBRARY[label], start=1
        )
    ]
    evidence_level = (
        "입력 재확인 후 패턴 가설 검토"
        if ood_status == "out_of_distribution"
        else "패턴 기반 점검 가설"
    )
    return {
        "predicted_label": label,
        "interpretation": f"원인 확정 불가 · {label} 패턴 기반 후보",
        "evidence_level": evidence_level,
        "observations": observations,
        "candidates": candidates,
        "disclaimer": (
            "후보 순서는 원인 확률이나 인과 추정값이 아닙니다. 장비·레시피·센서·"
            "유지보수·재측정 기록과 공정 엔지니어 검토로 실제 원인을 확인해야 합니다."
        ),
    }
