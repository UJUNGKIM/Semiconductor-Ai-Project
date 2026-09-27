"""WM-811K 웨이퍼 맵 형태 특징 추출 — 독립 검증 독립 방식.

기준선은 2채널 CNN이 화소에서 표현을 **학습**합니다.
독립 검증는 반대로, 결함 모양을 설명하는 기하 서술자를 **직접 정의**해서
표 데이터 분류기에 넣습니다. 장단점이 분명히 갈립니다.

- CNN: 표현을 스스로 찾지만 GPU가 필요하고 왜 그렇게 봤는지 설명이 간접적입니다.
- 형태 서술자: CPU로 돌고 각 특징이 무슨 뜻인지 바로 말할 수 있습니다.
  대신 정의하지 않은 패턴은 볼 수 없습니다.

설계한 서술자(총 {n}개)는 WM-811K 9개 클래스의 알려진 모양과 대응합니다.

- `Center` / `Donut`: 반경 방향 프로파일(중심에서 가장자리까지 8개 고리의 불량 비율)
- `Edge-Ring` / `Edge-Loc`: 바깥 20% 고리 집중도와 각도 분산
- `Loc` / `Random`: 연결 성분 개수와 최대 성분 비율
- `Scratch`: 최장 가로·세로·대각 연속 구간 길이와 최대 성분의 가늘고 긴 정도
- `Near-full`: 전체 불량 비율
- 크기·회전에 덜 민감하도록 Hu 불변 모멘트와 정렬된 각도 프로파일을 함께 씁니다.

웨이퍼 맵 화소 값: 0=웨이퍼 밖, 1=정상 die, 2=불량 die
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wm811k_common import BACKGROUND, FAIL_DIE  # noqa: E402

N_RADIAL_BINS = 8
N_ANGULAR_BINS = 12
EDGE_RADIUS_FRACTION = 0.80
CORE_RADIUS_FRACTION = 0.30


def feature_names() -> list[str]:
    """추출 순서와 정확히 같은 이름 목록."""
    names = [
        "die_count",
        "fail_count",
        "fail_ratio",
        "wafer_fill_ratio",
        "centroid_offset",
        "radius_mean",
        "radius_std",
        "edge_fail_ratio",
        "core_fail_ratio",
        "edge_to_core_ratio",
        "angular_resultant",
        "angular_entropy",
        "angular_max",
        "angular_min",
        "component_count",
        "largest_component_ratio",
        "second_component_ratio",
        "component_size_mean",
        "largest_elongation",
        "largest_extent",
        "largest_boundary_ratio",
        "longest_run_horizontal",
        "longest_run_vertical",
        "longest_run_diagonal",
        "bbox_fill_ratio",
    ]
    names += [f"radial_bin_{i}" for i in range(N_RADIAL_BINS)]
    names += [f"angular_sorted_{i}" for i in range(N_ANGULAR_BINS)]
    names += [f"hu_{i}" for i in range(7)]
    return names


def extract(wafer: np.ndarray) -> np.ndarray:
    """웨이퍼 맵 한 장에서 형태 특징 벡터를 만든다."""
    wafer = np.asarray(wafer)
    if wafer.ndim != 2:
        raise ValueError(f"2차원 웨이퍼 맵이어야 합니다: {wafer.shape}")

    inside = wafer != BACKGROUND
    fail = wafer == FAIL_DIE

    die_count = float(inside.sum())
    fail_count = float(fail.sum())
    height, width = wafer.shape
    wafer_fill_ratio = die_count / float(height * width)

    if die_count == 0:
        # 전부 배경인 비정상 입력. 0 벡터로 두되 예외는 내지 않는다.
        return np.zeros(len(feature_names()), dtype=float)

    fail_ratio = fail_count / die_count

    rows, cols = np.nonzero(inside)
    center_row, center_col = rows.mean(), cols.mean()
    radius_scale = max(
        np.sqrt(((rows - center_row) ** 2 + (cols - center_col) ** 2)).max(), 1e-9
    )

    fail_rows, fail_cols = np.nonzero(fail)
    if fail_count > 0:
        dr = fail_rows - center_row
        dc = fail_cols - center_col
        radii = np.sqrt(dr**2 + dc**2) / radius_scale
        angles = np.arctan2(dr, dc)
        centroid_offset = float(
            np.sqrt((dr.mean()) ** 2 + (dc.mean()) ** 2) / radius_scale
        )
        radius_mean = float(radii.mean())
        radius_std = float(radii.std())
    else:
        radii = np.zeros(0)
        angles = np.zeros(0)
        centroid_offset = radius_mean = radius_std = 0.0

    # --- 반경 프로파일: 고리마다 (불량 die / 전체 die) -------------------
    inside_radii = (
        np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2) / radius_scale
    )
    edges = np.linspace(0.0, 1.0, N_RADIAL_BINS + 1)
    radial_profile = np.zeros(N_RADIAL_BINS)
    for i in range(N_RADIAL_BINS):
        low, high = edges[i], edges[i + 1]
        in_ring = (inside_radii >= low) & (
            inside_radii < high if i < N_RADIAL_BINS - 1 else inside_radii <= high
        )
        total = in_ring.sum()
        if total == 0:
            continue
        if fail_count > 0:
            ring_fail = ((radii >= low) & (radii < high)).sum() if i < N_RADIAL_BINS - 1 else (
                (radii >= low) & (radii <= high)
            ).sum()
        else:
            ring_fail = 0
        radial_profile[i] = ring_fail / total

    edge_mask_inside = inside_radii >= EDGE_RADIUS_FRACTION
    core_mask_inside = inside_radii <= CORE_RADIUS_FRACTION
    edge_fail_ratio = _safe_ratio(
        ((radii >= EDGE_RADIUS_FRACTION).sum() if fail_count else 0),
        edge_mask_inside.sum(),
    )
    core_fail_ratio = _safe_ratio(
        ((radii <= CORE_RADIUS_FRACTION).sum() if fail_count else 0),
        core_mask_inside.sum(),
    )
    edge_to_core_ratio = edge_fail_ratio / core_fail_ratio if core_fail_ratio > 0 else (
        edge_fail_ratio * 10.0
    )

    # --- 각도 분포: 한쪽에 몰렸는지(Edge-Loc) 고르게 퍼졌는지(Edge-Ring) ---
    if fail_count > 0:
        resultant = float(np.abs(np.exp(1j * angles).mean()))
        counts, _ = np.histogram(angles, bins=N_ANGULAR_BINS, range=(-np.pi, np.pi))
        proportions = counts / counts.sum()
        nonzero = proportions[proportions > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(N_ANGULAR_BINS))
        angular_sorted = np.sort(proportions)[::-1]
        angular_max = float(proportions.max())
        angular_min = float(proportions.min())
    else:
        resultant = entropy = angular_max = angular_min = 0.0
        angular_sorted = np.zeros(N_ANGULAR_BINS)

    # --- 연결 성분: 한 덩어리인가(Loc) 흩어졌는가(Random) ------------------
    component_stats = _component_stats(fail)

    # --- 연속 구간: 긁힘(Scratch)은 길고 가는 선으로 나타난다 --------------
    runs = _longest_runs(fail)

    if fail_count > 0:
        bbox_area = float(
            (fail_rows.max() - fail_rows.min() + 1) * (fail_cols.max() - fail_cols.min() + 1)
        )
        bbox_fill_ratio = fail_count / bbox_area
    else:
        bbox_fill_ratio = 0.0

    values = [
        die_count,
        fail_count,
        fail_ratio,
        wafer_fill_ratio,
        centroid_offset,
        radius_mean,
        radius_std,
        edge_fail_ratio,
        core_fail_ratio,
        edge_to_core_ratio,
        resultant,
        entropy,
        angular_max,
        angular_min,
        component_stats["count"],
        component_stats["largest_ratio"],
        component_stats["second_ratio"],
        component_stats["size_mean"],
        component_stats["elongation"],
        component_stats["extent"],
        component_stats["boundary_ratio"],
        runs["horizontal"],
        runs["vertical"],
        runs["diagonal"],
        bbox_fill_ratio,
    ]
    values.extend(radial_profile.tolist())
    values.extend(angular_sorted.tolist())
    values.extend(_hu_moments(fail).tolist())

    vector = np.asarray(values, dtype=float)
    if vector.size != len(feature_names()):
        raise RuntimeError(
            f"특징 개수 불일치: {vector.size} != {len(feature_names())}"
        )
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def extract_many(wafers, progress_every: int = 0) -> np.ndarray:
    """여러 장을 한 번에 처리한다."""
    rows = []
    for index, wafer in enumerate(wafers, start=1):
        rows.append(extract(wafer))
        if progress_every and index % progress_every == 0:
            print(f"  형태 특징 {index:,}장 완료", flush=True)
    return np.vstack(rows) if rows else np.zeros((0, len(feature_names())))


# ------------------------------------------------------------------ 보조
def _safe_ratio(numerator, denominator) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _component_stats(mask: np.ndarray) -> dict:
    """불량 die 덩어리 통계.

    8-이웃 연결을 쓴다. 기본값인 4-이웃으로 하면 1칸 폭 대각선 긁힘이
    낱개 점으로 쪼개져 '가늘고 긴 덩어리'라는 신호가 통째로 사라진다.
    Scratch 클래스가 바로 그 모양이라 여기서는 대각선도 한 덩어리로 봐야 한다.
    (자동 테스트 `test_긁힘형_결함은_가늘고_길다`가 이 경우를 지킨다.)
    """
    from scipy import ndimage

    connectivity = np.ones((3, 3), dtype=int)
    total = float(mask.sum())
    if total == 0:
        return {
            "count": 0.0,
            "largest_ratio": 0.0,
            "second_ratio": 0.0,
            "size_mean": 0.0,
            "elongation": 0.0,
            "extent": 0.0,
            "boundary_ratio": 0.0,
        }

    labeled, count = ndimage.label(mask, structure=connectivity)
    sizes = np.bincount(labeled.ravel())[1:]
    order = np.argsort(sizes)[::-1]
    largest_label = int(order[0]) + 1
    largest_size = float(sizes[order[0]])
    second_size = float(sizes[order[1]]) if sizes.size > 1 else 0.0

    coords = np.argwhere(labeled == largest_label).astype(float)
    if len(coords) >= 2:
        centered = coords - coords.mean(axis=0)
        covariance = centered.T @ centered / len(coords)
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
        # 가늘고 길수록 1에 가깝다. 동그랄수록 0에 가깝다.
        elongation = float(
            1.0 - eigenvalues[1] / eigenvalues[0] if eigenvalues[0] > 1e-12 else 0.0
        )
        bbox_area = float(
            (coords[:, 0].max() - coords[:, 0].min() + 1)
            * (coords[:, 1].max() - coords[:, 1].min() + 1)
        )
        extent = largest_size / bbox_area if bbox_area else 0.0
    else:
        elongation = 0.0
        extent = 1.0

    # 덩어리 경계에 놓인 die 비율: 값이 크면 얇고 흩어진 모양이다.
    eroded = ndimage.binary_erosion(labeled == largest_label, structure=connectivity)
    boundary_ratio = float((largest_size - eroded.sum()) / largest_size)

    return {
        "count": float(count),
        "largest_ratio": largest_size / total,
        "second_ratio": second_size / total,
        "size_mean": float(sizes.mean()),
        "elongation": elongation,
        "extent": float(extent),
        "boundary_ratio": boundary_ratio,
    }


def _longest_runs(mask: np.ndarray) -> dict:
    """가로·세로·대각선 방향 최장 연속 불량 구간 길이(정규화)."""
    if not mask.any():
        return {"horizontal": 0.0, "vertical": 0.0, "diagonal": 0.0}

    scale = float(max(mask.shape))
    horizontal = _max_run_along_rows(mask)
    vertical = _max_run_along_rows(mask.T)
    diagonal = max(
        _max_run_along_rows(_diagonal_rows(mask)),
        _max_run_along_rows(_diagonal_rows(mask[:, ::-1])),
    )
    return {
        "horizontal": horizontal / scale,
        "vertical": vertical / scale,
        "diagonal": diagonal / scale,
    }


def _max_run_along_rows(rows) -> float:
    best = 0
    for row in rows:
        current = 0
        for value in np.asarray(row).ravel():
            current = current + 1 if value else 0
            best = max(best, current)
    return float(best)


def _diagonal_rows(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    return [np.diagonal(mask, offset) for offset in range(-height + 1, width)]


def _hu_moments(mask: np.ndarray) -> np.ndarray:
    """Hu 불변 모멘트 7개를 직접 계산한다(이동·크기·회전에 불변).

    OpenCV 없이 쓰기 위해 정의 그대로 구현했다. 부호를 살린 로그 스케일로
    돌려주어 값의 범위를 트리 모델이 다루기 쉽게 만든다.
    """
    image = mask.astype(float)
    total = image.sum()
    if total == 0:
        return np.zeros(7)

    rows, cols = np.indices(image.shape)
    mean_row = (rows * image).sum() / total
    mean_col = (cols * image).sum() / total
    dr = rows - mean_row
    dc = cols - mean_col

    def mu(p: int, q: int) -> float:
        return float(((dc**p) * (dr**q) * image).sum())

    def nu(p: int, q: int) -> float:
        return mu(p, q) / (total ** (1 + (p + q) / 2))

    n20, n02, n11 = nu(2, 0), nu(0, 2), nu(1, 1)
    n30, n03, n21, n12 = nu(3, 0), nu(0, 3), nu(2, 1), nu(1, 2)

    h = np.zeros(7)
    h[0] = n20 + n02
    h[1] = (n20 - n02) ** 2 + 4 * n11**2
    h[2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h[4] = (n30 - 3 * n12) * (n30 + n12) * (
        (n30 + n12) ** 2 - 3 * (n21 + n03) ** 2
    ) + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    h[5] = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (
        n30 + n12
    ) * (n21 + n03)
    h[6] = (3 * n21 - n03) * (n30 + n12) * (
        (n30 + n12) ** 2 - 3 * (n21 + n03) ** 2
    ) - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)

    return np.sign(h) * np.log1p(np.abs(h))


__doc__ = __doc__.replace("{n}", str(len(feature_names())))



