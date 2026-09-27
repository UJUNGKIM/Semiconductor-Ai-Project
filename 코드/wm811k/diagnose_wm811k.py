"""WM-811K inference, audit, Grad-CAM, Integrated Gradients, and Gradient SHAP utilities."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import colormaps
from torch.nn import functional as F

from train_wm811k_cnn import WaferCNN
from wafer_shape import wafer_shape_features


VALID_MAP_VALUES = np.array([0, 1, 2], dtype=np.uint8)


def validate_wafer_map(wafer: np.ndarray) -> np.ndarray:
    """Return a validated two-dimensional WM-811K map."""
    array = np.asarray(wafer)
    if array.ndim == 3 and 1 in (array.shape[0], array.shape[-1]):
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"웨이퍼 맵은 2차원이어야 합니다. 현재 형상: {array.shape}")
    if min(array.shape) < 4:
        raise ValueError(f"웨이퍼 맵 크기가 너무 작습니다: {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("웨이퍼 맵에는 숫자만 사용할 수 있습니다.")
    numeric = array.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all():
        raise ValueError("웨이퍼 맵에 결측치 또는 무한대가 있습니다.")
    rounded = np.rint(numeric)
    if not np.allclose(numeric, rounded):
        raise ValueError("웨이퍼 맵 값은 0, 1, 2 중 하나여야 합니다.")
    integers = rounded.astype(np.int16)
    invalid = sorted(set(np.unique(integers)).difference({0, 1, 2}))
    if invalid:
        raise ValueError(f"지원하지 않는 웨이퍼 값: {invalid}. 0, 1, 2만 사용하세요.")
    return integers.astype(np.uint8)


def parse_wafer_bytes(raw_bytes: bytes, filename: str) -> np.ndarray:
    """Parse one 2-D wafer map from a safe NPY, CSV, or whitespace text file."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".npy":
        wafer = np.load(BytesIO(raw_bytes), allow_pickle=False)
    elif suffix in {".csv", ".txt", ".data"}:
        delimiter = "," if suffix == ".csv" else None
        try:
            wafer = np.loadtxt(BytesIO(raw_bytes), delimiter=delimiter)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(
                "CSV/TXT는 헤더 없이 0, 1, 2 숫자로만 구성된 2차원 행렬이어야 합니다."
            ) from error
    else:
        raise ValueError("지원 형식은 .npy, .csv, .txt, .data입니다.")
    return validate_wafer_map(wafer)


def resize_nearest(wafer: np.ndarray, image_size: int = 64) -> np.ndarray:
    """Resize a categorical wafer map using nearest-neighbor indices."""
    wafer = validate_wafer_map(wafer)
    row_index = np.rint(np.linspace(0, wafer.shape[0] - 1, image_size)).astype(int)
    column_index = np.rint(
        np.linspace(0, wafer.shape[1] - 1, image_size)
    ).astype(int)
    return wafer[np.ix_(row_index, column_index)]


def canonical_wafer_shape_features(wafer: np.ndarray) -> dict[str, float | int]:
    """Return global shape descriptors on the canonical 64x64 representation."""
    return wafer_shape_features(resize_nearest(wafer))


def wafer_tensor(wafer: np.ndarray) -> torch.Tensor:
    """Build the two channels used during CNN training."""
    resized = resize_nearest(wafer)
    channels = np.stack((resized > 0, resized == 2), axis=0).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(channels))[None, ...]


def load_checkpoint(checkpoint_path: Path) -> tuple[WaferCNN, list[str], dict]:
    """Load and validate the committed CNN checkpoint on CPU."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {
        "model_state_dict",
        "class_names",
        "num_classes",
        "input_channels",
        "image_size",
        "best_epoch",
        "best_validation_macro_f1",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"체크포인트 필수 항목 누락: {sorted(missing)}")
    if checkpoint["input_channels"] != 2 or checkpoint["image_size"] != 64:
        raise ValueError("학습 입력 규격과 체크포인트가 일치하지 않습니다.")
    class_names = list(checkpoint["class_names"])
    if len(class_names) != int(checkpoint["num_classes"]):
        raise ValueError("체크포인트 클래스 정보가 일치하지 않습니다.")
    model = WaferCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, class_names, checkpoint


def load_ood_reference(metadata_path: Path, arrays_path: Path) -> dict[str, object]:
    """Load a pickle-free WM-811K OOD reference."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("지원하지 않는 WM-811K OOD 기준 버전입니다.")
    if metadata.get("fit_split") != "train":
        raise ValueError("OOD 기준은 train split으로 적합되어야 합니다.")
    if metadata.get("calibration_split") != "validation":
        raise ValueError("OOD 임계값은 validation split에서 고정되어야 합니다.")
    if metadata.get("test_used_for_threshold") is not False:
        raise ValueError("test를 사용해 OOD 임계값을 정한 기준은 사용할 수 없습니다.")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"class_means", "class_variances", "class_counts"}:
            raise ValueError("OOD NPZ 배열 구성이 잘못되었습니다.")
        metadata["class_means"] = arrays["class_means"].copy()
        metadata["class_variances"] = arrays["class_variances"].copy()
        metadata["class_counts"] = arrays["class_counts"].copy()
    means = np.asarray(metadata["class_means"])
    variances = np.asarray(metadata["class_variances"])
    if means.shape != (9, 256) or variances.shape != means.shape:
        raise ValueError("OOD 기준 배열 형상이 모델과 일치하지 않습니다.")
    if not np.isfinite(means).all() or not np.isfinite(variances).all():
        raise ValueError("OOD 기준에 유한하지 않은 값이 있습니다.")
    if np.any(variances <= 0):
        raise ValueError("OOD 기준 분산은 모두 양수여야 합니다.")
    thresholds = metadata["thresholds"]
    if not 0 < float(thresholds["review_threshold"]) < float(
        thresholds["ood_threshold"]
    ):
        raise ValueError("OOD 임계값 순서가 잘못되었습니다.")
    return metadata


def assess_wafer_ood_batch(
    model: WaferCNN,
    class_names: list[str],
    wafers: list[np.ndarray],
    reference: dict[str, object],
    batch_size: int = 64,
) -> list[dict[str, object]]:
    """Assess feature-space shift and train-range geometry for wafer maps."""
    if not wafers:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size는 1 이상이어야 합니다.")
    reference_names = list(reference["class_names"])
    if reference_names != list(class_names):
        raise ValueError("OOD 기준과 체크포인트 클래스 순서가 다릅니다.")
    means = np.asarray(reference["class_means"], dtype=np.float32)
    variances = np.asarray(reference["class_variances"], dtype=np.float32)
    thresholds = reference["thresholds"]
    review_threshold = float(thresholds["review_threshold"])
    ood_threshold = float(thresholds["ood_threshold"])
    geometry = reference["geometry_reference"]["global"]
    device = next(model.parameters()).device
    results: list[dict[str, object]] = []
    model.eval()

    for start in range(0, len(wafers), batch_size):
        wafer_chunk = wafers[start : start + batch_size]
        inputs = torch.cat([wafer_tensor(wafer) for wafer in wafer_chunk], dim=0).to(
            device
        )
        with torch.no_grad():
            feature_map = model.features(inputs)
            features = model.pool(feature_map).flatten(1).cpu().numpy()
        distances = np.sqrt(
            np.mean(
                np.square(features[:, None, :] - means[None, :, :])
                / variances[None, :, :],
                axis=2,
            )
        )
        nearest = distances.argmin(axis=1)
        scores = distances[np.arange(len(distances)), nearest]
        for wafer, nearest_id, score in zip(wafer_chunk, nearest, scores):
            resized = resize_nearest(wafer)
            wafer_area = int((resized > 0).sum())
            defect_count = int((resized == 2).sum())
            coverage = wafer_area / resized.size
            defect_ratio = defect_count / max(wafer_area, 1)
            warnings: list[str] = []
            for metric_name, value, label in (
                ("wafer_coverage", coverage, "웨이퍼 면적 비율"),
                ("defect_ratio", defect_ratio, "결함 die 비율"),
            ):
                bounds = geometry[metric_name]
                if value < float(bounds["lower_0_1pct"]) or value > float(
                    bounds["upper_99_9pct"]
                ):
                    warnings.append(f"{label}이 train 0.1~99.9% 범위를 벗어남")

            embedding_status = (
                "out_of_distribution"
                if score >= ood_threshold
                else "review"
                if score >= review_threshold
                else "in_distribution"
            )
            final_status = embedding_status
            if warnings and final_status == "in_distribution":
                final_status = "review"
            results.append(
                {
                    "ood_status": final_status,
                    "embedding_status": embedding_status,
                    "ood_score": float(score),
                    "review_threshold": review_threshold,
                    "ood_threshold": ood_threshold,
                    "nearest_reference_class": class_names[int(nearest_id)],
                    "wafer_coverage": float(coverage),
                    "defect_ratio": float(defect_ratio),
                    "geometry_warnings": warnings,
                }
            )
    return results


def canonical_wafer_sha256(wafer: np.ndarray) -> str:
    """Hash the canonical 64x64 categorical representation of one wafer."""
    canonical = np.ascontiguousarray(resize_nearest(wafer), dtype=np.uint8)
    return hashlib.sha256(canonical.tobytes()).hexdigest().upper()


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA-256 digest without loading a whole file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_wafer_excel_report(results: pd.DataFrame) -> bytes:
    """Build a readable Excel workbook for WM-811K batch diagnosis results."""
    if results.empty:
        raise ValueError("Excel 보고서로 저장할 배치 결과가 없습니다.")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        results.to_excel(writer, sheet_name="배치 진단 결과", index=False)
        workbook = writer.book
        worksheet = writer.sheets["배치 진단 결과"]
        worksheet.hide_gridlines(2)
        worksheet.freeze_panes(1, 2)
        worksheet.autofilter(0, 0, len(results), len(results.columns) - 1)

        header_format = workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#123B5D",
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
                "border": 1,
                "border_color": "#D9E2F3",
            }
        )
        text_format = workbook.add_format(
            {"valign": "vcenter", "text_wrap": True}
        )
        center_format = workbook.add_format(
            {"align": "center", "valign": "vcenter", "text_wrap": True}
        )
        probability_format = workbook.add_format(
            {"num_format": "0.00%", "align": "right", "valign": "vcenter"}
        )
        score_format = workbook.add_format(
            {"num_format": "0.0000", "align": "right", "valign": "vcenter"}
        )

        worksheet.set_row(0, 34, header_format)
        worksheet.set_default_row(34)
        width_by_column = {
            "파일명": 28,
            "입력 SHA-256": 34,
            "실제 클래스": 14,
            "판정": 10,
            "예측 클래스": 14,
            "보정 신뢰도": 13,
            "입력 신뢰도": 24,
            "OOD 점수": 12,
            "가까운 기준 클래스": 18,
            "검토 필요": 11,
            "처리 권고": 28,
            "원인 해석": 28,
            "주요 원인 후보": 48,
            "우선 확인 항목": 56,
            "형상 근거": 46,
            "입력 품질 경고": 48,
            "결함 die 비율": 15,
            "8-이웃 연결 영역 수": 18,
            "최대 연결 영역 점유율": 20,
            "가장자리 결함 구성비": 19,
            "웨이퍼 경계 결함률": 18,
            "결함 분포 범위": 17,
        }
        percentage_columns = {
            "보정 신뢰도",
            "결함 die 비율",
            "최대 연결 영역 점유율",
            "가장자리 결함 구성비",
            "웨이퍼 경계 결함률",
            "결함 분포 범위",
        }
        centered_columns = {
            "실제 클래스",
            "판정",
            "예측 클래스",
            "입력 신뢰도",
            "가까운 기준 클래스",
            "검토 필요",
        }
        for index, column in enumerate(results.columns):
            if column in percentage_columns:
                cell_format = probability_format
            elif column == "OOD 점수":
                cell_format = score_format
            elif column in centered_columns:
                cell_format = center_format
            else:
                cell_format = text_format
            width = width_by_column.get(
                str(column),
                min(40, max(12, len(str(column)) + 4)),
            )
            worksheet.set_column(index, index, width, cell_format)

        for index, column in enumerate(results.columns):
            worksheet.write(0, index, str(column), header_format)

    return output.getvalue()


def build_wafer_audit_report(
    *,
    source_name: str,
    results: pd.DataFrame,
    artifact_hashes: dict[str, str],
) -> bytes:
    """Build a self-contained escaped HTML audit report for wafer decisions."""
    required = {
        "파일명",
        "입력 SHA-256",
        "판정",
        "예측 클래스",
        "보정 신뢰도",
        "입력 신뢰도",
        "OOD 점수",
        "검토 필요",
        "처리 권고",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"감사 보고서 결과 열 누락: {sorted(missing)}")
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    normalized = results.to_csv(index=False, lineterminator="\n")
    result_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest().upper()
    defect_count = int((results["판정"] == "불량").sum())
    review_count = int((results["검토 필요"] == "예").sum())
    ood_count = int(results["입력 신뢰도"].astype(str).str.contains("분포 이탈").sum())
    table = results.to_html(
        index=False,
        escape=True,
        float_format=lambda value: f"{value:.5f}",
        border=0,
    )
    artifact_rows = "".join(
        f"<li>{html.escape(name)} SHA-256: <code>{html.escape(value)}</code></li>"
        for name, value in artifact_hashes.items()
    )
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>WM-811K 진단 감사 보고서</title>
<style>
body{{font-family:Arial,'Malgun Gothic',sans-serif;max-width:1200px;margin:32px auto;color:#1f2937;line-height:1.55}}
h1,h2{{color:#123b5d}} .cards{{display:flex;gap:12px;flex-wrap:wrap}}
.card{{border:1px solid #d1d5db;border-radius:10px;padding:12px 18px;min-width:150px}}
table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border-bottom:1px solid #ddd;padding:7px;text-align:left}}
.warning{{background:#fff3cd;border-left:5px solid #e0a800;padding:12px}} code{{word-break:break-all}}
</style></head><body>
<h1>WM-811K 설명가능 AI 진단 감사 보고서</h1>
<p>생성 시각: {html.escape(generated)}<br>입력: {html.escape(source_name)}<br>
정규화 결과 SHA-256: <code>{result_digest}</code></p>
<div class="cards"><div class="card"><b>전체 웨이퍼</b><br>{len(results):,}</div>
<div class="card"><b>불량 판정</b><br>{defect_count:,}</div>
<div class="card"><b>전문가 검토</b><br>{review_count:,}</div>
<div class="card"><b>OOD 자동판정 보류</b><br>{ood_count:,}</div></div>
<h2>모델·기준 무결성</h2><ul>{artifact_rows}</ul>
<h2>웨이퍼별 판정 기록</h2>{table}
<h2>해석 제한</h2><div class="warning">CNN 분류와 Grad-CAM·Gradient SHAP, 특징 공간 OOD는 모델 기반
점검 근거이며 물리적 불량 원인이나 공정 조정 지시를 증명하지 않습니다. 공간 셔플
stress는 실제 외부 장비 검증이 아닙니다. 저신뢰·OOD 결과와 신규 장비·lot는 공정
전문가가 별도로 확인해야 합니다.</div>
</body></html>"""
    return document.encode("utf-8")


def predict_with_gradcam(
    model: WaferCNN,
    class_names: list[str],
    wafer: np.ndarray,
    target_class: int | None = None,
    temperature: float = 1.0,
) -> dict[str, object]:
    """Predict one wafer and explain the selected class with Grad-CAM."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature는 0보다 큰 유한한 값이어야 합니다.")
    device = next(model.parameters()).device
    inputs = wafer_tensor(wafer).to(device)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        activations = model.features(inputs)
        activations.retain_grad()
        logits = model.classifier(model.pool(activations).flatten(1))
        probabilities = torch.softmax(logits / temperature, dim=1)
        predicted_class = int(probabilities.argmax(dim=1).item())
        selected_class = predicted_class if target_class is None else int(target_class)
        if not 0 <= selected_class < len(class_names):
            raise ValueError(f"설명 대상 클래스가 범위를 벗어났습니다: {selected_class}")
        logits[0, selected_class].backward()
        gradients = activations.grad
        if gradients is None:
            raise RuntimeError("Grad-CAM 기울기를 계산하지 못했습니다.")
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(
            heatmap, size=(64, 64), mode="bilinear", align_corners=False
        )[0, 0]
        maximum = float(heatmap.max().detach())
        if maximum > 0:
            heatmap = heatmap / maximum

    probability_values = probabilities.detach().cpu().numpy()[0]
    return {
        "predicted_class": predicted_class,
        "predicted_label": class_names[predicted_class],
        "confidence": float(probability_values[predicted_class]),
        "probabilities": probability_values,
        "selected_class": selected_class,
        "selected_label": class_names[selected_class],
        "heatmap": heatmap.detach().cpu().numpy(),
        "resized_map": resize_nearest(wafer),
    }


def predict_with_integrated_gradients(
    model: WaferCNN,
    class_names: list[str],
    wafer: np.ndarray,
    target_class: int | None = None,
    temperature: float = 1.0,
    steps: int = 24,
) -> dict[str, object]:
    """Predict one wafer and explain it with absolute Integrated Gradients."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature는 0보다 큰 유한한 값이어야 합니다.")
    if steps < 2:
        raise ValueError("Integrated Gradients steps는 2 이상이어야 합니다.")
    device = next(model.parameters()).device
    inputs = wafer_tensor(wafer).to(device)
    baseline = torch.zeros_like(inputs)
    with torch.no_grad():
        input_logits = model(inputs)
        baseline_logits = model(baseline)
        probabilities = torch.softmax(input_logits / temperature, dim=1)
        predicted_class = int(probabilities.argmax(dim=1).item())
    selected_class = predicted_class if target_class is None else int(target_class)
    if not 0 <= selected_class < len(class_names):
        raise ValueError(f"설명 대상 클래스가 범위를 벗어났습니다: {selected_class}")

    alphas = torch.linspace(0.0, 1.0, steps, device=device).view(-1, 1, 1, 1)
    scaled = (baseline + alphas * (inputs - baseline)).detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    selected_logits = model(scaled)[:, selected_class].sum()
    gradients = torch.autograd.grad(selected_logits, scaled, retain_graph=False)[0]
    trapezoids = (gradients[:-1] + gradients[1:]) * 0.5
    average_gradient = trapezoids.mean(dim=0, keepdim=True)
    signed_attribution = (inputs - baseline) * average_gradient
    heatmap = signed_attribution.abs().sum(dim=1)[0]
    maximum = float(heatmap.max().detach())
    if maximum > 0:
        heatmap = heatmap / maximum
    attribution_sum = float(signed_attribution.sum().detach())
    logit_delta = float(
        (input_logits[0, selected_class] - baseline_logits[0, selected_class]).detach()
    )
    probability_values = probabilities.detach().cpu().numpy()[0]
    return {
        "predicted_class": predicted_class,
        "predicted_label": class_names[predicted_class],
        "confidence": float(probability_values[predicted_class]),
        "probabilities": probability_values,
        "selected_class": selected_class,
        "selected_label": class_names[selected_class],
        "heatmap": heatmap.detach().cpu().numpy(),
        "resized_map": resize_nearest(wafer),
        "steps": steps,
        "baseline": "all_background_zero",
        "attribution_sum": attribution_sum,
        "selected_logit_delta": logit_delta,
        "completeness_delta": attribution_sum - logit_delta,
    }


GRADIENT_SHAP_BASELINE = "same_geometry_all_active_dies_normal"
GRADIENT_SHAP_METHOD = "expected_gradients_gradient_shap_approximation"
# Local-accuracy check on raw attributions: |sum(phi) - (f(x) - f(x'))| must stay
# within 5% of the logit change (the completeness check recommended for
# path-integral attributions), with a small floor for near-zero logit changes.
GRADIENT_SHAP_RELATIVE_TOLERANCE = 0.05
GRADIENT_SHAP_ABSOLUTE_TOLERANCE = 1e-3


def gradient_shap_baseline(inputs: torch.Tensor) -> torch.Tensor:
    """Keep the wafer outline and turn every defect die into a normal die."""
    baseline = inputs.clone()
    baseline[:, 1] = 0.0
    return baseline


def assess_additivity(
    attribution_sum: float,
    logit_delta: float,
    relative_tolerance: float = GRADIENT_SHAP_RELATIVE_TOLERANCE,
    absolute_tolerance: float = GRADIENT_SHAP_ABSOLUTE_TOLERANCE,
) -> dict[str, float | bool]:
    """Judge SHAP local accuracy from raw attributions without rescaling them."""
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("가산성 허용오차는 0 이상이어야 합니다.")
    residual = float(attribution_sum) - float(logit_delta)
    tolerance = max(float(absolute_tolerance), float(relative_tolerance) * abs(logit_delta))
    if abs(logit_delta) > 0:
        relative_residual = abs(residual) / abs(logit_delta)
    else:
        relative_residual = 0.0 if residual == 0 else float("inf")
    return {
        "raw_additivity_residual": residual,
        "relative_additivity_residual": float(relative_residual),
        "additivity_tolerance": tolerance,
        "additivity_relative_tolerance": float(relative_tolerance),
        "additivity_absolute_tolerance": float(absolute_tolerance),
        "additivity_check_passed": bool(
            np.isfinite(residual) and abs(residual) <= tolerance
        ),
    }


def expected_gradients_attribution(
    model: WaferCNN,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    target_class: int,
    samples: int,
    seed: int,
    local_smoothing: float = 0.0,
) -> torch.Tensor:
    """Estimate raw Expected-Gradients attributions for one input and baseline.

    Path positions are drawn with stratified uniform sampling, an unbiased
    Monte-Carlo estimate of the path expectation with lower variance than
    independent draws. The result is never rescaled to the logit change, so
    its sum can be checked against the model instead of being forced to match.
    """
    device = inputs.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    offsets = torch.rand(samples, generator=generator, device=device)
    alphas = ((torch.arange(samples, device=device) + offsets) / samples).view(
        -1, 1, 1, 1
    )
    delta = inputs - baseline
    interpolated = baseline + alphas * delta
    if local_smoothing:
        noise = torch.randn(
            interpolated.shape,
            generator=generator,
            device=device,
            dtype=interpolated.dtype,
        ) * local_smoothing
        interpolated = (interpolated + noise).clamp(0.0, 1.0)
    interpolated = interpolated.detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        selected_logits = model(interpolated)[:, target_class].sum()
        gradients = torch.autograd.grad(selected_logits, interpolated)[0]
    return (delta * gradients.mean(dim=0, keepdim=True)).detach()


def predict_with_gradient_shap(
    model: WaferCNN,
    class_names: list[str],
    wafer: np.ndarray,
    target_class: int | None = None,
    temperature: float = 1.0,
    samples: int = 64,
    max_samples: int = 256,
    seed: int = 42,
    local_smoothing: float = 0.0,
) -> dict[str, object]:
    """Explain one wafer with a deterministic Expected-Gradients SHAP estimate.

    The reference keeps the observed wafer outline and changes every active die
    to normal. Points on the reference-to-input path approximate the
    expectation used by Gradient SHAP. This makes the attribution question
    explicit: which defect-die locations move the selected-class logit away
    from an otherwise normal wafer with the same geometry?

    Attributions are returned in raw selected-class logit units (before
    temperature scaling). ``heatmap`` is a display-only magnitude map scaled to
    [0, 1]; the additivity check uses the raw attribution sum. When the check
    fails, the path sample count is doubled up to ``max_samples`` and every
    attempt is recorded; a result that still fails is reported as failing.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("온도는 0보다 큰 유한한 값이어야 합니다.")
    if samples < 4:
        raise ValueError("Gradient SHAP 표본 수는 4 이상이어야 합니다.")
    if max_samples < samples:
        raise ValueError("max_samples는 samples 이상이어야 합니다.")
    if not np.isfinite(local_smoothing) or local_smoothing < 0:
        raise ValueError("local_smoothing은 0 이상의 유한한 값이어야 합니다.")

    device = next(model.parameters()).device
    inputs = wafer_tensor(wafer).to(device)
    baseline = gradient_shap_baseline(inputs)
    with torch.no_grad():
        input_logits = model(inputs)
        baseline_logits = model(baseline)
        probabilities = torch.softmax(input_logits / temperature, dim=1)
        predicted_class = int(probabilities.argmax(dim=1).item())
    selected_class = predicted_class if target_class is None else int(target_class)
    if not 0 <= selected_class < len(class_names):
        raise ValueError(f"설명 대상 클래스가 범위를 벗어났습니다: {selected_class}")
    logit_delta = float(
        (input_logits[0, selected_class] - baseline_logits[0, selected_class]).item()
    )

    attempts: list[dict[str, float | int | bool]] = []
    current_samples = samples
    while True:
        attribution = expected_gradients_attribution(
            model,
            inputs,
            baseline,
            selected_class,
            samples=current_samples,
            seed=seed,
            local_smoothing=local_smoothing,
        )
        signed_map = attribution.sum(dim=1)[0].cpu().numpy().astype(np.float64)
        raw_attribution_sum = float(signed_map.sum())
        additivity = assess_additivity(raw_attribution_sum, logit_delta)
        attempts.append(
            {
                "samples": current_samples,
                "raw_additivity_residual": additivity["raw_additivity_residual"],
                "relative_additivity_residual": additivity["relative_additivity_residual"],
                "additivity_check_passed": additivity["additivity_check_passed"],
            }
        )
        if additivity["additivity_check_passed"] or current_samples * 2 > max_samples:
            break
        current_samples *= 2
    magnitude = np.abs(signed_map)
    maximum = float(magnitude.max())
    display_heatmap = magnitude / maximum if maximum > 0 else magnitude

    probability_values = probabilities.detach().cpu().numpy()[0]
    return {
        "predicted_class": predicted_class,
        "predicted_label": class_names[predicted_class],
        "confidence": float(probability_values[predicted_class]),
        "probabilities": probability_values,
        "selected_class": selected_class,
        "selected_label": class_names[selected_class],
        "attribution": attribution[0].cpu().numpy(),
        "signed_attribution_map": signed_map,
        "attribution_units": "selected_class_logit_before_temperature",
        "heatmap": display_heatmap.astype(np.float32),
        "display_normalization": "absolute_value_divided_by_maximum_for_color_only",
        "resized_map": resize_nearest(wafer),
        "samples": current_samples,
        "max_samples": max_samples,
        "sample_attempts": attempts,
        "seed": seed,
        "alpha_sampling": "stratified_uniform",
        "local_smoothing": local_smoothing,
        "baseline": GRADIENT_SHAP_BASELINE,
        "raw_attribution_sum": raw_attribution_sum,
        "positive_attribution_sum": float(signed_map[signed_map > 0].sum()),
        "negative_attribution_sum": float(signed_map[signed_map < 0].sum()),
        "selected_logit_delta": logit_delta,
        **additivity,
        "rescaled_to_logit_delta": False,
        "method": GRADIENT_SHAP_METHOD,
    }


def predict_wafer_batch(
    model: WaferCNN,
    class_names: list[str],
    wafers: list[np.ndarray],
    temperature: float = 1.0,
    batch_size: int = 64,
) -> list[dict[str, object]]:
    """Predict multiple wafer maps without Grad-CAM for operational triage."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature는 0보다 큰 유한한 값이어야 합니다.")
    if batch_size <= 0:
        raise ValueError("batch_size는 1 이상이어야 합니다.")
    if not wafers:
        return []

    device = next(model.parameters()).device
    predictions: list[dict[str, object]] = []
    model.eval()
    for start in range(0, len(wafers), batch_size):
        inputs = torch.cat(
            [wafer_tensor(wafer) for wafer in wafers[start : start + batch_size]],
            dim=0,
        ).to(device)
        with torch.no_grad():
            activations = model.features(inputs)
            logits = model.classifier(model.pool(activations).flatten(1))
            probabilities = torch.softmax(logits / temperature, dim=1).cpu().numpy()
        for probability_values in probabilities:
            predicted_class = int(probability_values.argmax())
            predictions.append(
                {
                    "predicted_class": predicted_class,
                    "predicted_label": class_names[predicted_class],
                    "confidence": float(probability_values[predicted_class]),
                    "probabilities": probability_values,
                }
            )
    return predictions


def wafer_rgb(wafer: np.ndarray) -> np.ndarray:
    """Render WM-811K values with fixed, interpretable colors."""
    wafer = validate_wafer_map(wafer)
    palette = np.array(
        [[20, 24, 33], [103, 164, 121], [220, 72, 72]], dtype=np.uint8
    )
    return palette[wafer]


def gradcam_overlay(wafer: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend a Grad-CAM heatmap over the categorical wafer map."""
    base = wafer_rgb(wafer).astype(np.float32) / 255.0
    heatmap = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    if heatmap.shape != wafer.shape:
        tensor = torch.from_numpy(heatmap)[None, None]
        heatmap = F.interpolate(
            tensor, size=wafer.shape, mode="bilinear", align_corners=False
        )[0, 0].numpy()
    color = colormaps["inferno"](heatmap)[..., :3].astype(np.float32)
    alpha = (0.15 + 0.55 * heatmap)[..., None]
    overlay = base * (1.0 - alpha) + color * alpha
    return np.rint(np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)


def artificial_demo_map(size: int = 32) -> np.ndarray:
    """Create a clearly labeled artificial ring-pattern input for UI testing."""
    yy, xx = np.ogrid[:size, :size]
    center = (size - 1) / 2
    radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    wafer = np.zeros((size, size), dtype=np.uint8)
    wafer[radius <= size * 0.44] = 1
    wafer[(radius >= size * 0.32) & (radius <= size * 0.42)] = 2
    return wafer


def class_report_table(report_path: Path) -> pd.DataFrame:
    """Load only the nine class rows from the saved test report."""
    report = pd.read_csv(report_path, index_col=0)
    return report.iloc[:9].reset_index(names="class")
