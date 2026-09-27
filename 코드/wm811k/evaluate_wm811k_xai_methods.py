"""Compare Grad-CAM, Integrated Gradients, and Gradient SHAP on a fixed WM-811K test sample.

The script is uploaded as a single file by ``노트북/WM811K_07_XAI_방법비교_Colab.ipynb``
and can also run locally on CPU. Gradient SHAP here mirrors
``diagnose_wm811k.predict_with_gradient_shap``: same-geometry all-normal
baseline, stratified Expected-Gradients sampling, raw (never rescaled)
attributions, and a 5% local-accuracy check with sample doubling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

CLASS_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
METHODS = ("Grad-CAM", "Integrated Gradients", "Gradient SHAP")
GRADIENT_SHAP_RELATIVE_TOLERANCE = 0.05
GRADIENT_SHAP_ABSOLUTE_TOLERANCE = 1e-3


def canonical_text_sha256(path: Path) -> str:
    """Hash text bytes after converting CRLF/CR to LF (newline-independent)."""
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_hash(path: Path, expected: str | None) -> str:
    actual = file_sha256(path)
    if expected is not None and actual != expected.lower():
        raise ValueError("배포 체크포인트 SHA-256과 업로드 파일이 다릅니다.")
    return actual


class WaferCNN(nn.Module):
    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.30), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)).flatten(1))


def load_model(path: Path, device: torch.device) -> tuple[WaferCNN, list[str]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    names = list(checkpoint["class_names"])
    if names != CLASS_NAMES or checkpoint["input_channels"] != 2 or checkpoint["image_size"] != 64:
        raise ValueError("체크포인트 규격이 예상과 다릅니다.")
    model = WaferCNN(len(names))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), names


def wafer_tensor(wafer: np.ndarray, device: torch.device) -> torch.Tensor:
    wafer = np.asarray(wafer, dtype=np.uint8)
    if wafer.shape != (64, 64) or not set(np.unique(wafer)).issubset({0, 1, 2}):
        raise ValueError("웨이퍼는 값 0/1/2의 64x64 배열이어야 합니다.")
    channels = np.stack((wafer > 0, wafer == 2), axis=0).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(channels))[None].to(device)


def _normalize(values: torch.Tensor) -> np.ndarray:
    heatmap = torch.nan_to_num(values.detach()).clamp(min=0)
    maximum = heatmap.max()
    if maximum > 0:
        heatmap = heatmap / maximum
    return heatmap.cpu().numpy().astype(np.float32)


def gradcam_heatmap(model: WaferCNN, inputs: torch.Tensor, target: int) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    activations = model.features(inputs)
    activations.retain_grad()
    logits = model.classifier(model.pool(activations).flatten(1))
    logits[0, target].backward()
    if activations.grad is None:
        raise RuntimeError("Grad-CAM 기울기를 계산하지 못했습니다.")
    weights = activations.grad.mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    heatmap = F.interpolate(heatmap, (64, 64), mode="bilinear", align_corners=False)[0, 0]
    return _normalize(heatmap)


def integrated_gradients_heatmap(model: WaferCNN, inputs: torch.Tensor, target: int, steps: int = 24) -> np.ndarray:
    """Absolute two-channel Integrated Gradients using an all-background baseline."""
    if steps < 2:
        raise ValueError("steps는 2 이상이어야 합니다.")
    baseline = torch.zeros_like(inputs)
    alphas = torch.linspace(0, 1, steps, device=inputs.device).view(-1, 1, 1, 1)
    scaled = (baseline + alphas * (inputs - baseline)).detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    gradients = torch.autograd.grad(model(scaled)[:, target].sum(), scaled)[0]
    average_gradient = ((gradients[:-1] + gradients[1:]) * 0.5).mean(dim=0, keepdim=True)
    attribution = ((inputs - baseline) * average_gradient).abs().sum(dim=1)[0]
    return _normalize(attribution)


def gradient_shap_attribution(
    model: WaferCNN,
    inputs: torch.Tensor,
    target: int,
    samples: int = 64,
    max_samples: int = 256,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Raw signed Gradient SHAP map and its local-accuracy record for one wafer."""
    if samples < 4 or max_samples < samples:
        raise ValueError("Gradient SHAP 표본 수 설정이 잘못되었습니다.")
    baseline = inputs.clone()
    baseline[:, 1] = 0.0
    with torch.no_grad():
        logit_delta = float((model(inputs)[0, target] - model(baseline)[0, target]).item())
    delta = inputs - baseline
    attempts = []
    current = samples
    while True:
        generator = torch.Generator(device=inputs.device)
        generator.manual_seed(seed)
        offsets = torch.rand(current, generator=generator, device=inputs.device)
        alphas = ((torch.arange(current, device=inputs.device) + offsets) / current).view(-1, 1, 1, 1)
        path = (baseline + alphas * delta).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        gradients = torch.autograd.grad(model(path)[:, target].sum(), path)[0]
        signed = (delta * gradients.mean(dim=0, keepdim=True)).sum(dim=1)[0].detach().cpu().numpy().astype(np.float64)
        residual = float(signed.sum()) - logit_delta
        tolerance = max(GRADIENT_SHAP_ABSOLUTE_TOLERANCE, GRADIENT_SHAP_RELATIVE_TOLERANCE * abs(logit_delta))
        relative = abs(residual) / abs(logit_delta) if abs(logit_delta) > 0 else (0.0 if residual == 0 else float("inf"))
        passed = bool(np.isfinite(residual) and abs(residual) <= tolerance)
        attempts.append(current)
        if passed or current * 2 > max_samples:
            break
        current *= 2
    return signed, {
        "gradient_shap_samples": current,
        "gradient_shap_attempts": "|".join(map(str, attempts)),
        "selected_logit_delta": logit_delta,
        "raw_attribution_sum": float(signed.sum()),
        "raw_additivity_residual": residual,
        "relative_additivity_residual": relative,
        "additivity_tolerance": tolerance,
        "additivity_check_passed": passed,
    }


def flip_ranked_defects(wafer: np.ndarray, scores: np.ndarray, fraction: float, largest: bool) -> np.ndarray | None:
    """Turn the highest- or lowest-scored defect dies into normal dies."""
    defects = np.flatnonzero((wafer == 2).ravel())
    if len(defects) == 0:
        return None
    count = max(1, round(len(defects) * fraction))
    order = np.argsort(scores.ravel()[defects], kind="stable")
    chosen = defects[order[-count:] if largest else order[:count]]
    result = wafer.copy().ravel()
    result[chosen] = 1
    return result.reshape(wafer.shape)


def flip_random_defects(wafer: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray | None:
    defects = np.flatnonzero((wafer == 2).ravel())
    if len(defects) == 0:
        return None
    count = max(1, round(len(defects) * fraction))
    result = wafer.copy().ravel()
    result[rng.choice(defects, count, replace=False)] = 1
    return result.reshape(wafer.shape)


def fixed_class_sample(assignments: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    required = {"array_index", "label_id", "failure_type", "split"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"분할표 필수 열 누락: {sorted(missing)}")
    test = assignments.loc[assignments["split"].eq("test")]
    if sorted(test["label_id"].unique().tolist()) != list(range(9)):
        raise ValueError("test split에 9개 클래스가 모두 있어야 합니다.")
    pieces = []
    for class_id in range(9):
        group = test.loc[test["label_id"].eq(class_id)]
        pieces.append(group.sample(min(per_class, len(group)), random_state=seed + class_id))
    sampled = pd.concat(pieces, ignore_index=True).sort_values(["label_id", "array_index"]).reset_index(drop=True)
    sampled["selection_seed"] = seed
    sampled["class_test_count"] = sampled["label_id"].map(test.groupby("label_id").size())
    sampled["class_sample_count"] = sampled["label_id"].map(sampled.groupby("label_id").size())
    return sampled


def mask_ranked(wafer: np.ndarray, heatmap: np.ndarray, fraction: float, largest: bool) -> np.ndarray:
    active = np.flatnonzero((wafer > 0).ravel())
    count = max(1, round(len(active) * fraction))
    order = np.argsort(heatmap.ravel()[active], kind="stable")
    chosen = active[order[-count:] if largest else order[:count]]
    result = wafer.copy().ravel()
    result[chosen] = 0
    return result.reshape(wafer.shape)


def mask_random(wafer: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    active = np.flatnonzero((wafer > 0).ravel())
    count = max(1, round(len(active) * fraction))
    result = wafer.copy().ravel()
    result[rng.choice(active, count, replace=False)] = 0
    return result.reshape(wafer.shape)


def probabilities(model: WaferCNN, wafers: list[np.ndarray], device: torch.device, temperature: float, batch: int = 256) -> np.ndarray:
    values = []
    for start in range(0, len(wafers), batch):
        inputs = torch.cat([wafer_tensor(x, device) for x in wafers[start:start + batch]])
        with torch.no_grad():
            values.append(torch.softmax(model(inputs) / temperature, dim=1).cpu().numpy())
    return np.concatenate(values)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, repeats: int = 2000) -> list[float]:
    draws = rng.choice(np.asarray(values, dtype=float), size=(repeats, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def evaluate(args: argparse.Namespace) -> dict:
    checkpoint_hash = verify_checkpoint_hash(
        args.checkpoint, getattr(args, "expected_checkpoint_sha256", None)
    )
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, names = load_model(args.checkpoint, device)
    maps = np.load(args.maps, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")
    selected = fixed_class_sample(pd.read_csv(args.assignments), args.per_class, args.seed)
    indices = selected["array_index"].to_numpy(dtype=np.int64)
    if len(maps) != len(labels) or not np.array_equal(labels[indices].astype(int), selected["label_id"].to_numpy(dtype=int)):
        raise ValueError("전처리 배열과 분할표가 일치하지 않습니다.")
    wafers = [np.asarray(maps[i], dtype=np.uint8).copy() for i in indices]
    base = probabilities(model, wafers, device, args.temperature)
    predictions = base.argmax(axis=1)
    heatmaps = {method: [] for method in METHODS}
    gradient_shap_signed = []
    gradient_shap_records = []
    for i, wafer in enumerate(wafers):
        inputs = wafer_tensor(wafer, device)
        target = int(predictions[i])
        heatmaps["Grad-CAM"].append(gradcam_heatmap(model, inputs, target))
        heatmaps["Integrated Gradients"].append(integrated_gradients_heatmap(model, inputs, target, args.ig_steps))
        signed, additivity = gradient_shap_attribution(
            model, inputs, target, args.gradient_shap_samples, args.gradient_shap_max_samples, args.seed
        )
        magnitude = np.abs(signed)
        heatmaps["Gradient SHAP"].append((magnitude / magnitude.max() if magnitude.max() > 0 else magnitude).astype(np.float32))
        gradient_shap_signed.append(signed)
        gradient_shap_records.append(additivity)
        if (i + 1) % 25 == 0 or i + 1 == len(wafers):
            print(f"설명 계산 {i + 1}/{len(wafers)}", flush=True)
    # Ranking scores for the defect-flip test: unsigned maps for Grad-CAM and IG,
    # signed contributions to the predicted-class logit for Gradient SHAP.
    flip_scores = {
        "Grad-CAM": heatmaps["Grad-CAM"],
        "Integrated Gradients": heatmaps["Integrated Gradients"],
        "Gradient SHAP": gradient_shap_signed,
    }

    rng = np.random.default_rng(args.seed)
    random_wafers = [mask_random(wafer, args.mask_fraction, rng) for wafer in wafers for _ in range(args.random_repeats)]
    random_probs = probabilities(model, random_wafers, device, args.temperature)
    flip_rng = np.random.default_rng(args.seed + 2000)
    flip_random_sets = [
        [flip_random_defects(wafer, args.mask_fraction, flip_rng) for _ in range(args.random_repeats)]
        for wafer in wafers
    ]
    flippable = [sets[0] is not None for sets in flip_random_sets]
    flip_random_probs = probabilities(
        model, [flipped for sets in flip_random_sets for flipped in sets if flipped is not None], device, args.temperature
    ) if any(flippable) else np.empty((0, len(names)))
    flip_offsets = np.cumsum([0] + [args.random_repeats if ok else 0 for ok in flippable])
    rows = []
    for method, maps_for_method in heatmaps.items():
        top = probabilities(model, [mask_ranked(w, h, args.mask_fraction, True) for w, h in zip(wafers, maps_for_method)], device, args.temperature)
        bottom = probabilities(model, [mask_ranked(w, h, args.mask_fraction, False) for w, h in zip(wafers, maps_for_method)], device, args.temperature)
        flip_positions = [i for i, ok in enumerate(flippable) if ok]
        flip_top = probabilities(model, [flip_ranked_defects(wafers[i], flip_scores[method][i], args.mask_fraction, True) for i in flip_positions], device, args.temperature) if flip_positions else np.empty((0, len(names)))
        flip_bottom = probabilities(model, [flip_ranked_defects(wafers[i], flip_scores[method][i], args.mask_fraction, False) for i in flip_positions], device, args.temperature) if flip_positions else np.empty((0, len(names)))
        flip_row = {position: row for row, position in enumerate(flip_positions)}
        for i, (wafer, heatmap) in enumerate(zip(wafers, maps_for_method)):
            target = int(predictions[i]); confidence = float(base[i, target])
            random_target = random_probs[i * args.random_repeats:(i + 1) * args.random_repeats, target]
            top_drop = confidence - float(top[i, target]); random_drop = confidence - float(random_target.mean()); bottom_drop = confidence - float(bottom[i, target])
            active = wafer > 0; active_fraction = float(active.mean()); mass = float(heatmap.sum()); active_mass = float(heatmap[active].sum() / mass) if mass else 0.0
            row = {"method": method, "array_index": int(indices[i]), "true_class": names[int(selected.iloc[i]["label_id"])], "predicted_class": names[target], "correct": target == int(selected.iloc[i]["label_id"]), "baseline_confidence": confidence, "active_attribution_mass": active_mass, "active_attribution_lift": active_mass / active_fraction if active_fraction else np.nan, "top10_confidence_drop": top_drop, "random10_confidence_drop_mean": random_drop, "bottom10_confidence_drop": bottom_drop, "top_vs_random_advantage": top_drop - random_drop, "top_vs_bottom_advantage": top_drop - bottom_drop, "top10_prediction_changed": int(top[i].argmax()) != target}
            row["defect_die_count"] = int((wafer == 2).sum())
            if flippable[i]:
                flip_index = flip_row[i]
                flip_random_target = flip_random_probs[flip_offsets[i]:flip_offsets[i + 1], target]
                flip_top_drop = confidence - float(flip_top[flip_index, target])
                flip_random_drop = confidence - float(flip_random_target.mean())
                flip_bottom_drop = confidence - float(flip_bottom[flip_index, target])
                row.update({"flip_top10_confidence_drop": flip_top_drop, "flip_random10_confidence_drop_mean": flip_random_drop, "flip_bottom10_confidence_drop": flip_bottom_drop, "flip_top_vs_random_advantage": flip_top_drop - flip_random_drop, "flip_top_vs_bottom_advantage": flip_top_drop - flip_bottom_drop})
            else:
                row.update({key: np.nan for key in FLIP_METRICS})
            if method == "Gradient SHAP":
                row.update(gradient_shap_records[i])
            rows.append(row)

    records = pd.DataFrame(rows)
    class_summary = records.groupby(["method", "true_class"], sort=False)[LEGACY_METRICS + FLIP_METRICS].mean().reset_index()
    summary = {"schema_version": 1, "purpose": "post_selection_class_balanced_xai_comparison", "used_for_model_selection": False, "thresholds_retuned": False, "population_representative": False, "sample_count": len(selected), "per_class_cap": args.per_class, "mask_fraction": args.mask_fraction, "random_repeats": args.random_repeats, "integrated_gradients_steps": args.ig_steps, "random_seed": args.seed, "device": str(device), "methods": {}, "limitations": ["클래스별 동일 상한 표본이므로 실제 test 클래스 비율의 전체 평균이 아닙니다.", "0 기준 Integrated Gradients와 die 제거는 실제 공정 개입이 아닌 민감도 점검입니다.", "설명법은 물리적 원인이나 인과관계를 증명하지 않습니다."]}
    summary.update({
        "schema_version": 3,
        "methods_compared": list(METHODS),
        "checkpoint_sha256": checkpoint_hash,
        "assignments_sha256": file_sha256(args.assignments),
        "assignments_canonical_sha256": canonical_text_sha256(args.assignments),
        "maps_sha256": file_sha256(args.maps),
        "labels_sha256": file_sha256(args.labels),
        "temperature": args.temperature,
        "defect_flip_fraction": args.mask_fraction,
        "defect_flip_note": "상위·하위·무작위 결함 die를 정상 die로 바꿉니다. Gradient SHAP 기준(같은 형상의 모든 die 정상)과 같은 방향의 교란입니다.",
    })
    boot_rng = np.random.default_rng(args.seed + 1000)
    # Legacy metrics keep the historical bootstrap stream: Grad-CAM, IG, paired IG-Grad-CAM.
    for method in ("Grad-CAM", "Integrated Gradients"):
        summary["methods"][method] = _legacy_method_summary(records, class_summary, method, boot_rng)
    paired = records.pivot(index="array_index", columns="method", values="top_vs_random_advantage")
    delta = paired["Integrated Gradients"] - paired["Grad-CAM"]
    summary["paired_ig_minus_gradcam"] = float(delta.mean())
    summary["paired_ig_minus_gradcam_95ci"] = bootstrap_ci(delta.to_numpy(), boot_rng)
    summary["methods"]["Gradient SHAP"] = _legacy_method_summary(records, class_summary, "Gradient SHAP", boot_rng)
    for method in METHODS:
        part = records.loc[records["method"].eq(method)].dropna(subset=["flip_top_vs_random_advantage"])
        macro = class_summary.loc[class_summary["method"].eq(method), FLIP_METRICS].mean(numeric_only=True, skipna=True)
        summary["methods"][method]["defect_flip"] = {"evaluated_wafers": int(len(part)), "macro_top_vs_random_advantage": float(macro["flip_top_vs_random_advantage"]), "top_vs_random_95ci": bootstrap_ci(part["flip_top_vs_random_advantage"].to_numpy(), boot_rng), "macro_top_vs_bottom_advantage": float(macro["flip_top_vs_bottom_advantage"]), "top_exceeds_random_rate": float((part["flip_top_vs_random_advantage"] > 0).mean()), "top_exceeds_bottom_rate": float((part["flip_top_vs_bottom_advantage"] > 0).mean())}
    flip_paired = records.pivot(index="array_index", columns="method", values="flip_top_vs_random_advantage").dropna()
    for other, key in (("Integrated Gradients", "integrated_gradients"), ("Grad-CAM", "gradcam")):
        difference = flip_paired["Gradient SHAP"] - flip_paired[other]
        summary[f"paired_gradient_shap_minus_{key}_defect_flip"] = float(difference.mean())
        summary[f"paired_gradient_shap_minus_{key}_defect_flip_95ci"] = bootstrap_ci(difference.to_numpy(), boot_rng)
    shap_part = records.loc[records["method"].eq("Gradient SHAP")]
    relative = shap_part["relative_additivity_residual"].to_numpy(dtype=float)
    summary["gradient_shap"] = {
        "baseline": "same_geometry_all_active_dies_normal",
        "estimator": "stratified_expected_gradients",
        "attribution_units": "selected_class_logit_before_temperature",
        "rescaled_to_logit_delta": False,
        "initial_samples": args.gradient_shap_samples,
        "max_samples": args.gradient_shap_max_samples,
        "seed": args.seed,
        "relative_tolerance": GRADIENT_SHAP_RELATIVE_TOLERANCE,
        "absolute_tolerance": GRADIENT_SHAP_ABSOLUTE_TOLERANCE,
        "additivity_pass_count": int(shap_part["additivity_check_passed"].sum()),
        "additivity_pass_rate": float(shap_part["additivity_check_passed"].mean()),
        "relative_residual_median": float(np.median(relative)),
        "relative_residual_p90": float(np.quantile(relative, 0.9)),
        "relative_residual_max": float(relative.max()),
        "samples_used_counts": {str(key): int(value) for key, value in shap_part["gradient_shap_samples"].astype(int).value_counts().sort_index().items()},
    }
    summary["limitations"] = summary["limitations"] + [
        "Gradient SHAP 가산성 점검은 사후 보정 전 원시 attribution 합계로 계산했습니다. 유한 표본 Expected Gradients 근사라 잔차가 남을 수 있으며 허용오차를 넘은 웨이퍼는 통과로 세지 않습니다.",
        "결함 die → 정상 die 교란은 모델 민감도 점검이며 실제 공정 개입이나 원인 확인이 아닙니다.",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "xai_selected_test_samples.csv", index=False)
    records.to_csv(args.output_dir / "xai_method_records.csv", index=False)
    class_summary.to_csv(args.output_dir / "xai_method_class_summary.csv", index=False)
    (args.output_dir / "xai_method_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    colors = ["#4c78a8", "#f28e2b", "#54a24b"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    legacy_plot = class_summary.pivot(index="true_class", columns="method", values="top_vs_random_advantage").reindex(index=CLASS_NAMES, columns=list(METHODS))
    legacy_plot.plot(kind="bar", ax=axes[0, 0], color=colors); axes[0, 0].axhline(0, color="black", lw=.8); axes[0, 0].set(title="Top 10% die removal advantage over random", xlabel="True class", ylabel="Confidence-drop advantage"); axes[0, 0].tick_params(axis="x", rotation=35)
    flip_plot = class_summary.pivot(index="true_class", columns="method", values="flip_top_vs_random_advantage").reindex(index=CLASS_NAMES, columns=list(METHODS))
    flip_plot.plot(kind="bar", ax=axes[0, 1], color=colors); axes[0, 1].axhline(0, color="black", lw=.8); axes[0, 1].set(title="Top 10% defect-to-normal flip advantage over random", xlabel="True class", ylabel="Confidence-drop advantage"); axes[0, 1].tick_params(axis="x", rotation=35)
    axes[1, 0].boxplot([records.loc[records["method"].eq(x), "flip_top_vs_random_advantage"].dropna() for x in METHODS], tick_labels=list(METHODS), showmeans=True); axes[1, 0].axhline(0, color="black", lw=.8); axes[1, 0].set(title="Per-wafer defect-flip faithfulness", ylabel="Top-minus-random confidence drop")
    axes[1, 1].hist(np.clip(relative, 0, 0.5), bins=25, color=colors[2]); axes[1, 1].axvline(GRADIENT_SHAP_RELATIVE_TOLERANCE, color="black", ls="--", lw=1); axes[1, 1].set(title="Gradient SHAP raw relative additivity residual", xlabel="|sum(phi) - logit delta| / |logit delta| (clipped at 0.5)", ylabel="Wafers")
    fig.tight_layout(); fig.savefig(args.output_dir / "xai_method_comparison_dashboard.png", dpi=170); plt.close(fig)
    gs = summary["gradient_shap"]
    lines = ["# WM-811K 설명 방법 비교", "", f"고정 test 표본 {len(selected):,}개에서 Grad-CAM, Integrated Gradients, Gradient SHAP을 비교했습니다.", "클래스별 최대 표본 수가 같으므로 실제 test 분포 평균이 아닌 클래스 균형 점검입니다.", "", "## Gradient SHAP 가산성(원시 attribution)", "", f"- 허용오차: 로짓 변화량의 {gs['relative_tolerance']:.0%} (최소 {gs['absolute_tolerance']})", f"- 통과: {gs['additivity_pass_count']}/{len(shap_part)} ({gs['additivity_pass_rate']:.1%})", f"- 상대 잔차 중앙값/90%/최대: {gs['relative_residual_median']:.4f}/{gs['relative_residual_p90']:.4f}/{gs['relative_residual_max']:.4f}", f"- 사용 표본 수 분포: {gs['samples_used_counts']}", ""]
    for method, value in summary["methods"].items():
        ci = value["top_vs_random_95ci"]; flip = value["defect_flip"]; flip_ci = flip["top_vs_random_95ci"]
        lines += [f"## {method}", "", f"- die 제거: 클래스 평균 상위-무작위 효과 {value['macro_top_vs_random_advantage']:+.4f} (bootstrap 95% {ci[0]:+.4f}~{ci[1]:+.4f})", f"- die 제거: 상위 제거가 무작위보다 큰 표본 {value['top_exceeds_random_rate']:.1%}", f"- 결함→정상 교란: 클래스 평균 상위-무작위 효과 {flip['macro_top_vs_random_advantage']:+.4f} (bootstrap 95% {flip_ci[0]:+.4f}~{flip_ci[1]:+.4f}, {flip['evaluated_wafers']}개)", f"- 결함→정상 교란: 상위가 무작위보다 큰 표본 {flip['top_exceeds_random_rate']:.1%}", ""]
    lines += ["세 방법 모두 실제 공정 개입이나 물리적 원인을 증명하지 않습니다. 위치 참고와 전문가 검토 보조에만 사용합니다."]
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


LEGACY_METRICS = ["active_attribution_mass", "active_attribution_lift", "top10_confidence_drop", "random10_confidence_drop_mean", "bottom10_confidence_drop", "top_vs_random_advantage", "top_vs_bottom_advantage", "top10_prediction_changed"]
FLIP_METRICS = ["flip_top10_confidence_drop", "flip_random10_confidence_drop_mean", "flip_bottom10_confidence_drop", "flip_top_vs_random_advantage", "flip_top_vs_bottom_advantage"]


def _legacy_method_summary(records: pd.DataFrame, class_summary: pd.DataFrame, method: str, boot_rng: np.random.Generator) -> dict:
    part = records.loc[records["method"].eq(method)]
    macro = class_summary.loc[class_summary["method"].eq(method), LEGACY_METRICS].mean(numeric_only=True)
    return {"macro_top_vs_random_advantage": float(macro["top_vs_random_advantage"]), "top_vs_random_95ci": bootstrap_ci(part["top_vs_random_advantage"].to_numpy(), boot_rng), "macro_top_vs_bottom_advantage": float(macro["top_vs_bottom_advantage"]), "macro_active_attribution_lift": float(macro["active_attribution_lift"]), "top_exceeds_random_rate": float((part["top_vs_random_advantage"] > 0).mean()), "top_exceeds_bottom_rate": float((part["top_vs_bottom_advantage"] > 0).mean())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True); parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", type=str)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--temperature", type=float, default=1.0); parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--ig-steps", type=int, default=24); parser.add_argument("--mask-fraction", type=float, default=0.10)
    parser.add_argument("--random-repeats", type=int, default=5); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-shap-samples", type=int, default=64); parser.add_argument("--gradient-shap-max-samples", type=int, default=256)
    parsed = parser.parse_args()
    if parsed.per_class < 1 or parsed.temperature <= 0 or parsed.ig_steps < 2 or not 0 < parsed.mask_fraction < 1 or parsed.random_repeats < 1:
        parser.error("평가 매개변수가 유효하지 않습니다.")
    if parsed.gradient_shap_samples < 4 or parsed.gradient_shap_max_samples < parsed.gradient_shap_samples:
        parser.error("Gradient SHAP 표본 수 설정이 유효하지 않습니다.")
    print(json.dumps(evaluate(parsed), ensure_ascii=False, indent=2))
