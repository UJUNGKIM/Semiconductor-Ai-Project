"""Build a leakage-safe feature-space OOD reference for the WM-811K CNN."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPLITS = ("train", "validation", "test")


class WaferCNN(nn.Module):
    """Checkpoint-compatible compact WM-811K CNN."""

    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.30), nn.Linear(256, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(self.pool(features).flatten(1))


def load_checkpoint(checkpoint_path: Path) -> tuple[WaferCNN, list[str], dict]:
    """Load the selected checkpoint without depending on another project module."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {
        "model_state_dict",
        "class_names",
        "num_classes",
        "input_channels",
        "image_size",
        "best_epoch",
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OodWaferDataset(Dataset):
    """Read fixed 64x64 categorical wafer maps lazily from a memory map."""

    def __init__(
        self,
        maps_path: Path,
        indices: np.ndarray,
        labels: np.ndarray,
        *,
        spatial_shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.maps_path = str(maps_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.spatial_shuffle = spatial_shuffle
        self.seed = seed
        self._maps: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def _open_maps(self) -> np.ndarray:
        if self._maps is None:
            self._maps = np.load(self.maps_path, mmap_mode="r", allow_pickle=False)
        return self._maps

    def __getitem__(self, position: int) -> tuple[torch.Tensor, int, int, float, float]:
        array_index = int(self.indices[position])
        wafer = np.asarray(self._open_maps()[array_index], dtype=np.uint8).copy()
        if self.spatial_shuffle:
            rng = np.random.default_rng(self.seed + array_index)
            wafer = rng.permutation(wafer.reshape(-1)).reshape(wafer.shape)
        wafer_area = int((wafer > 0).sum())
        defect_count = int((wafer == 2).sum())
        coverage = wafer_area / wafer.size
        defect_ratio = defect_count / max(wafer_area, 1)
        channels = np.stack((wafer > 0, wafer == 2), axis=0).astype(np.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(channels)),
            int(self.labels[position]),
            array_index,
            float(coverage),
            float(defect_ratio),
        )


def fit_class_reference(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    shrinkage: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit regularized diagonal Gaussian references for each known class."""
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features와 labels의 행 수가 일치해야 합니다.")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage는 0과 1 사이여야 합니다.")
    global_variance = features.var(axis=0, dtype=np.float64) + 1e-6
    means = np.zeros((num_classes, features.shape[1]), dtype=np.float32)
    variances = np.zeros_like(means)
    counts = np.zeros(num_classes, dtype=np.int64)
    for class_id in range(num_classes):
        subset = np.asarray(features[labels == class_id], dtype=np.float64)
        if len(subset) < 2:
            raise ValueError(f"train 클래스 {class_id}의 표본이 2개 미만입니다.")
        counts[class_id] = len(subset)
        means[class_id] = subset.mean(axis=0).astype(np.float32)
        local_variance = subset.var(axis=0)
        regularized = (1.0 - shrinkage) * local_variance + shrinkage * global_variance
        variances[class_id] = np.maximum(regularized, 1e-6).astype(np.float32)
    return means, variances, counts


def embedding_distance_scores(
    features: np.ndarray,
    class_means: np.ndarray,
    class_variances: np.ndarray,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Return minimum normalized diagonal-Mahalanobis distance and nearest class."""
    if class_means.shape != class_variances.shape:
        raise ValueError("class_means와 class_variances 형상이 다릅니다.")
    if chunk_size <= 0:
        raise ValueError("chunk_size는 1 이상이어야 합니다.")
    scores = np.empty(len(features), dtype=np.float32)
    nearest = np.empty(len(features), dtype=np.int64)
    for start in range(0, len(features), chunk_size):
        stop = min(start + chunk_size, len(features))
        chunk = features[start:stop]
        distances = np.sqrt(
            np.mean(
                np.square(chunk[:, None, :] - class_means[None, :, :])
                / class_variances[None, :, :],
                axis=2,
            )
        )
        chunk_nearest = distances.argmin(axis=1).astype(np.int64)
        nearest[start:stop] = chunk_nearest
        scores[start:stop] = distances[np.arange(len(distances)), chunk_nearest]
    return scores, nearest


def calibrate_score_thresholds(
    validation_scores: np.ndarray,
    review_quantile: float,
    ood_quantile: float,
) -> dict[str, float]:
    """Fix review and OOD cutoffs using validation scores only."""
    scores = np.asarray(validation_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("validation_scores는 비어 있지 않은 유한한 1차원 배열이어야 합니다.")
    if not 0.5 < review_quantile < ood_quantile < 1.0:
        raise ValueError("0.5 < review_quantile < ood_quantile < 1이어야 합니다.")
    return {
        "review_quantile": float(review_quantile),
        "ood_quantile": float(ood_quantile),
        "review_threshold": float(np.quantile(scores, review_quantile)),
        "ood_threshold": float(np.quantile(scores, ood_quantile)),
    }


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Extract pooled CNN features and basic wafer geometry statistics."""
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    prediction_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    coverage_chunks: list[np.ndarray] = []
    defect_ratio_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, labels, indices, coverage, defect_ratio in loader:
            images = images.to(device, non_blocking=True)
            feature_map = model.features(images)
            features = model.pool(feature_map).flatten(1)
            logits = model.classifier(features)
            feature_chunks.append(features.cpu().numpy().astype(np.float32))
            prediction_chunks.append(logits.argmax(dim=1).cpu().numpy())
            label_chunks.append(labels.numpy())
            index_chunks.append(indices.numpy())
            coverage_chunks.append(coverage.numpy())
            defect_ratio_chunks.append(defect_ratio.numpy())
    return {
        "features": np.concatenate(feature_chunks),
        "labels": np.concatenate(label_chunks).astype(np.int64),
        "predictions": np.concatenate(prediction_chunks).astype(np.int64),
        "array_indices": np.concatenate(index_chunks).astype(np.int64),
        "wafer_coverage": np.concatenate(coverage_chunks).astype(np.float32),
        "defect_ratio": np.concatenate(defect_ratio_chunks).astype(np.float32),
    }


def geometry_reference(
    labels: np.ndarray,
    coverage: np.ndarray,
    defect_ratio: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    """Store robust train-only ranges for future input-quality checks."""
    metrics = {"wafer_coverage": coverage, "defect_ratio": defect_ratio}

    def bounds(values: np.ndarray) -> dict[str, float]:
        return {
            "lower_0_1pct": float(np.quantile(values, 0.001)),
            "lower_1pct": float(np.quantile(values, 0.01)),
            "median": float(np.median(values)),
            "upper_99pct": float(np.quantile(values, 0.99)),
            "upper_99_9pct": float(np.quantile(values, 0.999)),
        }

    return {
        "global": {name: bounds(values) for name, values in metrics.items()},
        "by_class": {
            class_name: {
                name: bounds(values[labels == class_id])
                for name, values in metrics.items()
            }
            for class_id, class_name in enumerate(class_names)
        },
    }


def score_status(scores: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    return np.where(
        scores >= thresholds["ood_threshold"],
        "out_of_distribution",
        np.where(scores >= thresholds["review_threshold"], "review", "in_distribution"),
    )


def split_metrics(
    name: str,
    data: dict[str, np.ndarray],
    scores: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, object]:
    status = score_status(scores, thresholds)
    return {
        "split": name,
        "rows": int(len(scores)),
        "accuracy": float(np.mean(data["labels"] == data["predictions"])),
        "mean_score": float(np.mean(scores)),
        "review_rate": float(np.mean(status == "review")),
        "ood_rate": float(np.mean(status == "out_of_distribution")),
        "automatic_coverage": float(np.mean(status == "in_distribution")),
    }


def make_loader(
    maps_path: Path,
    subset: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    *,
    spatial_shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    dataset = OodWaferDataset(
        maps_path,
        subset["array_index"].to_numpy(),
        subset["label_id"].to_numpy(),
        spatial_shuffle=spatial_shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    for path in (args.maps, args.labels, args.assignments, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    assignments = pd.read_csv(args.assignments)
    required = {"array_index", "label_id", "failure_type", "lot_name", "split"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"split_assignments.csv 필수 열 누락: {sorted(missing)}")
    if set(assignments["split"].unique()) != set(SPLITS):
        raise ValueError("train/validation/test split이 모두 필요합니다.")
    if assignments["array_index"].duplicated().any():
        raise ValueError("array_index가 중복됩니다.")

    labels = np.load(args.labels, mmap_mode="r", allow_pickle=False)
    map_shape = np.load(args.maps, mmap_mode="r", allow_pickle=False).shape
    if len(labels) != map_shape[0] or tuple(map_shape[1:]) != (64, 64):
        raise ValueError(f"전처리 배열 형상이 예상과 다릅니다: maps={map_shape}, labels={labels.shape}")
    assignment_indices = assignments["array_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        np.asarray(labels[assignment_indices], dtype=np.int64),
        assignments["label_id"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("labels.npy와 split_assignments.csv 라벨이 일치하지 않습니다.")

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. Colab 런타임을 T4 GPU로 변경하세요.")
    device = torch.device(requested_device)
    model, class_names, checkpoint = load_checkpoint(args.checkpoint)
    model = model.to(device)
    num_classes = len(class_names)
    assignment_class_names = (
        assignments[["label_id", "failure_type"]]
        .drop_duplicates()
        .sort_values("label_id")["failure_type"]
        .tolist()
    )
    if assignment_class_names != class_names:
        raise ValueError(
            "split_assignments.csv의 label_id 순서와 체크포인트 클래스 순서가 다릅니다: "
            f"{assignment_class_names} != {class_names}"
        )
    print(f"device={device}, classes={num_classes}, maps={map_shape[0]:,}")

    extracted: dict[str, dict[str, np.ndarray]] = {}
    for split_name in SPLITS:
        subset = assignments.loc[assignments["split"] == split_name].reset_index(drop=True)
        print(f"extracting {split_name}: {len(subset):,}")
        extracted[split_name] = extract_features(
            model,
            make_loader(
                args.maps, subset, args.batch_size, args.num_workers, device, seed=args.seed
            ),
            device,
        )

    train = extracted["train"]
    class_means, class_variances, class_counts = fit_class_reference(
        train["features"], train["labels"], num_classes, args.shrinkage
    )
    scores: dict[str, np.ndarray] = {}
    nearest: dict[str, np.ndarray] = {}
    for split_name in SPLITS:
        scores[split_name], nearest[split_name] = embedding_distance_scores(
            extracted[split_name]["features"], class_means, class_variances
        )
    thresholds = calibrate_score_thresholds(
        scores["validation"], args.review_quantile, args.ood_quantile
    )

    validation_subset = assignments.loc[assignments["split"] == "validation"].copy()
    stress_subset = validation_subset.sample(
        n=min(args.stress_samples, len(validation_subset)), random_state=args.seed
    ).reset_index(drop=True)
    print(f"extracting spatial-shuffle stress set: {len(stress_subset):,}")
    stress = extract_features(
        model,
        make_loader(
            args.maps,
            stress_subset,
            args.batch_size,
            args.num_workers,
            device,
            spatial_shuffle=True,
            seed=args.seed,
        ),
        device,
    )
    stress_scores, stress_nearest = embedding_distance_scores(
        stress["features"], class_means, class_variances
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = {
        "schema_version": 1,
        "method": "minimum_regularized_diagonal_mahalanobis",
        "feature_layer": "adaptive_average_pool_256",
        "feature_dim": int(class_means.shape[1]),
        "class_names": class_names,
        "variance_shrinkage": float(args.shrinkage),
        "thresholds": thresholds,
        "geometry_reference": geometry_reference(
            train["labels"], train["wafer_coverage"], train["defect_ratio"], class_names
        ),
        "fit_split": "train",
        "calibration_split": "validation",
        "test_used_for_threshold": False,
        "seed": int(args.seed),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "assignments_sha256": sha256_file(args.assignments),
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
    }
    np.savez_compressed(
        args.output_dir / "ood_reference.npz",
        class_means=class_means,
        class_variances=class_variances,
        class_counts=class_counts,
    )
    (args.output_dir / "ood_reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    metric_rows = [
        split_metrics(name, extracted[name], scores[name], thresholds) for name in SPLITS
    ]
    stress_status = score_status(stress_scores, thresholds)
    stress_metrics = {
        "split": "spatial_shuffle_stress",
        "rows": int(len(stress_scores)),
        "mean_score": float(np.mean(stress_scores)),
        "review_or_ood_rate": float(np.mean(stress_status != "in_distribution")),
        "ood_detection_rate": float(np.mean(stress_status == "out_of_distribution")),
    }
    pd.DataFrame(metric_rows + [stress_metrics]).to_csv(
        args.output_dir / "ood_split_metrics.csv", index=False
    )

    score_frames = []
    for split_name in ("validation", "test"):
        data = extracted[split_name]
        score_frames.append(
            pd.DataFrame(
                {
                    "split": split_name,
                    "array_index": data["array_indices"],
                    "true_label_id": data["labels"],
                    "predicted_label_id": data["predictions"],
                    "nearest_reference_label_id": nearest[split_name],
                    "ood_score": scores[split_name],
                    "ood_status": score_status(scores[split_name], thresholds),
                }
            )
        )
    score_frames.append(
        pd.DataFrame(
            {
                "split": "spatial_shuffle_stress",
                "array_index": stress["array_indices"],
                "true_label_id": stress["labels"],
                "predicted_label_id": stress["predictions"],
                "nearest_reference_label_id": stress_nearest,
                "ood_score": stress_scores,
                "ood_status": stress_status,
            }
        )
    )
    pd.concat(score_frames, ignore_index=True).to_csv(
        args.output_dir / "ood_score_records.csv", index=False
    )

    figure, axis = plt.subplots(figsize=(9, 5))
    upper = float(
        np.quantile(np.concatenate((scores["validation"], scores["test"], stress_scores)), 0.995)
    )
    bins = np.linspace(0.0, upper, 70)
    axis.hist(scores["validation"], bins=bins, density=True, alpha=0.55, label="validation")
    axis.hist(scores["test"], bins=bins, density=True, alpha=0.45, label="test")
    axis.hist(stress_scores, bins=bins, density=True, alpha=0.40, label="spatial-shuffle stress")
    axis.axvline(thresholds["review_threshold"], color="#e69f00", linestyle="--", label="review")
    axis.axvline(thresholds["ood_threshold"], color="#d55e00", linestyle="--", label="OOD")
    axis.set(xlabel="Feature-space OOD score", ylabel="Density", title="WM-811K OOD calibration")
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "ood_score_distribution.png", dpi=180)
    plt.close(figure)

    summary: dict[str, object] = {
        "schema_version": 1,
        "method": reference["method"],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "class_names": class_names,
        "split_rows": {
            name: int((assignments["split"] == name).sum()) for name in SPLITS
        },
        "threshold_selection": {
            **thresholds,
            "calibration_split": "validation",
            "test_used_for_threshold": False,
        },
        "split_metrics": metric_rows,
        "stress_metrics": stress_metrics,
        "limitations": [
            "Validation 분위수로 고정한 연구용 분포 이탈 기준입니다.",
            "공간 셔플 stress는 실제 외부 장비 데이터가 아니며 안전성 증명의 대체물이 아닙니다.",
            "신규 장비와 신규 lot에서 별도 외부 검증이 필요합니다.",
        ],
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "assignments_sha256": reference["assignments_sha256"],
    }
    (args.output_dir / "ood_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# WM-811K OOD 기준 결과\n\n"
        "- 기준 적합: train CNN pooled feature\n"
        "- 임계값 고정: validation 점수 분위수\n"
        "- test 사용: 고정 후 평가만 수행\n"
        "- stress: validation 웨이퍼 픽셀 공간 셔플(실제 외부 데이터 아님)\n\n"
        "이 기준은 연구·시연용이며 신규 장비 데이터의 외부 검증을 대체하지 않습니다.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--review-quantile", type=float, default=0.95)
    parser.add_argument("--ood-quantile", type=float, default=0.99)
    parser.add_argument("--shrinkage", type=float, default=0.10)
    parser.add_argument("--stress-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
