"""Build train-only reference artifacts for advanced SECOM diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer, StandardScaler

from advanced_diagnosis import sha256_file
from diagnose_secom import shap_matrix
from train_compare_models import RANDOM_STATE, TEST_SIZE, load_data, structural_filter


def upper_threshold(values: np.ndarray, quantile: float, minimum: float = 0.0) -> float:
    value = max(float(np.quantile(values, quantile)), minimum)
    return float(value)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    model_dir = project_dir / "결과물" / "secom" / "dual_model_results"
    output_dir = project_dir / "결과물" / "secom" / "advanced_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("[1/6] Loading the fixed train split and saved finalists...")
    X_raw, y, _ = load_data(project_dir)
    X, _, _ = structural_filter(X_raw)
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    raw_train = X_raw.loc[X_train.index]
    bundles = {
        model_name: joblib.load(model_dir / f"{model_name.lower()}_model.joblib")
        for model_name in ("CatBoost", "XGBoost")
    }
    catboost_bundle = bundles["CatBoost"]
    input_features = list(catboost_bundle["input_features"])
    if input_features != list(bundles["XGBoost"]["input_features"]):
        raise ValueError("두 모델의 구조 필터 입력 변수가 다릅니다.")

    print("[2/6] Fitting train-only OOD reference distributions...")
    train_values = X_train[input_features].to_numpy(dtype=float)
    medians = np.asarray(catboost_bundle["imputer"].statistics_, dtype=float)
    filled = train_values.copy()
    rows, columns = np.where(np.isnan(filled))
    filled[rows, columns] = medians[columns]
    q01 = np.nanquantile(train_values, 0.01, axis=0)
    q05 = np.nanquantile(train_values, 0.05, axis=0)
    q25 = np.nanquantile(train_values, 0.25, axis=0)
    q75 = np.nanquantile(train_values, 0.75, axis=0)
    q95 = np.nanquantile(train_values, 0.95, axis=0)
    q99 = np.nanquantile(train_values, 0.99, axis=0)
    iqr = np.maximum(q75 - q25, 1e-9)

    ood_scaler = StandardScaler().fit(filled)
    scaled = ood_scaler.transform(filled)
    ood_pca = PCA(n_components=20, random_state=RANDOM_STATE).fit(scaled)
    embedding = ood_pca.transform(scaled)
    neighbour_distances = NearestNeighbors(n_neighbors=2).fit(embedding).kneighbors(
        embedding
    )[0][:, 1]
    missing_rates = raw_train.isna().mean(axis=1).to_numpy(dtype=float)
    observed = np.isfinite(train_values)
    outside = observed & ((train_values < q01) | (train_values > q99))
    range_rates = outside.sum(axis=1) / np.maximum(observed.sum(axis=1), 1)

    distance_review = upper_threshold(neighbour_distances, 0.95)
    distance_outlier = max(
        upper_threshold(neighbour_distances, 0.99), distance_review * 1.05
    )
    missing_review = upper_threshold(missing_rates, 0.95, 0.01)
    missing_outlier = max(
        upper_threshold(missing_rates, 0.99, 0.02), missing_review + 1 / 590
    )
    range_review = upper_threshold(range_rates, 0.95, 0.02)
    range_outlier = max(
        upper_threshold(range_rates, 0.99, 0.04), range_review + 1 / len(input_features)
    )
    ood_thresholds = {
        "distance_review": distance_review,
        "distance_outlier": distance_outlier,
        "missing_review": missing_review,
        "missing_outlier": missing_outlier,
        "range_review": range_review,
        "range_outlier": range_outlier,
    }

    print("[3/6] Computing CatBoost SHAP vectors for real train defects...")
    cat_imputed = catboost_bundle["imputer"].transform(X_train[input_features])
    cat_ready = catboost_bundle["selector"].transform(cat_imputed)
    defect_mask = y_train.to_numpy(dtype=int) == 1
    defect_ready = cat_ready[defect_mask]
    defect_row_ids = X_train.index.to_numpy()[defect_mask]
    defect_shap = shap_matrix(catboost_bundle["model"], defect_ready)

    global_importance = np.abs(defect_shap).mean(axis=0)
    signature_feature_indices = np.argsort(global_importance)[::-1][:20]
    signature_scaler = Normalizer(norm="l2").fit(
        defect_shap[:, signature_feature_indices]
    )
    signature_scaled = signature_scaler.transform(
        defect_shap[:, signature_feature_indices]
    )
    signature_components = min(5, len(defect_shap) - 1, signature_scaled.shape[1])
    signature_pca = PCA(
        n_components=signature_components, random_state=RANDOM_STATE
    ).fit(signature_scaled)
    signature_embedding = signature_pca.transform(signature_scaled)

    print("[4/6] Selecting a compact, reproducible failure-pattern clustering...")
    candidates = {}
    best = None
    for clusters in (2, 3, 4):
        model = KMeans(n_clusters=clusters, random_state=RANDOM_STATE, n_init=30)
        labels = model.fit_predict(signature_embedding)
        counts = np.bincount(labels, minlength=clusters)
        score = float(silhouette_score(signature_embedding, labels))
        candidates[clusters] = {
            "silhouette": score,
            "minimum_cluster_size": int(counts.min()),
        }
        if counts.min() >= 5 and (best is None or score > best[0]):
            best = (score, model, labels)
    if best is None:
        raise RuntimeError("최소 5개 실제 불량을 갖는 군집 구성을 찾지 못했습니다.")
    silhouette, signature_kmeans, signature_labels = best
    stability_scores = []
    rng = np.random.default_rng(RANDOM_STATE)
    subsample_size = max(int(round(len(signature_embedding) * 0.8)), 10)
    for repeat in range(50):
        selected = rng.choice(
            len(signature_embedding), size=subsample_size, replace=False
        )
        repeated = KMeans(
            n_clusters=signature_kmeans.n_clusters,
            random_state=RANDOM_STATE + repeat + 1,
            n_init=20,
        ).fit(signature_embedding[selected])
        stability_scores.append(
            adjusted_rand_score(
                signature_labels,
                repeated.predict(signature_embedding),
            )
        )
    signature_stability = float(np.mean(stability_scores))

    selected_features = list(catboost_bundle["selected_features"])
    signature_details = {}
    signature_rows = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for cluster in range(signature_kmeans.n_clusters):
        mask = signature_labels == cluster
        mean_shap = defect_shap[mask].mean(axis=0)
        positive = np.argsort(mean_shap)[::-1]
        negative = np.argsort(mean_shap)
        risk_features = [
            selected_features[index]
            for index in positive
            if mean_shap[index] > 0
        ][:5]
        protective_features = [
            selected_features[index]
            for index in negative
            if mean_shap[index] < 0
        ][:5]
        member_positions = np.flatnonzero(mask)
        centroid = signature_kmeans.cluster_centers_[cluster]
        representative_position = member_positions[
            np.argmin(
                np.linalg.norm(signature_embedding[member_positions] - centroid, axis=1)
            )
        ]
        detail = {
            "pattern": f"패턴 {letters[cluster]}",
            "training_defect_rows": int(mask.sum()),
            "top_risk_features": risk_features,
            "top_protective_features": protective_features,
            "representative_row_id": int(defect_row_ids[representative_position]),
        }
        signature_details[cluster] = detail
        signature_rows.append(
            {
                "cluster": cluster,
                **detail,
                "share": float(mask.mean()),
                "silhouette": silhouette,
                "top_risk_features": "; ".join(risk_features),
                "top_protective_features": "; ".join(protective_features),
            }
        )

    print("[5/6] Saving reference artifact and pattern map...")
    model_hashes = {
        model_name: sha256_file(
            model_dir / f"{model_name.lower()}_model.joblib"
        )
        for model_name in ("CatBoost", "XGBoost")
    }
    reference = {
        "version": 1,
        "random_state": RANDOM_STATE,
        "input_features": input_features,
        "raw_features": list(X_raw.columns),
        "feature_index": {feature: index for index, feature in enumerate(input_features)},
        "medians": medians,
        "q01": q01,
        "q05": q05,
        "q25": q25,
        "median": medians,
        "q75": q75,
        "q95": q95,
        "q99": q99,
        "iqr": iqr,
        "ood_scaler": ood_scaler,
        "ood_pca": ood_pca,
        "reference_embedding": embedding.astype("float32"),
        "ood_thresholds": ood_thresholds,
        "signature_model": "CatBoost",
        "signature_feature_indices": signature_feature_indices,
        "signature_scaler": signature_scaler,
        "signature_pca": signature_pca,
        "signature_kmeans": signature_kmeans,
        "signature_details": signature_details,
        "model_hashes": model_hashes,
    }
    joblib.dump(reference, output_dir / "advanced_reference.joblib", compress=3)
    pd.DataFrame(signature_rows).to_csv(
        output_dir / "failure_signatures.csv", index=False
    )

    figure, axis = plt.subplots(figsize=(8.5, 6.2))
    second = signature_embedding[:, 1] if signature_embedding.shape[1] > 1 else np.zeros(len(signature_embedding))
    scatter = axis.scatter(
        signature_embedding[:, 0],
        second,
        c=signature_labels,
        cmap="tab10",
        s=46,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.4,
    )
    axis.set_title("SECOM train defects: SHAP failure-pattern candidates")
    axis.set_xlabel("SHAP PCA component 1")
    axis.set_ylabel("SHAP PCA component 2")
    handles, _ = scatter.legend_elements()
    axis.legend(
        handles,
        [signature_details[index]["pattern"] for index in range(signature_kmeans.n_clusters)],
        title="Pattern candidate",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "failure_signature_map.png", dpi=190)
    plt.close(figure)

    metadata = {
        "version": 1,
        "random_state": RANDOM_STATE,
        "fit_scope": "fixed_train_only",
        "train_rows": int(len(X_train)),
        "train_defect_rows": int(defect_mask.sum()),
        "retained_features": len(input_features),
        "ood_pca_components": int(ood_pca.n_components_),
        "ood_pca_explained_variance": float(ood_pca.explained_variance_ratio_.sum()),
        "ood_thresholds": ood_thresholds,
        "signature_clusters": int(signature_kmeans.n_clusters),
        "signature_silhouette": silhouette,
        "signature_bootstrap_stability_ari": signature_stability,
        "signature_candidates": candidates,
        "model_hashes": model_hashes,
        "physical_causality_claimed": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = [
        "# SECOM 고급 진단 안전장치",
        "",
        f"- 참조 범위: 고정 train {len(X_train):,}행만 사용",
        f"- OOD 공간: 표준화 후 PCA {ood_pca.n_components_}차원",
        f"- 실제 train 불량 패턴: {int(defect_mask.sum())}행 → {signature_kmeans.n_clusters}개 후보 군집",
        f"- 군집 silhouette: {silhouette:.4f}",
        f"- 80% subsample bootstrap 안정성 ARI: {signature_stability:.4f}",
        "- 군집 표현: 평균 |SHAP| 상위 20개 → L2 정규화 → PCA 5차원",
        "- 반사실적 후보: train 5/25/50/75/95% 분위수 안에서만 탐색",
        "- 자동판정 보류: 결측률·범위 이탈·최근접 거리의 train 기준 초과 시 적용",
        "",
        "패턴은 SHAP 유사성에 따른 탐색적 후보이며 실제 고장 유형이나 물리적 원인을 의미하지 않습니다.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("[6/6] Advanced diagnostic artifacts complete:", output_dir)


if __name__ == "__main__":
    main()
