"""Evaluate whether unsupervised region families recover physical exposure state.

The 129 region families are frozen discovery features.  A 136-photo board review
is used only for development; the later blind review supplies unseen relation
labels.  No image encoder, region proposal, or graph is recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260928
C_GRID = (0.01, 0.1, 1.0, 10.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def family_features(assignments: pd.DataFrame, ids: list[str]) -> tuple[np.ndarray, list[str]]:
    index = {value: pos for pos, value in enumerate(ids)}
    counts = np.zeros((len(ids), 129), dtype=np.float32)
    strength = np.zeros_like(counts)
    isolates = np.zeros((len(ids), 1), dtype=np.float32)
    for row in assignments.itertuples():
        parent = str(row.parent_id)
        if parent not in index:
            continue
        pos = index[parent]
        family = int(row.main_family_internal)
        if family < 0:
            isolates[pos, 0] += 1.0
        else:
            counts[pos, family] += 1.0
            strength[pos, family] = max(strength[pos, family], float(row.weighted_degree))
    presence = (counts > 0).astype(np.float32)
    normalized_counts = np.log1p(counts)
    totals = normalized_counts.sum(axis=1, keepdims=True)
    normalized_counts = normalized_counts / np.maximum(totals, 1.0)
    isolate_flag = (isolates > 0).astype(np.float32)
    features = np.concatenate([presence, normalized_counts, strength, isolate_flag], axis=1)
    names = (
        [f"F{family:04d}_presence" for family in range(129)]
        + [f"F{family:04d}_relative_count" for family in range(129)]
        + [f"F{family:04d}_max_graph_strength" for family in range(129)]
        + ["unresolved_isolate_present"]
    )
    return features, names


def make_pipeline(kind: str, dino_dim: int, siglip_dim: int, region_dim: int, c: float) -> Pipeline:
    logistic = LogisticRegression(
        C=c,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=SEED,
    )
    if kind == "global_dinov3":
        transform = ColumnTransformer([
            ("dino", Pipeline([("pca", PCA(n_components=32, random_state=SEED))]), slice(0, dino_dim)),
        ], remainder="drop")
    elif kind == "global_siglip2":
        transform = ColumnTransformer([
            ("siglip", Pipeline([("pca", PCA(n_components=32, random_state=SEED))]),
             slice(dino_dim, dino_dim + siglip_dim)),
        ], remainder="drop")
    elif kind == "global_fused":
        transform = ColumnTransformer([
            ("global", Pipeline([("pca", PCA(n_components=48, random_state=SEED))]),
             slice(0, dino_dim + siglip_dim)),
        ], remainder="drop")
    elif kind == "region_families":
        transform = ColumnTransformer([
            ("region", Pipeline([
                ("variance", VarianceThreshold()),
                ("scale", StandardScaler()),
            ]), slice(dino_dim + siglip_dim, dino_dim + siglip_dim + region_dim)),
        ], remainder="drop")
    elif kind == "region_plus_global":
        transform = ColumnTransformer([
            ("global", Pipeline([("pca", PCA(n_components=48, random_state=SEED))]),
             slice(0, dino_dim + siglip_dim)),
            ("region", Pipeline([
                ("variance", VarianceThreshold()),
                ("scale", StandardScaler()),
            ]), slice(dino_dim + siglip_dim, dino_dim + siglip_dim + region_dim)),
        ], remainder="drop")
    else:
        raise ValueError(kind)
    return Pipeline([("transform", transform), ("logistic", logistic)])


def choose_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    measured = [balanced_accuracy_score(y, scores >= value) for value in candidates]
    best = max(measured)
    tied = [float(value) for value, metric in zip(candidates, measured) if metric == best]
    return min(tied, key=lambda value: abs(value - 0.5))


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float, budget: int = 10) -> dict[str, float]:
    order = np.argsort(-scores)
    k = min(budget, len(y))
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y, scores >= threshold)),
        "brier": float(brier_score_loss(y, scores)),
        "threshold_from_development": float(threshold),
        "mounted_found_at_10": int(y[order[:k]].sum()),
        "precision_at_10": float(y[order[:k]].mean()),
    }


def bootstrap_metrics(y: np.ndarray, scores: np.ndarray, threshold: float, repeats: int = 10000) -> dict:
    rng = np.random.default_rng(SEED)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rows = []
    for _ in range(repeats):
        picked = np.r_[rng.choice(positive, len(positive), replace=True),
                       rng.choice(negative, len(negative), replace=True)]
        row = metrics(y[picked], scores[picked], threshold)
        rows.append((row["roc_auc"], row["average_precision"], row["balanced_accuracy"]))
    values = np.asarray(rows)
    return {
        name: {"low_95": float(np.quantile(values[:, index], 0.025)),
               "high_95": float(np.quantile(values[:, index], 0.975))}
        for index, name in enumerate(("roc_auc", "average_precision", "balanced_accuracy"))
    }


def paired_bootstrap_auc(y: np.ndarray, proposed: np.ndarray, baseline: np.ndarray,
                         repeats: int = 10000) -> dict:
    rng = np.random.default_rng(SEED + 7)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    deltas = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        picked = np.r_[rng.choice(positive, len(positive), replace=True),
                       rng.choice(negative, len(negative), replace=True)]
        deltas[index] = roc_auc_score(y[picked], proposed[picked]) - roc_auc_score(y[picked], baseline[picked])
    return {
        "mean_delta": float(deltas.mean()),
        "low_95": float(np.quantile(deltas, 0.025)),
        "high_95": float(np.quantile(deltas, 0.975)),
        "fraction_delta_above_zero": float((deltas > 0).mean()),
    }


@dataclass
class ModelResult:
    name: str
    c: float
    threshold: float
    oof_auc: float
    scores: np.ndarray
    pipeline: Pipeline


def fit_model(name: str, x_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray,
              x_test: np.ndarray, dims: tuple[int, int, int]) -> ModelResult:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(x_train, y_train, groups))
    best_c, best_auc = None, -np.inf
    for c in C_GRID:
        oof = np.full(len(y_train), np.nan, dtype=np.float64)
        for fit, valid in splits:
            pipe = make_pipeline(name, *dims, c)
            pipe.fit(x_train[fit], y_train[fit])
            oof[valid] = pipe.predict_proba(x_train[valid])[:, 1]
        auc = roc_auc_score(y_train, oof)
        if auc > best_auc + 1e-12 or (abs(auc - best_auc) <= 1e-12 and (best_c is None or c < best_c)):
            best_c, best_auc = c, auc
    final_oof = np.full(len(y_train), np.nan, dtype=np.float64)
    for fit, valid in splits:
        pipe = make_pipeline(name, *dims, best_c)
        pipe.fit(x_train[fit], y_train[fit])
        final_oof[valid] = pipe.predict_proba(x_train[valid])[:, 1]
    threshold = choose_threshold(y_train, final_oof)
    final = make_pipeline(name, *dims, best_c)
    final.fit(x_train, y_train)
    return ModelResult(name, float(best_c), threshold, float(best_auc),
                       final.predict_proba(x_test)[:, 1], final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-labels", type=Path, required=True)
    parser.add_argument("--blind-evaluation", type=Path, required=True)
    parser.add_argument("--region-assignments", type=Path, required=True)
    parser.add_argument("--dino-features", type=Path, required=True)
    parser.add_argument("--dino-index", type=Path, required=True)
    parser.add_argument("--siglip-features", type=Path, required=True)
    parser.add_argument("--siglip-index", type=Path, required=True)
    parser.add_argument("--near-duplicates", type=Path, required=True)
    parser.add_argument("--family-census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    development = pd.read_csv(args.development_labels)
    development = development[development.relation_status.isin([
        "mounted_populated_circuitry_identifiable_unit", "detached_part_or_board", "multi_item_scene"
    ])].copy()
    development["label"] = (development.relation_status == "mounted_populated_circuitry_identifiable_unit").astype(int)
    development = development.rename(columns={"image_id": "canonical_id"})
    blind = pd.read_csv(args.blind_evaluation)
    conflicts = blind.groupby("canonical_id").label.nunique()
    if int((conflicts > 1).sum()) != 0:
        raise ValueError("Blind labels conflict across selection arms")
    blind = blind.drop_duplicates("canonical_id").copy()
    blind = blind[blind.label.isin(["M", "X"])].copy()
    blind["target"] = (blind.label == "M").astype(int)

    train_ids = development.canonical_id.astype(str).tolist()
    test_ids = blind.canonical_id.astype(str).tolist()
    if set(train_ids) & set(test_ids):
        raise ValueError("Development and blind identities overlap")
    all_ids = train_ids + test_ids

    near = pd.read_csv(args.near_duplicates)
    uf = UnionFind(all_ids + near.source_image_id.astype(str).tolist() + near.target_image_id.astype(str).tolist())
    for row in near[near.phash_hamming <= 5].itertuples():
        uf.union(str(row.source_image_id), str(row.target_image_id))
    groups = np.asarray([uf.find(value) for value in train_ids])
    train_roots = {uf.find(value) for value in train_ids}
    test_linked_to_train = np.asarray([uf.find(value) in train_roots for value in test_ids])

    dino_index = pd.read_csv(args.dino_index)
    sha_column = 'sha256' if 'sha256' in dino_index else 'source_sha256'
    dino_by_sha = dict(zip(dino_index[sha_column].astype(str), dino_index.row.astype(int)))
    siglip_index = pd.read_csv(args.siglip_index)
    siglip_by_id = dict(zip(siglip_index.canonical_id.astype(str), siglip_index.row.astype(int)))
    dino_all = np.load(args.dino_features, mmap_mode="r")
    siglip_all = np.load(args.siglip_features, mmap_mode="r")
    if dino_all.shape != (3931, 1024) or siglip_all.shape != (3931, 1152):
        raise ValueError("Frozen feature shapes changed")
    sha_by_id = dict(zip(development.canonical_id.astype(str), development.source_sha256.astype(str)))
    sha_by_id.update(dict(zip(blind.canonical_id.astype(str), blind.source_sha256.astype(str))))
    dino_rows = [dino_by_sha[sha_by_id[value]] for value in all_ids]
    siglip_rows = [siglip_by_id[value] for value in all_ids]
    dino = normalize_rows(dino_all[dino_rows])
    siglip = normalize_rows(siglip_all[siglip_rows])

    assignments = pd.read_csv(args.region_assignments)
    region, region_names = family_features(assignments, all_ids)
    x = np.concatenate([dino, siglip, region], axis=1).astype(np.float32)
    n_train = len(train_ids)
    x_train, x_test = x[:n_train], x[n_train:]
    y_train = development.label.to_numpy(np.int64)
    y_test = blind.target.to_numpy(np.int64)
    dims = (dino.shape[1], siglip.shape[1], region.shape[1])

    model_names = ["global_dinov3", "global_siglip2", "global_fused", "region_families", "region_plus_global"]
    results = [fit_model(name, x_train, y_train, groups, x_test, dims) for name in model_names]
    rows = []
    predictions = blind[["canonical_id", "source_sha256", "source_relative_path", "label", "target"]].copy()
    uncertainty = {}
    for result in results:
        measured = metrics(y_test, result.scores, result.threshold)
        rows.append({"model": result.name, "development_oof_auc": result.oof_auc,
                     "selected_C": result.c, **measured})
        predictions[f"score_{result.name}"] = result.scores
        uncertainty[result.name] = bootstrap_metrics(y_test, result.scores, result.threshold)

    result_map = {result.name: result for result in results}
    comparison = {
        "region_families_minus_global_fused_auc": paired_bootstrap_auc(
            y_test, result_map["region_families"].scores, result_map["global_fused"].scores),
        "region_plus_global_minus_global_fused_auc": paired_bootstrap_auc(
            y_test, result_map["region_plus_global"].scores, result_map["global_fused"].scores),
    }

    sensitivity_rows = []
    keep = ~test_linked_to_train
    for result in results:
        if keep.sum() and y_test[keep].sum() and (~y_test[keep].astype(bool)).sum():
            sensitivity_rows.append({
                "model": result.name,
                "excluded_test_phash_linked_to_development": int(test_linked_to_train.sum()),
                **metrics(y_test[keep], result.scores[keep], result.threshold),
            })

    census = pd.read_csv(args.family_census).set_index("family_internal")
    region_model = result_map["region_families"].pipeline
    transformer = region_model.named_steps["transform"].named_transformers_["region"]
    support_mask = transformer.named_steps["variance"].get_support()
    coefficients = region_model.named_steps["logistic"].coef_[0]
    kept_names = np.asarray(region_names)[support_mask]
    coef_rows = []
    for name, value in zip(kept_names, coefficients):
        family = int(name[1:5]) if name.startswith("F") else -1
        coef_rows.append({
            "feature": name,
            "family_internal": family,
            "descriptive_name_posthoc": census.loc[family, "visual_name_id"] if family >= 0 else "region isolat",
            "coefficient_standardized": float(value),
        })
    coefficients_frame = pd.DataFrame(coef_rows).sort_values("coefficient_standardized", ascending=False)

    metrics_path = args.output / "model_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    predictions_path = args.output / "blind_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    sensitivity_path = args.output / "source_control_metrics.csv"
    pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
    coefficients_path = args.output / "region_feature_coefficients.csv"
    coefficients_frame.to_csv(coefficients_path, index=False)

    existing_stage_one = json.loads((args.blind_evaluation.parent / "evaluation.json").read_text(encoding="utf-8"))
    summary = {
        "status": "complete_unseen_label_holdout",
        "scientific_question": "Can frozen unsupervised region-family signatures recover mounted-versus-unowned board relation beyond whole-image embeddings?",
        "development": {"n": int(len(y_train)), "mounted": int(y_train.sum()), "unowned": int((1-y_train).sum())},
        "blind_test": {"n": int(len(y_test)), "mounted": int(y_test.sum()), "unowned": int((1-y_test).sum()),
                       "excluded_N_and_U_for_relation_test": int(111-len(y_test))},
        "test_phash_linked_to_development": int(test_linked_to_train.sum()),
        "models": {row["model"]: row for row in rows},
        "bootstrap_95": uncertainty,
        "paired_auc_differences": comparison,
        "stage_one_existing_equal_budget": {
            "global_dinov3_visible_board_at_40": existing_stage_one["arms"]["dinov3_seed"]["visible_board_confirmed"],
            "random_same_parent_group_visible_board_at_40": existing_stage_one["arms"]["random_same_parent_group"]["visible_board_confirmed"],
            "global_dinov3_mounted_at_40": existing_stage_one["arms"]["dinov3_seed"]["mounted_confirmed"],
            "random_same_parent_group_mounted_at_40": existing_stage_one["arms"]["random_same_parent_group"]["mounted_confirmed"],
        },
        "claim_limit": "Single-agent, transductive collection evaluation. Blind relation labels were unseen during development; test images still participated without labels in the frozen unsupervised collection structure.",
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in {
            "development_labels": args.development_labels,
            "blind_evaluation": args.blind_evaluation,
            "region_assignments": args.region_assignments,
            "dino_features": args.dino_features,
            "dino_index": args.dino_index,
            "siglip_features": args.siglip_features,
            "siglip_index": args.siglip_index,
            "near_duplicates": args.near_duplicates,
            "family_census": args.family_census,
        }.items()},
    }
    for path in (metrics_path, predictions_path, sensitivity_path, coefficients_path):
        summary.setdefault("outputs", {})[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
