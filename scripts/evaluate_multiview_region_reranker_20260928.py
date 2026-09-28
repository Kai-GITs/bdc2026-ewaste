"""Target-separated multiview reranking over the complete r2 discovery bank.

The r2 partition stays the unsupervised discovery layer.  A small pairwise
scorer is fitted only from pre-existing r3 high-confidence supports after all
42 audit parents have been removed.  The scorer therefore never receives an
audit target.  Results remain an adaptive transductive collection diagnostic:
the r2 graph itself was built on the full collection before this evaluation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 260928
FEATURE_NAMES = [
    "dino_foreground_cosine",
    "dino_box_cosine",
    "siglip2_crop_cosine",
    "minimum_view_cosine",
    "mean_view_cosine",
    "view_cosine_std",
    "foreground_box_gap",
    "foreground_siglip_gap",
    "box_siglip_gap",
    "absolute_log_area_ratio",
    "centroid_distance",
    "same_resolution_status",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def parent_fold(parent_id: str) -> int:
    return int(hashlib.sha256(parent_id.encode("utf-8")).hexdigest()[:8], 16) % 5


def pair_features(
    left: np.ndarray,
    right: np.ndarray,
    foreground: np.ndarray,
    box: np.ndarray,
    siglip: np.ndarray,
    area: np.ndarray,
    centroid: np.ndarray,
    resolution: np.ndarray,
) -> np.ndarray:
    fg = np.einsum("ij,ij->i", foreground[left], foreground[right])
    bx = np.einsum("ij,ij->i", box[left], box[right])
    sg = np.einsum("ij,ij->i", siglip[left], siglip[right])
    views = np.stack([fg, bx, sg], axis=1)
    return np.column_stack(
        [
            fg,
            bx,
            sg,
            views.min(axis=1),
            views.mean(axis=1),
            views.std(axis=1),
            np.abs(fg - bx),
            np.abs(fg - sg),
            np.abs(bx - sg),
            np.abs(np.log(np.maximum(area[left], 1e-8) / np.maximum(area[right], 1e-8))),
            np.linalg.norm(centroid[left] - centroid[right], axis=1),
            (resolution[left] == resolution[right]).astype(np.float32),
        ]
    ).astype(np.float32)


def one_to_many_features(
    query: int,
    references: np.ndarray,
    foreground: np.ndarray,
    box: np.ndarray,
    siglip: np.ndarray,
    area: np.ndarray,
    centroid: np.ndarray,
    resolution: np.ndarray,
) -> np.ndarray:
    return pair_features(
        np.full(len(references), query, dtype=np.int64),
        references,
        foreground,
        box,
        siglip,
        area,
        centroid,
        resolution,
    )


def sampled_pairs(
    indices: np.ndarray,
    labels: np.ndarray,
    parents: np.ndarray,
    foreground: np.ndarray,
    siglip: np.ndarray,
    rng: np.random.Generator,
    max_positive_per_family: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positives: set[tuple[int, int]] = set()
    for label in sorted(set(labels[indices])):
        members = indices[labels[indices] == label]
        candidates = [(int(a), int(b)) for a, b in combinations(members.tolist(), 2)
                      if parents[a] != parents[b]]
        if len(candidates) > max_positive_per_family:
            chosen = rng.choice(len(candidates), size=max_positive_per_family, replace=False)
            candidates = [candidates[int(i)] for i in chosen]
        positives.update((min(a, b), max(a, b)) for a, b in candidates)

    negatives: set[tuple[int, int]] = set()
    local_fg = foreground[indices]
    local_sg = siglip[indices]
    for matrix in (local_fg @ local_fg.T, local_sg @ local_sg.T):
        np.fill_diagonal(matrix, -np.inf)
        for local_anchor, anchor in enumerate(indices):
            valid = (labels[indices] != labels[anchor]) & (parents[indices] != parents[anchor])
            order = np.argsort(np.where(valid, matrix[local_anchor], -np.inf))[-2:]
            for local_other in order:
                other = int(indices[int(local_other)])
                if np.isfinite(matrix[local_anchor, local_other]):
                    negatives.add((min(int(anchor), other), max(int(anchor), other)))
    # One random different-family negative per anchor preserves broad coverage.
    for anchor in indices:
        pool = indices[(labels[indices] != labels[anchor]) & (parents[indices] != parents[anchor])]
        if len(pool):
            other = int(rng.choice(pool))
            negatives.add((min(int(anchor), other), max(int(anchor), other)))

    positive_list = sorted(positives)
    negative_list = sorted(negatives)
    if len(negative_list) > 2 * len(positive_list):
        chosen = rng.choice(len(negative_list), size=2 * len(positive_list), replace=False)
        negative_list = [negative_list[int(i)] for i in sorted(chosen)]
    pairs = positive_list + negative_list
    left = np.array([a for a, _ in pairs], dtype=np.int64)
    right = np.array([b for _, b in pairs], dtype=np.int64)
    target = np.r_[np.ones(len(positive_list), dtype=np.int8),
                   np.zeros(len(negative_list), dtype=np.int8)]
    return left, right, target


def choose_threshold(confidence: np.ndarray, correct: np.ndarray, minimum_precision: float = 0.80) -> dict:
    best = None
    for threshold in sorted(set(float(v) for v in confidence)):
        emitted = confidence >= threshold
        count = int(emitted.sum())
        if not count:
            continue
        precision = float(correct[emitted].mean())
        if precision + 1e-12 < minimum_precision:
            continue
        candidate = (count, -threshold, precision)
        if best is None or candidate > best[0]:
            best = (candidate, threshold, count, precision)
    if best is None:
        return {"threshold": 1.0, "emitted": 0, "precision": None,
                "minimum_precision_target": minimum_precision}
    return {"threshold": float(best[1]), "emitted": int(best[2]),
            "precision": float(best[3]), "minimum_precision_target": minimum_precision}


def ranking_metrics(rows: list[dict], visible_total: int) -> dict:
    claimed = np.array([row["claim_emitted"] for row in rows], dtype=bool)
    correct = np.array([row["correct_visible_family"] for row in rows], dtype=bool)
    top5 = np.array([row["visible_family_in_top5"] for row in rows], dtype=bool)
    return {
        "claims": int(claimed.sum()),
        "correct_visible_family_claims": int(correct.sum()),
        "visible_family_precision": float(correct[claimed].mean()) if claimed.any() else None,
        "visible_family_completeness": float(correct.sum() / max(visible_total, 1)),
        "visible_family_top5_recall": float(top5.sum() / max(visible_total, 1)),
        "operator_actions_observed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    project = Path(__file__).resolve().parents[1]
    local = project / ".local/final_study_20260928"
    protocol_path = args.output / "protocol.json"
    audit_path = project / "experiments/final_study_20260928/mask_correspondence_audit_r1/adjudicated_manifest.csv"
    assignments_path = local / "region-family-graph-object-filtered-r2/assignments.csv.gz"
    supports_path = project / "experiments/final_study_20260928/region_family_consensus_r3/photo_concept_support.csv.gz"
    adjudication_path = project / "experiments/final_study_20260928/region_family_consensus_r3/family_adjudication_r3.csv"

    audit = pd.read_csv(audit_path)
    assignments = pd.read_csv(assignments_path).sort_values("graph_row").reset_index(drop=True)
    if not np.array_equal(assignments.graph_row.to_numpy(), np.arange(len(assignments))):
        raise ValueError("r2 graph rows must be contiguous")
    region_to_graph = assignments.set_index("region_id").graph_row.to_dict()
    supports = pd.read_csv(supports_path)
    core = supports[supports.configuration_core_support.eq(1)].drop_duplicates(
        ["parent_id", "descriptive_family_key", "support_region_id"]
    )
    adjudication = pd.read_csv(adjudication_path)
    family_to_key = adjudication.set_index("family_internal").descriptive_family_key.to_dict()

    foreground_all = np.load(local / "composition-region-bank-r1/dinov3_foreground.npy", mmap_mode="r")
    box_all = np.load(local / "dinov3-region-box-r1/dinov3_box.npy", mmap_mode="r")
    siglip_all = np.load(local / "siglip2-region-crop-r1/siglip2_crop.npy", mmap_mode="r")
    feature_rows = assignments.feature_row.to_numpy(np.int64)
    foreground = normalize(foreground_all[feature_rows])
    box = normalize(box_all[feature_rows])
    siglip = normalize(siglip_all[feature_rows])
    area = assignments.area_fraction.to_numpy(np.float32)
    centroid = assignments[["centroid_x", "centroid_y"]].to_numpy(np.float32)
    resolution = assignments.resolution_status.astype(str).to_numpy()
    parents = assignments.parent_id.astype(str).to_numpy()
    source_sha = assignments.source_sha256.astype(str).to_numpy()
    family = assignments.main_family_internal.to_numpy(np.int32)

    audit_parents = set(audit.parent_id.astype(str))
    core = core[core.support_region_id.isin(region_to_graph)].copy()
    core["graph_row"] = core.support_region_id.map(region_to_graph).astype(int)
    core = core[~core.parent_id.astype(str).isin(audit_parents)].copy()
    r3_indices = core.graph_row.to_numpy(np.int64)
    r3_label_by_region = core.set_index("support_region_id").descriptive_family_key.to_dict()
    semantic_labels = np.array([r3_label_by_region.get(region, "") for region in assignments.region_id], dtype=object)
    if any(not semantic_labels[index] for index in r3_indices):
        raise ValueError("Missing r3 fit labels")
    train = r3_indices[np.array([parent_fold(parents[index]) != 0 for index in r3_indices])]
    validation = r3_indices[np.array([parent_fold(parents[index]) == 0 for index in r3_indices])]
    if len(train) < 300 or len(validation) < 50:
        raise ValueError("Deterministic parent split is unexpectedly small")

    rng = np.random.default_rng(SEED)
    train_left, train_right, train_target = sampled_pairs(
        train, semantic_labels, parents, foreground, siglip, rng
    )
    val_left, val_right, val_target = sampled_pairs(
        validation, semantic_labels, parents, foreground, siglip, rng
    )
    train_features = pair_features(
        train_left, train_right, foreground, box, siglip, area, centroid, resolution
    )
    val_features = pair_features(
        val_left, val_right, foreground, box, siglip, area, centroid, resolution
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                              random_state=SEED)),
        ]
    )
    model.fit(train_features, train_target)
    val_pair_probability = model.predict_proba(val_features)[:, 1]

    # Parent-held-out semantic retrieval calibrates the claim threshold.  The
    # r3 label is used as a validation target only; it is never an input feature.
    validation_rows = []
    train_labels = semantic_labels[train]
    for query in validation:
        refs = train[(parents[train] != parents[query]) & (source_sha[train] != source_sha[query])]
        if semantic_labels[query] not in set(semantic_labels[refs]):
            continue
        probability = model.predict_proba(
            one_to_many_features(query, refs, foreground, box, siglip, area, centroid, resolution)
        )[:, 1]
        order = np.argsort(probability)[::-1]
        best = int(refs[order[0]])
        validation_rows.append({
            "query_region_id": assignments.iloc[query].region_id,
            "expected_family": semantic_labels[query],
            "predicted_family": semantic_labels[best],
            "confidence": float(probability[order[0]]),
            "correct": bool(semantic_labels[best] == semantic_labels[query]),
        })
    validation_frame = pd.DataFrame(validation_rows)
    threshold = choose_threshold(
        validation_frame.confidence.to_numpy(float), validation_frame.correct.to_numpy(bool)
    )

    audit = audit.copy()
    audit["query_graph_row"] = audit.region_id.map(region_to_graph)
    if audit.query_graph_row.isna().any():
        raise ValueError("Audit region absent from active r2 bank")
    audit["query_graph_row"] = audit.query_graph_row.astype(int)
    audit["expected_family_internal"] = audit.family_internal.astype(int)
    audit["gold_visible_family"] = (
        audit.semantic_correspondence.eq("yes")
        & audit.failure_mode.isin(["none", "proposal", "source_redundancy"])
    )
    visible_total = int(audit.gold_visible_family.sum())

    arms = [
        "dino_foreground_nn_all_r2",
        "dino_box_nn_all_r2",
        "siglip2_crop_nn_all_r2",
        "equal_multiview_nn_all_r2",
        "fitted_multiview_nn_all_r2",
        "fitted_multiview_thresholded_all_r2",
    ]
    predictions: list[dict] = []
    foreground_top5: list[dict] = []
    family_values = sorted(set(int(v) for v in family))
    for audit_row in audit.itertuples():
        query = int(audit_row.query_graph_row)
        references = np.flatnonzero(
            (parents != parents[query]) & (source_sha != source_sha[query])
        )
        feature_matrix = one_to_many_features(
            query, references, foreground, box, siglip, area, centroid, resolution
        )
        fitted = model.predict_proba(feature_matrix)[:, 1]
        score_by_arm = {
            "dino_foreground_nn_all_r2": feature_matrix[:, 0],
            "dino_box_nn_all_r2": feature_matrix[:, 1],
            "siglip2_crop_nn_all_r2": feature_matrix[:, 2],
            "equal_multiview_nn_all_r2": feature_matrix[:, 4],
            "fitted_multiview_nn_all_r2": fitted,
            "fitted_multiview_thresholded_all_r2": fitted,
        }
        for arm in arms:
            scores = score_by_arm[arm]
            order = np.argsort(scores)[::-1]
            candidate = int(references[order[0]])
            # Family ranks use maximum member score.  This permits discovery
            # families with only two parents and does not reward family size.
            family_scores = {
                value: float(scores[family[references] == value].max())
                for value in family_values if np.any(family[references] == value)
            }
            ranked_families = sorted(family_scores, key=lambda value: family_scores[value], reverse=True)
            expected = int(audit_row.expected_family_internal)
            expected_rank = ranked_families.index(expected) + 1 if expected in ranked_families else None
            emitted = True
            if arm == "fitted_multiview_thresholded_all_r2":
                emitted = bool(float(scores[order[0]]) >= threshold["threshold"])
            correct_visible = bool(
                emitted and audit_row.gold_visible_family and int(family[candidate]) == expected
            )
            key = family_to_key.get(int(family[candidate]))
            if not isinstance(key, str) or not key.strip():
                key = f"r2_family_{int(family[candidate]):04d}"
            predictions.append({
                "arm": arm,
                "query_parent_id": parents[query],
                "query_region_id": assignments.iloc[query].region_id,
                "query_family_internal": expected,
                "query_visual_name_id": audit_row.visual_name_id,
                "gold_visible_family": bool(audit_row.gold_visible_family),
                "query_semantic_correspondence": audit_row.semantic_correspondence,
                "query_failure_mode": audit_row.failure_mode,
                "candidate_parent_id": parents[candidate],
                "candidate_region_id": assignments.iloc[candidate].region_id,
                "candidate_family_internal": int(family[candidate]),
                "candidate_family_key": key,
                "candidate_score": float(scores[order[0]]),
                "dino_foreground_cosine": float(feature_matrix[order[0], 0]),
                "dino_box_cosine": float(feature_matrix[order[0], 1]),
                "siglip2_crop_cosine": float(feature_matrix[order[0], 2]),
                "claim_emitted": emitted,
                "correct_visible_family": correct_visible,
                "expected_family_rank": expected_rank,
                "visible_family_in_top5": bool(audit_row.gold_visible_family and expected_rank is not None and expected_rank <= 5),
            })
            if arm == "dino_foreground_nn_all_r2":
                used_families: set[int] = set()
                family_rank = 0
                for candidate_order in order:
                    top_candidate = int(references[candidate_order])
                    candidate_family = int(family[top_candidate])
                    if candidate_family in used_families:
                        continue
                    used_families.add(candidate_family)
                    family_rank += 1
                    foreground_top5.append({
                        "query_parent_id": parents[query],
                        "query_region_id": assignments.iloc[query].region_id,
                        "query_family_internal": expected,
                        "query_visual_name_id": audit_row.visual_name_id,
                        "gold_visible_family": bool(audit_row.gold_visible_family),
                        "family_rank": family_rank,
                        "candidate_parent_id": parents[top_candidate],
                        "candidate_region_id": assignments.iloc[top_candidate].region_id,
                        "candidate_family_internal": candidate_family,
                        "dino_foreground_cosine": float(scores[candidate_order]),
                        "is_expected_family": bool(candidate_family == expected),
                    })
                    if family_rank == 5:
                        break

    prediction_frame = pd.DataFrame(predictions)
    prediction_frame.to_csv(args.output / "audit_predictions.csv", index=False)
    pd.DataFrame(foreground_top5).to_csv(args.output / "dino_foreground_top5_families.csv", index=False)
    validation_frame.to_csv(args.output / "threshold_validation_queries.csv", index=False)
    metrics = {arm: ranking_metrics(
        prediction_frame[prediction_frame.arm.eq(arm)].to_dict("records"), visible_total
    ) for arm in arms}

    scaler = model.named_steps["scale"]
    logistic = model.named_steps["logistic"]
    model_record = {
        "feature_names": FEATURE_NAMES,
        "standard_scaler_mean": scaler.mean_.tolist(),
        "standard_scaler_scale": scaler.scale_.tolist(),
        "logistic_coefficients": logistic.coef_[0].tolist(),
        "logistic_intercept": float(logistic.intercept_[0]),
        "claim_threshold": threshold,
        "seed": SEED,
    }
    (args.output / "model_parameters.json").write_text(
        json.dumps(model_record, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "status": "complete_target_separated_multiview_reranker_diagnostic",
        "protocol_sha256": digest(protocol_path),
        "scope": "adaptive_transductive_collection_diagnostic",
        "discovery_layer": "complete r2 partition with automatic family IDs retained",
        "claim_layer": "r3 supports train a post-discovery pair scorer; audit targets remain evaluation-only",
        "fit": {
            "r3_rows_after_audit_parent_exclusion": int(len(r3_indices)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "train_pairs": int(len(train_target)),
            "validation_pairs": int(len(val_target)),
            "validation_pair_roc_auc": float(roc_auc_score(val_target, val_pair_probability)),
            "validation_pair_average_precision": float(average_precision_score(val_target, val_pair_probability)),
            "validation_retrieval_queries": int(len(validation_frame)),
            "validation_retrieval_top1": float(validation_frame.correct.mean()),
            "threshold": threshold,
        },
        "audit": {
            "queries": int(len(audit)),
            "visible_family_targets": visible_total,
            "strict_claimable_targets": int(audit.claim_interpretation.eq("supported").sum()),
            "all_42_parents_excluded_from_fit": True,
            "query_parent_and_exact_image_excluded_from_reference": True,
            "target_used_by_predictor": False,
            "arms": metrics,
        },
        "interpretation_limit": (
            "A recovered r2 family ID is not yet a semantic finding. Retrieved query-candidate pairs "
            "must be inspected on original pixels before the reranker supports a claim or flagship relation."
        ),
        "inputs": {
            str(path.relative_to(project)): digest(path) for path in [
                audit_path,
                assignments_path,
                supports_path,
                adjudication_path,
                local / "composition-region-bank-r1/dinov3_foreground.npy",
                local / "dinov3-region-box-r1/dinov3_box.npy",
                local / "siglip2-region-crop-r1/siglip2_crop.npy",
            ]
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "validation_pair_auc": summary["fit"]["validation_pair_roc_auc"],
        "visible_targets": visible_total,
        "arms": {name: {
            "correct": value["correct_visible_family_claims"],
            "claims": value["claims"],
            "top5": value["visible_family_top5_recall"],
        } for name, value in metrics.items()},
    }))


if __name__ == "__main__":
    main()
