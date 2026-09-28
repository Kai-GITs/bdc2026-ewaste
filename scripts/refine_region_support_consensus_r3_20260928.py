"""Refine visually adjudicated region support with an independent crop view.

The r2 graph was fitted with DINOv3 foreground weighted descriptors.  This
script does not refit that graph.  It uses the already extracted box crop
descriptor as a held out mechanism check for each proposed support.  A support
is called core in r3 only when its local graph affinity, foreground to medoid
similarity, and box to medoid similarity are all at or above their own family
median.  The medians are sensitivity cutoffs, not calibrated probabilities.

Two visual interpretation errors found by reviewing configuration pairs are
also recorded here: family 32 is screen content rather than visible damage,
and family 37 is pictorial content rather than a printer media path.  Earlier
r2 outputs remain untouched as an audit trail.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


CORRECTIONS = {
    32: (
        "screen_content_shortcut",
        "Garis antarmuka, foto, dan area tampilan aktif bercampur; audit pasangan "
        "menunjukkan contoh tanpa kerusakan fisik yang terlihat.",
    ),
    37: (
        "pictorial_content_shortcut",
        "Crop gambar pada kertas atau layar menyatukan printer, televisi, ponsel, "
        "dan pemutar; bukan jalur media pencetak yang konsisten.",
    ),
}
SEED = 20260928


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def percentile_against(sorted_values: np.ndarray, value: float) -> float:
    if not len(sorted_values):
        return float("nan")
    return float(np.searchsorted(sorted_values, value, side="right") / len(sorted_values))


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 2000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 100):
        stop = min(start + 100, draws)
        sample = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[sample].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def json_counts(values) -> str:
    return json.dumps(dict(sorted(Counter(values).items())), separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foreground", type=Path, required=True)
    parser.add_argument("--box", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--concept-support", type=Path, required=True)
    parser.add_argument("--global-assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    foreground = np.load(args.foreground, mmap_mode="r")
    box = np.load(args.box, mmap_mode="r")
    if foreground.shape != box.shape or foreground.shape[1] != 1024:
        raise ValueError(f"Feature mismatch: foreground={foreground.shape}, box={box.shape}")

    assignments = pd.read_csv(args.assignments)
    families = pd.read_csv(args.families).set_index("family_internal", drop=False)
    adjudication = pd.read_csv(args.adjudication)
    support = pd.read_csv(args.concept_support)
    global_rows = pd.read_csv(args.global_assignments)
    if len(assignments) != 6514 or len(families) != 129 or len(global_rows) != 3931:
        raise ValueError("Unexpected input cohort size")
    if int(assignments.feature_row.max()) >= foreground.shape[0]:
        raise ValueError("Assignment feature row exceeds descriptor bank")

    # Per family, measure member to medoid similarity in both representations.
    family_stats: dict[int, dict] = {}
    diagnostics = []
    for family, group in assignments[assignments.main_family_internal >= 0].groupby("main_family_internal"):
        family = int(family)
        feature_rows = group.feature_row.to_numpy(np.int64)
        medoid_row = int(families.loc[family, "medoid_feature_row"])
        fg_medoid = np.asarray(foreground[medoid_row], dtype=np.float32)
        box_medoid = np.asarray(box[medoid_row], dtype=np.float32)
        fg_sim = np.asarray(foreground[feature_rows], dtype=np.float32) @ fg_medoid
        box_sim = np.asarray(box[feature_rows], dtype=np.float32) @ box_medoid
        local = group.median_within_family_dino_similarity.to_numpy(np.float64)
        finite_local = local[np.isfinite(local)]
        record = {
            "family_internal": family,
            "members": len(group),
            "unique_parents": int(group.parent_id.nunique()),
            "medoid_feature_row": medoid_row,
            "foreground_medoid_median": float(np.median(fg_sim)),
            "box_medoid_median": float(np.median(box_sim)),
            "local_affinity_median": float(np.median(finite_local)),
            "foreground_minus_box_mean": float(np.mean(fg_sim - box_sim)),
            "foreground_box_correlation": float(np.corrcoef(fg_sim, box_sim)[0, 1]) if len(group) > 1 else float("nan"),
        }
        family_stats[family] = {
            **record,
            "fg_sorted": np.sort(fg_sim),
            "box_sorted": np.sort(box_sim),
            "local_sorted": np.sort(finite_local),
        }
        diagnostics.append(record)

    diagnostic_path = args.output / "family_cross_view_diagnostics.csv"
    pd.DataFrame(diagnostics).sort_values("family_internal").to_csv(diagnostic_path, index=False)

    # Freeze correction decisions without editing the prior r2 adjudication.
    revised_adjudication = adjudication.copy()
    revised_adjudication["r3_decision"] = revised_adjudication.decision
    revised_adjudication["r3_rejection_reason"] = revised_adjudication.rejection_reason.fillna("")
    revised_adjudication["r3_review_note"] = revised_adjudication.review_note
    for family, (reason, note) in CORRECTIONS.items():
        mask = revised_adjudication.family_internal == family
        if int(mask.sum()) != 1:
            raise ValueError(f"Missing corrected family {family}")
        revised_adjudication.loc[mask, "r3_decision"] = "rejected_configuration_pair_review"
        revised_adjudication.loc[mask, "r3_rejection_reason"] = reason
        revised_adjudication.loc[mask, "r3_review_note"] = note
        revised_adjudication.loc[mask, "include_in_configuration"] = 0
    adjudication_path = args.output / "family_adjudication_r3.csv"
    revised_adjudication.to_csv(adjudication_path, index=False)

    assignment_by_region = assignments.set_index("region_id")
    retained_rows = []
    rejected_supports = Counter()
    for row in support.to_dict("records"):
        family = int(row["source_family_internal"])
        if family in CORRECTIONS:
            rejected_supports[CORRECTIONS[family][0]] += 1
            continue
        region = row["support_region_id"]
        if region not in assignment_by_region.index:
            raise ValueError(f"Support region absent from assignments: {region}")
        assignment = assignment_by_region.loc[region]
        feature_row = int(assignment.feature_row)
        stats = family_stats[family]
        medoid_row = int(stats["medoid_feature_row"])
        fg_sim = float(np.dot(
            np.asarray(foreground[feature_row], dtype=np.float32),
            np.asarray(foreground[medoid_row], dtype=np.float32),
        ))
        box_sim = float(np.dot(
            np.asarray(box[feature_row], dtype=np.float32),
            np.asarray(box[medoid_row], dtype=np.float32),
        ))
        local = float(row["median_neighbor_affinity"])
        fg_pct = percentile_against(stats["fg_sorted"], fg_sim)
        box_pct = percentile_against(stats["box_sorted"], box_sim)
        local_pct = percentile_against(stats["local_sorted"], local)
        old_core = int(row["configuration_core_support"])
        consensus_strength = min(fg_pct, box_pct, local_pct)
        new_core = int(old_core == 1 and consensus_strength >= 0.5)
        row.update({
            "r2_affinity_core_support": old_core,
            "foreground_medoid_similarity": fg_sim,
            "foreground_family_median": stats["foreground_medoid_median"],
            "foreground_family_percentile": fg_pct,
            "box_medoid_similarity": box_sim,
            "box_family_median": stats["box_medoid_median"],
            "box_family_percentile": box_pct,
            "local_affinity_family_percentile": local_pct,
            "two_view_consensus_strength": consensus_strength,
            "configuration_core_support": new_core,
            "support_status": (
                "r3_core_local_foreground_box_at_or_above_family_medians"
                if new_core else "r3_peripheral_failed_one_or_more_median_sensitivity_checks"
            ),
        })
        retained_rows.append(row)

    support_r3 = pd.DataFrame(retained_rows)
    support_path = args.output / "photo_concept_support.csv.gz"
    support_r3.to_csv(support_path, index=False, compression="gzip")

    # Concept summaries use only consensus core supports; peripheral evidence is
    # retained in the support table for sensitivity analysis.
    global_community = dict(zip(global_rows.canonical_id, global_rows.community_leiden_fused))
    core = support_r3[support_r3.configuration_core_support == 1].copy()
    concept_records = []
    adjudication_lookup = revised_adjudication.set_index("family_internal")
    for key, group in core.groupby("descriptive_family_key"):
        parents = sorted(group.parent_id.unique())
        communities = [int(global_community[parent]) for parent in parents]
        counts = Counter(communities)
        probabilities = np.asarray(list(counts.values()), dtype=np.float64) / len(parents)
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        normalized_entropy = entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
        source_families = sorted(int(value) for value in group.source_family_internal.unique())
        exemplar = group.iloc[0]
        concept_records.append({
            "descriptive_family_key": key,
            "descriptive_name_id": exemplar.descriptive_name_id,
            "scientific_role": exemplar.scientific_role,
            "source_internal_families": json.dumps(source_families, separators=(",", ":")),
            "core_unique_parents": len(parents),
            "global_communities_supported": len(counts),
            "largest_global_community_share": max(counts.values()) / len(parents),
            "normalized_global_community_entropy": normalized_entropy,
            "global_community_counts": json.dumps(dict(sorted(counts.items())), separators=(",", ":")),
            "consensus_strength_median": float(group.two_view_consensus_strength.median()),
        })
    concept_path = args.output / "conceptual_family_summary.csv"
    pd.DataFrame(concept_records).sort_values("descriptive_family_key").to_csv(concept_path, index=False)

    # Foreground vs box diagnostic on the frozen foreground selected edge set.
    # This is intentionally labelled selection conditional rather than an
    # unbiased model benchmark.
    edges = pd.read_csv(args.edges)
    source_rows = edges.source_feature_row.to_numpy(np.int64)
    target_rows = edges.target_feature_row.to_numpy(np.int64)
    edge_fg = np.einsum(
        "ij,ij->i",
        np.asarray(foreground[source_rows], dtype=np.float32),
        np.asarray(foreground[target_rows], dtype=np.float32),
    )
    edge_box = np.einsum(
        "ij,ij->i",
        np.asarray(box[source_rows], dtype=np.float32),
        np.asarray(box[target_rows], dtype=np.float32),
    )
    rng = np.random.default_rng(SEED)
    graph_feature_rows = assignments.feature_row.to_numpy(np.int64)
    graph_parents = assignments.parent_id.to_numpy()
    source_graph_rows = edges.source_graph_row.to_numpy(np.int64)
    negative_graph_rows = rng.integers(0, len(assignments), size=len(edges))
    conflict = graph_parents[negative_graph_rows] == graph_parents[source_graph_rows]
    while conflict.any():
        negative_graph_rows[conflict] = rng.integers(0, len(assignments), size=int(conflict.sum()))
        conflict = graph_parents[negative_graph_rows] == graph_parents[source_graph_rows]
    negative_rows = graph_feature_rows[negative_graph_rows]
    negative_fg = np.einsum(
        "ij,ij->i",
        np.asarray(foreground[source_rows], dtype=np.float32),
        np.asarray(foreground[negative_rows], dtype=np.float32),
    )
    negative_box = np.einsum(
        "ij,ij->i",
        np.asarray(box[source_rows], dtype=np.float32),
        np.asarray(box[negative_rows], dtype=np.float32),
    )
    labels = np.concatenate([np.ones(len(edges)), np.zeros(len(edges))])
    foreground_auc = float(roc_auc_score(labels, np.concatenate([edge_fg, negative_fg])))
    box_auc = float(roc_auc_score(labels, np.concatenate([edge_box, negative_box])))
    fg_margin = edge_fg - negative_fg
    box_margin = edge_box - negative_box
    margin_advantage = fg_margin - box_margin
    diagnostic_metrics = {
        "status": "complete_selection_conditional_mask_box_diagnostic",
        "edge_count": len(edges),
        "negative_count": len(edges),
        "negative_sampling": "fixed-seed random cross-parent region per frozen foreground-selected edge source",
        "selection_condition": "Edges and families were fitted from foreground descriptors; results are a mechanism diagnostic, not an unbiased benchmark.",
        "foreground_edge_similarity_mean": float(edge_fg.mean()),
        "box_edge_similarity_mean": float(edge_box.mean()),
        "foreground_negative_similarity_mean": float(negative_fg.mean()),
        "box_negative_similarity_mean": float(negative_box.mean()),
        "foreground_edge_negative_margin_mean": float(fg_margin.mean()),
        "box_edge_negative_margin_mean": float(box_margin.mean()),
        "foreground_margin_advantage_over_box_mean": float(margin_advantage.mean()),
        "foreground_margin_advantage_bootstrap_95pct": bootstrap_mean_ci(margin_advantage, rng),
        "foreground_edge_vs_negative_auc": foreground_auc,
        "box_edge_vs_negative_auc": box_auc,
        "auc_difference_foreground_minus_box": foreground_auc - box_auc,
    }
    diagnostic_metrics_path = args.output / "mask_box_diagnostic.json"
    diagnostic_metrics_path.write_text(json.dumps(diagnostic_metrics, indent=2) + "\n", encoding="utf-8")

    core_counts = Counter(core.groupby("parent_id").size())
    retained_family_count = int((revised_adjudication.include_in_configuration == 1).sum())
    metrics = {
        "status": "complete_cross_view_consensus_refinement_r3",
        "input_support_rows": len(support),
        "supports_removed_by_visual_pair_correction": int(sum(rejected_supports.values())),
        "visual_pair_correction_counts": dict(rejected_supports),
        "retained_internal_families": retained_family_count,
        "retained_concepts_with_any_core_support": int(core.descriptive_family_key.nunique()),
        "retained_support_rows_including_peripheral": len(support_r3),
        "consensus_core_support_rows": len(core),
        "canonical_photos_with_consensus_core_support": int(core.parent_id.nunique()),
        "photos_by_distinct_consensus_core_concept_count": dict(sorted((int(k), int(v)) for k, v in core_counts.items())),
        "sensitivity_definition": "r2 local affinity, foreground to medoid, and box to medoid are each at or above their family median",
        "confidence_warning": "Median consensus is a fixed sensitivity analysis, not calibrated probability or component identity.",
        "inputs": {name: {"path": str(path), "sha256": digest(path)} for name, path in {
            "foreground": args.foreground,
            "box": args.box,
            "assignments": args.assignments,
            "edges": args.edges,
            "families": args.families,
            "adjudication": args.adjudication,
            "concept_support": args.concept_support,
            "global_assignments": args.global_assignments,
        }.items()},
        "outputs": {},
    }
    for path in (diagnostic_path, adjudication_path, support_path, concept_path, diagnostic_metrics_path):
        metrics["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**metrics, "mask_box_diagnostic": diagnostic_metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
