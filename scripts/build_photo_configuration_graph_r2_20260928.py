"""Build the conservative r2 photo-configuration graph from adjudicated concepts.

Only retained visible concepts enter this graph.  Multiple unsupervised clusters
collapsed to one descriptive concept count once per photo.  An edge requires at
least two shared concepts backed by distinct regions in each photo.  It remains
a same-scene configuration candidate; physical ownership is unresolved.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np


SEEDS = (11, 23, 37)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def comb2(value):
    value = np.asarray(value, dtype=np.float64)
    return value * (value - 1.0) / 2.0


def adjusted_rand(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2:
        return 1.0
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    cells = comb2(table).sum()
    rows = comb2(table.sum(axis=1)).sum()
    cols = comb2(table.sum(axis=0)).sum()
    total = comb2(len(a))
    expected = rows * cols / max(total, 1.0)
    maximum = 0.5 * (rows + cols)
    return float((cells - expected) / (maximum - expected)) if maximum != expected else 1.0


def cluster(n, edge_i, edge_j, weights, seed):
    labels = np.full(n, -1, dtype=np.int32)
    if not len(edge_i):
        return labels
    degree = np.bincount(np.concatenate([edge_i, edge_j]), minlength=n)
    active = np.flatnonzero(degree > 0)
    local = np.full(n, -1, dtype=np.int32)
    local[active] = np.arange(len(active), dtype=np.int32)
    graph = ig.Graph(
        n=len(active),
        edges=list(zip(local[edge_i].tolist(), local[edge_j].tolist())),
        directed=False,
    )
    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        weights=weights.tolist(),
        resolution_parameter=1.0,
        seed=seed,
        n_iterations=-1,
    )
    labels[active] = np.asarray(partition.membership, dtype=np.int32)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-support", type=Path, required=True)
    parser.add_argument("--concept-summary", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--global-assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    support_rows = read_csv(args.concept_support)
    concept_rows = read_csv(args.concept_summary)
    manifest = json.loads(args.image_manifest.read_text(encoding="utf-8"))
    global_rows = read_csv(args.global_assignments)
    assert len(manifest) == 3961
    canonical_ids = sorted({row["exact_duplicate_canonical_id"] for row in manifest})
    assert len(canonical_ids) == 3931
    parent_index = {parent: index for index, parent in enumerate(canonical_ids)}
    global_community = {
        row["canonical_id"]: int(row["community_leiden_fused"]) for row in global_rows
    }
    assert set(global_community) == set(canonical_ids)

    support = defaultdict(dict)
    concept_name = {}
    concept_role = {}
    for row in concept_rows:
        key = row["descriptive_family_key"]
        concept_name[key] = row["descriptive_name_id"]
        concept_role[key] = row["scientific_role"]
    peripheral_support_rows = 0
    for row in support_rows:
        if row["configuration_core_support"] != "1":
            peripheral_support_rows += 1
            continue
        parent = row["parent_id"]
        key = row["descriptive_family_key"]
        assert parent in parent_index and key in concept_name
        assert key not in support[parent]
        support[parent][key] = row
    for parent, entries in support.items():
        region_ids = [row["support_region_id"] for row in entries.values()]
        assert len(region_ids) == len(set(region_ids)), parent

    concepts = sorted(concept_name)
    n = len(canonical_ids)
    df = Counter()
    pair_count = Counter()
    pair_global_cross = Counter()
    for parent in canonical_ids:
        keys = sorted(support[parent])
        df.update(keys)
        for first in range(len(keys)):
            for second in range(first + 1, len(keys)):
                pair = (keys[first], keys[second])
                pair_count[pair] += 1
                # One photo has one global community.  This counter is retained
                # for schema symmetry and explicit downstream interpretation.
                pair_global_cross[pair] += 0
    weights = {
        key: min(3.0, math.log((n + 1) / (df[key] + 1)) + 1.0) for key in concepts
    }

    cooccurrence_path = args.output / "concept_cooccurrence.csv"
    with cooccurrence_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "concept_a", "name_a_id", "role_a", "concept_b", "name_b_id", "role_b",
            "photos_with_both", "weight_a", "weight_b",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (a, b), count in sorted(pair_count.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({
                "concept_a": a, "name_a_id": concept_name[a], "role_a": concept_role[a],
                "concept_b": b, "name_b_id": concept_name[b], "role_b": concept_role[b],
                "photos_with_both": count, "weight_a": weights[a], "weight_b": weights[b],
            })

    support_count = np.asarray([len(support[parent]) for parent in canonical_ids], dtype=np.int16)
    total_weight = np.asarray([
        sum(weights[key] for key in support[parent]) for parent in canonical_ids
    ], dtype=np.float64)
    shared = np.zeros((n, n), dtype=np.uint8)
    numerator = np.zeros((n, n), dtype=np.float32)
    for key in concepts:
        indices = np.asarray([
            parent_index[parent] for parent in canonical_ids if key in support[parent]
        ], dtype=np.int32)
        if len(indices) < 2:
            continue
        ix = np.ix_(indices, indices)
        shared[ix] += 1
        numerator[ix] += weights[key]

    upper_i, upper_j = np.triu_indices(n, k=1)
    keep = shared[upper_i, upper_j] >= 2
    edge_i = upper_i[keep].astype(np.int32)
    edge_j = upper_j[keep].astype(np.int32)
    shared_count = shared[edge_i, edge_j].astype(np.int16)
    intersection_weight = numerator[edge_i, edge_j].astype(np.float64)
    denominator = total_weight[edge_i] + total_weight[edge_j] - intersection_weight
    assert (denominator > 0).all()
    jaccard = intersection_weight / denominator
    del shared, numerator, upper_i, upper_j, keep

    labels_by_seed = {
        seed: cluster(n, edge_i, edge_j, jaccard, seed) for seed in SEEDS
    }
    primary = labels_by_seed[11]
    shared3_mask = shared_count >= 3
    jaccard25_mask = jaccard >= 0.25
    shared3_labels = cluster(
        n, edge_i[shared3_mask], edge_j[shared3_mask], jaccard[shared3_mask], 11
    )
    jaccard25_labels = cluster(
        n, edge_i[jaccard25_mask], edge_j[jaccard25_mask], jaccard[jaccard25_mask], 11
    )

    edge_path = args.output / "configuration_edges.csv.gz"
    cross_global = 0
    with gzip.open(edge_path, "wt", encoding="utf-8", newline="") as handle:
        fields = [
            "source_parent", "target_parent", "shared_concept_count", "weighted_jaccard",
            "shared_concept_keys", "shared_concept_names_id", "source_support_regions",
            "target_support_regions", "source_global_community_internal",
            "target_global_community_internal", "cross_global_community",
            "relation_status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (i, j) in enumerate(zip(edge_i, edge_j)):
            first, second = canonical_ids[int(i)], canonical_ids[int(j)]
            common = sorted(set(support[first]) & set(support[second]))
            assert len(common) == int(shared_count[index])
            source_regions = [support[first][key]["support_region_id"] for key in common]
            target_regions = [support[second][key]["support_region_id"] for key in common]
            assert len(set(source_regions)) == len(common)
            assert len(set(target_regions)) == len(common)
            is_cross = int(global_community[first] != global_community[second])
            cross_global += is_cross
            writer.writerow({
                "source_parent": first,
                "target_parent": second,
                "shared_concept_count": len(common),
                "weighted_jaccard": float(jaccard[index]),
                "shared_concept_keys": json.dumps(common, separators=(",", ":")),
                "shared_concept_names_id": json.dumps(
                    [concept_name[key] for key in common], ensure_ascii=False, separators=(",", ":")
                ),
                "source_support_regions": json.dumps(source_regions, separators=(",", ":")),
                "target_support_regions": json.dumps(target_regions, separators=(",", ":")),
                "source_global_community_internal": global_community[first],
                "target_global_community_internal": global_community[second],
                "cross_global_community": is_cross,
                "relation_status": "shared_visible_configuration_candidate_ownership_unresolved",
            })

    degree = np.bincount(np.concatenate([edge_i, edge_j]), minlength=n) if len(edge_i) else np.zeros(n, dtype=int)
    parent_path = args.output / "photo_configuration_assignments.csv"
    with parent_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "parent_id", "retained_concept_count", "configuration_eligible",
            "configuration_degree", "configuration_main_internal",
            "configuration_seed23_internal", "configuration_seed37_internal",
            "configuration_shared3_internal", "configuration_jaccard025_internal",
            "global_community_internal", "evidence_status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, parent in enumerate(canonical_ids):
            writer.writerow({
                "parent_id": parent,
                "retained_concept_count": int(support_count[index]),
                "configuration_eligible": int(support_count[index] >= 2),
                "configuration_degree": int(degree[index]),
                "configuration_main_internal": int(primary[index]),
                "configuration_seed23_internal": int(labels_by_seed[23][index]),
                "configuration_seed37_internal": int(labels_by_seed[37][index]),
                "configuration_shared3_internal": int(shared3_labels[index]),
                "configuration_jaccard025_internal": int(jaccard25_labels[index]),
                "global_community_internal": global_community[parent],
                "evidence_status": "positive_visible_support_only; physical ownership unresolved",
            })

    active = np.flatnonzero(primary >= 0)
    seed_ari = [
        adjusted_rand(labels_by_seed[SEEDS[i]][active], labels_by_seed[SEEDS[j]][active])
        for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))
    ] if len(active) else []
    metrics = {
        "status": "complete_adjudicated_photo_configuration_graph_r2",
        "parent_records": len(manifest),
        "canonical_photos": n,
        "retained_concepts": len(concepts),
        "positive_photo_concept_supports": int(support_count.sum()),
        "peripheral_supports_excluded_from_configuration": peripheral_support_rows,
        "photos_without_retained_concept_support": int((support_count == 0).sum()),
        "photos_with_at_least_two_retained_concepts": int((support_count >= 2).sum()),
        "configuration_edges_shared_at_least_two": len(edge_i),
        "configuration_active_photos": len(active),
        "configuration_unresolved_photos": int((primary < 0).sum()),
        "configuration_communities": len(set(primary[active])) if len(active) else 0,
        "seed_pairwise_ari": seed_ari,
        "cross_global_community_edges": cross_global,
        "cross_global_community_edge_fraction": cross_global / len(edge_i) if len(edge_i) else 0.0,
        "sensitivity": {
            "shared_at_least_three_edges": int(shared3_mask.sum()),
            "shared_at_least_three_active_photos": int((shared3_labels >= 0).sum()),
            "weighted_jaccard_at_least_0_25_edges": int(jaccard25_mask.sum()),
            "weighted_jaccard_at_least_0_25_active_photos": int((jaccard25_labels >= 0).sum()),
        },
        "concept_pair_counts": {
            f"{a}|{b}": count for (a, b), count in sorted(
                pair_count.items(), key=lambda item: (-item[1], item[0])
            )
        },
        "family_weight": "min(3, log((N+1)/(df+1))+1), N=3,931 canonical photos",
        "configuration_definition": (
            "At least two distinct adjudicated visible concepts, each backed by a distinct "
            "region at or above its source family's median within-family affinity in both "
            "photos; weighted Jaccard over positive core support only."
        ),
        "missingness": "No support is unresolved and is never interpreted as physical absence.",
        "ownership_limit": (
            "Edges show shared visible configuration support. They do not establish same unit, "
            "attachment, compatibility, quantity, hidden content, value, toxicity, or safety."
        ),
        "source_scene_limit": (
            "Byte-exact duplicates are consolidated. Other shared-source scenes remain unresolved."
        ),
        "inputs": {
            "concept_support_sha256": digest(args.concept_support),
            "concept_summary_sha256": digest(args.concept_summary),
            "image_manifest_sha256": digest(args.image_manifest),
            "global_assignments_sha256": digest(args.global_assignments),
        },
        "igraph": ig.__version__,
        "leidenalg": getattr(la, "__version__", "unknown"),
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    outputs = [cooccurrence_path, edge_path, parent_path, metrics_path]
    (args.output / "artifact_manifest.json").write_text(
        json.dumps(
            {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in outputs},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
