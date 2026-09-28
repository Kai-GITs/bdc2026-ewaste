"""Build the object-filtered unsupervised BDC region-family graph.

Topology comes only from mutual DINOv3 foreground neighbors across different
parents.  SigLIP 2 crop ranks reweight those same edges without adding target
classes or deleting DINO edges.  Contributions are normalized per parent before
Leiden clustering; isolated regions remain unresolved.
"""

import csv
import gzip
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ["HF_HOME"] = "/research-cache/hf"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

import igraph as ig
import leidenalg as la
import numpy as np
import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


START = time.monotonic()
TOKEN = os.environ["HF_TOKEN"]
REPO = "Fin000/codex-research-workspace"
PRIMARY_PREFIX = "bdc2026-ewaste/final-study-20260928/composition-region-bank-r1"
SIG_PREFIX = "bdc2026-ewaste/final-study-20260928/siglip2-region-crop-r1"
FILTER_PREFIX = "bdc2026-ewaste/final-study-20260928/region-object-filter-r2"
PRIMARY_REV = "eb364930b8679c5817d44730233cdb9899d37034"
SIG_REV = "949ca4ef03f0464c93d1b37853137f7c824c0c09"
FILTER_REV = "f7c08dc74a442fffbfdefba5230c278ed621fd3b"
PRIMARY_REGIONS_SHA = "a9ee2b70b43cc05e65a56babbda818cb9ff81674808b67128684c2ab4bdcf071"
PRIMARY_FEATURE_SHA = "7adb08f1f8577bc73b5bff8ef58ab7d6c5e4dfcc384a9db7178abaa2a870bc56"
SIG_FEATURE_SHA = "fbd9ee772ab6bf5cfc5a172a6f8e450514f3785bd6cdf24df69c3850f062d9be"
FILTER_SHA = "d8505afd58fa207579587c2d1cd9b99e7aaa9b44b9b19a1ab50e59b22098f083"
DEST = "bdc2026-ewaste/final-study-20260928/region-family-graph-object-filtered-r2"
OUT = Path("/research-cache/bdc2026/final-study-20260928/region-family-graph-object-filtered-r2")
OUT.mkdir(parents=True, exist_ok=True)

K = 15
RAW_K = 512
ALTERNATIVES = 30
BLOCK = 512
RESOLUTIONS = (0.5, 1.0, 1.5)
SEEDS = (11, 23, 37)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def comb2(value):
    value = np.asarray(value, dtype=np.float64)
    return value * (value - 1.0) / 2.0


def adjusted_rand(labels_a, labels_b):
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    assert labels_a.shape == labels_b.shape
    n = len(labels_a)
    if n < 2:
        return 1.0
    _, ai = np.unique(labels_a, return_inverse=True)
    _, bi = np.unique(labels_b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    sum_cells = comb2(table).sum()
    sum_a = comb2(table.sum(axis=1)).sum()
    sum_b = comb2(table.sum(axis=0)).sum()
    total = comb2(n)
    expected = sum_a * sum_b / max(total, 1.0)
    maximum = 0.5 * (sum_a + sum_b)
    denominator = maximum - expected
    return float((sum_cells - expected) / denominator) if denominator else 1.0


def private_file(prefix, name, revision, expected_sha):
    path = Path(hf_hub_download(
        REPO, f"{prefix}/{name}", repo_type="dataset", revision=revision, token=TOKEN))
    assert digest(path) == expected_sha, name
    return path


assert all(not value.startswith("__") for value in (
    PRIMARY_REV, SIG_REV, FILTER_REV, PRIMARY_REGIONS_SHA,
    PRIMARY_FEATURE_SHA, SIG_FEATURE_SHA, FILTER_SHA))
api = HfApi(token=TOKEN)
assert api.repo_info(REPO, repo_type="dataset").private
regions_path = private_file(PRIMARY_PREFIX, "regions.jsonl.gz", PRIMARY_REV, PRIMARY_REGIONS_SHA)
dino_path = private_file(PRIMARY_PREFIX, "dinov3_foreground.npy", PRIMARY_REV, PRIMARY_FEATURE_SHA)
sig_path = private_file(SIG_PREFIX, "siglip2_crop.npy", SIG_REV, SIG_FEATURE_SHA)
filter_path = private_file(
    FILTER_PREFIX, "eligible_regions.csv.gz", FILTER_REV, FILTER_SHA)
with gzip.open(regions_path, "rt", encoding="utf-8") as handle:
    all_regions = [json.loads(line) for line in handle]
assert [row["feature_row"] for row in all_regions] == list(range(len(all_regions)))
with gzip.open(filter_path, "rt", encoding="utf-8", newline="") as handle:
    eligible_rows = list(csv.DictReader(handle))
source_feature_rows = np.asarray(
    [int(row["feature_row"]) for row in eligible_rows], dtype=np.int32)
assert len(source_feature_rows) == 6514
assert len(set(source_feature_rows.tolist())) == len(source_feature_rows)
regions = [all_regions[index] for index in source_feature_rows]
n = len(regions)
full_dino = np.load(dino_path, mmap_mode="r")
full_sig = np.load(sig_path, mmap_mode="r")
assert full_dino.shape == (len(all_regions), 1024)
assert full_sig.shape == (len(all_regions), 1152)
dino = np.asarray(full_dino[source_feature_rows], dtype=np.float16)
sig = np.asarray(full_sig[source_feature_rows], dtype=np.float16)
assert dino.shape == (n, 1024) and sig.shape == (n, 1152)
assert np.isfinite(dino).all() and np.isfinite(sig).all()

parent_names = [row["parent_id"] for row in regions]
parent_lookup = {name: index for index, name in enumerate(sorted(set(parent_names)))}
parents = np.asarray([parent_lookup[name] for name in parent_names], dtype=np.int32)
parent_region_counts = np.bincount(parents)
max_regions_per_parent = int(parent_region_counts.max())
assert max_regions_per_parent <= 21
assert (RAW_K - 1) // max_regions_per_parent >= K


def exact_neighbor_view(features, dino_targets=None):
    """Return unique-parent neighbors and pair-specific ranks from exact search."""
    matrix = torch.from_numpy(np.array(features, dtype=np.float16, copy=True)).cuda()
    matrix = torch.nn.functional.normalize(matrix.float(), dim=1).half()
    neighbors = np.full((n, K), -1, dtype=np.int32)
    unique_ranks = np.zeros((n, K), dtype=np.uint16)
    raw_ranks = np.zeros((n, K), dtype=np.uint16)
    similarities = np.full((n, K), np.nan, dtype=np.float16)
    alternatives = np.full((n, ALTERNATIVES), -1, dtype=np.int32)
    alternative_for = np.full((n, ALTERNATIVES), -1, dtype=np.int16)
    alternative_raw_rank = np.zeros((n, ALTERNATIVES), dtype=np.uint16)
    target_pair_rank = (np.full((n, K), RAW_K + 1, dtype=np.uint16)
                        if dino_targets is not None else None)
    target_parent_rank = (np.full((n, K), RAW_K + 1, dtype=np.uint16)
                          if dino_targets is not None else None)
    target_is_parent_representative = (np.zeros((n, K), dtype=np.uint8)
                                       if dino_targets is not None else None)
    search_k = min(n, RAW_K + 1)
    with torch.inference_mode():
        for begin in range(0, n, BLOCK):
            end = min(begin + BLOCK, n)
            scores = matrix[begin:end] @ matrix.T
            values, indices = torch.topk(scores, k=search_k, dim=1, largest=True, sorted=True)
            values = values.float().cpu().numpy()
            indices = indices.cpu().numpy()
            for offset, source_index in enumerate(range(begin, end)):
                seen_parent = {}
                alt_count = 0
                cross_rank = 0
                target_slots = ({int(value): slot for slot, value in enumerate(
                    dino_targets[source_index]) if value >= 0}
                    if dino_targets is not None else {})
                for raw_rank, (target_index, similarity) in enumerate(
                        zip(indices[offset], values[offset]), start=1):
                    target_index = int(target_index)
                    if target_index == source_index or parents[target_index] == parents[source_index]:
                        continue
                    cross_rank += 1
                    parent = int(parents[target_index])
                    first_for_parent = parent not in seen_parent
                    if first_for_parent:
                        slot = len(seen_parent)
                        seen_parent[parent] = target_index
                        if slot < K:
                            neighbors[source_index, slot] = target_index
                            unique_ranks[source_index, slot] = slot + 1
                            raw_ranks[source_index, slot] = cross_rank
                            similarities[source_index, slot] = similarity
                    else:
                        selected_target = seen_parent[parent]
                        selected_positions = np.flatnonzero(neighbors[source_index] == selected_target)
                        if len(selected_positions) and alt_count < ALTERNATIVES:
                            alternatives[source_index, alt_count] = target_index
                            alternative_for[source_index, alt_count] = int(selected_positions[0])
                            alternative_raw_rank[source_index, alt_count] = cross_rank
                            alt_count += 1
                    if target_index in target_slots:
                        slot = target_slots[target_index]
                        target_pair_rank[source_index, slot] = min(cross_rank, RAW_K + 1)
                        target_parent_rank[source_index, slot] = min(
                            list(seen_parent).index(parent) + 1, RAW_K + 1)
                        target_is_parent_representative[source_index, slot] = int(
                            seen_parent[parent] == target_index)
                    if (len(seen_parent) >= K and alt_count >= ALTERNATIVES and
                            (dino_targets is None or all(
                                target_pair_rank[source_index, slot] <= RAW_K
                                for slot in target_slots.values()))):
                        break
                assert (neighbors[source_index] >= 0).all(), source_index
            print(json.dumps({"event": "neighbor_block", "begin": begin, "end": end,
                              "elapsed_seconds": round(time.monotonic() - START, 1)}), flush=True)
            del scores, values, indices
    del matrix
    torch.cuda.empty_cache()
    return {
        "neighbors": neighbors,
        "unique_ranks": unique_ranks,
        "raw_ranks": raw_ranks,
        "similarities": similarities,
        "alternatives": alternatives,
        "alternative_for": alternative_for,
        "alternative_raw_rank": alternative_raw_rank,
        "target_pair_rank": target_pair_rank,
        "target_parent_rank": target_parent_rank,
        "target_is_parent_representative": target_is_parent_representative,
    }


dino_view = exact_neighbor_view(dino)
sig_view = exact_neighbor_view(sig, dino_targets=dino_view["neighbors"])

src, dst = [], []
rank_src, rank_dst, sig_rank_src, sig_rank_dst = [], [], [], []
dino_sim_src, dino_sim_dst = [], []
sig_parent_rep_src, sig_parent_rep_dst = [], []
for i in range(n):
    for position, j in enumerate(dino_view["neighbors"][i]):
        j = int(j)
        if j <= i:
            continue
        reverse = np.flatnonzero(dino_view["neighbors"][j] == i)
        if not len(reverse):
            continue
        reverse_position = int(reverse[0])
        src.append(i)
        dst.append(j)
        rank_src.append(position + 1)
        rank_dst.append(reverse_position + 1)
        sig_rank_src.append(int(sig_view["target_pair_rank"][i, position]))
        sig_rank_dst.append(int(sig_view["target_pair_rank"][j, reverse_position]))
        dino_sim_src.append(float(dino_view["similarities"][i, position]))
        dino_sim_dst.append(float(dino_view["similarities"][j, reverse_position]))
        sig_parent_rep_src.append(int(
            sig_view["target_is_parent_representative"][i, position]))
        sig_parent_rep_dst.append(int(
            sig_view["target_is_parent_representative"][j, reverse_position]))

src = np.asarray(src, dtype=np.int32)
dst = np.asarray(dst, dtype=np.int32)
rank_src = np.asarray(rank_src, dtype=np.float32)
rank_dst = np.asarray(rank_dst, dtype=np.float32)
sig_rank_src = np.asarray(sig_rank_src, dtype=np.float32)
sig_rank_dst = np.asarray(sig_rank_dst, dtype=np.float32)
dino_sim_src = np.asarray(dino_sim_src, dtype=np.float32)
dino_sim_dst = np.asarray(dino_sim_dst, dtype=np.float32)
sig_parent_rep_src = np.asarray(sig_parent_rep_src, dtype=np.uint8)
sig_parent_rep_dst = np.asarray(sig_parent_rep_dst, dtype=np.uint8)
assert len(src) and (parents[src] != parents[dst]).all()


def normalized_weights(effective_src, effective_dst):
    directed_src = 1.0 / effective_src
    directed_dst = 1.0 / effective_dst
    totals = np.zeros(len(parent_lookup), dtype=np.float64)
    np.add.at(totals, parents[src], directed_src)
    np.add.at(totals, parents[dst], directed_dst)
    norm_src = directed_src / totals[parents[src]]
    norm_dst = directed_dst / totals[parents[dst]]
    check = np.zeros(len(parent_lookup), dtype=np.float64)
    np.add.at(check, parents[src], norm_src)
    np.add.at(check, parents[dst], norm_dst)
    assert np.allclose(check[totals > 0], 1.0, atol=1e-7)
    edge = 0.5 * (norm_src + norm_dst)
    return edge.astype(np.float64), totals


dino_weights, dino_parent_totals = normalized_weights(rank_src, rank_dst)
effective_src = np.sqrt(rank_src * sig_rank_src)
effective_dst = np.sqrt(rank_dst * sig_rank_dst)
reweighted, fused_parent_totals = normalized_weights(effective_src, effective_dst)
assert np.isfinite(dino_weights).all() and np.isfinite(reweighted).all()

degree = np.bincount(np.concatenate([src, dst]), minlength=n)
active = np.flatnonzero(degree > 0)
isolates = np.flatnonzero(degree == 0)
local_index = np.full(n, -1, dtype=np.int32)
local_index[active] = np.arange(len(active), dtype=np.int32)
edge_pairs = list(zip(local_index[src].tolist(), local_index[dst].tolist()))
graph = ig.Graph(n=len(active), edges=edge_pairs, directed=False)


def partitions(weights, label):
    output = {}
    for resolution in RESOLUTIONS:
        for seed in SEEDS:
            partition = la.find_partition(
                graph, la.RBConfigurationVertexPartition,
                weights=weights.tolist(), resolution_parameter=resolution,
                seed=seed, n_iterations=-1)
            labels = np.full(n, -1, dtype=np.int32)
            labels[active] = np.asarray(partition.membership, dtype=np.int32)
            key = f"{label}_r{resolution:g}_s{seed}"
            output[key] = labels
    return output


assignments = {}
assignments.update(partitions(dino_weights, "dino"))
assignments.update(partitions(reweighted, "reweighted"))
main_key = "reweighted_r1_s11"
main = assignments[main_key]


def pairwise_stability(prefix, resolution):
    keys = [f"{prefix}_r{resolution:g}_s{seed}" for seed in SEEDS]
    values = []
    for first in range(len(keys)):
        for second in range(first + 1, len(keys)):
            values.append(adjusted_rand(assignments[keys[first]][active],
                                        assignments[keys[second]][active]))
    return values


stability = {}
for prefix in ("dino", "reweighted"):
    for resolution in RESOLUTIONS:
        values = pairwise_stability(prefix, resolution)
        stability[f"{prefix}_resolution_{resolution:g}"] = {
            "pairwise_ari": values, "minimum_ari": min(values),
            "mean_ari": float(np.mean(values)),
        }
partition_summary = {
    key: {
        "families": len(set(values[active])),
        "largest_family": int(max(Counter(values[active]).values())),
    }
    for key, values in assignments.items()
}


incident_weights = [[] for _ in range(n)]
incident_similarities = [[] for _ in range(n)]
weighted_degree = np.zeros(n, dtype=np.float64)
for edge_index, (i, j) in enumerate(zip(src, dst)):
    if main[i] == main[j] and main[i] >= 0:
        weight = float(reweighted[edge_index])
        similarity = 0.5 * (float(dino_sim_src[edge_index]) +
                            float(dino_sim_dst[edge_index]))
        incident_weights[i].append(weight)
        incident_weights[j].append(weight)
        incident_similarities[i].append(similarity)
        incident_similarities[j].append(similarity)
        weighted_degree[i] += weight
        weighted_degree[j] += weight

support_weight = np.asarray([
    np.median(values) if values else np.nan for values in incident_weights], dtype=np.float64)
support_similarity = np.asarray([
    np.median(values) if values else np.nan for values in incident_similarities], dtype=np.float64)

assignment_path = OUT / "assignments.csv.gz"
assignment_fields = [
    "graph_row", "feature_row", "region_id", "parent_id", "source_sha256",
    "resolution_status", "area_fraction", "centroid_x", "centroid_y",
    "main_family_internal", "median_within_family_edge_weight",
    "median_within_family_dino_similarity", "weighted_degree",
] + sorted(assignments)
with gzip.open(assignment_path, "wt", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=assignment_fields)
    writer.writeheader()
    for index, row in enumerate(regions):
        record = {
            "graph_row": index,
            "feature_row": row["feature_row"],
            "region_id": row["region_id"],
            "parent_id": row["parent_id"],
            "source_sha256": row["source_sha256"],
            "resolution_status": row["resolution_status"],
            "area_fraction": row["area_fraction"],
            "centroid_x": row["centroid_xy_normalized"][0],
            "centroid_y": row["centroid_xy_normalized"][1],
            "main_family_internal": int(main[index]),
            "median_within_family_edge_weight": (
                "" if not np.isfinite(support_weight[index]) else support_weight[index]),
            "median_within_family_dino_similarity": (
                "" if not np.isfinite(support_similarity[index]) else support_similarity[index]),
            "weighted_degree": weighted_degree[index],
        }
        record.update({key: int(values[index]) for key, values in assignments.items()})
        writer.writerow(record)

edge_path = OUT / "edges.csv.gz"
with gzip.open(edge_path, "wt", encoding="utf-8", newline="") as handle:
    fields = ["source_graph_row", "target_graph_row", "source_feature_row",
              "target_feature_row", "source_parent", "target_parent",
              "dino_rank_source", "dino_rank_target", "siglip_pair_rank_source",
              "siglip_pair_rank_target", "siglip_parent_representative_source",
              "siglip_parent_representative_target", "dino_similarity_source",
              "dino_similarity_target", "dino_parent_normalized_weight",
              "two_view_parent_normalized_weight"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for edge_index, (i, j) in enumerate(zip(src, dst)):
        writer.writerow({
            "source_graph_row": int(i), "target_graph_row": int(j),
            "source_feature_row": int(regions[i]["feature_row"]),
            "target_feature_row": int(regions[j]["feature_row"]),
            "source_parent": regions[i]["parent_id"],
            "target_parent": regions[j]["parent_id"],
            "dino_rank_source": int(rank_src[edge_index]),
            "dino_rank_target": int(rank_dst[edge_index]),
            "siglip_pair_rank_source": int(sig_rank_src[edge_index]),
            "siglip_pair_rank_target": int(sig_rank_dst[edge_index]),
            "siglip_parent_representative_source": int(sig_parent_rep_src[edge_index]),
            "siglip_parent_representative_target": int(sig_parent_rep_dst[edge_index]),
            "dino_similarity_source": float(dino_sim_src[edge_index]),
            "dino_similarity_target": float(dino_sim_dst[edge_index]),
            "dino_parent_normalized_weight": float(dino_weights[edge_index]),
            "two_view_parent_normalized_weight": float(reweighted[edge_index]),
        })

np.savez_compressed(
    OUT / "neighbor_evidence.npz",
    source_feature_rows=source_feature_rows,
    dino_neighbors=dino_view["neighbors"],
    dino_unique_parent_ranks=dino_view["unique_ranks"],
    dino_raw_cross_parent_ranks=dino_view["raw_ranks"],
    dino_similarities=dino_view["similarities"],
    dino_alternative_indices=dino_view["alternatives"],
    dino_alternative_for_slot=dino_view["alternative_for"],
    dino_alternative_raw_rank=dino_view["alternative_raw_rank"],
    siglip_neighbors=sig_view["neighbors"],
    siglip_similarities=sig_view["similarities"],
    siglip_pair_rank_for_dino=sig_view["target_pair_rank"],
    siglip_parent_rank_for_dino=sig_view["target_parent_rank"],
    siglip_is_parent_representative_for_dino=sig_view["target_is_parent_representative"],
)

families = []
for family in sorted(set(main) - {-1}):
    members = np.flatnonzero(main == family)
    family_parents = np.unique(parents[members])
    medoid = int(members[np.argmax(weighted_degree[members])])
    finite_support = support_similarity[members][np.isfinite(support_similarity[members])]
    areas = np.asarray([regions[index]["area_fraction"] for index in members], dtype=float)
    families.append({
        "family_internal": int(family),
        "regions": len(members),
        "unique_parents": len(family_parents),
        "coarse_regions": sum(
            regions[index]["resolution_status"] == "coarse_only_under_24px"
            for index in members),
        "area_fraction_q10": float(np.quantile(areas, .1)),
        "area_fraction_median": float(np.median(areas)),
        "area_fraction_q90": float(np.quantile(areas, .9)),
        "median_within_family_dino_similarity": (
            float(np.median(finite_support)) if len(finite_support) else ""),
        "medoid_graph_row": medoid,
        "medoid_feature_row": int(regions[medoid]["feature_row"]),
        "medoid_region_id": regions[medoid]["region_id"],
        "medoid_parent_id": regions[medoid]["parent_id"],
    })
with (OUT / "family_summary.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(families[0]))
    writer.writeheader()
    writer.writerows(families)

components = graph.connected_components()
main_sizes = Counter(main[active])
dino_main = assignments["dino_r1_s11"]
metrics = {
    "status": "complete_unsupervised_region_family_graph",
    "input_regions": len(all_regions),
    "eligible_regions": n,
    "regions": n,
    "parents_with_eligible_regions": len(parent_lookup),
    "canonical_parents_retained_in_global_graph": 3931,
    "mutual_cross_parent_edges": len(src),
    "active_regions": len(active),
    "unresolved_isolates": len(isolates),
    "connected_components_active_graph": len(components),
    "neighbor_k": K,
    "raw_exact_candidate_k": RAW_K,
    "max_regions_per_parent": max_regions_per_parent,
    "raw_k_distinct_parent_guarantee": (RAW_K - 1) // max_regions_per_parent,
    "topology": "mutual DINOv3 foreground exact-kNN across different parents; one selected region per neighbor parent",
    "dino_weight": "directed reciprocal DINO unique-parent rank, normalized so each parent with mutual edges has total outgoing mass one; endpoint mean",
    "two_view_weight": "directed reciprocal sqrt(DINO unique-parent rank * pair-specific SigLIP cross-parent rank capped at 513), parent-normalized; endpoint mean",
    "main_partition": main_key,
    "main_families": len(main_sizes),
    "main_family_size_quantiles": np.quantile(
        np.asarray(list(main_sizes.values())), [0, .1, .25, .5, .75, .9, 1]).tolist(),
    "dino_main_families": len(set(dino_main[active])),
    "dino_vs_two_view_main_ari": adjusted_rand(dino_main[active], main[active]),
    "stability": stability,
    "partition_summary": partition_summary,
    "siglip_pair_rank_censored_fraction": float(np.mean(np.concatenate([
        sig_rank_src, sig_rank_dst]) > RAW_K)),
    "siglip_parent_representative_fraction": float(np.mean(np.concatenate([
        sig_parent_rep_src, sig_parent_rep_dst]))),
    "parent_normalization_check": {
        "dino_nonzero_parents": int((dino_parent_totals > 0).sum()),
        "two_view_nonzero_parents": int((fused_parent_totals > 0).sum()),
    },
    "interpretation_limit": "Families are unsupervised visual hypotheses. Region masks, graph edges, and family membership do not prove component identity, physical ownership, hidden content, value, toxicity, or safety.",
    "primary_revision": PRIMARY_REV,
    "siglip_revision": SIG_REV,
    "filter_revision": FILTER_REV,
    "torch": torch.__version__,
    "igraph": ig.__version__,
    "leidenalg": getattr(la, "__version__", "unknown"),
    "gpu": torch.cuda.get_device_name(0),
    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    "elapsed_seconds": round(time.monotonic() - START, 2),
}
write_json(OUT / "metrics.json", metrics)
write_json(OUT / "provenance.json", {
    "primary_region_bank": {"repo": REPO, "prefix": PRIMARY_PREFIX,
                            "revision": PRIMARY_REV,
                            "regions_sha256": PRIMARY_REGIONS_SHA,
                            "features_sha256": PRIMARY_FEATURE_SHA},
    "siglip_crop_bank": {"repo": REPO, "prefix": SIG_PREFIX,
                         "revision": SIG_REV, "features_sha256": SIG_FEATURE_SHA},
    "class_free_object_filter": {"repo": REPO, "prefix": FILTER_PREFIX,
                                  "revision": FILTER_REV,
                                  "eligible_rows_sha256": FILTER_SHA},
    "algorithm": {"exact_raw_neighbors": RAW_K, "mutual_k": K,
                  "resolutions": RESOLUTIONS, "seeds": SEEDS,
                  "leiden_partition": "RBConfigurationVertexPartition"},
})

names = ["assignments.csv.gz", "edges.csv.gz", "neighbor_evidence.npz",
         "family_summary.csv", "metrics.json", "provenance.json"]
checks = {name: {"bytes": (OUT / name).stat().st_size,
                 "sha256": digest(OUT / name)} for name in names}
commit = api.create_commit(
    REPO, repo_type="dataset",
    operations=[CommitOperationAdd(path_in_repo=f"{DEST}/{name}",
                                   path_or_fileobj=str(OUT / name)) for name in names],
    commit_message="Private BDC unsupervised region-family graph")
remote = {item.path: item for item in api.get_paths_info(
    REPO, [f"{DEST}/{name}" for name in names], repo_type="dataset", revision=commit.oid)}
for name in names:
    info = remote[f"{DEST}/{name}"]
    assert info.size == checks[name]["bytes"]
    if info.lfs is not None:
        assert info.lfs.sha256 == checks[name]["sha256"]
    else:
        path = OUT / name
        blob = hashlib.sha1(
            f"blob {path.stat().st_size}\0".encode() + path.read_bytes()).hexdigest()
        assert info.blob_id == blob
print(json.dumps({"event": "complete_private_verified", "revision": commit.oid,
                  "prefix": DEST, "regions": n, "mutual_edges": len(src),
                  "active_regions": len(active), "unresolved_isolates": len(isolates),
                  "main_families": len(main_sizes),
                  "elapsed_seconds": metrics["elapsed_seconds"], "files": checks}), flush=True)
