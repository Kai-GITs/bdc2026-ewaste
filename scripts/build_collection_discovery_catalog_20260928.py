"""Build a queryable catalog for all active r2 region-family members."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value, fallback: str) -> str:
    return str(value).strip() if isinstance(value, str) and value.strip() else fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-assignments", type=Path, required=True)
    parser.add_argument("--family-summary", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--r3-supports", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--global-assignments", type=Path, required=True)
    parser.add_argument("--community-names", type=Path, required=True)
    parser.add_argument("--top-family-candidates", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assignments = pd.read_csv(args.r2_assignments)
    assignments = assignments[assignments.main_family_internal.ge(0)].copy()
    summary = pd.read_csv(args.family_summary).set_index("family_internal")
    adjudication = pd.read_csv(args.adjudication).set_index("family_internal")
    supports = pd.read_csv(args.r3_supports)
    core = supports[supports.configuration_core_support.eq(1)].copy()
    core_by_region = core.sort_values("two_view_consensus_strength", ascending=False).drop_duplicates(
        "support_region_id"
    ).set_index("support_region_id")
    manifest = json.loads(args.image_manifest.read_text(encoding="utf-8"))
    image_by_parent = {row["parent_image_id"]: row for row in manifest}
    global_assignments = pd.read_csv(args.global_assignments).set_index("canonical_id")
    community_names = pd.read_csv(args.community_names).set_index("community_id")

    wanted = set(assignments.region_id)
    regions = {}
    with gzip.open(args.regions, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["region_id"] in wanted:
                regions[row["region_id"]] = row
    if wanted - set(regions):
        raise ValueError(f"missing {len(wanted - set(regions))} active region records")

    features = np.asarray(np.load(args.features, mmap_mode="r"), dtype=np.float32)
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    neighbor_map: dict[str, list[dict]] = {}
    for _, group in assignments.groupby("main_family_internal", sort=True):
        rows = group.reset_index(drop=True)
        matrix = features[rows.feature_row.astype(int).to_numpy()]
        similarity = matrix @ matrix.T
        parents = rows.parent_id.astype(str).to_numpy()
        similarity[parents[:, None] == parents[None, :]] = -np.inf
        for index, region_id in enumerate(rows.region_id):
            eligible = np.flatnonzero(np.isfinite(similarity[index]))
            top = eligible[np.argsort(-similarity[index, eligible])[:3]]
            neighbor_map[region_id] = [
                {"region_id": str(rows.iloc[position].region_id),
                 "similarity": round(float(similarity[index, position]), 6)}
                for position in top
            ]

    # Collection-query candidates are computed over the complete active r2
    # bank.  Each rank represents a different family; query parents and exact
    # image duplicates are excluded.  This is the mechanism evaluated in the
    # target-separated 42-query audit, not a new-photo predictor.
    active = assignments.reset_index(drop=True)
    active_matrix = features[active.feature_row.astype(int).to_numpy()]
    active_parents = active.parent_id.astype(str).to_numpy()
    active_sha = active.source_sha256.astype(str).to_numpy()
    active_family = active.main_family_internal.astype(int).to_numpy()
    active_global = np.asarray([
        int(global_assignments.loc[parent].community_leiden_fused)
        for parent in active_parents
    ])
    candidate_map: dict[str, list[dict]] = {}
    limit = max(1, min(int(args.top_family_candidates), 10))
    block = 256
    for start in range(0, len(active), block):
        stop = min(start + block, len(active))
        similarity = active_matrix[start:stop] @ active_matrix.T
        for local_index, graph_index in enumerate(range(start, stop)):
            valid = ((active_parents != active_parents[graph_index])
                     & (active_sha != active_sha[graph_index]))
            scores = np.where(valid, similarity[local_index], -np.inf)
            candidate_count = min(512, len(scores))
            shortlist = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
            shortlist = shortlist[np.argsort(-scores[shortlist])]
            # Retain one candidate per family.  When that family occurs in a
            # different whole-image community, prefer the cross-context
            # example even if a near-duplicate scene has a higher cosine.
            # This makes the retrieval useful for relation inspection rather
            # than turning it into a nearest-duplicate browser.
            best_cross_context: dict[int, int] = {}
            best_fallback: dict[int, int] = {}
            for position in shortlist:
                if not np.isfinite(scores[position]):
                    continue
                family_id = int(active_family[position])
                best_fallback.setdefault(family_id, int(position))
                if active_global[position] != active_global[graph_index]:
                    best_cross_context.setdefault(family_id, int(position))
            representatives = [
                best_cross_context.get(family_id, position)
                for family_id, position in best_fallback.items()
            ]
            representatives.sort(key=lambda position: -scores[position])
            selected = []
            for position in representatives[:limit]:
                family_id = int(active_family[position])
                selected.append({
                    "region_id": str(active.iloc[position].region_id),
                    "family_internal": family_id,
                    "similarity": round(float(scores[position]), 6),
                    "same_family": bool(family_id == int(active_family[graph_index])),
                    "cross_context_priority": bool(
                        active_global[position] != active_global[graph_index]
                    ),
                })
            if len(selected) < limit:
                raise ValueError("candidate shortlist did not span enough r2 families")
            candidate_map[str(active.iloc[graph_index].region_id)] = selected

    families = []
    for family, row in summary.sort_index().iterrows():
        adjudicated = adjudication.loc[family]
        key = clean_text(adjudicated.descriptive_family_key, f"r2_family_{int(family):04d}")
        name = clean_text(adjudicated.descriptive_name_id, f"Pola belum terselesaikan F{int(family):04d}")
        families.append({
            "family_internal": int(family),
            "machine_family_id": f"F{int(family):04d}",
            "family_key": key,
            "family_name": name,
            "scientific_role": clean_text(adjudicated.scientific_role, "unresolved_pattern"),
            "r3_decision": clean_text(adjudicated.r3_decision, "unresolved"),
            "r3_rejection_reason": clean_text(adjudicated.r3_rejection_reason, ""),
            "review_note": clean_text(adjudicated.r3_review_note, ""),
            "regions": int(row.regions),
            "unique_parents": int(row.unique_parents),
            "median_family_similarity": round(float(row.median_within_family_dino_similarity), 6),
            "medoid_region_id": str(row.medoid_region_id),
            "r3_core_regions": int((core.source_family_internal == family).sum()),
        })
    family_by_id = {row["family_internal"]: row for row in families}

    items = []
    for row in assignments.sort_values(["main_family_internal", "region_id"]).itertuples():
        family = family_by_id[int(row.main_family_internal)]
        region = regions[row.region_id]
        image = image_by_parent[str(row.parent_id)]
        global_row = global_assignments.loc[str(row.parent_id)]
        community_id = int(global_row.community_leiden_fused)
        community = community_names.loc[community_id]
        is_core = row.region_id in core_by_region.index
        items.append({
            "region_id": str(row.region_id),
            "parent_id": str(row.parent_id),
            "family_internal": int(row.main_family_internal),
            "machine_family_id": family["machine_family_id"],
            "family_key": family["family_key"],
            "family_name": family["family_name"],
            "scientific_role": family["scientific_role"],
            "r3_decision": family["r3_decision"],
            "is_r3_core": bool(is_core),
            "source_relative_path": str(image["source_relative_path"]).replace("\\", "/"),
            "source_sha256": str(image["source_sha256"]),
            "global_community": community_id,
            "global_context_name": str(community.descriptive_name),
            "width": int(image["native_width"]),
            "height": int(image["native_height"]),
            "bbox_xyxy": region["mask_bbox_xyxy_native"],
            "mask_sha256": region["mask_sha256"],
            "area_fraction": round(float(row.area_fraction), 8),
            "family_affinity": round(float(row.median_within_family_dino_similarity), 6),
            "neighbors": neighbor_map[str(row.region_id)],
            "candidates": candidate_map[str(row.region_id)],
        })

    payload = {
        "schema_version": 1,
        "scope": "transductive catalog of the already analyzed BDC image collection",
        "summary": {
            "received_photo_records": len(manifest),
            "canonical_photos": len({row["exact_duplicate_canonical_id"] for row in manifest}),
            "active_regions": len(items),
            "isolated_regions_excluded": int((pd.read_csv(args.r2_assignments).main_family_internal < 0).sum()),
            "r2_families": len(families),
            "r3_retained_families": int(sum(row["r3_decision"] == "retained" for row in families)),
            "r3_core_families": int(core.source_family_internal.nunique()),
            "r3_core_descriptive_families": int(core.descriptive_family_key.nunique()),
            "r3_core_regions": int(sum(row["is_r3_core"] for row in items)),
            "r3_core_photos": len({row["parent_id"] for row in items if row["is_r3_core"]}),
            "query_candidates_per_region": limit,
        },
        "revisions": {
            "r2_partition": "region-family-graph-object-filtered-r2",
            "r3_adjudication": "region-family-consensus-r3-20260928",
        },
        "limitations": [
            "Family membership is unsupervised and transductive; this catalog does not process a new photo.",
            "Semantic family names were assigned after discovery by one Codex agent.",
            "r2 members remain discovery candidates; r3 core is a stricter support subset, not human observation.",
            "Region proximity does not prove component identity, ownership, hidden content, or physical-unit count.",
        ],
        "families": families,
        "items": items,
        "inputs": {
            str(path): sha256(path) for path in [
                args.r2_assignments, args.family_summary, args.adjudication,
                args.r3_supports, args.regions, args.features, args.image_manifest,
                args.global_assignments, args.community_names,
            ]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    receipt = {
        "catalog": str(args.output),
        "sha256": sha256(args.output),
        "bytes": args.output.stat().st_size,
        **payload["summary"],
    }
    args.output.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
