"""Build a complete discovery census for all 129 r2 region families.

This census keeps the unsupervised machine family separate from later visual
interpretation.  It includes every family regardless of size and carries r3
claim status only as a separate field.  No embedding, mask, or graph is refit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PARTITION_COLUMNS = [
    "dino_r0.5_s11", "dino_r0.5_s23", "dino_r0.5_s37",
    "dino_r1_s11", "dino_r1_s23", "dino_r1_s37",
    "dino_r1.5_s11", "dino_r1.5_s23", "dino_r1.5_s37",
    "reweighted_r0.5_s11", "reweighted_r0.5_s23", "reweighted_r0.5_s37",
    "reweighted_r1_s11", "reweighted_r1_s23", "reweighted_r1_s37",
    "reweighted_r1.5_s11", "reweighted_r1.5_s23", "reweighted_r1.5_s37",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def normalized_entropy(counts: Counter) -> float:
    if len(counts) <= 1:
        return 0.0
    values = np.asarray(list(counts.values()), dtype=np.float64)
    values /= values.sum()
    return float(-(values * np.log(values)).sum() / math.log(len(values)))


def context_from_path(value: str) -> str:
    name = Path(value).stem.lower()
    patterns = (
        (r"battery|baterai", "baterai"),
        (r"pcb|circuit|motherboard", "papan_rangkaian"),
        (r"printer|print", "pencetak"),
        (r"television|\btv\b|bravia", "televisi"),
        (r"mobile|phone|ponsel|smartphone", "ponsel"),
        (r"keyboard", "papan_ketik"),
        (r"mouse|tetikus", "tetikus"),
        (r"microwave", "microwave"),
        (r"washing|washer|mesin.?cuci", "mesin_cuci"),
        (r"player|radio|gramophone|turntable|record", "audio_pemutar"),
        (r"laptop|notebook", "laptop"),
        (r"monitor|screen|display|layar", "monitor_layar"),
        (r"socket|switch|outlet|stopkontak|sakelar", "soket_sakelar"),
    )
    for pattern, label in patterns:
        if re.search(pattern, name):
            return label
    return "nama_sumber_tidak_terstruktur"


def max_jaccard(group: pd.DataFrame, column: str, totals: dict) -> float:
    counts = group[column].value_counts()
    best = 0.0
    for label, intersection in counts.items():
        if int(label) < 0:
            continue
        union = len(group) + totals[column].get(label, 0) - int(intersection)
        if union:
            best = max(best, int(intersection) / union)
    return float(best)


def stability_score(value: float) -> int:
    if value >= 0.75:
        return 4
    if value >= 0.60:
        return 3
    if value >= 0.45:
        return 2
    if value >= 0.30:
        return 1
    return 0


def evidence_tier(parents: int, stability: float) -> str:
    if parents >= 50 and stability >= 0.60:
        return "broad_recurrence"
    if parents >= 20:
        return "moderate_recurrence"
    if parents >= 5:
        return "limited_recurrence"
    return "rare_two_to_four_parent_hypothesis"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--global-assignments", type=Path, required=True)
    parser.add_argument("--community-names", type=Path, required=True)
    parser.add_argument("--cross-view", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument("--r3-adjudication", type=Path, required=True)
    parser.add_argument("--r3-support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(args.assignments)
    families = pd.read_csv(args.families)
    global_rows = pd.read_csv(args.global_assignments)
    community_names = pd.read_csv(args.community_names).set_index("community_id")
    cross_view = pd.read_csv(args.cross_view)
    review = pd.read_csv(args.review_manifest)
    manual = pd.read_csv(args.manual_review)
    r3_adjudication = pd.read_csv(args.r3_adjudication)
    r3_support = pd.read_csv(args.r3_support)

    expected_ids = set(range(129))
    for name, frame in (("families", families), ("manual", manual), ("cross_view", cross_view)):
        ids = set(frame.family_internal.astype(int))
        if ids != expected_ids or len(frame) != 129:
            raise ValueError(f"{name} does not cover exactly all 129 families")
    if len(assignments) != 6514 or int((assignments.main_family_internal < 0).sum()) != 421:
        raise ValueError("Unexpected r2 assignment cohort")

    totals = {
        column: assignments[column].value_counts().to_dict() for column in PARTITION_COLUMNS
    }
    global_lookup = global_rows.set_index("canonical_id")
    r3_lookup = r3_adjudication.set_index("family_internal")
    support_group = r3_support.groupby("source_family_internal")
    records = []
    stability_records = []
    example_records = []

    for family in range(129):
        group = assignments[assignments.main_family_internal == family].copy()
        summary = families[families.family_internal == family].iloc[0]
        manual_row = manual[manual.family_internal == family].iloc[0]
        cross = cross_view[cross_view.family_internal == family].iloc[0]
        if len(group) != int(summary.regions):
            raise ValueError(f"Family size mismatch for {family}")

        partition_jaccard = {
            column: max_jaccard(group, column, totals) for column in PARTITION_COLUMNS
            if column != "reweighted_r1_s11"
        }
        stability_values = np.asarray(list(partition_jaccard.values()), dtype=np.float64)
        same_resolution_seed_min = min(
            partition_jaccard["reweighted_r1_s23"],
            partition_jaccard["reweighted_r1_s37"],
        )
        resolution_min = min(
            value for key, value in partition_jaccard.items()
            if key.startswith("reweighted_r0.5") or key.startswith("reweighted_r1.5")
        )
        view_agreement = partition_jaccard["dino_r1_s11"]
        stability_median = float(np.median(stability_values))
        stability_minimum = float(np.min(stability_values))
        stability_records.append({
            "family_internal": family,
            **partition_jaccard,
        })

        parent_rows = group.drop_duplicates("parent_id").set_index("parent_id")
        joined = parent_rows.join(
            global_lookup[["community_leiden_fused", "source_relative_path"]], how="left"
        )
        if joined.community_leiden_fused.isna().any():
            raise ValueError(f"Missing global parent for family {family}")
        global_counts = Counter(int(value) for value in joined.community_leiden_fused)
        global_named_counts = {
            community_names.loc[key, "descriptive_name"]: int(value)
            for key, value in sorted(global_counts.items())
        }
        source_context_counts = Counter(
            context_from_path(value) for value in joined.source_relative_path
        )

        family_examples = review[review.family_internal == family].copy()
        if family_examples.empty:
            raise ValueError(f"No rendered review examples for family {family}")
        for _, row in family_examples.iterrows():
            example_records.append({
                "family_internal": family,
                "machine_family_id": f"F{family:04d}",
                "visual_name_id": manual_row.visual_name_id,
                "role": row.role,
                "parent_id": row.parent_id,
                "source_relative_path": row.source_relative_path,
                "source_sha256": row.source_sha256,
                "region_id": row.region_id,
                "feature_row": int(row.feature_row),
                "resolution_status": row.resolution_status,
            })
        example_json = json.dumps(
            [
                {
                    "role": row.role,
                    "parent_id": row.parent_id,
                    "path": row.source_relative_path,
                    "region_id": row.region_id,
                }
                for _, row in family_examples.iterrows()
            ], ensure_ascii=False, separators=(",", ":"),
        )

        r3_row = r3_lookup.loc[family]
        if family in support_group.groups:
            supports = support_group.get_group(family)
            r3_core = int((supports.configuration_core_support == 1).sum())
            r3_peripheral = int((supports.configuration_core_support == 0).sum())
        else:
            r3_core = 0
            r3_peripheral = 0

        stable_score = stability_score(stability_median)
        weights = {
            "coherence": 0.22,
            "cross_context": 0.18,
            "novelty": 0.22,
            "interpretability": 0.13,
            "usefulness": 0.18,
            "stability": 0.07,
        }
        base_score = 100.0 * (
            weights["coherence"] * float(manual_row.coherence_0_4)
            + weights["cross_context"] * float(manual_row.cross_context_0_4)
            + weights["novelty"] * float(manual_row.novelty_vs_global_0_4)
            + weights["interpretability"] * float(manual_row.interpretability_0_4)
            + weights["usefulness"] * float(manual_row.analytical_usefulness_0_4)
            + weights["stability"] * stable_score
        ) / 4.0
        interest_score = max(0.0, base_score - 15.0 * float(manual_row.shortcut_risk_0_4) / 4.0)

        records.append({
            "family_internal": family,
            "machine_family_id": f"F{family:04d}",
            "visual_name_id": manual_row.visual_name_id,
            "discovery_roles": manual_row.discovery_roles,
            "regions": int(summary.regions),
            "unique_parent_photos": int(summary.unique_parents),
            "evidence_strength_tier": evidence_tier(int(summary.unique_parents), stability_median),
            "global_community_span": len(global_counts),
            "global_community_entropy_normalized": normalized_entropy(global_counts),
            "largest_global_community_share": max(global_counts.values()) / sum(global_counts.values()),
            "global_community_counts": json.dumps(global_named_counts, ensure_ascii=False, separators=(",", ":")),
            "source_product_context_span": len(source_context_counts),
            "source_product_context_counts": json.dumps(dict(source_context_counts.most_common()), ensure_ascii=False, separators=(",", ":")),
            "same_resolution_seed_jaccard_min": same_resolution_seed_min,
            "resolution_jaccard_min": resolution_min,
            "dino_vs_two_view_jaccard": view_agreement,
            "partition_jaccard_median": stability_median,
            "partition_jaccard_minimum": stability_minimum,
            "partition_jaccard_all": json.dumps(partition_jaccard, separators=(",", ":")),
            "foreground_medoid_similarity_median": float(cross.foreground_medoid_median),
            "box_medoid_similarity_median": float(cross.box_medoid_median),
            "foreground_minus_box_mean": float(cross.foreground_minus_box_mean),
            "foreground_box_correlation": float(cross.foreground_box_correlation),
            "rendered_examples": len(family_examples),
            "medoid_and_diverse_examples": example_json,
            "coherence_0_4": int(manual_row.coherence_0_4),
            "cross_context_0_4": int(manual_row.cross_context_0_4),
            "novelty_vs_global_0_4": int(manual_row.novelty_vs_global_0_4),
            "interpretability_0_4": int(manual_row.interpretability_0_4),
            "analytical_usefulness_0_4": int(manual_row.analytical_usefulness_0_4),
            "shortcut_risk_0_4": int(manual_row.shortcut_risk_0_4),
            "relation_tags": manual_row.relation_tags,
            "visual_review_note": manual_row.visual_review_note,
            "r3_decision": r3_row.r3_decision,
            "r3_rejection_reason": r3_row.r3_rejection_reason,
            "r3_core_supports": r3_core,
            "r3_peripheral_supports": r3_peripheral,
            "discovery_interest_score_0_100": interest_score,
            "semantic_status": "agent_evaluated_offline_from_original_photo_context_and_exact_region_crop",
            "claim_limit": "discovery_hypothesis_not_component_identity_or_hidden_content",
        })

    census = pd.DataFrame(records)
    census["discovery_interest_rank"] = census.discovery_interest_score_0_100.rank(
        method="min", ascending=False
    ).astype(int)
    census = census.sort_values("family_internal")
    census_path = args.output / "family_census.csv"
    census.to_csv(census_path, index=False)

    stability_path = args.output / "partition_stability_by_family.csv"
    pd.DataFrame(stability_records).sort_values("family_internal").to_csv(stability_path, index=False)
    examples_path = args.output / "family_example_manifest.csv"
    pd.DataFrame(example_records).sort_values(["family_internal", "role"]).to_csv(examples_path, index=False)

    candidates = census[
        ~census.discovery_roles.str.contains("shortcut_or_scene")
    ].sort_values(
        ["discovery_interest_score_0_100", "partition_jaccard_median", "unique_parent_photos"],
        ascending=[False, False, False],
    )
    ranking_path = args.output / "candidate_ranking.csv"
    candidates.to_csv(ranking_path, index=False)

    top = candidates.head(20)
    review_path = args.output / "CENSUS_REVIEW.md"
    lines = [
        "# Sensus discovery keluarga region r2",
        "",
        "Sensus ini mencakup seluruh 129 keluarga r2 tanpa ambang minimum jumlah foto. "
        "ID mesin dipertahankan di data pendamping; nama deskriptif dan penilaian visual "
        "adalah interpretasi pascadiscovery.",
        "",
        "## Pemisahan discovery dan klaim",
        "",
        "- **Discovery:** semua keluarga r2, dukungan periferal, partisi multiresolusi, "
        "serta 421 region terisolasi tetap dicatat.",
        "- **Klaim:** status r3, kontrol foreground-versus-box, inspeksi contoh, dan "
        "counterexample menentukan hasil yang boleh dibawa ke naskah.",
        "- Skor minat adalah rubric agen untuk memprioritaskan inspeksi; bukan probabilitas, "
        "ground truth, atau ukuran signifikansi.",
        "",
        "## Cakupan",
        "",
        f"- Keluarga: {len(census)}",
        f"- Region aktif dalam keluarga: {int(census.regions.sum()):,}",
        "- Region terisolasi di luar keluarga: 421",
        f"- Keluarga dua sampai empat foto: {int((census.unique_parent_photos <= 4).sum())}",
        f"- Keluarga dengan peran shortcut/scene: {int(census.discovery_roles.str.contains('shortcut_or_scene').sum())}",
        "",
        "## Dua puluh kandidat pertama untuk pengembangan",
        "",
        "| Rank | Nama deskriptif | Foto | Span global | Peran | Skor minat | Status r3 |",
        "| ---: | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {int(row.discovery_interest_rank)} | {row.visual_name_id} | "
            f"{int(row.unique_parent_photos)} | {int(row.global_community_span)} | "
            f"{row.discovery_roles.replace('|', ', ')} | "
            f"{row.discovery_interest_score_0_100:.1f} | {row.r3_decision} |"
        )
    lines += [
        "",
        "## Batas interpretasi",
        "",
        "Jumlah foto menyatakan kekuatan recurrence dan tidak menjadi syarat kelayakan discovery. "
        "Keluarga kecil tetap hipotesis sampai contoh dan sumber tambahan mendukungnya. Mask dan "
        "kemiripan visual tidak membuktikan identitas komponen, kepemilikan fisik, isi tersembunyi, "
        "jumlah unit, nilai, toksisitas, atau keselamatan.",
    ]
    review_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metrics = {
        "status": "complete_r2_region_family_discovery_census",
        "families": len(census),
        "active_regions": int(census.regions.sum()),
        "unresolved_isolates": int((assignments.main_family_internal < 0).sum()),
        "families_with_2_to_4_unique_parents": int((census.unique_parent_photos <= 4).sum()),
        "families_with_5_to_19_unique_parents": int(census.unique_parent_photos.between(5, 19).sum()),
        "families_with_at_least_20_unique_parents": int((census.unique_parent_photos >= 20).sum()),
        "families_spanning_multiple_global_communities": int((census.global_community_span >= 2).sum()),
        "shortcut_or_scene_families": int(census.discovery_roles.str.contains("shortcut_or_scene").sum()),
        "rare_families_retained_for_discovery": int((census.unique_parent_photos <= 4).sum()),
        "interest_ranking": "agent rubric: coherence 22%, cross-context 18%, novelty 22%, interpretability 13%, usefulness 18%, partition stability 7%, minus up to 15 points shortcut risk; size excluded",
        "inputs": {name: {"path": str(path), "sha256": digest(path)} for name, path in {
            "assignments": args.assignments,
            "families": args.families,
            "global_assignments": args.global_assignments,
            "community_names": args.community_names,
            "cross_view": args.cross_view,
            "review_manifest": args.review_manifest,
            "manual_review": args.manual_review,
            "r3_adjudication": args.r3_adjudication,
            "r3_support": args.r3_support,
        }.items()},
        "outputs": {},
    }
    for path in (census_path, stability_path, examples_path, ranking_path, review_path):
        metrics["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
