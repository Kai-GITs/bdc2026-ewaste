"""Freeze the image-first post-discovery vocabulary for the r2 region graph.

The mapping below records a complete agent visual review of all 22 evidence
sheets.  It never changes the unsupervised fit.  Repeated clusters that describe
the same visible structure are collapsed before multipart counting so that, for
example, several key-matrix clusters cannot manufacture a configuration.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


REVIEWER = "Codex agent visual review"
REVIEW_DATE = "2026-09-28"
MIN_CONFIGURATION_PARENTS = 20

# family_id: (concept_key, Indonesian descriptive name, scientific role, note)
RETAINED = {
    0: ("circuit_board_surface", "permukaan papan sirkuit terpopulasi", "cross_category_visible_part", "Jalur, pad, dan komponen papan konsisten lintas foto."),
    13: ("circuit_board_surface", "permukaan papan sirkuit terpopulasi", "cross_category_visible_part", "Varian papan dalam rakitan dan tumpukan; digabung dengan keluarga papan lain."),
    41: ("integrated_circuit_package", "kemasan sirkuit terpadu pada papan", "cross_category_visible_part", "Kemasan IC persegi terlihat berulang pada papan berbeda."),
    1: ("dark_display_surface", "permukaan layar gelap", "cross_category_visible_part", "Permukaan layar gelap pada perangkat genggam; isi tersembunyi tetap tidak diketahui."),
    11: ("dark_display_surface", "permukaan layar gelap", "cross_category_visible_part", "Panel gelap pada monitor dan televisi; digabung sebagai bukti permukaan layar."),
    21: ("active_display_surface", "permukaan layar aktif", "cross_category_visible_part", "Area tampilan aktif pada beberapa bentuk perangkat."),
    32: ("visible_display_damage", "kerusakan tampak pada layar", "cross_category_visible_condition", "Garis, pecah, atau gangguan tampilan terlihat pada layar."),
    34: ("visible_display_damage", "kerusakan tampak pada layar", "cross_category_visible_condition", "Retak layar pada perangkat berbeda; digabung dengan keluarga kerusakan layar."),
    52: ("visible_display_damage", "kerusakan tampak pada layar", "cross_category_visible_condition", "Panel rusak atau bergaris; dukungan 21 foto dipertahankan."),
    5: ("key_matrix", "matriks tombol", "cross_category_visible_part", "Kisi tombol konsisten; beberapa keluarga crop digabung sebelum hitung konfigurasi."),
    9: ("key_matrix", "matriks tombol", "cross_category_visible_part", "Fragmen kisi tombol konsisten pada keyboard berbeda."),
    48: ("key_matrix", "matriks tombol", "cross_category_visible_part", "Kisi tombol berulang; digabung dengan keluarga matriks tombol lain."),
    10: ("battery_pack_surface", "permukaan kemasan baterai", "cross_category_visible_part", "Casing, label, atau terminal baterai yang benar-benar terlihat."),
    49: ("display_bezel_or_stand", "bingkai atau penyangga layar", "cross_category_visible_part", "Bingkai bawah dan penyangga layar konsisten."),
    3: ("printer_control_or_output_zone", "zona kontrol atau keluaran pencetak", "within_product_configuration_part", "Panel depan, kontrol, atau keluaran pada pencetak."),
    37: ("printer_media_path", "jalur media pencetak", "within_product_configuration_part", "Keluaran atau baki media terlihat pada pencetak."),
    43: ("printer_media_path", "jalur media pencetak", "within_product_configuration_part", "Tutup atau jalur kertas; digabung dengan jalur media pencetak."),
    6: ("mouse_control_shell", "cangkang dan kontrol mouse", "within_product_configuration_part", "Cangkang, tombol, atau roda mouse terlihat konsisten."),
    7: ("audio_front_panel", "panel depan perangkat audio", "within_product_configuration_part", "Speaker dan kontrol depan pada perangkat audio portabel."),
    16: ("crt_display_front", "muka layar tabung", "within_product_configuration_part", "Permukaan layar dan kontrol muka perangkat CRT."),
    24: ("microwave_door_or_cavity", "pintu atau rongga microwave", "within_product_configuration_part", "Area pintu/rongga microwave konsisten."),
    20: ("microwave_control_panel", "panel kontrol microwave", "within_product_configuration_part", "Panel tombol/kenop microwave konsisten."),
    27: ("laptop_input_deck", "dek masukan laptop", "within_product_configuration_part", "Keyboard dan dek masukan laptop."),
    23: ("laptop_input_deck", "dek masukan laptop", "within_product_configuration_part", "Keyboard/touchpad laptop; digabung agar tidak menjadi dua bagian palsu."),
    33: ("laptop_outer_lid", "penutup luar laptop", "within_product_configuration_part", "Permukaan luar dan logo penutup laptop."),
    39: ("laptop_display_assembly", "rakitan layar laptop", "within_product_configuration_part", "Panel dan bingkai layar laptop."),
    36: ("laptop_edge_ports", "tepi dan port laptop", "within_product_configuration_part", "Tepi perangkat dan deret port pada laptop."),
    30: ("washing_machine_access_door", "pintu akses mesin cuci", "within_product_configuration_part", "Pintu bundar mesin cuci terlihat konsisten."),
    45: ("washing_machine_access_door", "pintu akses mesin cuci", "within_product_configuration_part", "Bingkai pintu mesin cuci; digabung dengan pintu akses."),
    40: ("washing_machine_control_top", "panel atas mesin cuci", "within_product_configuration_part", "Panel atau bagian atas mesin cuci."),
    22: ("desktop_internal_chassis", "sasis internal komputer meja", "within_product_configuration_part", "Sasis dan rakitan internal komputer meja terlihat."),
    26: ("handheld_back_casing", "casing belakang perangkat genggam", "within_product_configuration_part", "Permukaan casing belakang ponsel; bukan bukti isi."),
    44: ("turntable_record_area", "area piringan pemutar rekaman", "within_product_configuration_part", "Tutup/piringan perangkat pemutar rekaman."),
    46: ("turntable_record_area", "area piringan pemutar rekaman", "within_product_configuration_part", "Piringan dan rekaman; digabung dengan area pemutar rekaman."),
}

# Every family with >=20 unique parents that is not retained has an explicit
# image-first rejection reason.  Lower-support hypotheses remain in companion
# data, but are not used for multipart claims.
REJECTED_HIGH_SUPPORT = {
    2: ("scene_fragment", "Potongan tumpukan e-waste dan perangkat campuran, bukan bagian yang koheren."),
    4: ("incoherent_local_texture", "Crop kain, perangkat, dan permukaan adegan bercampur."),
    8: ("generic_surface", "Permukaan gelap/abu dan tepi perangkat bercampur tanpa struktur khusus."),
    12: ("background_or_frame", "Area putih, lantai, dan bingkai kecil mendominasi."),
    14: ("scene_fragment", "Potongan kamera, pencetak, tumpukan, dan meja bercampur."),
    15: ("incoherent_local_texture", "Tepi speaker, layar, dan furnitur bercampur."),
    17: ("incoherent_local_texture", "Kabel, sisi mouse, dan tepi layar tidak membentuk satu struktur."),
    18: ("generic_device_edge", "Tepi bawah/samping beberapa pencetak dan perangkat tidak cukup spesifik."),
    19: ("generic_appliance_surface", "Permukaan samping terang beberapa peralatan bercampur."),
    25: ("scene_or_packaging", "Kemasan, tumpukan, dan permukaan adegan mendominasi."),
    28: ("generic_label_or_top_patch", "Crop persegi panjang menyatukan label/permukaan atas baterai, audio, pencetak, dan layar; tidak sah sebagai keluarga baterai."),
    29: ("mixed_accessories", "Remote, kabel, mouse, dan aksesori lain bercampur."),
    31: ("incoherent_internal_fragment", "Fragmen pembongkaran laptop, slot, dan tumpukan tidak koheren."),
    35: ("scene_or_hand", "Tangan, toko, dan konteks ponsel mendominasi."),
    38: ("incoherent_small_mark", "Label, ujung baterai, jam, dan panel listrik bercampur."),
    42: ("generic_appliance_surface", "Panel samping terang dari beberapa jenis peralatan bercampur."),
    47: ("outdoor_scene_fragment", "Tanah, tumbuhan, layar, dan objek terbakar bercampur."),
    50: ("ground_or_base", "Tanah/basis di bawah perangkat, bukan bagian elektronik."),
}

LOW_SUPPORT_NAMES = {
    51: ("wall_switch_or_socket", "saklar atau soket dinding"),
    54: ("mouse_control_shell", "cangkang dan kontrol mouse"),
    55: ("mouse_control_element", "elemen kontrol mouse"),
    56: ("crt_display_front", "muka layar tabung"),
    57: ("mouse_control_shell", "cangkang dan kontrol mouse"),
    59: ("key_matrix", "matriks tombol"),
    60: ("microwave_door_or_cavity", "pintu atau rongga microwave"),
    62: ("key_matrix", "matriks tombol"),
    63: ("circuit_board_surface", "permukaan papan sirkuit"),
    64: ("visible_display_damage", "kerusakan tampak pada layar"),
    69: ("key_matrix", "matriks tombol"),
    70: ("gramophone_horn", "corong gramofon"),
}


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


def finite(value, default=-1.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--global-assignments", type=Path, required=True)
    parser.add_argument("--review-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    families = read_csv(args.families)
    assignments = read_csv(args.assignments)
    global_rows = read_csv(args.global_assignments)
    review_summary = json.loads(args.review_summary.read_text(encoding="utf-8"))
    assert review_summary["families_rendered"] == len(families) == 129
    assert review_summary["sheets"] == 22
    family_ids = {int(row["family_internal"]) for row in families}
    assert set(RETAINED).issubset(family_ids)
    assert set(REJECTED_HIGH_SUPPORT).issubset(family_ids)

    global_community = {
        row["canonical_id"]: int(row["community_leiden_fused"]) for row in global_rows
    }
    assert len(global_community) == 3931
    by_family = defaultdict(list)
    for row in assignments:
        family = int(row["main_family_internal"])
        if family >= 0:
            by_family[family].append(row)

    adjudication = []
    for row in sorted(families, key=lambda item: int(item["family_internal"])):
        family = int(row["family_internal"])
        parents = int(row["unique_parents"])
        if family in RETAINED:
            key, name, role, note = RETAINED[family]
            decision = "retained"
            rejection = ""
            include = 1
        elif family in REJECTED_HIGH_SUPPORT:
            rejection, note = REJECTED_HIGH_SUPPORT[family]
            key, name, role = "", "", "not_retained"
            decision = "rejected_visual_review"
            include = 0
        else:
            key, name = LOW_SUPPORT_NAMES.get(family, ("", ""))
            role = "low_support_hypothesis"
            decision = "not_retained_low_support"
            rejection = "support_below_20_unique_parents"
            note = (
                "Visual hypothesis remains in companion data; insufficient unique-parent "
                "support for configuration or flagship claims."
            )
            include = 0
        if parents >= MIN_CONFIGURATION_PARENTS:
            assert family in RETAINED or family in REJECTED_HIGH_SUPPORT
        adjudication.append({
            "family_internal": family,
            "descriptive_family_key": key,
            "descriptive_name_id": name,
            "scientific_role": role,
            "decision": decision,
            "include_in_configuration": include,
            "rejection_reason": rejection,
            "review_note": note,
            "regions": int(row["regions"]),
            "unique_parents": parents,
            "median_within_family_dino_similarity": row["median_within_family_dino_similarity"],
            "reviewer": REVIEWER,
            "review_date": REVIEW_DATE,
            "evidence_status": "agent_evaluated_offline_from_original_photo_context_and_crop",
        })

    adjudication_path = args.output / "family_adjudication.csv"
    with adjudication_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(adjudication[0]))
        writer.writeheader()
        writer.writerows(adjudication)

    retained_by_family = {
        int(row["family_internal"]): row for row in adjudication
        if row["include_in_configuration"] == 1
    }
    photo_concepts = defaultdict(dict)
    for row in assignments:
        family = int(row["main_family_internal"])
        mapping = retained_by_family.get(family)
        if mapping is None:
            continue
        parent = row["parent_id"]
        key = mapping["descriptive_family_key"]
        affinity = finite(row["median_within_family_dino_similarity"])
        family_median = finite(mapping["median_within_family_dino_similarity"])
        is_core = int(affinity >= family_median)
        evidence = {
            "parent_id": parent,
            "descriptive_family_key": key,
            "descriptive_name_id": mapping["descriptive_name_id"],
            "scientific_role": mapping["scientific_role"],
            "source_family_internal": family,
            "support_region_id": row["region_id"],
            "support_feature_row": int(row["feature_row"]),
            "median_neighbor_affinity": affinity,
            "family_median_neighbor_affinity": family_median,
            "configuration_core_support": is_core,
            "weighted_degree": finite(row["weighted_degree"]),
            "support_status": (
                "core_positive_support_at_or_above_family_median_affinity"
                if is_core else "peripheral_positive_support_below_family_median_affinity"
            ),
        }
        prior = photo_concepts[parent].get(key)
        if prior is None or (
            evidence["configuration_core_support"], evidence["median_neighbor_affinity"],
            evidence["weighted_degree"]
        ) > (
            prior["configuration_core_support"], prior["median_neighbor_affinity"],
            prior["weighted_degree"]
        ):
            photo_concepts[parent][key] = evidence

    support_path = args.output / "photo_concept_support.csv.gz"
    support_rows = [
        evidence
        for parent in sorted(photo_concepts)
        for evidence in sorted(photo_concepts[parent].values(), key=lambda item: item["descriptive_family_key"])
    ]
    with gzip.open(support_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(support_rows[0]))
        writer.writeheader()
        writer.writerows(support_rows)

    concept_to_families = defaultdict(list)
    for family, row in retained_by_family.items():
        concept_to_families[row["descriptive_family_key"]].append(family)
    concept_summary = []
    for key, internal_families in sorted(concept_to_families.items()):
        supports = [row for row in support_rows if row["descriptive_family_key"] == key]
        parent_ids = {row["parent_id"] for row in supports}
        community_counts = Counter(global_community[parent] for parent in parent_ids)
        total = sum(community_counts.values())
        probabilities = [count / total for count in community_counts.values()]
        entropy = -sum(value * math.log(value) for value in probabilities)
        normalized_entropy = entropy / math.log(len(community_counts)) if len(community_counts) > 1 else 0.0
        exemplar = retained_by_family[internal_families[0]]
        concept_summary.append({
            "descriptive_family_key": key,
            "descriptive_name_id": exemplar["descriptive_name_id"],
            "scientific_role": exemplar["scientific_role"],
            "source_internal_families": json.dumps(sorted(internal_families), separators=(",", ":")),
            "unique_parents": len(parent_ids),
            "global_communities_supported": len(community_counts),
            "largest_global_community_share": max(community_counts.values()) / total,
            "normalized_global_community_entropy": normalized_entropy,
            "global_community_counts": json.dumps(dict(sorted(community_counts.items())), separators=(",", ":")),
        })

    concept_path = args.output / "conceptual_family_summary.csv"
    with concept_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(concept_summary[0]))
        writer.writeheader()
        writer.writerows(concept_summary)

    core_photo_concepts = {
        parent: {key: value for key, value in concepts.items()
                 if value["configuration_core_support"] == 1}
        for parent, concepts in photo_concepts.items()
    }
    core_photo_concepts = {parent: value for parent, value in core_photo_concepts.items() if value}
    per_photo_counts = Counter(len(value) for value in photo_concepts.values())
    core_per_photo_counts = Counter(len(value) for value in core_photo_concepts.values())
    metrics = {
        "status": "complete_post_discovery_agent_visual_adjudication",
        "review_scope": {
            "families": len(families),
            "sheets": review_summary["sheets"],
            "examples": review_summary["review_examples"],
            "evidence": "original-photo context plus exact region crop",
            "reviewer": REVIEWER,
            "date": REVIEW_DATE,
        },
        "minimum_unique_parents_for_configuration": MIN_CONFIGURATION_PARENTS,
        "high_support_families": sum(int(row["unique_parents"]) >= MIN_CONFIGURATION_PARENTS for row in adjudication),
        "retained_internal_families": len(retained_by_family),
        "rejected_high_support_families": len(REJECTED_HIGH_SUPPORT),
        "conceptual_visible_families_after_collapse": len(concept_summary),
        "positive_photo_concept_supports": len(support_rows),
        "core_photo_concept_supports": sum(len(value) for value in core_photo_concepts.values()),
        "canonical_photos_with_retained_support": len(photo_concepts),
        "canonical_photos_without_retained_support": 3931 - len(photo_concepts),
        "photos_by_distinct_retained_concept_count": dict(sorted(per_photo_counts.items())),
        "canonical_photos_with_core_support": len(core_photo_concepts),
        "photos_by_distinct_core_concept_count": dict(sorted(core_per_photo_counts.items())),
        "cross_category_concepts": sum(row["scientific_role"].startswith("cross_category") for row in concept_summary),
        "within_product_configuration_concepts": sum(row["scientific_role"] == "within_product_configuration_part" for row in concept_summary),
        "interpretation_limit": (
            "Names are post-discovery interpretations from an offline agent visual review. "
            "Core is an at-or-above-family-median affinity sensitivity definition, not calibrated confidence. "
            "Support means visible resemblance to a retained family, not component identity, "
            "physical ownership, quantity, hidden content, value, toxicity, or safety."
        ),
        "inputs": {
            "families_sha256": digest(args.families),
            "assignments_sha256": digest(args.assignments),
            "global_assignments_sha256": digest(args.global_assignments),
            "review_summary_sha256": digest(args.review_summary),
        },
        "outputs": {},
    }
    for path in (adjudication_path, support_path, concept_path):
        metrics["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
