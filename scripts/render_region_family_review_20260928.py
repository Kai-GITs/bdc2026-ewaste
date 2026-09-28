"""Render every learned region family with original-photo context and crops.

The sheets are diagnostic evidence for naming and rejection.  They include the
graph medoid, strong members, geometric variation, and low-affinity boundary
members from distinct parents whenever possible.  Internal family identifiers
remain confined to these review artifacts and machine-readable tables.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def finite(value, default=-1.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def choose_members(members, medoid_row, count=6):
    by_row = {int(row["feature_row"]): row for row in members}
    medoid = by_row[int(medoid_row)]
    candidates = [
        ("medoid", medoid),
        ("dukungan kuat", max(members, key=lambda row: finite(row["weighted_degree"]))),
        ("bagian kecil", min(members, key=lambda row: finite(row["area_fraction"], 2.0))),
        ("bagian besar", max(members, key=lambda row: finite(row["area_fraction"]))),
        ("variasi posisi", max(members, key=lambda row: (
            (finite(row["centroid_x"]) - finite(medoid["centroid_x"])) ** 2 +
            (finite(row["centroid_y"]) - finite(medoid["centroid_y"])) ** 2))),
        ("batas afinitas", min(
            [row for row in members if row["median_within_family_dino_similarity"]]
            or members,
            key=lambda row: finite(row["median_within_family_dino_similarity"], 2.0))),
    ]
    chosen, used_rows, used_parents = [], set(), set()
    for role, row in candidates:
        index = int(row["feature_row"])
        if index in used_rows:
            continue
        if row["parent_id"] in used_parents and len({item["parent_id"] for item in members}) >= count:
            alternatives = sorted(
                [item for item in members if item["parent_id"] not in used_parents and
                 int(item["feature_row"]) not in used_rows],
                key=lambda item: -finite(item["weighted_degree"]))
            if alternatives:
                row = alternatives[0]
                index = int(row["feature_row"])
        chosen.append((role, row))
        used_rows.add(index)
        used_parents.add(row["parent_id"])
    for row in sorted(members, key=lambda item: -finite(item["weighted_degree"])):
        if len(chosen) >= count:
            break
        index = int(row["feature_row"])
        if index not in used_rows and row["parent_id"] not in used_parents:
            chosen.append(("anggota tambahan", row))
            used_rows.add(index)
            used_parents.add(row["parent_id"])
    return chosen[:count]


def fit(image, size, fill=(245, 245, 242)):
    canvas = Image.new("RGB", size, fill)
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    left = (size[0] - copy.width) // 2
    top = (size[1] - copy.height) // 2
    canvas.paste(copy, (left, top))
    return canvas, (left, top, copy.width, copy.height)


def cell(image, box, title, size=(285, 230)):
    top_height = 130
    output = Image.new("RGB", size, "white")
    context, geometry = fit(image, (size[0], top_height))
    left, top, width, height = geometry
    sx, sy = width / image.width, height / image.height
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(context)
    draw.rectangle((left + x0 * sx, top + y0 * sy,
                    left + x1 * sx, top + y1 * sy), outline=(0, 132, 145), width=3)
    output.paste(context, (0, 0))
    crop = image.crop((max(0, math.floor(x0)), max(0, math.floor(y0)),
                       min(image.width, math.ceil(x1)), min(image.height, math.ceil(y1))))
    crop_view, _ = fit(crop, (size[0], 72), fill=(236, 242, 242))
    output.paste(crop_view, (0, top_height))
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, top_height, size[0] - 1, top_height + 71), outline=(0, 132, 145), width=1)
    draw.text((7, 207), title[:30], fill=(25, 37, 43), font=font(15, bold=True))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--family-ranking", type=Path,
        help="Optional stability CSV; when set, render review_priority=1 families in its row order.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    assignments = rows(args.assignments)
    family_summary = rows(args.families)
    region_rows = jsonl(args.regions)
    region_by_row = {int(row["feature_row"]): row for row in region_rows}
    assignment_rows = [int(row["feature_row"]) for row in assignments]
    assert len(assignment_rows) == len(set(assignment_rows))
    assert all(index in region_by_row for index in assignment_rows)
    manifest = json.loads(args.image_manifest.read_text(encoding="utf-8"))
    path_by_sha = {}
    for item in manifest:
        path_by_sha.setdefault(item["source_sha256"], args.raw_root / item["source_relative_path"])

    by_family = defaultdict(list)
    for row in assignments:
        family = int(row["main_family_internal"])
        if family >= 0:
            by_family[family].append(row)
    summary_by_family = {int(row["family_internal"]): row for row in family_summary}
    assert set(by_family) == set(summary_by_family)
    order = sorted(by_family, key=lambda family: (
        -int(summary_by_family[family]["unique_parents"]),
        -finite(summary_by_family[family]["median_within_family_dino_similarity"]),
        family))
    if args.family_ranking:
        ranking = rows(args.family_ranking)
        order = [
            int(row["family_internal"]) for row in ranking
            if row.get("review_priority") == "1" and int(row["family_internal"]) in by_family
        ]
        assert len(order) == len(set(order)) and order

    sheet_width = 1800
    header = 82
    row_height = 276
    families_per_sheet = 6
    manifest_rows = []
    output_files = []
    for page_start in range(0, len(order), families_per_sheet):
        page_families = order[page_start:page_start + families_per_sheet]
        sheet = Image.new("RGB", (sheet_width, header + row_height * len(page_families)), (248, 247, 243))
        draw = ImageDraw.Draw(sheet)
        draw.text((24, 18), "Audit visual keluarga region: konteks foto asli + crop box mask",
                  fill=(18, 32, 38), font=font(20, bold=True))
        draw.text((24, 42), "Medoid, variasi, dan batas dipakai untuk menerima/menolak hipotesis; mask bukan identitas komponen.",
                  fill=(64, 76, 80), font=font(15))
        for local_row, family in enumerate(page_families):
            y = header + local_row * row_height
            info = summary_by_family[family]
            draw.rectangle((0, y, sheet_width - 1, y + row_height - 1), outline=(210, 214, 211), width=1)
            label = (f"F{family:04d} | {info['regions']} region | "
                     f"{info['unique_parents']} foto | sim median "
                     f"{finite(info['median_within_family_dino_similarity']):.3f}")
            draw.text((14, y + 8), label, fill=(20, 45, 49), font=font(16, bold=True))
            selected = choose_members(by_family[family], info["medoid_feature_row"])
            for column, (role, row) in enumerate(selected):
                feature_row = int(row["feature_row"])
                region = region_by_row[feature_row]
                source = path_by_sha[row["source_sha256"]]
                assert source.is_file() and digest(source) == row["source_sha256"]
                with Image.open(source) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                box = tuple(float(value) for value in region["mask_bbox_xyxy_native"])
                view = cell(image, box, role)
                sheet.paste(view, (14 + column * 296, y + 36))
                manifest_rows.append({
                    "sheet": page_start // families_per_sheet + 1,
                    "family_internal": family,
                    "role": role,
                    "feature_row": feature_row,
                    "region_id": row["region_id"],
                    "parent_id": row["parent_id"],
                    "source_relative_path": str(source.relative_to(args.raw_root)).replace("\\", "/"),
                    "source_sha256": row["source_sha256"],
                    "mask_bbox_xyxy_native": json.dumps(box),
                    "resolution_status": row["resolution_status"],
                })
        name = f"family_review_{page_start // families_per_sheet + 1:03d}.jpg"
        sheet.save(args.output / name, quality=92, subsampling=0)
        output_files.append(name)

    with (args.output / "review_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    report = {
        "status": "rendered_for_visual_adjudication",
        "families_rendered": len(order),
        "sheets": len(output_files),
        "review_examples": len(manifest_rows),
        "ordering": "unique parent support descending, then median within-family DINO similarity",
        "selection": "medoid, high support, area extremes, centroid variation and low-affinity boundary with distinct parents when available",
        "claim_limit": "These sheets support later human/agent interpretation; they do not turn masks or family IDs into component identities.",
        "files": {name: {"bytes": (args.output / name).stat().st_size,
                          "sha256": digest(args.output / name)} for name in output_files},
    }
    if args.family_ranking:
        report["ordering"] = "review_priority=1 in stability-screen rank order"
        report["family_ranking_sha256"] = digest(args.family_ranking)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
