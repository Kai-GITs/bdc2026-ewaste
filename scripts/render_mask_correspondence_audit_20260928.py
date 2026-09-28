"""Render actual SAM masks for a targeted proposal/correspondence audit.

No proposal or embedding is recomputed.  The renderer decodes the immutable
mask RLE and places the original image with a translucent mask, the isolated
mask crop, and the box-only crop side by side for the same region.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


TARGET_FAMILIES = [41, 34, 52, 64, 13, 0, 119, 124, 94, 99, 116, 122]
HARD_NEGATIVE_FAMILIES = [28, 32, 37, 73]
TEAL = (0, 136, 145)


def font(size: int, bold: bool = False):
    for path in (
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def decode_mask(row: dict) -> np.ndarray:
    counts = np.frombuffer(
        zlib.decompress(base64.b64decode(row["mask_rle_zlib_base64"])), dtype="<u4"
    )
    if len(counts) != int(row["mask_rle_count_length"]):
        raise ValueError("RLE count length mismatch")
    values = np.arange(len(counts), dtype=np.uint8) % 2
    flat = np.repeat(values, counts.astype(np.int64))
    height, width = map(int, row["mask_shape_hw"])
    if len(flat) != height * width:
        raise ValueError("Decoded mask size mismatch")
    mask = flat.reshape(height, width)
    hasher = hashlib.sha256()
    hasher.update(struct.pack("<II", height, width))
    hasher.update(np.packbits(mask, bitorder="little").tobytes())
    encoded = hasher.hexdigest()
    if encoded != row["mask_sha256"]:
        raise ValueError("Decoded mask hash mismatch")
    return mask


def fit(image: Image.Image, size: tuple[int, int], fill=(246, 247, 244)):
    canvas = Image.new("RGB", size, fill)
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    left = (size[0] - copy.width) // 2
    top = (size[1] - copy.height) // 2
    canvas.paste(copy, (left, top))
    return canvas, (left, top, copy.width, copy.height)


def original_with_mask(original: Image.Image, mask: Image.Image) -> Image.Image:
    rgba = original.convert("RGBA")
    overlay = Image.new("RGBA", original.size, (0, 0, 0, 0))
    color = Image.new("RGBA", original.size, (*TEAL, 118))
    overlay.paste(color, (0, 0), mask)
    composed = Image.alpha_composite(rgba, overlay).convert("RGB")
    # Draw the actual mask contour rather than only its bounding rectangle.
    mask_array = np.asarray(mask, dtype=np.uint8) > 0
    edge = mask_array ^ (
        np.pad(mask_array[1:, :], ((0, 1), (0, 0)), constant_values=False)
        & np.pad(mask_array[:-1, :], ((1, 0), (0, 0)), constant_values=False)
        & np.pad(mask_array[:, 1:], ((0, 0), (0, 1)), constant_values=False)
        & np.pad(mask_array[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    )
    edge_image = Image.fromarray((edge * 255).astype(np.uint8), mode="L")
    contour = Image.new("RGB", original.size, TEAL)
    composed.paste(contour, (0, 0), edge_image)
    return composed


def masked_crop(original: Image.Image, mask: Image.Image, box: list[float]) -> Image.Image:
    xyxy = tuple(int(round(value)) for value in box)
    crop = original.crop(xyxy).convert("RGBA")
    mask_crop = mask.crop(xyxy)
    neutral = Image.new("RGBA", crop.size, (235, 238, 234, 255))
    neutral.paste(crop, (0, 0), mask_crop)
    return neutral.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-family", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    review = pd.read_csv(args.review_manifest)
    manual = pd.read_csv(args.manual_review).set_index("family_internal")
    source_manifest_raw = json.loads(args.image_manifest.read_text(encoding="utf-8"))
    source_manifest = {}
    for row in source_manifest_raw:
        source_manifest.setdefault(row["exact_duplicate_canonical_id"], row)

    ordered = TARGET_FAMILIES + HARD_NEGATIVE_FAMILIES
    selected = []
    for family in ordered:
        rows = review[review.family_internal == family].copy()
        role_order = {"medoid": 0, "variasi posisi": 1, "batas afinitas": 2,
                      "bagian besar": 3, "bagian kecil": 4, "anggota tambahan": 5}
        rows["role_order"] = rows.role.map(role_order).fillna(9)
        rows = rows.sort_values(["role_order", "parent_id"]).drop_duplicates("parent_id")
        rows = rows.head(args.examples_per_family)
        selected.extend(rows.to_dict("records"))
    selected_ids = {row["region_id"] for row in selected}

    proposals = {}
    with gzip.open(args.proposals, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["region_id"] in selected_ids:
                proposals[row["region_id"]] = row
    if set(proposals) != selected_ids:
        missing = sorted(selected_ids - set(proposals))
        raise ValueError(f"Missing proposals: {missing[:5]}")

    rows_by_family = {}
    for row in selected:
        rows_by_family.setdefault(int(row["family_internal"]), []).append(row)

    width, header, family_height = 1800, 88, 420
    families_per_sheet = 2
    evidence_rows = []
    sheet_paths = []
    for start in range(0, len(ordered), families_per_sheet):
        family_ids = ordered[start:start + families_per_sheet]
        canvas = Image.new("RGB", (width, header + family_height * len(family_ids)), (249, 248, 244))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 13), "Audit mekanisme: foto + mask aktual + crop", font=font(24, True), fill=(18, 38, 45))
        draw.text((24, 49), "Overlay hijau adalah mask SAM tersimpan; panel terisolasi mempertahankan piksel di dalam mask.",
                  font=font(15), fill=(65, 78, 82))
        for local, family in enumerate(family_ids):
            y = header + local * family_height
            draw.rectangle((0, y, width - 1, y + family_height - 1), outline=(207, 211, 208), width=1)
            family_type = "kandidat" if family in TARGET_FAMILIES else "hard negative"
            name = manual.loc[family, "visual_name_id"]
            draw.text((18, y + 10), f"{name}  | {family_type}", font=font(18, True), fill=(20, 45, 49))
            for column, item in enumerate(rows_by_family[family]):
                proposal = proposals[item["region_id"]]
                source_row = source_manifest[item["parent_id"]]
                source = args.raw_root / source_row["source_relative_path"]
                if not source.is_file() or digest(source) != source_row["source_sha256"]:
                    raise ValueError(f"Source verification failed: {source}")
                with Image.open(source) as opened:
                    original = ImageOps.exif_transpose(opened).convert("RGB")
                mask_array = decode_mask(proposal)
                mask_processing = Image.fromarray((mask_array * 255).astype(np.uint8), mode="L")
                mask_native = mask_processing.resize(original.size, Image.Resampling.NEAREST)
                overlay = original_with_mask(original, mask_native)
                box = [float(value) for value in proposal["mask_bbox_xyxy_native"]]
                isolated = masked_crop(original, mask_native, box)
                box_crop = original.crop(tuple(int(round(value)) for value in box)).convert("RGB")

                x = 18 + column * 585
                ov, _ = fit(overlay, (270, 270))
                iso, _ = fit(isolated, (135, 270))
                bx, _ = fit(box_crop, (135, 270))
                canvas.paste(ov, (x, y + 55))
                canvas.paste(iso, (x + 278, y + 55))
                canvas.paste(bx, (x + 421, y + 55))
                draw.text((x, y + 332), "foto + mask", font=font(12, True), fill=(35, 59, 63))
                draw.text((x + 278, y + 332), "mask crop", font=font(12, True), fill=(35, 59, 63))
                draw.text((x + 421, y + 332), "box crop", font=font(12, True), fill=(35, 59, 63))
                draw.text((x, y + 356), str(item["role"]), font=font(12), fill=(70, 82, 85))

                evidence_rows.append({
                    "sheet": start // families_per_sheet + 1,
                    "family_internal": family,
                    "machine_family_id": f"F{family:04d}",
                    "visual_name_id": name,
                    "audit_group": family_type,
                    "role": item["role"],
                    "parent_id": item["parent_id"],
                    "region_id": item["region_id"],
                    "source_relative_path": source_row["source_relative_path"],
                    "source_sha256": source_row["source_sha256"],
                    "mask_sha256": proposal["mask_sha256"],
                    "mask_area_fraction": proposal["area_fraction"],
                    "sam_score": proposal["score"],
                    "proposal_alignment": "pending_visual_adjudication",
                    "semantic_correspondence": "pending_visual_adjudication",
                    "failure_mode": "pending_visual_adjudication",
                })
        path = args.output / f"mask_correspondence_audit_{start // families_per_sheet + 1:02d}.jpg"
        canvas.save(path, quality=94, subsampling=0)
        sheet_paths.append(path)

    manifest_path = args.output / "audit_manifest.csv"
    pd.DataFrame(evidence_rows).to_csv(manifest_path, index=False)
    summary = {
        "status": "rendered_targeted_mask_correspondence_audit",
        "candidate_families": TARGET_FAMILIES,
        "hard_negative_families": HARD_NEGATIVE_FAMILIES,
        "examples": len(evidence_rows),
        "sheets": len(sheet_paths),
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)} for path in sheet_paths},
        "manifest": {"bytes": manifest_path.stat().st_size, "sha256": digest(manifest_path)},
        "claim_limit": "The render enables proposal and correspondence adjudication; it does not itself verify the semantic family name.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
