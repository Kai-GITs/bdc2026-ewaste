"""Verify and exercise the official SAM 3 static-image path on BDC exemplars.

This is a bounded engineering gate.  The prompts are class-free multiscale boxes;
the resulting masks are geometry candidates and never component identities.
"""

import contextlib
import csv
import gc
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

os.environ["HF_HOME"] = "/research-cache/hf"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from torch._subclasses.fake_tensor import FakeTensorMode


START = time.monotonic()
TOKEN = os.environ["HF_TOKEN"]
PRIVATE_REPO = "Fin000/codex-research-workspace"
PREFIX = "bdc2026-ewaste/final-study-20260928/sam3-image-compat-r1"

SOURCE_LOCAL = os.environ.get("BDC_SOURCE_ZIP")
SOURCE_REPO = os.environ.get("BDC_SOURCE_DATASET_REPO")
SOURCE_FILE = os.environ.get("BDC_SOURCE_DATASET_FILE", "bdc-official-2026.zip")
SOURCE_REV = os.environ.get("BDC_SOURCE_DATASET_REVISION")
SOURCE_BYTES = 1368394793
SOURCE_MANIFEST_SHA = "697f8ef83052c860782263c7218b0a9bdf7a1611586734b21ed7846f032e5a81"
ROI_PATH = "bdc2026-ewaste/track-a/runs/20260924-054638-189f34ad/roi_index.csv"
ROI_SHA = "d4d0b6f67ce9821f38fa3f02ccd7251f09b71a17059882257ab6d720408ff0df"

STATIC_MIRROR = "cubert-gmbh/sam3"
STATIC_REV = "6d25af14a085ff9d3e1342c35bae7c87de4811f4"
STATIC_FILE = "sam3.pt"
STATIC_BYTES = 3450062241
STATIC_SHA = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
VIDEO_MIRROR = "AEmotionStudio/sam3.1"
VIDEO_REV = "694239a1479aab8fd1317c87c433c58acd7c6eab"
VIDEO_FILE = "sam3.1_multiplex.pt"
VIDEO_SHA = "0567c0d8ca9caf686ec36fecbe1a7d7b5ec71e2305cfbbd9624a851c63e0fe71"

CODE_COMMIT = "2345a4ad109ac29c569da749c91d84f10dc08c40"
CODE_URL = f"https://github.com/facebookresearch/sam3/archive/{CODE_COMMIT}.zip"
CODE_SENTINELS = {
    "sam3/model_builder.py": "d71d6d3e485ec3eae48bbc2ba676f401b5853d65c4195a91d077b04da38121c2",
    "sam3/model/sam3_image_processor.py": "d8738a0efb6138b01c0dc5deceffd29de9e675860a9a1ed3822b08766373333b",
    "LICENSE": "4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be",
    "pyproject.toml": "255f5d8d1db011459878e3de296a6afc84e03a8dda2bfa756a4de304aaff6366",
}

EXEMPLARS = [
    ("fbb7c5787a6a1641", "sel baterai lepas"),
    ("c31ee235e538f8f4", "ponsel tertutup"),
    ("962ccd67e797f53d", "laptop terbuka"),
    ("49fc9b181dbb0307", "adegan reparasi campuran"),
    ("a6b052df8e569bd6", "mesin cuci"),
]
OUT = Path("/research-cache/bdc2026/final-study-20260928/sam3-image-compat-r1")
OUT.mkdir(parents=True, exist_ok=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalized_license(path):
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def electronic_relative(name):
    normalized = name.replace("\\", "/")
    marker = "train/1_Electronic/"
    position = normalized.find(marker)
    return None if position < 0 else normalized[position + len("train/"):]


def prepare_official_source():
    cache = Path("/research-cache/source")
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"sam3-{CODE_COMMIT}.zip"
    source = cache / f"sam3-{CODE_COMMIT}"

    def source_valid():
        return source.is_dir() and all(
            (source / name).is_file() and digest(source / name) == expected
            for name, expected in CODE_SENTINELS.items()
        )

    if not source_valid():
        if source.exists():
            shutil.rmtree(source)
        if not archive.exists() or not zipfile.is_zipfile(archive):
            tmp = archive.with_suffix(".download")
            if tmp.exists():
                tmp.unlink()
            urllib.request.urlretrieve(CODE_URL, tmp)
            tmp.replace(archive)
        with zipfile.ZipFile(archive) as zf:
            assert zf.testzip() is None
            top = zf.namelist()[0].split("/")[0]
            stage = cache / (top + "-extracting")
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir()
            zf.extractall(stage)
            extracted = stage / top
            extracted.replace(source)
            stage.rmdir()
    assert source_valid()
    return source, archive


source_root, source_archive = prepare_official_source()
sys.path.insert(0, str(source_root))
from sam3 import __version__ as sam3_version
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

api = HfApi(token=TOKEN)
assert api.repo_info(PRIVATE_REPO, repo_type="dataset").private
checkpoint = Path(hf_hub_download(
    STATIC_MIRROR, STATIC_FILE, revision=STATIC_REV, token=False))
assert checkpoint.stat().st_size == STATIC_BYTES
assert digest(checkpoint) == STATIC_SHA

static_license = Path(hf_hub_download(
    STATIC_MIRROR, "LICENSE", revision=STATIC_REV, token=False))
video_license = Path(hf_hub_download(
    VIDEO_MIRROR, "LICENSE", revision=VIDEO_REV, token=False))
license_text = normalized_license(source_root / "LICENSE")
assert normalized_license(static_license) == license_text
assert normalized_license(video_license) == license_text

# Inspect state-dict keys without allocating checkpoint tensors.  Loading for
# inference remains the official builder's responsibility below.
with FakeTensorMode():
    checkpoint_meta = torch.load(
        checkpoint, map_location="cpu", weights_only=True, mmap=True)
if "model" in checkpoint_meta and isinstance(checkpoint_meta["model"], dict):
    checkpoint_meta = checkpoint_meta["model"]
raw_keys = tuple(checkpoint_meta.keys())
prefix_counts = {
    "detector": sum("detector" in key for key in raw_keys),
    "tracker": sum("tracker" in key for key in raw_keys),
    "other": sum("detector" not in key and "tracker" not in key for key in raw_keys),
}
del checkpoint_meta
gc.collect()

assert torch.cuda.is_available()
load_capture = io.StringIO()
with contextlib.redirect_stdout(load_capture):
    model = build_sam3_image_model(
        device="cuda", eval_mode=True, checkpoint_path=str(checkpoint),
        load_from_HF=False, enable_segmentation=True,
        enable_inst_interactivity=True, compile=False)
load_log = load_capture.getvalue().strip()

transformed = {
    key.replace("detector.", "")
    for key in raw_keys if "detector" in key
}
transformed.update({
    key.replace("tracker.", "inst_interactive_predictor.model.")
    for key in raw_keys if "tracker" in key
})
model_keys = set(model.state_dict())
missing_keys = sorted(model_keys - transformed)
unexpected_keys = sorted(transformed - model_keys)
assert not missing_keys, missing_keys[:20]
assert not unexpected_keys, unexpected_keys[:20]
assert "missing_keys=" not in load_log
processor = Sam3Processor(model, device="cuda")

if SOURCE_LOCAL:
    dataset_zip = Path(SOURCE_LOCAL)
else:
    if not SOURCE_REPO:
        raise RuntimeError("Set BDC_SOURCE_ZIP or BDC_SOURCE_DATASET_REPO.")
    dataset_zip = Path(hf_hub_download(
        SOURCE_REPO, SOURCE_FILE, repo_type="dataset", revision=SOURCE_REV,
        token=os.environ.get("BDC_SOURCE_DATASET_TOKEN") or False))
assert dataset_zip.stat().st_size == SOURCE_BYTES
manifest_path = Path("/research-cache/bdc2026/source-verify/electronic_image_hashes.csv")
if not manifest_path.exists():
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dataset_zip) as archive:
        members = [(m, electronic_relative(m.filename)) for m in archive.infolist()
                   if not m.is_dir()]
        members = [(m, relative) for m, relative in members if relative is not None]
        assert len(members) == 3961
        rows = []
        for member, relative in members:
            h = hashlib.sha256()
            with archive.open(member) as reader:
                for block in iter(lambda: reader.read(1 << 20), b""):
                    h.update(block)
            rows.append((relative, member.file_size, h.hexdigest(), member.filename))
        rows.sort(key=lambda row: row[0])
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "bytes", "sha256", "zip_member"))
        writer.writerows(rows)
assert digest(manifest_path) == SOURCE_MANIFEST_SHA
with manifest_path.open(encoding="utf-8", newline="") as handle:
    source_by_sha = {row["sha256"]: row for row in csv.DictReader(handle)}
assert len(source_by_sha) == 3931

roi_path = Path(hf_hub_download(
    PRIVATE_REPO, ROI_PATH, repo_type="dataset", token=TOKEN))
assert digest(roi_path) == ROI_SHA
canonical_to_sha = {}
with roi_path.open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        prior = canonical_to_sha.setdefault(row["canonical_id"], row["source_sha256"])
        assert prior == row["source_sha256"]
target_ids = [ident for ident, _ in EXEMPLARS]
assert all(ident in canonical_to_sha for ident in target_ids)


def grid_boxes(width, height):
    boxes, names = [], []
    for grid in (1, 2, 4):
        for row in range(grid):
            for col in range(grid):
                margin = 0.0 if grid == 1 else 0.125 / grid
                x0 = max(0.0, col / grid - margin)
                y0 = max(0.0, row / grid - margin)
                x1 = min(1.0, (col + 1) / grid + margin)
                y1 = min(1.0, (row + 1) / grid + margin)
                boxes.append([x0 * width, y0 * height, x1 * width, y1 * height])
                names.append("whole" if grid == 1 else f"g{grid}_r{row}_c{col}")
    return np.asarray(boxes, dtype=np.float32), names


def to_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def deduplicate(masks, records):
    valid = [i for i, row in enumerate(records)
             if 0.002 <= row["area_fraction"] <= 0.95 and row["bbox_min_side"] >= 12]
    ordered = sorted(valid, key=lambda i: (-records[i]["score"],
                                            -records[i]["area_fraction"], i))
    kept = []
    for idx in ordered:
        duplicate_of = None
        area_i = int(masks[idx].sum())
        for other in kept:
            intersection = int(np.logical_and(masks[idx], masks[other]).sum())
            area_j = int(masks[other].sum())
            union = area_i + area_j - intersection
            iou = intersection / max(union, 1)
            containment = intersection / max(min(area_i, area_j), 1)
            area_ratio = min(area_i, area_j) / max(area_i, area_j, 1)
            if iou >= 0.85 or (containment >= 0.98 and area_ratio >= 0.80):
                duplicate_of = other
                break
        records[idx]["duplicate_of_prompt"] = (
            records[duplicate_of]["prompt_name"] if duplicate_of is not None else "")
        records[idx]["kept"] = duplicate_of is None
        if duplicate_of is None:
            kept.append(idx)
    for idx, row in enumerate(records):
        row.setdefault("duplicate_of_prompt", "")
        row.setdefault("kept", False)
    return kept


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
          "#56B4E9", "#F0E442", "#6A3D9A")
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
FONT_BOLD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)


def make_panel(image, masks, records, kept, label):
    thumb = ImageOps.contain(image, (500, 270), Image.Resampling.LANCZOS)
    original = Image.new("RGB", (520, 285), "#f7f7f3")
    offset = ((520 - thumb.width) // 2, 5 + (270 - thumb.height) // 2)
    original.paste(thumb, offset)
    overlay = original.convert("RGBA")
    shown = sorted(kept, key=lambda i: records[i]["score"], reverse=True)[:8]
    for j, idx in enumerate(shown):
        small = Image.fromarray(masks[idx].astype(np.uint8) * 255).resize(
            thumb.size, Image.Resampling.NEAREST)
        alpha = small.point(lambda value: 42 if value else 0)
        fill = Image.new("RGBA", thumb.size, COLORS[j % len(COLORS)])
        fill.putalpha(alpha)
        layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        layer.alpha_composite(fill, offset)
        edge = np.asarray(small.filter(ImageFilter.MaxFilter(5)), dtype=np.int16) - np.asarray(
            small.filter(ImageFilter.MinFilter(5)), dtype=np.int16)
        edge_img = Image.fromarray(np.clip(edge, 0, 255).astype(np.uint8))
        outline = Image.new("RGBA", thumb.size, COLORS[j % len(COLORS)])
        outline.putalpha(edge_img)
        layer.alpha_composite(outline, offset)
        overlay = Image.alpha_composite(overlay, layer)
    panel = Image.new("RGB", (1060, 330), "white")
    panel.paste(original, (5, 38))
    panel.paste(overlay.convert("RGB"), (535, 38))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), label, font=FONT_BOLD, fill="#173642")
    draw.text((535, 9), f"{len(kept)} kandidat unik; {len(shown)} ditampilkan",
              font=FONT, fill="#334e58")
    draw.text((8, 309), "foto masukan", font=FONT, fill="#334e58")
    draw.text((535, 309), "overlay mask; warna tidak menyatakan identitas",
              font=FONT, fill="#334e58")
    return panel


all_rows, image_rows, panels = [], [], []
with zipfile.ZipFile(dataset_zip) as archive, torch.inference_mode():
    for canonical_id, label in EXEMPLARS:
        source_sha = canonical_to_sha[canonical_id]
        source_row = source_by_sha[source_sha]
        raw = archive.read(source_row["zip_member"])
        assert hashlib.sha256(raw).hexdigest() == source_sha
        with Image.open(io.BytesIO(raw)) as opened:
            native = ImageOps.exif_transpose(opened).convert("RGB")
        native_size = native.size
        scale = min(1.0, 1280 / max(native.size))
        if scale < 1:
            image = native.resize((max(1, round(native.width * scale)),
                                   max(1, round(native.height * scale))),
                                  Image.Resampling.LANCZOS)
        else:
            image = native
        boxes, prompt_names = grid_boxes(image.width, image.height)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            masks, scores, _ = model.predict_inst(
                state, point_coords=None, point_labels=None, box=boxes,
                multimask_output=False)
        masks = to_numpy(masks)
        scores = to_numpy(scores).reshape(-1)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        assert masks.ndim == 3 and masks.shape[0] == len(boxes)
        assert scores.shape == (len(boxes),)
        masks = masks.astype(bool)
        records = []
        for index, (mask, score, prompt_name, prompt_box) in enumerate(
                zip(masks, scores, prompt_names, boxes)):
            bbox = mask_bbox(mask)
            area = int(mask.sum())
            row = {
                "canonical_id": canonical_id,
                "source_sha256": source_sha,
                "prompt_index": index,
                "prompt_name": prompt_name,
                "prompt_box_xyxy_processing": [round(float(v), 2) for v in prompt_box],
                "score": round(float(score), 6),
                "mask_bbox_xyxy_processing": bbox,
                "bbox_min_side": min(bbox[2] - bbox[0], bbox[3] - bbox[1]),
                "area_pixels": area,
                "area_fraction": round(area / (image.width * image.height), 7),
                "kept": False,
                "duplicate_of_prompt": "",
            }
            records.append(row)
        kept = deduplicate(masks, records)
        non_global = [idx for idx in kept if .003 <= records[idx]["area_fraction"] <= .80]
        image_rows.append({
            "canonical_id": canonical_id,
            "description": label,
            "source_sha256": source_sha,
            "native_width": native_size[0],
            "native_height": native_size[1],
            "processing_width": image.width,
            "processing_height": image.height,
            "prompts": len(boxes),
            "unique_candidates": len(kept),
            "non_global_candidates": len(non_global),
            "max_score": round(float(scores.max()), 6),
        })
        panels.append(make_panel(
            image, masks, records, kept,
            f"{label} | {image.width}x{image.height} | {canonical_id}"))
        all_rows.extend(records)
        del state, masks, scores
        torch.cuda.empty_cache()
        print(json.dumps({"event": "segmented", "canonical_id": canonical_id,
                          "kept": len(kept), "non_global": len(non_global),
                          "elapsed_seconds": round(time.monotonic() - START, 1)}), flush=True)

sheet = Image.new("RGB", (1060, 330 * len(panels)), "white")
for index, panel in enumerate(panels):
    sheet.paste(panel, (0, index * 330))
sheet.save(OUT / "review.jpg", quality=92, subsampling=0)

with (OUT / "proposals.csv").open("w", encoding="utf-8", newline="") as handle:
    fields = list(all_rows[0])
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in all_rows:
        serial = dict(row)
        serial["prompt_box_xyxy_processing"] = json.dumps(serial["prompt_box_xyxy_processing"])
        serial["mask_bbox_xyxy_processing"] = json.dumps(serial["mask_bbox_xyxy_processing"])
        writer.writerow(serial)
write_json(OUT / "image_summary.json", image_rows)

compatible_images = sum(row["unique_candidates"] > 0 for row in image_rows)
images_with_non_global = sum(row["non_global_candidates"] > 0 for row in image_rows)
assert compatible_images == len(EXEMPLARS)
assert images_with_non_global >= 3
provenance = {
    "model_family_attribution": "Segment Anything Model 3, Meta",
    "static_checkpoint_download_source": STATIC_MIRROR,
    "static_checkpoint_revision": STATIC_REV,
    "static_checkpoint_file": STATIC_FILE,
    "static_checkpoint_sha256": STATIC_SHA,
    "upstream_identity_claim": "not made; runtime compatibility was verified by key coverage and inference",
    "sam31_mirror_considered": VIDEO_MIRROR,
    "sam31_mirror_revision": VIDEO_REV,
    "sam31_checkpoint_file": VIDEO_FILE,
    "sam31_checkpoint_sha256_recorded_not_downloaded_here": VIDEO_SHA,
    "sam31_decision": "video multiplex checkpoint; not substituted for the static-image model",
    "official_code_repository": "facebookresearch/sam3",
    "official_code_commit": CODE_COMMIT,
    "official_code_archive_url": CODE_URL,
    "official_code_archive_sha256": digest(source_archive),
    "official_code_sentinel_sha256": CODE_SENTINELS,
    "license": "SAM License from the official code; mirror LICENSE text matched after newline normalization",
    "weights_redistributed": False,
}
write_json(OUT / "provenance.json", provenance)
metrics = {
    "status": "engineering_compatibility_pass_visual_review_pending",
    "compatible_images": compatible_images,
    "images_expected": len(EXEMPLARS),
    "images_with_non_global_candidate": images_with_non_global,
    "prompts_per_image": 21,
    "proposal_rows": len(all_rows),
    "unique_candidates": sum(row["unique_candidates"] for row in image_rows),
    "checkpoint_key_counts": prefix_counts,
    "missing_model_keys": missing_keys,
    "unexpected_model_keys": unexpected_keys,
    "official_loader_log": load_log,
    "prompting": "class-free whole, 2x2 and 4x4 overlapping boxes through official static SAM-style image interactivity",
    "interpretation_limit": "Masks are candidate visible geometry, not automatic component identities or hidden-content evidence.",
    "source_revision": SOURCE_REV,
    "source_manifest_sha256": SOURCE_MANIFEST_SHA,
    "roi_index_sha256": ROI_SHA,
    "checkpoint_mirror": STATIC_MIRROR,
    "checkpoint_revision": STATIC_REV,
    "checkpoint_sha256": STATIC_SHA,
    "official_code_commit": CODE_COMMIT,
    "sam3_version": sam3_version,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    "elapsed_seconds": round(time.monotonic() - START, 2),
}
write_json(OUT / "metrics.json", metrics)

files = sorted(OUT.iterdir())
operations = [CommitOperationAdd(
    path_in_repo=f"{PREFIX}/{path.name}", path_or_fileobj=str(path))
    for path in files]
commit = api.create_commit(
    PRIVATE_REPO, repo_type="dataset", operations=operations,
    commit_message="Private BDC SAM3 static-image compatibility evidence")
remote = {item.path: item for item in api.get_paths_info(
    PRIVATE_REPO, [f"{PREFIX}/{path.name}" for path in files],
    repo_type="dataset", revision=commit.oid)}
for path in files:
    info = remote[f"{PREFIX}/{path.name}"]
    assert info.size == path.stat().st_size
    if info.lfs is not None:
        assert info.lfs.sha256 == digest(path)
    else:
        blob = hashlib.sha1(
            f"blob {path.stat().st_size}\0".encode() + path.read_bytes()).hexdigest()
        assert info.blob_id == blob
print(json.dumps({
    "event": "complete_private_verified",
    "revision": commit.oid,
    "prefix": PREFIX,
    "compatible_images": compatible_images,
    "images_with_non_global_candidate": images_with_non_global,
    "unique_candidates": metrics["unique_candidates"],
    "elapsed_seconds": metrics["elapsed_seconds"],
    "files": {path.name: {"bytes": path.stat().st_size,
                           "sha256": digest(path)} for path in files},
}), flush=True)
