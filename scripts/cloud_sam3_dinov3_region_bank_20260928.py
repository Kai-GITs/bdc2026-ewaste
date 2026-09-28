"""Build the primary full-cohort BDC region bank.

SAM 3 supplies class-free static-image geometry from fixed multiscale boxes.
DINOv3-L supplies foreground-weighted dense descriptors.  Every prompt/result is
retained; masks never become component labels and low-quality candidates are not
silently promoted into the descriptor bank.
"""

import base64
import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

os.environ["HF_HOME"] = "/research-cache/hf"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import timm
import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from PIL import Image, ImageOps
from safetensors.torch import load_file


START = time.monotonic()
TOKEN = os.environ["HF_TOKEN"]
PRIVATE_REPO = "Fin000/codex-research-workspace"
PREFIX = "bdc2026-ewaste/final-study-20260928/composition-region-bank-r1"
COMPAT_REV = "6a5309158cb27eecb7a14a977feebd8e2d910d84"

SOURCE_LOCAL = os.environ.get("BDC_SOURCE_ZIP")
SOURCE_REPO = os.environ.get("BDC_SOURCE_DATASET_REPO")
SOURCE_FILE = os.environ.get("BDC_SOURCE_DATASET_FILE", "bdc-official-2026.zip")
SOURCE_REV = os.environ.get("BDC_SOURCE_DATASET_REVISION")
SOURCE_BYTES = 1368394793
SOURCE_MANIFEST_SHA = "697f8ef83052c860782263c7218b0a9bdf7a1611586734b21ed7846f032e5a81"
ROI_PATH = "bdc2026-ewaste/track-a/runs/20260924-054638-189f34ad/roi_index.csv"
ROI_SHA = "d4d0b6f67ce9821f38fa3f02ccd7251f09b71a17059882257ab6d720408ff0df"

SAM_REPO = "cubert-gmbh/sam3"
SAM_REV = "6d25af14a085ff9d3e1342c35bae7c87de4811f4"
SAM_FILE = "sam3.pt"
SAM_BYTES = 3450062241
SAM_SHA = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
CODE_COMMIT = "2345a4ad109ac29c569da749c91d84f10dc08c40"
CODE_URL = f"https://github.com/facebookresearch/sam3/archive/{CODE_COMMIT}.zip"
CODE_SENTINELS = {
    "sam3/model_builder.py": "d71d6d3e485ec3eae48bbc2ba676f401b5853d65c4195a91d077b04da38121c2",
    "sam3/model/sam3_image_processor.py": "d8738a0efb6138b01c0dc5deceffd29de9e675860a9a1ed3822b08766373333b",
    "LICENSE": "4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be",
}

DINO_REPO = "timm/vit_large_patch16_dinov3.lvd1689m"
DINO_REV = "30c1109559f65dea34316b0d4842d35c5771fe11"
DINO_SHA = "45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941"
DINO_SIDE = 384
DINO_GRID = 24
DINO_PATCH = 16

SCORE_THRESHOLD = 0.5
MIN_AREA = 0.002
MAX_DETAIL_AREA = 0.90
NMS_IOU = 0.85
CONTAINMENT_DUP = 0.95
CONTAINMENT_AREA_RATIO = 0.80
SHARD_SIZE = 64
OUT = Path("/research-cache/bdc2026/final-study-20260928/composition-region-bank-r1")
SHARDS = OUT / "shards"
OUT.mkdir(parents=True, exist_ok=True)
SHARDS.mkdir(parents=True, exist_ok=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_jsonl_gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_official_source():
    cache = Path("/research-cache/source")
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"sam3-{CODE_COMMIT}.zip"
    source = cache / f"sam3-{CODE_COMMIT}"

    def valid():
        return source.is_dir() and all(
            (source / name).is_file() and digest(source / name) == expected
            for name, expected in CODE_SENTINELS.items())

    if not valid():
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
            (stage / top).replace(source)
            stage.rmdir()
    assert valid()
    return source


source_root = prepare_official_source()
sys.path.insert(0, str(source_root))
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

api = HfApi(token=TOKEN)
assert api.repo_info(PRIVATE_REPO, repo_type="dataset").private
checkpoint = Path(hf_hub_download(SAM_REPO, SAM_FILE, revision=SAM_REV, token=False))
assert checkpoint.stat().st_size == SAM_BYTES and digest(checkpoint) == SAM_SHA

assert torch.cuda.is_available()
sam_model = build_sam3_image_model(
    device="cuda", eval_mode=True, checkpoint_path=str(checkpoint),
    load_from_HF=False, enable_segmentation=True,
    enable_inst_interactivity=True, compile=False)
sam_processor = Sam3Processor(sam_model, device="cuda")

dino_weights = Path(hf_hub_download(
    DINO_REPO, "model.safetensors", revision=DINO_REV, token=False))
assert digest(dino_weights) == DINO_SHA
dino_model = timm.create_model(
    "vit_large_patch16_dinov3", pretrained=False, num_classes=0,
    global_pool="avg", dynamic_img_size=True)
dino_model.load_state_dict(load_file(str(dino_weights)), strict=True)
dino_model = dino_model.eval().cuda()
torch.set_num_threads(4)

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
assert manifest_path.is_file() and digest(manifest_path) == SOURCE_MANIFEST_SHA
with manifest_path.open(encoding="utf-8", newline="") as handle:
    manifest = list(csv.DictReader(handle))
assert len(manifest) == 3961
source_by_sha = {}
for row in manifest:
    source_by_sha.setdefault(row["sha256"], row)
assert len(source_by_sha) == 3931

roi_path = Path(hf_hub_download(
    PRIVATE_REPO, ROI_PATH, repo_type="dataset", token=TOKEN))
assert digest(roi_path) == ROI_SHA
sha_to_id = {}
with roi_path.open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        old = sha_to_id.setdefault(row["source_sha256"], row["canonical_id"])
        assert old == row["canonical_id"]
assert set(sha_to_id) == set(source_by_sha)
items = [(sha_to_id[sha], sha, source_by_sha[sha]["zip_member"])
         for sha in sorted(source_by_sha)]
assert len(items) == 3931

MEAN = np.asarray([.485, .456, .406], dtype=np.float32)
STD = np.asarray([.229, .224, .225], dtype=np.float32)


def load_image(archive, item):
    canonical_id, expected_sha, member = item
    raw = archive.read(member)
    assert hashlib.sha256(raw).hexdigest() == expected_sha
    with Image.open(io.BytesIO(raw)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    return image


def prepare_dino(image):
    scale = min(DINO_SIDE / image.width, DINO_SIDE / image.height)
    new_w = max(1, round(image.width * scale))
    new_h = max(1, round(image.height * scale))
    left = (DINO_SIDE - new_w) // 2
    top = (DINO_SIDE - new_h) // 2
    canvas = Image.new("RGB", (DINO_SIDE, DINO_SIDE), (124, 116, 104))
    canvas.paste(image.resize((new_w, new_h), Image.Resampling.BICUBIC), (left, top))
    array = (np.asarray(canvas, dtype=np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    return tensor, (new_w, new_h, left, top)


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


def rle_encoded(mask):
    flat = mask.reshape(-1).astype(np.uint8)
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], changes, [flat.size]))).astype("<u4")
    if flat[0] == 1:
        counts = np.concatenate((np.zeros(1, dtype="<u4"), counts))
    assert int(counts.astype(np.uint64).sum()) == flat.size
    return base64.b64encode(zlib.compress(counts.tobytes(), level=9)).decode("ascii"), len(counts)


def mask_sha(mask):
    h = hashlib.sha256()
    h.update(struct.pack("<II", mask.shape[0], mask.shape[1]))
    h.update(np.packbits(mask, bitorder="little").tobytes())
    return h.hexdigest()


def select_masks(masks, records):
    valid = [i for i, row in enumerate(records)
             if MIN_AREA <= row["area_fraction"] <= 0.95 and row["bbox_min_side_processing"] > 0]
    ordered = sorted(valid, key=lambda i: (-records[i]["score"],
                                            -records[i]["area_fraction"], i))
    kept = []
    for idx in ordered:
        duplicate = None
        area_i = int(masks[idx].sum())
        for other in kept:
            intersection = int(np.logical_and(masks[idx], masks[other]).sum())
            area_j = int(masks[other].sum())
            union = area_i + area_j - intersection
            iou = intersection / max(union, 1)
            containment = intersection / max(min(area_i, area_j), 1)
            area_ratio = min(area_i, area_j) / max(area_i, area_j, 1)
            if iou >= NMS_IOU or (
                    containment > CONTAINMENT_DUP and area_ratio > CONTAINMENT_AREA_RATIO):
                duplicate = other
                break
        records[idx]["nms_kept"] = duplicate is None
        records[idx]["duplicate_of_prompt"] = (
            records[duplicate]["prompt_name"] if duplicate is not None else None)
        if duplicate is None:
            kept.append(idx)
    selected = []
    for idx, row in enumerate(records):
        row.setdefault("nms_kept", False)
        row.setdefault("duplicate_of_prompt", None)
        reasons = []
        if row["area_pixels"] == 0:
            reasons.append("empty_mask")
        if row["area_fraction"] < MIN_AREA:
            reasons.append("below_min_area")
        if row["area_fraction"] > MAX_DETAIL_AREA:
            reasons.append("global_object")
        if row["score"] < SCORE_THRESHOLD:
            reasons.append("below_score_threshold")
        if not row["nms_kept"]:
            reasons.append("near_duplicate")
        row["selection_status"] = "selected" if not reasons else "+".join(reasons)
        if not reasons:
            selected.append(idx)
    return selected


def pool_foreground(patches, mask, geometry):
    new_w, new_h, left, top = geometry
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (new_w, new_h), Image.Resampling.NEAREST)
    canvas = np.zeros((DINO_SIDE, DINO_SIDE), dtype=np.float32)
    canvas[top:top + new_h, left:left + new_w] = np.asarray(resized, dtype=np.float32) / 255.0
    weights = canvas.reshape(DINO_GRID, DINO_PATCH, DINO_GRID, DINO_PATCH).mean(axis=(1, 3)).reshape(-1)
    fallback = False
    if float(weights.sum()) <= 1e-8:
        y, x = np.unravel_index(np.argmax(canvas), canvas.shape)
        patch_index = min(DINO_GRID - 1, y // DINO_PATCH) * DINO_GRID + min(
            DINO_GRID - 1, x // DINO_PATCH)
        weights[patch_index] = 1.0
        fallback = True
    vector = (patches * weights[:, None]).sum(axis=0) / weights.sum()
    vector /= max(float(np.linalg.norm(vector)), 1e-8)
    return vector.astype(np.float16), {
        "foreground_patch_weight_sum": round(float(weights.sum()), 6),
        "foreground_nonzero_patches": int((weights > 0).sum()),
        "foreground_pool_fallback": fallback,
    }


def native_box(process_box, process_size, native_size):
    sx = native_size[0] / process_size[0]
    sy = native_size[1] / process_size[1]
    x0, y0, x1, y1 = process_box
    return [round(x0 * sx, 2), round(y0 * sy, 2),
            round(x1 * sx, 2), round(y1 * sy, 2)]


def parent_relations(selected, masks, records):
    output = []
    for child in selected:
        child_area = int(masks[child].sum())
        candidates = []
        for parent in selected:
            if parent == child:
                continue
            parent_area = int(masks[parent].sum())
            if parent_area <= child_area:
                continue
            intersection = int(np.logical_and(masks[child], masks[parent]).sum())
            containment = intersection / max(child_area, 1)
            area_ratio = child_area / max(parent_area, 1)
            if containment >= 0.90 and area_ratio <= 0.80:
                candidates.append((parent_area, parent, containment, area_ratio))
        if not candidates:
            continue
        candidates.sort()
        _, parent, containment, area_ratio = candidates[0]
        conflict = False
        if len(candidates) > 1:
            second = candidates[1][1]
            a = int(np.logical_and(masks[parent], masks[second]).sum())
            nested = a / max(min(int(masks[parent].sum()), int(masks[second].sum())), 1)
            conflict = nested < 0.90
        output.append({
            "child_region_id": records[child]["region_id"],
            "parent_region_id": records[parent]["region_id"],
            "relation": "geometric_parent_candidate",
            "child_containment": round(containment, 6),
            "child_parent_area_ratio": round(area_ratio, 6),
            "candidate_count": len(candidates),
            "conflict": conflict,
            "claim_limit": "geometric containment; not verified physical ownership",
        })
    return output


def shard_ok(folder, expected_start):
    marker = folder / "complete.json"
    if not marker.is_file():
        return None
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if meta["feature_start"] != expected_start:
        return None
    for name, expected in meta["files"].items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            return None
    return meta


feature_offset = 0
shard_metas = []
image_counts = []
with zipfile.ZipFile(dataset_zip) as archive, torch.inference_mode():
    for shard_begin in range(0, len(items), SHARD_SIZE):
        shard_end = min(shard_begin + SHARD_SIZE, len(items))
        shard_id = shard_begin // SHARD_SIZE
        folder = SHARDS / f"shard_{shard_id:03d}"
        prior = shard_ok(folder, feature_offset)
        if prior is not None:
            shard_metas.append(prior)
            feature_offset = prior["feature_end"]
            print(json.dumps({"event": "reuse_shard", "shard": shard_id,
                              "images": shard_end, "regions": feature_offset}), flush=True)
            continue

        stage = SHARDS / f"shard_{shard_id:03d}.building"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        image_data = []
        dino_inputs = []
        for item in items[shard_begin:shard_end]:
            canonical_id, source_sha, _ = item
            native = load_image(archive, item)
            native_size = native.size
            sam_scale = min(1.0, 1280 / max(native.size))
            if sam_scale < 1:
                sam_image = native.resize((max(1, round(native.width * sam_scale)),
                                           max(1, round(native.height * sam_scale))),
                                          Image.Resampling.LANCZOS)
            else:
                sam_image = native
            boxes, prompt_names = grid_boxes(sam_image.width, sam_image.height)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                state = sam_processor.set_image(sam_image)
                masks, scores, _ = sam_model.predict_inst(
                    state, point_coords=None, point_labels=None, box=boxes,
                    multimask_output=False)
            masks = to_numpy(masks)
            scores = to_numpy(scores).reshape(-1)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            assert masks.shape == (21, sam_image.height, sam_image.width)
            assert scores.shape == (21,)
            masks = masks.astype(bool)
            records = []
            for prompt_index, (mask, score, prompt_name, prompt_box) in enumerate(
                    zip(masks, scores, prompt_names, boxes)):
                bbox = mask_bbox(mask)
                native_bbox = native_box(bbox, sam_image.size, native_size)
                area = int(mask.sum())
                msha = mask_sha(mask)
                proposal_id = hashlib.sha256(
                    f"{source_sha}|{prompt_name}|{msha}".encode()).hexdigest()[:24]
                cx = ((bbox[0] + bbox[2]) / 2) / sam_image.width
                cy = ((bbox[1] + bbox[3]) / 2) / sam_image.height
                encoded_rle, rle_length = rle_encoded(mask)
                record = {
                    "proposal_id": proposal_id,
                    "region_id": proposal_id,
                    "parent_id": canonical_id,
                    "canonical_id": canonical_id,
                    "source_sha256": source_sha,
                    "prompt_index": prompt_index,
                    "prompt_name": prompt_name,
                    "prompt_box_xyxy_processing": [round(float(v), 2) for v in prompt_box],
                    "prompt_box_xyxy_native": native_box(prompt_box, sam_image.size, native_size),
                    "score": round(float(score), 6),
                    "mask_bbox_xyxy_processing": bbox,
                    "mask_bbox_xyxy_native": native_bbox,
                    "bbox_min_side_processing": min(bbox[2] - bbox[0], bbox[3] - bbox[1]),
                    "bbox_min_side_native": round(min(native_bbox[2] - native_bbox[0],
                                                      native_bbox[3] - native_bbox[1]), 2),
                    "area_pixels": area,
                    "area_fraction": round(area / (sam_image.width * sam_image.height), 8),
                    "centroid_xy_normalized": [round(cx, 6), round(cy, 6)],
                    "mask_shape_hw": [sam_image.height, sam_image.width],
                    "mask_rle_order": "row-major, alternating zero/one runs starting with zero",
                    "mask_rle_encoding": "little-endian uint32 counts, zlib level 9, base64",
                    "mask_rle_count_length": rle_length,
                    "mask_rle_zlib_base64": encoded_rle,
                    "mask_sha256": msha,
                    "native_width": native_size[0],
                    "native_height": native_size[1],
                    "processing_width": sam_image.width,
                    "processing_height": sam_image.height,
                    "nms_kept": False,
                    "duplicate_of_prompt": None,
                    "selection_status": None,
                    "resolution_status": "fine_visible_geometry" if min(
                        native_bbox[2] - native_bbox[0], native_bbox[3] - native_bbox[1]) >= 24
                        else "coarse_only_under_24px",
                    "feature_row": None,
                }
                records.append(record)
            selected = select_masks(masks, records)
            dino_tensor, dino_geometry = prepare_dino(native)
            dino_inputs.append(dino_tensor)
            image_data.append({
                "item": item, "native_size": native_size, "sam_size": sam_image.size,
                "masks": masks, "records": records, "selected": selected,
                "dino_geometry": dino_geometry,
            })
            del state

        patch_chunks = []
        for micro_begin in range(0, len(dino_inputs), 8):
            dino_batch = torch.stack(dino_inputs[micro_begin:micro_begin + 8]).cuda(
                non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                tokens = dino_model.forward_features(dino_batch)
            assert tokens.shape[1:] == (5 + DINO_GRID * DINO_GRID, 1024)
            patch_chunks.append(tokens[:, 5:, :].float().cpu().numpy())
            del dino_batch, tokens
        patch_batch = np.concatenate(patch_chunks, axis=0)
        assert patch_batch.shape == (len(image_data), DINO_GRID * DINO_GRID, 1024)
        del patch_chunks

        proposal_rows, region_rows, relation_rows, summary_rows, features = [], [], [], [], []
        for image_index, data in enumerate(image_data):
            canonical_id, source_sha, _ = data["item"]
            patches = patch_batch[image_index]
            for selected_index in data["selected"]:
                record = data["records"][selected_index]
                vector, pool_info = pool_foreground(
                    patches, data["masks"][selected_index], data["dino_geometry"])
                feature_row = feature_offset + len(features)
                record["feature_row"] = feature_row
                region = {key: value for key, value in record.items()
                          if key not in {"mask_rle_zlib_base64", "mask_rle_order",
                                         "mask_rle_encoding", "mask_rle_count_length"}}
                region.update(pool_info)
                region["descriptor"] = "DINOv3-L foreground-weighted final patch tokens; L2 normalized"
                region_rows.append(region)
                features.append(vector)
            relation_rows.extend(parent_relations(
                data["selected"], data["masks"], data["records"]))
            proposal_rows.extend(data["records"])
            summary = {
                "canonical_id": canonical_id,
                "source_sha256": source_sha,
                "native_width": data["native_size"][0],
                "native_height": data["native_size"][1],
                "processing_width": data["sam_size"][0],
                "processing_height": data["sam_size"][1],
                "prompts": 21,
                "nms_kept": sum(row["nms_kept"] for row in data["records"]),
                "selected_regions": len(data["selected"]),
                "coarse_selected_regions": sum(
                    data["records"][idx]["resolution_status"] == "coarse_only_under_24px"
                    for idx in data["selected"]),
                "global_object_proposals": sum(
                    row["area_fraction"] > MAX_DETAIL_AREA for row in data["records"]),
                "max_score": max(row["score"] for row in data["records"]),
            }
            summary_rows.append(summary)

        feature_array = (np.stack(features).astype(np.float16) if features
                         else np.empty((0, 1024), dtype=np.float16))
        assert feature_array.shape[0] == len(region_rows)
        if len(feature_array):
            norms = np.linalg.norm(feature_array.astype(np.float32), axis=1)
            assert norms.min() > .98 and np.isfinite(feature_array).all()
        write_jsonl_gz(stage / "proposals.jsonl.gz", proposal_rows)
        write_jsonl_gz(stage / "regions.jsonl.gz", region_rows)
        write_jsonl_gz(stage / "parent_relations.jsonl.gz", relation_rows)
        np.save(stage / "foreground.npy", feature_array, allow_pickle=False)
        with (stage / "image_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        files = ["proposals.jsonl.gz", "regions.jsonl.gz", "parent_relations.jsonl.gz",
                 "foreground.npy", "image_summary.csv"]
        meta = {
            "shard": shard_id,
            "image_begin": shard_begin,
            "image_end": shard_end,
            "feature_start": feature_offset,
            "feature_end": feature_offset + len(region_rows),
            "proposal_rows": len(proposal_rows),
            "region_rows": len(region_rows),
            "relation_rows": len(relation_rows),
            "files": {name: {"bytes": (stage / name).stat().st_size,
                              "sha256": digest(stage / name)} for name in files},
        }
        write_json(stage / "complete.json", meta)
        if folder.exists():
            shutil.rmtree(folder)
        stage.replace(folder)
        shard_metas.append(meta)
        feature_offset = meta["feature_end"]
        print(json.dumps({"event": "complete_shard", "shard": shard_id,
                          "images": shard_end, "regions": feature_offset,
                          "elapsed_seconds": round(time.monotonic() - START, 1)}), flush=True)
        del image_data, dino_inputs, patch_batch, feature_array
        torch.cuda.empty_cache()

assert shard_metas[-1]["image_end"] == len(items)
assert all(meta["feature_start"] == (0 if i == 0 else shard_metas[i - 1]["feature_end"])
           for i, meta in enumerate(shard_metas))
total_regions = shard_metas[-1]["feature_end"]

final_feature = np.lib.format.open_memmap(
    OUT / "dinov3_foreground.npy", mode="w+", dtype=np.float16,
    shape=(total_regions, 1024))
all_summaries = []
with gzip.open(OUT / "proposals.jsonl.gz", "wt", encoding="utf-8", newline="\n") as proposals_out, \
     gzip.open(OUT / "regions.jsonl.gz", "wt", encoding="utf-8", newline="\n") as regions_out, \
     gzip.open(OUT / "parent_relations.jsonl.gz", "wt", encoding="utf-8", newline="\n") as relations_out:
    for meta in shard_metas:
        folder = SHARDS / f"shard_{meta['shard']:03d}"
        block = np.load(folder / "foreground.npy", mmap_mode="r")
        final_feature[meta["feature_start"]:meta["feature_end"]] = block
        for source_name, target in [
                ("proposals.jsonl.gz", proposals_out),
                ("regions.jsonl.gz", regions_out),
                ("parent_relations.jsonl.gz", relations_out)]:
            with gzip.open(folder / source_name, "rt", encoding="utf-8") as source_handle:
                shutil.copyfileobj(source_handle, target)
        with (folder / "image_summary.csv").open(encoding="utf-8", newline="") as handle:
            all_summaries.extend(csv.DictReader(handle))
final_feature.flush()
assert len(all_summaries) == len(items)
assert np.isfinite(final_feature).all()
assert np.linalg.norm(np.asarray(final_feature, dtype=np.float32), axis=1).min() > .98

with (OUT / "image_summary.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(all_summaries[0]))
    writer.writeheader()
    writer.writerows(all_summaries)

selected_counts = np.asarray([int(row["selected_regions"]) for row in all_summaries])
coarse_counts = np.asarray([int(row["coarse_selected_regions"]) for row in all_summaries])
metrics = {
    "status": "complete_primary_region_bank",
    "canonical_images_completed": len(items),
    "canonical_images": len(items),
    "parent_records": len(manifest),
    "proposal_rows": sum(meta["proposal_rows"] for meta in shard_metas),
    "selected_regions": total_regions,
    "parent_relation_rows": sum(meta["relation_rows"] for meta in shard_metas),
    "images_without_selected_region": int((selected_counts == 0).sum()),
    "selected_regions_per_image_quantiles": np.quantile(
        selected_counts, [0, .1, .25, .5, .75, .9, 1]).tolist(),
    "coarse_selected_regions": int(coarse_counts.sum()),
    "selection": {
        "score_threshold": SCORE_THRESHOLD,
        "minimum_area_fraction": MIN_AREA,
        "maximum_detail_area_fraction": MAX_DETAIL_AREA,
        "mask_nms_iou": NMS_IOU,
        "duplicate_containment": CONTAINMENT_DUP,
        "duplicate_area_ratio": CONTAINMENT_AREA_RATIO,
    },
    "proposal_method": "official SAM3 static image interactivity; class-free whole, 2x2 and 4x4 overlapping box prompts",
    "descriptor_method": "DINOv3-L 384 aspect-preserving letterbox; final dense patch tokens weighted by identically transformed foreground mask; no prefix/register/padding tokens; L2",
    "interpretation_limit": "Masks and families are model proposals. They do not establish component identity, physical ownership, hidden content, value, toxicity, or safety.",
    "source_revision": SOURCE_REV,
    "source_manifest_sha256": SOURCE_MANIFEST_SHA,
    "roi_index_sha256": ROI_SHA,
    "sam_checkpoint_mirror": SAM_REPO,
    "sam_checkpoint_revision": SAM_REV,
    "sam_checkpoint_sha256": SAM_SHA,
    "sam_official_code_commit": CODE_COMMIT,
    "sam_compatibility_private_revision": COMPAT_REV,
    "dinov3_model": DINO_REPO,
    "dinov3_revision": DINO_REV,
    "dinov3_sha256": DINO_SHA,
    "feature_shape": [total_regions, 1024],
    "shards": len(shard_metas),
    "torch": torch.__version__,
    "timm": timm.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    "elapsed_seconds": round(time.monotonic() - START, 2),
}
write_json(OUT / "metrics.json", metrics)
write_json(OUT / "provenance.json", {
    "source": {"repo": SOURCE_REPO, "revision": SOURCE_REV,
               "manifest_sha256": SOURCE_MANIFEST_SHA},
    "sam": {"attribution": "Meta Segment Anything Model 3",
            "download_mirror": SAM_REPO, "revision": SAM_REV,
            "checkpoint": SAM_FILE, "sha256": SAM_SHA,
            "official_code_commit": CODE_COMMIT,
            "claim": "runtime-compatible mirror; no upstream byte-identity claim"},
    "dinov3": {"model": DINO_REPO, "revision": DINO_REV,
               "sha256": DINO_SHA},
    "weights_redistributed": False,
})

names = ["proposals.jsonl.gz", "regions.jsonl.gz", "parent_relations.jsonl.gz",
         "dinov3_foreground.npy", "image_summary.csv", "metrics.json", "provenance.json"]
checks = {name: {"bytes": (OUT / name).stat().st_size,
                 "sha256": digest(OUT / name)} for name in names}
operations = [CommitOperationAdd(
    path_in_repo=f"{PREFIX}/{name}", path_or_fileobj=str(OUT / name))
    for name in names]
commit = api.create_commit(
    PRIVATE_REPO, repo_type="dataset", operations=operations,
    commit_message="Private full BDC SAM3 and DINOv3 primary region bank")
remote = {item.path: item for item in api.get_paths_info(
    PRIVATE_REPO, [f"{PREFIX}/{name}" for name in names],
    repo_type="dataset", revision=commit.oid)}
for name in names:
    info = remote[f"{PREFIX}/{name}"]
    assert info.size == checks[name]["bytes"]
    if info.lfs is not None:
        assert info.lfs.sha256 == checks[name]["sha256"]
    else:
        path = OUT / name
        blob = hashlib.sha1(
            f"blob {path.stat().st_size}\0".encode() + path.read_bytes()).hexdigest()
        assert info.blob_id == blob
print(json.dumps({
    "event": "complete_private_verified",
    "revision": commit.oid,
    "prefix": PREFIX,
    "canonical_images_completed": len(items),
    "selected_regions": total_regions,
    "images_without_selected_region": metrics["images_without_selected_region"],
    "elapsed_seconds": metrics["elapsed_seconds"],
    "files": checks,
}), flush=True)
