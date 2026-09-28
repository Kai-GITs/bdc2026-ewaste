"""Embed every selected SAM3 region box with the frozen DINOv3-L encoder.

This is the matched box-only control for the primary foreground-weighted bank.
The exact native SAM mask bounding box is cropped, aspect-preserving letterboxed
to 384 pixels, and pooled over valid content patch tokens only.  It is resumable
by immutable feature-row shards and never assigns a component label.
"""

import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import time
import zipfile
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
REGION_PREFIX = "bdc2026-ewaste/final-study-20260928/composition-region-bank-r1"
REGION_REV = "eb364930b8679c5817d44730233cdb9899d37034"
REGION_ROWS_SHA = "a9ee2b70b43cc05e65a56babbda818cb9ff81674808b67128684c2ab4bdcf071"
DEST = "bdc2026-ewaste/final-study-20260928/dinov3-region-box-r1"

SOURCE_LOCAL = os.environ.get("BDC_SOURCE_ZIP")
SOURCE_REPO = os.environ.get("BDC_SOURCE_DATASET_REPO")
SOURCE_FILE = os.environ.get("BDC_SOURCE_DATASET_FILE", "bdc-official-2026.zip")
SOURCE_REV = os.environ.get("BDC_SOURCE_DATASET_REVISION")
SOURCE_BYTES = 1368394793
SOURCE_MANIFEST_SHA = "697f8ef83052c860782263c7218b0a9bdf7a1611586734b21ed7846f032e5a81"

MODEL = "timm/vit_large_patch16_dinov3.lvd1689m"
MODEL_REV = "30c1109559f65dea34316b0d4842d35c5771fe11"
MODEL_SHA = "45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941"
SIDE = 384
PATCH = 16
GRID = SIDE // PATCH
BATCH = 16
SHARD_SIZE = 1024
OUT = Path("/research-cache/bdc2026/final-study-20260928/dinov3-region-box-r1")
SHARDS = OUT / "shards"
OUT.mkdir(parents=True, exist_ok=True)
SHARDS.mkdir(parents=True, exist_ok=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl_gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def shard_ok(folder, begin, end):
    marker = folder / "complete.json"
    if not marker.is_file():
        return None
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if meta["feature_begin"] != begin or meta["feature_end"] != end:
        return None
    for name, expected in meta["files"].items():
        path = folder / name
        if (not path.is_file() or path.stat().st_size != expected["bytes"] or
                digest(path) != expected["sha256"]):
            return None
    return meta


api = HfApi(token=TOKEN)
assert api.repo_info(PRIVATE_REPO, repo_type="dataset").private
assert not REGION_REV.startswith("__"), "Pin the completed primary region revision before launch"
region_path = Path(hf_hub_download(
    PRIVATE_REPO, f"{REGION_PREFIX}/regions.jsonl.gz", repo_type="dataset",
    revision=REGION_REV, token=TOKEN))
assert digest(region_path) == REGION_ROWS_SHA
with gzip.open(region_path, "rt", encoding="utf-8") as handle:
    regions = [json.loads(line) for line in handle]
assert regions and [row["feature_row"] for row in regions] == list(range(len(regions)))

if SOURCE_LOCAL:
    source_path = Path(SOURCE_LOCAL)
else:
    if not SOURCE_REPO:
        raise RuntimeError("Set BDC_SOURCE_ZIP or BDC_SOURCE_DATASET_REPO.")
    source_path = Path(hf_hub_download(
        SOURCE_REPO, SOURCE_FILE, repo_type="dataset", revision=SOURCE_REV,
        token=os.environ.get("BDC_SOURCE_DATASET_TOKEN") or False))
assert source_path.stat().st_size == SOURCE_BYTES
manifest_path = Path("/research-cache/bdc2026/source-verify/electronic_image_hashes.csv")
assert manifest_path.is_file() and digest(manifest_path) == SOURCE_MANIFEST_SHA
with manifest_path.open(encoding="utf-8", newline="") as handle:
    manifest = list(csv.DictReader(handle))
source_by_sha = {}
for row in manifest:
    source_by_sha.setdefault(row["sha256"], row["zip_member"])
assert len(manifest) == 3961 and len(source_by_sha) == 3931
assert {row["source_sha256"] for row in regions}.issubset(source_by_sha)

weights = Path(hf_hub_download(MODEL, "model.safetensors", revision=MODEL_REV, token=False))
assert digest(weights) == MODEL_SHA
assert torch.cuda.is_available()
model = timm.create_model(
    "vit_large_patch16_dinov3", pretrained=False, num_classes=0,
    global_pool="avg", dynamic_img_size=True)
model.load_state_dict(load_file(str(weights)), strict=True)
model = model.eval().cuda()
torch.set_num_threads(4)

MEAN = np.asarray([.485, .456, .406], dtype=np.float32)
STD = np.asarray([.229, .224, .225], dtype=np.float32)


def load_parent(archive, source_sha):
    raw = archive.read(source_by_sha[source_sha])
    assert hashlib.sha256(raw).hexdigest() == source_sha
    with Image.open(io.BytesIO(raw)) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def exact_crop(image, row):
    x0, y0, x1, y1 = row["mask_bbox_xyxy_native"]
    coords = (max(0, min(image.width - 1, math.floor(x0))),
              max(0, min(image.height - 1, math.floor(y0))),
              max(1, min(image.width, math.ceil(x1))),
              max(1, min(image.height, math.ceil(y1))))
    if coords[2] <= coords[0]:
        coords = (coords[0], coords[1], min(image.width, coords[0] + 1), coords[3])
    if coords[3] <= coords[1]:
        coords = (coords[0], coords[1], coords[2], min(image.height, coords[1] + 1))
    crop = image.crop(coords)
    assert crop.width > 0 and crop.height > 0
    return crop, coords


def prepare(crop):
    scale = min(SIDE / crop.width, SIDE / crop.height)
    width = max(1, round(crop.width * scale))
    height = max(1, round(crop.height * scale))
    left = (SIDE - width) // 2
    top = (SIDE - height) // 2
    canvas = Image.new("RGB", (SIDE, SIDE), (124, 116, 104))
    canvas.paste(crop.resize((width, height), Image.Resampling.BICUBIC), (left, top))
    array = (np.asarray(canvas, dtype=np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    valid = np.zeros((SIDE, SIDE), dtype=np.float32)
    valid[top:top + height, left:left + width] = 1.0
    patch_weights = valid.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3)).reshape(-1)
    assert patch_weights.sum() > 0
    return tensor, patch_weights


metas = []
with zipfile.ZipFile(source_path) as archive, torch.inference_mode():
    for begin in range(0, len(regions), SHARD_SIZE):
        end = min(begin + SHARD_SIZE, len(regions))
        shard_id = begin // SHARD_SIZE
        folder = SHARDS / f"shard_{shard_id:03d}"
        prior = shard_ok(folder, begin, end)
        if prior is not None:
            metas.append(prior)
            print(json.dumps({"event": "reuse_shard", "shard": shard_id,
                              "regions": end}), flush=True)
            continue
        stage = SHARDS / f"shard_{shard_id:03d}.building"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        vectors, row_records = [], []
        parent_cache = {}
        for batch_begin in range(begin, end, BATCH):
            batch_end = min(batch_begin + BATCH, end)
            prepared, batch_meta = [], []
            for row in regions[batch_begin:batch_end]:
                sha = row["source_sha256"]
                if sha not in parent_cache:
                    parent_cache = {sha: load_parent(archive, sha)}
                image = parent_cache[sha]
                assert image.size == (row["native_width"], row["native_height"])
                crop, coords = exact_crop(image, row)
                tensor, patch_weights = prepare(crop)
                prepared.append((tensor, patch_weights))
                batch_meta.append((row, coords, crop.size))
            batch = torch.stack([item[0] for item in prepared]).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                tokens = model.forward_features(batch)
            assert tokens.shape == (batch_end - batch_begin, 5 + GRID * GRID, 1024)
            patches = tokens[:, 5:].float().cpu().numpy()
            for local_index, ((_, patch_weights), (row, coords, crop_size)) in enumerate(
                    zip(prepared, batch_meta)):
                vector = (patches[local_index] * patch_weights[:, None]).sum(axis=0)
                vector /= max(float(patch_weights.sum()), 1e-8)
                vector /= max(float(np.linalg.norm(vector)), 1e-8)
                vectors.append(vector.astype(np.float16))
                row_records.append({
                    "feature_row": row["feature_row"],
                    "region_id": row["region_id"],
                    "parent_id": row["parent_id"],
                    "source_sha256": row["source_sha256"],
                    "mask_bbox_xyxy_native": row["mask_bbox_xyxy_native"],
                    "crop_xyxy_native_integer": list(coords),
                    "crop_width": crop_size[0],
                    "crop_height": crop_size[1],
                    "valid_patch_weight_sum": round(float(patch_weights.sum()), 6),
                    "resolution_status": row["resolution_status"],
                })
            del batch, tokens, patches
        array = np.stack(vectors).astype(np.float16)
        assert array.shape == (end - begin, 1024) and np.isfinite(array).all()
        assert np.linalg.norm(array.astype(np.float32), axis=1).min() > .98
        np.save(stage / "box.npy", array, allow_pickle=False)
        write_jsonl_gz(stage / "rows.jsonl.gz", row_records)
        files = ["box.npy", "rows.jsonl.gz"]
        meta = {
            "shard": shard_id, "feature_begin": begin, "feature_end": end,
            "files": {name: {"bytes": (stage / name).stat().st_size,
                              "sha256": digest(stage / name)} for name in files},
        }
        write_json(stage / "complete.json", meta)
        if folder.exists():
            shutil.rmtree(folder)
        stage.replace(folder)
        metas.append(meta)
        print(json.dumps({"event": "complete_shard", "shard": shard_id,
                          "regions": end,
                          "elapsed_seconds": round(time.monotonic() - START, 1)}), flush=True)
        torch.cuda.empty_cache()

assert metas and metas[-1]["feature_end"] == len(regions)
final = np.lib.format.open_memmap(
    OUT / "dinov3_box.npy", mode="w+", dtype=np.float16,
    shape=(len(regions), 1024))
with gzip.open(OUT / "rows.jsonl.gz", "wt", encoding="utf-8", newline="\n") as target:
    for meta in metas:
        folder = SHARDS / f"shard_{meta['shard']:03d}"
        final[meta["feature_begin"]:meta["feature_end"]] = np.load(
            folder / "box.npy", mmap_mode="r")
        with gzip.open(folder / "rows.jsonl.gz", "rt", encoding="utf-8") as source:
            shutil.copyfileobj(source, target)
final.flush()
assert np.isfinite(final).all()
assert np.linalg.norm(np.asarray(final, dtype=np.float32), axis=1).min() > .98

metrics = {
    "status": "complete_box_control",
    "selected_regions": len(regions),
    "feature_shape": [len(regions), 1024],
    "input_primary_revision": REGION_REV,
    "input_regions_sha256": REGION_ROWS_SHA,
    "model": MODEL,
    "model_revision": MODEL_REV,
    "model_sha256": MODEL_SHA,
    "transform": "exact native SAM mask bounding box; aspect-preserving bicubic letterbox to 384; ImageNet normalization",
    "pooling": "DINOv3-L final dense patch tokens weighted by valid crop content; prefix/register/padding excluded; L2",
    "comparison_role": "box-only control for the foreground-weighted descriptor; no semantic label",
    "source_revision": SOURCE_REV,
    "source_manifest_sha256": SOURCE_MANIFEST_SHA,
    "batch_size": BATCH,
    "shards": len(metas),
    "torch": torch.__version__,
    "timm": timm.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    "elapsed_seconds": round(time.monotonic() - START, 2),
}
write_json(OUT / "metrics.json", metrics)
write_json(OUT / "provenance.json", {
    "primary_region_bank": {"repo": PRIVATE_REPO, "prefix": REGION_PREFIX,
                            "revision": REGION_REV, "regions_sha256": REGION_ROWS_SHA},
    "source": {"repo": SOURCE_REPO, "revision": SOURCE_REV,
               "manifest_sha256": SOURCE_MANIFEST_SHA},
    "dinov3": {"model": MODEL, "revision": MODEL_REV, "sha256": MODEL_SHA},
})

names = ["dinov3_box.npy", "rows.jsonl.gz", "metrics.json", "provenance.json"]
checks = {name: {"bytes": (OUT / name).stat().st_size,
                 "sha256": digest(OUT / name)} for name in names}
commit = api.create_commit(
    PRIVATE_REPO, repo_type="dataset",
    operations=[CommitOperationAdd(path_in_repo=f"{DEST}/{name}",
                                   path_or_fileobj=str(OUT / name)) for name in names],
    commit_message="Private full BDC DINOv3 region box control")
remote = {item.path: item for item in api.get_paths_info(
    PRIVATE_REPO, [f"{DEST}/{name}" for name in names],
    repo_type="dataset", revision=commit.oid)}
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
                  "prefix": DEST, "selected_regions": len(regions),
                  "elapsed_seconds": metrics["elapsed_seconds"], "files": checks}), flush=True)
