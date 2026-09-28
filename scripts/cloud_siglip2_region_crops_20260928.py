"""Embed every selected SAM3 region box with official SigLIP 2 NaFlex.

The semantic view uses the same exact native bounding box as the DINO box
control.  AutoProcessor supplies NaFlex spatial shapes and attention masks with
256 patches; no legacy SigLIP resize, text prompt, or target class is used.
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
from collections import Counter
from pathlib import Path

os.environ["HF_HOME"] = "/research-cache/hf"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor, __version__ as transformers_version


START = time.monotonic()
TOKEN = os.environ["HF_TOKEN"]
PRIVATE_REPO = "Fin000/codex-research-workspace"
REGION_PREFIX = "bdc2026-ewaste/final-study-20260928/composition-region-bank-r1"
REGION_REV = "eb364930b8679c5817d44730233cdb9899d37034"
REGION_ROWS_SHA = "a9ee2b70b43cc05e65a56babbda818cb9ff81674808b67128684c2ab4bdcf071"
DEST = "bdc2026-ewaste/final-study-20260928/siglip2-region-crop-r1"

SOURCE_LOCAL = os.environ.get("BDC_SOURCE_ZIP")
SOURCE_REPO = os.environ.get("BDC_SOURCE_DATASET_REPO")
SOURCE_FILE = os.environ.get("BDC_SOURCE_DATASET_FILE", "bdc-official-2026.zip")
SOURCE_REV = os.environ.get("BDC_SOURCE_DATASET_REVISION")
SOURCE_BYTES = 1368394793
SOURCE_MANIFEST_SHA = "697f8ef83052c860782263c7218b0a9bdf7a1611586734b21ed7846f032e5a81"

MODEL = "google/siglip2-so400m-patch16-naflex"
MODEL_REV = "cc24074f717b612951c2dead130904ab9b65a81e"
MODEL_SHA = "11a61a2068800d5f4f35cb041c1fea25de86ce87725e4c97a9ed046d0b22c076"
MAX_PATCHES = 256
BATCH = 16
SHARD_SIZE = 1024
OUT = Path("/research-cache/bdc2026/final-study-20260928/siglip2-region-crop-r1")
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

weight_path = Path(hf_hub_download(MODEL, "model.safetensors", revision=MODEL_REV, token=False))
assert digest(weight_path) == MODEL_SHA
processor = AutoProcessor.from_pretrained(MODEL, revision=MODEL_REV, trust_remote_code=False)
model = AutoModel.from_pretrained(
    MODEL, revision=MODEL_REV, trust_remote_code=False,
    use_safetensors=True, dtype=torch.float16).eval().cuda()
assert "Siglip2ImageProcessor" in type(processor.image_processor).__name__
assert processor.image_processor.max_num_patches == MAX_PATCHES
assert processor.image_processor.patch_size == 16
torch.set_num_threads(4)


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
        shard_shapes = Counter()
        parent_cache = {}
        for batch_begin in range(begin, end, BATCH):
            batch_end = min(batch_begin + BATCH, end)
            crops, batch_meta = [], []
            for row in regions[batch_begin:batch_end]:
                sha = row["source_sha256"]
                if sha not in parent_cache:
                    parent_cache = {sha: load_parent(archive, sha)}
                image = parent_cache[sha]
                assert image.size == (row["native_width"], row["native_height"])
                crop, coords = exact_crop(image, row)
                crops.append(crop)
                batch_meta.append((row, coords, crop.size))
            batch = processor(images=crops, max_num_patches=MAX_PATCHES, return_tensors="pt")
            assert {"pixel_values", "pixel_attention_mask", "spatial_shapes"}.issubset(batch)
            for shape in batch["spatial_shapes"].tolist():
                shard_shapes[tuple(shape)] += 1
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                result = model.get_image_features(**batch)
            encoded = result.pooler_output if hasattr(result, "pooler_output") else result
            encoded = encoded.float().cpu().numpy()
            assert encoded.shape == (batch_end - batch_begin, 1152)
            assert np.isfinite(encoded).all()
            encoded /= np.maximum(np.linalg.norm(encoded, axis=1, keepdims=True), 1e-8)
            vectors.extend(encoded.astype(np.float16))
            for row, coords, crop_size in batch_meta:
                row_records.append({
                    "feature_row": row["feature_row"],
                    "region_id": row["region_id"],
                    "parent_id": row["parent_id"],
                    "source_sha256": row["source_sha256"],
                    "mask_bbox_xyxy_native": row["mask_bbox_xyxy_native"],
                    "crop_xyxy_native_integer": list(coords),
                    "crop_width": crop_size[0],
                    "crop_height": crop_size[1],
                    "resolution_status": row["resolution_status"],
                })
            del batch, result, encoded
        array = np.stack(vectors).astype(np.float16)
        assert array.shape == (end - begin, 1152) and np.isfinite(array).all()
        assert np.linalg.norm(array.astype(np.float32), axis=1).min() > .98
        np.save(stage / "crop.npy", array, allow_pickle=False)
        write_jsonl_gz(stage / "rows.jsonl.gz", row_records)
        files = ["crop.npy", "rows.jsonl.gz"]
        meta = {
            "shard": shard_id, "feature_begin": begin, "feature_end": end,
            "spatial_shape_counts": {
                f"{key[0]}x{key[1]}": value for key, value in sorted(shard_shapes.items())},
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
    OUT / "siglip2_crop.npy", mode="w+", dtype=np.float16,
    shape=(len(regions), 1152))
with gzip.open(OUT / "rows.jsonl.gz", "wt", encoding="utf-8", newline="\n") as target:
    for meta in metas:
        folder = SHARDS / f"shard_{meta['shard']:03d}"
        final[meta["feature_begin"]:meta["feature_end"]] = np.load(
            folder / "crop.npy", mmap_mode="r")
        with gzip.open(folder / "rows.jsonl.gz", "rt", encoding="utf-8") as source:
            shutil.copyfileobj(source, target)
final.flush()
assert np.isfinite(final).all()
assert np.linalg.norm(np.asarray(final, dtype=np.float32), axis=1).min() > .98
shape_counts = Counter()
for meta in metas:
    shape_counts.update(meta["spatial_shape_counts"])

metrics = {
    "status": "complete_semantic_crop_view",
    "selected_regions": len(regions),
    "feature_shape": [len(regions), 1152],
    "input_primary_revision": REGION_REV,
    "input_regions_sha256": REGION_ROWS_SHA,
    "model": MODEL,
    "model_revision": MODEL_REV,
    "model_sha256": MODEL_SHA,
    "processor": type(processor.image_processor).__name__,
    "transform": "exact native SAM mask bounding box; official SigLIP 2 NaFlex AutoProcessor; max_num_patches=256",
    "pooling": "AutoModel.get_image_features(...).pooler_output; L2 normalization",
    "analysis_role": "second semantic rank view only; no text prompt or target label",
    "source_revision": SOURCE_REV,
    "source_manifest_sha256": SOURCE_MANIFEST_SHA,
    "spatial_shape_counts": dict(sorted(shape_counts.items())),
    "batch_size": BATCH,
    "shards": len(metas),
    "torch": torch.__version__,
    "transformers": transformers_version,
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
    "siglip2": {"model": MODEL, "revision": MODEL_REV, "sha256": MODEL_SHA,
                "processor": type(processor.image_processor).__name__,
                "max_num_patches": MAX_PATCHES},
})

names = ["siglip2_crop.npy", "rows.jsonl.gz", "metrics.json", "provenance.json"]
checks = {name: {"bytes": (OUT / name).stat().st_size,
                 "sha256": digest(OUT / name)} for name in names}
commit = api.create_commit(
    PRIVATE_REPO, repo_type="dataset",
    operations=[CommitOperationAdd(path_in_repo=f"{DEST}/{name}",
                                   path_or_fileobj=str(OUT / name)) for name in names],
    commit_message="Private full BDC SigLIP2 region crop view")
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
