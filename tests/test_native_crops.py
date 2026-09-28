import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "native_crops.py"


@pytest.fixture
def crop_case(tmp_path):
    source = tmp_path / "train"
    source.mkdir()
    path = source / "example.png"
    image = Image.new("RGB", (90, 50))
    image.putdata([(x, y, (x + y) % 256) for y in range(50) for x in range(90)])
    image.save(path)
    row = {
        "id": hashlib.sha256(b"example.png").hexdigest()[:16],
        "path": "example.png",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return source, tmp_path / "manifest.json", tmp_path / "crops", row


def run_crop(case, *, records=None, output=None):
    source, manifest, default_output, row = case
    manifest.write_text(json.dumps(records if records is not None else [row]), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source),
            "--manifest",
            str(manifest),
            "--output",
            str(output or default_output),
            "--ids",
            row["id"],
            "--size",
            "32",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_native_crop_preserves_pixels_and_source(crop_case):
    source, _, output, row = crop_case
    original = (source / row["path"]).read_bytes()
    result = run_crop(crop_case)
    assert result.returncode == 0, result.stderr
    with Image.open(source / row["path"]) as full, Image.open(output / f"{row['id']}.png") as crop:
        assert crop.size == (32, 32)
        assert crop.tobytes() == full.crop((29, 9, 61, 41)).tobytes()
    assert (source / row["path"]).read_bytes() == original
    receipt = json.loads((output / "receipt.json").read_text())[0]
    assert receipt["crop_xyxy"] == [29, 9, 61, 41]
    assert receipt["resampled"] is False
    assert receipt["status"] == "PENDING_VISUAL_INSPECTION"


def test_changed_source_is_rejected(crop_case):
    source, _, output, row = crop_case
    Image.new("RGB", (90, 50), "blue").save(source / row["path"])
    result = run_crop(crop_case)
    assert result.returncode != 0
    assert "Source bytes differ" in result.stderr
    assert not list(output.glob("*.png"))


def test_source_escape_is_rejected(crop_case):
    source, _, output, row = crop_case
    outside = source.parent / "outside.png"
    outside.write_bytes((source / row["path"]).read_bytes())
    row["path"] = "../outside.png"
    result = run_crop(crop_case)
    assert result.returncode != 0
    assert "escapes the dataset root" in result.stderr
    assert not list(output.glob("*.png"))


@pytest.mark.parametrize("mode", ["duplicate", "unsafe"])
def test_invalid_manifest_ids_are_rejected(crop_case, mode):
    row = crop_case[3]
    if mode == "unsafe":
        row["id"] = "../outside"
    result = run_crop(crop_case, records=[row, row] if mode == "duplicate" else [row])
    assert result.returncode != 0
    assert "Manifest IDs must be unique" in result.stderr
    assert not crop_case[2].exists()


def test_output_inside_source_is_rejected(crop_case):
    output = crop_case[0] / "crops"
    result = run_crop(crop_case, output=output)
    assert result.returncode != 0
    assert "Output must be outside" in result.stderr
    assert not output.exists()


def test_existing_output_is_not_overwritten(crop_case):
    output = crop_case[2]
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("preserve me", encoding="utf-8")
    result = run_crop(crop_case)
    assert result.returncode != 0
    assert marker.read_text() == "preserve me"
    assert list(output.iterdir()) == [marker]
