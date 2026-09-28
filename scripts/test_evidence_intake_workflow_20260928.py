"""Exercise the frozen-collection evidence workflow through its HTTP API."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def request(base: str, path: str, payload: dict | None = None) -> tuple[bytes, dict]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    with urlopen(Request(base.rstrip("/") + path, data=data, headers=headers), timeout=30) as response:
        return response.read(), dict(response.headers.items())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8877")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(request(args.base_url, "/api/collection/summary")[0])
    families = json.loads(request(args.base_url, "/api/collection/families")[0])
    family_id = int(families[0]["family_internal"])
    item_page = json.loads(request(
        args.base_url,
        "/api/collection/items?" + urlencode({"family_internal": family_id, "limit": 1}),
    )[0])
    query_id = item_page["items"][0]["region_id"]
    result = json.loads(request(
        args.base_url, "/api/collection/query?" + urlencode({"region_id": query_id})
    )[0])
    if len(result["candidates"]) != 5:
        raise AssertionError("query did not return five family candidates")
    candidate = result["candidates"][0]
    if candidate["parent_id"] == result["query"]["parent_id"]:
        raise AssertionError("query parent leaked into candidates")
    if candidate["source_sha256"] == result["query"]["source_sha256"]:
        raise AssertionError("exact image duplicate leaked into candidates")
    if candidate["global_community"] == result["query"]["global_community"]:
        raise AssertionError("first candidate did not use available cross-context priority")

    image_path = args.image_root / result["query"]["source_relative_path"]
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if image_hash != result["query"]["source_sha256"]:
        raise AssertionError("catalog source hash does not match the source bytes")
    photo_match = json.loads(request(
        args.base_url, "/api/collection/photo?" + urlencode({"sha256": image_hash})
    )[0])
    if photo_match["new_photo_inference_performed"] or not photo_match["items"]:
        raise AssertionError("exact-byte photo match contract was not preserved")

    created = json.loads(request(args.base_url, "/api/collection/dossiers", {
        "batch_id": "BDC-FUNCTIONAL-TEST-20260928",
        "region_ids": [query_id, candidate["region_id"]],
    })[0])
    corrections = [
        {"entry_id": entry["entry_id"], "changes": {
            "evidence_status": "observed_manual",
            "review_status": "reviewed",
            "review_note": "Functional test of a visible cross-context relation.",
        }}
        for entry in created["state"]["entries"]
    ]
    revised = json.loads(request(
        args.base_url,
        f"/api/dossiers/{created['dossier_id']}/corrections",
        {"reason": "functional workflow test", "corrections": corrections},
    )[0])
    if revised["revision"] != 2 or len(revised["state"]["entries"]) != 2:
        raise AssertionError("dossier revision did not preserve the two selected regions")

    exports: dict[str, dict] = {}
    for fmt in ("json", "csv", "pdf"):
        value, headers = request(
            args.base_url,
            f"/api/dossiers/{created['dossier_id']}/export?" + urlencode({"format": fmt}),
        )
        path = args.output_dir / f"functional_export.{fmt}"
        path.write_bytes(value)
        exports[fmt] = {
            "path": str(path),
            "bytes": len(value),
            "sha256": sha256_bytes(value),
            "content_type": headers.get("Content-Type"),
        }
    exported_json = json.loads((args.output_dir / "functional_export.json").read_text(encoding="utf-8"))
    exported_csv = list(csv.DictReader(io.StringIO(
        (args.output_dir / "functional_export.csv").read_text(encoding="utf-8-sig")
    )))
    exported_pdf = (args.output_dir / "functional_export.pdf").read_bytes()
    if len(exported_json["entries"]) != 2 or len(exported_csv) != 2 or not exported_pdf.startswith(b"%PDF"):
        raise AssertionError("one or more dossier exports are malformed")

    receipt = {
        "status": "pass",
        "scope": "functional test over the frozen transductive collection catalog",
        "collection": {
            "canonical_photos": summary["canonical_photos"],
            "active_regions": summary["active_regions"],
            "r2_families": summary["r2_families"],
        },
        "query_region_id": query_id,
        "candidate_region_id": candidate["region_id"],
        "candidate_crosses_global_context": True,
        "exact_byte_photo_match_count": len(photo_match["items"]),
        "new_photo_inference_performed": False,
        "dossier_id": created["dossier_id"],
        "dossier_revision": revised["revision"],
        "entry_count": len(revised["state"]["entries"]),
        "exports": exports,
    }
    receipt_path = args.output_dir / "functional_test_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
