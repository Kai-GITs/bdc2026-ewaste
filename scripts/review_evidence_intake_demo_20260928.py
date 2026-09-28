"""Create the fixed human-review actions for the real-photo intake demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "93bcd9f51f027f41": "Papan sirkuit terlihat dalam tumpukan rakitan",
    "297b5776844a8d24": "Papan sirkuit lepas terlihat",
    "f14e54d29cd238a9": "Kemasan sirkuit terpadu terlihat pada papan",
    "e31f2b952c857db8": "Kerusakan fisik terlihat pada layar ponsel",
    "ee677fbd3c36d3fd": "Kerusakan fisik terlihat pada layar televisi",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--created", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    created = json.loads(args.created.read_text(encoding="utf-8-sig"))
    corrections = []
    for entry in created["state"]["entries"]:
        photo = entry["photo_id"]
        if photo in LABELS:
            changes = {
                "visible_part_label": LABELS[photo],
                "evidence_status": "observed_manual",
                "review_status": "accepted",
                "review_note": "Dikonfirmasi dari foto asli dan mask tersimpan pada audit demo.",
            }
            reason = "visual confirmation from original photo and stored mask"
        else:
            changes = {
                "evidence_status": "unresolved",
                "review_status": "needs_additional_evidence",
                "review_note": "Layar gelap terlihat, tetapi foto statis belum membedakan perangkat mati dan gangguan tampilan.",
            }
            reason = "static photo cannot resolve display state"
        corrections.append({"entry_id": entry["entry_id"], "changes": changes, "reason": reason})
    args.output.write_text(json.dumps(corrections, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "corrections": len(corrections), "accepted": len(LABELS), "unresolved": 1}))


if __name__ == "__main__":
    main()
