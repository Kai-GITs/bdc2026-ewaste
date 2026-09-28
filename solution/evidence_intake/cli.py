"""Command line entry point for dossier creation, revision, and export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import DossierStore, build_dossier, dossier_csv, dossier_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("evidence_intake.sqlite3"))
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create")
    create.add_argument("batch", type=Path)
    revise = sub.add_parser("revise")
    revise.add_argument("dossier_id")
    revise.add_argument("corrections", type=Path)
    export = sub.add_parser("export")
    export.add_argument("dossier_id")
    export.add_argument("output", type=Path)
    export.add_argument("--revision", type=int)
    args = parser.parse_args()
    store = DossierStore(args.db)

    if args.action == "create":
        state = build_dossier(json.loads(args.batch.read_text(encoding="utf-8")))
        result = store.create(state)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.action == "revise":
        corrections = json.loads(args.corrections.read_text(encoding="utf-8"))
        result = store.revise(args.dossier_id, corrections)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        state = store.get(args.dossier_id, args.revision)["state"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".json":
            args.output.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        elif args.output.suffix.lower() == ".csv":
            args.output.write_bytes(dossier_csv(state))
        elif args.output.suffix.lower() == ".pdf":
            args.output.write_bytes(dossier_pdf(state))
        else:
            raise ValueError("output extension must be .json, .csv, or .pdf")
        print(json.dumps({"status": "exported", "path": str(args.output)}))


if __name__ == "__main__":
    main()
