"""Local task-oriented workspace for reviewing evidence intake dossiers."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .catalog import (
    build_batch_from_regions,
    catalog_summary,
    family_items,
    list_families,
    load_catalog,
    photo_items_by_sha,
    query_region,
)
from .core import DossierStore, build_dossier, dossier_csv, dossier_pdf
from .priorities import load_priorities
from .planner import load_policy, plan_inspection
from .inspection import InspectionSessions, inspection_csv


class WorkspaceHandler(BaseHTTPRequestHandler):
    store: DossierStore
    web_root: Path
    image_root: Path | None
    catalog: dict | None
    priorities: dict = {"scope": "not_configured", "items": []}
    inspection_weights: dict = {}
    sessions: InspectionSessions | None = None

    def _json(self, value, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _bytes(self, payload: bytes, content_type: str, filename: str | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 10_000_000:
            raise ValueError("request body is too large")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _state(self, dossier_id: str, query: dict[str, list[str]]):
        revision = int(query["revision"][0]) if query.get("revision") else None
        return self.store.get(dossier_id, revision)

    def do_GET(self):
        parsed = urlparse(self.path)
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        query = parse_qs(parsed.query)
        try:
            if len(parts) == 4 and parts[:2] == ['api', 'inspection'] and parts[3] == 'export' and self.sessions:
                export = self.sessions.export(parts[2])
                fmt = query.get('format', ['json'])[0]
                filename = f"Hasil_Inspeksi_{export['session_id']}_v{export['version']}"
                if fmt == 'json':
                    return self._bytes((json.dumps(export, ensure_ascii=False, indent=2)+'\n').encode('utf-8'), 'application/json; charset=utf-8', filename+'.json')
                if fmt == 'csv':
                    return self._bytes(inspection_csv(export), 'text/csv; charset=utf-8', filename+'.csv')
                raise ValueError('Inspection export format must be json or csv')
            if parsed.path == '/api/inspection/session' and self.sessions:
                return self._json(self.sessions.get(query.get('id',[None])[0]))
            if parsed.path == "/api/collection/assembly-priorities":
                return self._json(self.priorities)
            if parsed.path == "/api/collection/summary" and self.catalog:
                return self._json(catalog_summary(self.catalog))
            if parsed.path == "/api/collection/families" and self.catalog:
                return self._json(list_families(self.catalog))
            if parsed.path == "/api/collection/items" and self.catalog:
                family_key = query.get("family_key", [""])[0]
                family_internal = query.get("family_internal", [None])[0]
                if not family_key and family_internal is None:
                    raise ValueError("family_key or family_internal is required")
                return self._json(family_items(
                    self.catalog, family_key or None,
                    int(query.get("limit", ["24"])[0]), int(query.get("offset", ["0"])[0]),
                    int(family_internal) if family_internal is not None else None,
                ))
            if parsed.path == "/api/collection/query" and self.catalog:
                region_id = query.get("region_id", [""])[0]
                if not region_id:
                    raise ValueError("region_id is required")
                return self._json(query_region(self.catalog, region_id))
            if parsed.path == "/api/collection/photo" and self.catalog:
                sha256 = query.get("sha256", [""])[0]
                return self._json(photo_items_by_sha(self.catalog, sha256))
            if parsed.path == "/api/dossiers":
                summaries = []
                for item in self.store.list():
                    state = item["state"]
                    summaries.append({
                        "dossier_id": item["dossier_id"], "revision": item["revision"],
                        "batch_id": state["batch_id"], "photos_received": state["photos_received"],
                        "entries": len(state["entries"]), "review_status": state["review"]["status"],
                    })
                return self._json(summaries)
            if len(parts) >= 3 and parts[:2] == ["api", "dossiers"]:
                item = self._state(parts[2], query)
                if len(parts) == 3:
                    return self._json(item)
                if len(parts) == 4 and parts[3] == "export":
                    fmt = query.get("format", ["json"])[0]
                    state = item["state"]
                    base = f"{state['batch_id']}-r{item['revision']}"
                    if fmt == "json":
                        return self._bytes(
                            (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
                            "application/json; charset=utf-8", f"{base}.json")
                    if fmt == "csv":
                        return self._bytes(dossier_csv(state), "text/csv; charset=utf-8", f"{base}.csv")
                    if fmt == "pdf":
                        return self._bytes(dossier_pdf(state), "application/pdf", f"{base}.pdf")
                    raise ValueError("format must be json, csv, or pdf")
            if len(parts) >= 2 and parts[0] == "images" and self.image_root:
                relative = Path(*parts[1:])
                target = (self.image_root / relative).resolve()
                root = self.image_root.resolve()
                if target != root and root not in target.parents:
                    raise ValueError("image path escaped the configured root")
                if not target.is_file():
                    raise FileNotFoundError(target)
                return self._bytes(target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            name = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
            target = (self.web_root / name).resolve()
            if self.web_root.resolve() not in target.parents or not target.is_file():
                raise FileNotFoundError(target)
            return self._bytes(target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        except (KeyError, FileNotFoundError) as exc:
            self._json({"error": str(exc)}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 400)

    def do_POST(self):
        parsed = urlparse(self.path)
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        try:
            if parts == ['api','inspection','session'] and self.sessions:
                return self._json(self.sessions.create(self._body().get('budget',25)),201)
            if len(parts)==4 and parts[:2]==['api','inspection'] and self.sessions:
                body=self._body()
                if parts[3]=='review':
                    return self._json(self.sessions.review(parts[2],body['version'],body['photo_id'],body['outcome'],str(body.get('note',''))))
                if parts[3]=='next':
                    return self._json(self.sessions.next_batch(parts[2],body['version']))
            if parts == ["api", "collection", "inspection-plan"] and self.catalog:
                body = self._body()
                plan = plan_inspection(self.catalog, body.get('budget',25), self.inspection_weights,
                                       reviewed=body.get('reviewed_photo_ids'), eligible=body.get('eligible_photo_ids'),
                                       family_ids=body.get('family_ids'))
                if body.get('save') and plan['items']:
                    regions = [r for item in plan['items'] for r in item['region_ids']]
                    state = build_dossier(build_batch_from_regions(self.catalog, regions, str(body.get('batch_id') or 'BDC-INSPEKSI')))
                    state['inspection_plan'] = plan
                    plan['dossier'] = self.store.create(state, reason='automated discovery inspection plan')
                return self._json(plan)
            if parts == ["api", "collection", "dossiers"] and self.catalog:
                body = self._body()
                batch = build_batch_from_regions(
                    self.catalog, body.get("region_ids", []),
                    str(body.get("batch_id") or "BDC-COLLECTION-SELECTION"),
                )
                return self._json(self.store.create(build_dossier(batch)), 201)
            if parts == ["api", "dossiers"]:
                state = build_dossier(self._body())
                return self._json(self.store.create(state), 201)
            if len(parts) == 4 and parts[:2] == ["api", "dossiers"] and parts[3] == "corrections":
                body = self._body()
                result = self.store.revise(
                    parts[2], body.get("corrections", []), str(body.get("reason") or "reviewer correction"))
                return self._json(result, 201)
            self._json({"error": "unknown endpoint"}, 404)
        except KeyError as exc:
            self._json({"error": str(exc)}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 400)

    def log_message(self, message, *args):
        print(f"workspace: {message % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("evidence_intake.sqlite3"))
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--priorities", type=Path, help="Frozen context-score CSV; labels are never exposed")
    parser.add_argument("--inspection-policy", type=Path, help="Existing family census for explicit post-discovery utility weights")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    WorkspaceHandler.store = DossierStore(args.db)
    WorkspaceHandler.web_root = Path(__file__).with_name("web")
    WorkspaceHandler.image_root = args.image_root
    WorkspaceHandler.catalog = load_catalog(args.catalog) if args.catalog else None
    WorkspaceHandler.priorities = load_priorities(args.priorities)
    WorkspaceHandler.inspection_weights = load_policy(args.inspection_policy)
    if WorkspaceHandler.catalog:
        WorkspaceHandler.sessions = InspectionSessions(args.db,WorkspaceHandler.catalog,WorkspaceHandler.inspection_weights)
    server = ThreadingHTTPServer((args.host, args.port), WorkspaceHandler)
    print(json.dumps({"status": "serving", "url": f"http://{args.host}:{args.port}",
                      "db": str(args.db), "image_root": str(args.image_root) if args.image_root else None,
                      "catalog": str(args.catalog) if args.catalog else None}))
    server.serve_forever()


if __name__ == "__main__":
    main()
