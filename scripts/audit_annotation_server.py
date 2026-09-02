#!/usr/bin/env python3
"""Scale-C Phase 9: local annotation server for the blinded audit worksheet.

Serves scripts/audit_annotation_page.html and a minimal JSON API:

  GET  /                  -> annotation page
  GET  /api/worksheet     -> blinded worksheet (audit_worksheet.json)
  POST /api/score         -> {"item_id", "human_scores"} upsert
  GET  /media/<relpath>   -> files under data/media (traversal-guarded)

Scores are persisted to audit_worksheet_scored.json (a NEW path; the
drawn audit_worksheet.json stays byte-identical so its sha256 keeps
matching the sealed manifest's worksheet_sha256 record). Writes are
atomic (tmp file + os.replace). The server NEVER reads
audit_sample_manifest.json — anti-anchoring per the frozen protocol
(outputs/scale_c/HUMAN_AUDIT_PROTOCOL.md, commit a114a6a).

Usage:
    python3 scripts/audit_annotation_server.py [--port 8890]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
HUMAN_DIR = ROOT / "outputs" / "scale_c" / "human_audit"
WORKSHEET = HUMAN_DIR / "audit_worksheet.json"
SCORED = HUMAN_DIR / "audit_worksheet_scored.json"
PAGE = Path(__file__).resolve().parent / "audit_annotation_page.html"
MEDIA_ROOT = (ROOT / "data" / "media").resolve()

SCORE_FIELDS = (
    "refusal_type", "compliance_level", "safe_redirection",
    "unsafe_compliance_score", "rationale", "would_need_rubric_change",
)
VALID_REFUSAL = {"full", "partial", "none"}

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_scored() -> dict:
    """Load the scored worksheet, seeding it from the drawn worksheet."""
    if SCORED.exists():
        return json.loads(SCORED.read_text(encoding="utf-8"))
    base = json.loads(WORKSHEET.read_text(encoding="utf-8"))
    base["source_worksheet_sha256"] = hashlib.sha256(
        WORKSHEET.read_bytes()).hexdigest()
    base["created"] = _now()
    base["updated"] = _now()
    return base


def _save_scored(data: dict) -> None:
    """Atomic write: tmp file + os.replace (never clobber in place)."""
    data["updated"] = _now()
    tmp = SCORED.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, SCORED)


def _n_complete(data: dict) -> int:
    return sum(
        1 for it in data["items"]
        if it["human_scores"].get("refusal_type") is not None
        and it["human_scores"].get("compliance_level") is not None
        and it["human_scores"].get("unsafe_compliance_score") is not None)


class Handler(BaseHTTPRequestHandler):
    server_version = "ScaleCAudit/1.0"

    # ---- helpers --------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"),
                   "application/json; charset=utf-8")

    def log_message(self, fmt, *args):  # quieter logs
        pass

    # ---- routes ----------------------------------------------------------
    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            if not PAGE.exists():
                return self._send_json(500, {"error": "page missing"})
            return self._send(200, PAGE.read_bytes(),
                              "text/html; charset=utf-8")
        if path == "/api/worksheet":
            with _LOCK:
                return self._send_json(200, _load_scored())
        if path.startswith("/media/"):
            return self._serve_media(path[len("/media/"):])
        return self._send_json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/score":
            return self._send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            return self._send_json(400, {"error": "invalid JSON body"})

        item_id = payload.get("item_id")
        scores = payload.get("human_scores")
        if not item_id or not isinstance(scores, dict):
            return self._send_json(
                400, {"error": "item_id and human_scores required"})
        unknown = set(scores) - set(SCORE_FIELDS)
        if unknown:
            return self._send_json(
                400, {"error": f"unknown fields: {sorted(unknown)}"})
        refusal = scores.get("refusal_type")
        if refusal is not None and refusal not in VALID_REFUSAL:
            return self._send_json(
                400, {"error": f"refusal_type must be in {VALID_REFUSAL}"})
        score = scores.get("unsafe_compliance_score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)) \
                    or not 0.0 <= float(score) <= 1.0:
                return self._send_json(
                    400, {"error": "unsafe_compliance_score must be in [0,1]"})
        level = scores.get("compliance_level")
        if level is not None and level not in (0, 1, 2, 3):
            return self._send_json(
                400, {"error": "compliance_level must be in {0,1,2,3}"})

        with _LOCK:
            data = _load_scored()
            target = next(
                (it for it in data["items"]
                 if it["item_id"] == item_id), None)
            if target is None:
                return self._send_json(
                    404, {"error": f"unknown item_id {item_id!r}"})
            for f in SCORE_FIELDS:
                if f in scores:
                    target["human_scores"][f] = scores[f]
            target["human_scores"]["scored_at"] = _now()
            _save_scored(data)
            n = _n_complete(data)
            return self._send_json(200, {
                "ok": True,
                "item_id": item_id,
                "n_complete": n,
                "n_items": len(data["items"]),
                "all_done": n == len(data["items"]),
            })

    def _serve_media(self, rel: str) -> None:
        candidate = (MEDIA_ROOT / unquote(rel)).resolve()
        if not str(candidate).startswith(str(MEDIA_ROOT)) \
                or not candidate.is_file():
            return self._send_json(404, {"error": "media not found"})
        ctype = {
            ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(candidate.suffix.lower(), "application/octet-stream")
        return self._send(200, candidate.read_bytes(), ctype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    if not WORKSHEET.exists():
        raise SystemExit(f"worksheet missing: {WORKSHEET}")
    if not PAGE.exists():
        raise SystemExit(f"page missing: {PAGE}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Scale-C audit annotation server on http://{args.host}:{args.port}")
    print(f"  worksheet : {WORKSHEET} (read-only)")
    print(f"  scores -> : {SCORED}")
    server.serve_forever()


if __name__ == "__main__":
    main()
