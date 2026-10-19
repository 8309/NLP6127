from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def config():
    with (ROOT / "config.toml").open("rb") as f:
        return tomllib.load(f)


def words(text):
    return re.findall(r"[a-z0-9]+(?:[.'/-][a-z0-9]+)*", text.lower())


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def database(path=None):
    path = Path(path or DATA / "annotations.sqlite3")
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY, record_id TEXT NOT NULL, annotator TEXT NOT NULL,
        mode TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)""")
    db.execute("CREATE TABLE IF NOT EXISTS suggestion_views (record_id TEXT, annotator TEXT, created_at TEXT, PRIMARY KEY(record_id,annotator))")
    db.commit()
    return db


def mark_suggestion_view(record_id, annotator, path=None):
    if annotator.strip():
        with database(path) as db:
            db.execute("INSERT OR IGNORE INTO suggestion_views VALUES (?,?,?)", (record_id, annotator.strip(), now()))


def has_suggestion_view(record_id, annotator, path=None):
    with database(path) as db:
        return bool(db.execute("SELECT 1 FROM suggestion_views WHERE record_id=? AND annotator=?", (record_id, annotator.strip())).fetchone())


def save_review(record_id, annotator, mode, payload, path=None):
    if not annotator.strip() or mode not in {"human_independent", "human_reviewed"}:
        raise ValueError("A human annotator ID and valid provenance are required")
    with database(path) as db:
        seen = db.execute("SELECT 1 FROM suggestion_views WHERE record_id=? AND annotator=?", (record_id, annotator.strip())).fetchone()
        assisted_before = db.execute("SELECT 1 FROM reviews WHERE record_id=? AND annotator=? AND mode='human_reviewed'", (record_id, annotator.strip())).fetchone()
        if seen or assisted_before:
            mode = "human_reviewed"
        db.execute("INSERT INTO reviews (record_id,annotator,mode,payload,created_at) VALUES (?,?,?,?,?)",
                   (record_id, annotator.strip(), mode, json.dumps(payload, ensure_ascii=False), now()))


def latest_reviews(path=None):
    with database(path) as db:
        rows = db.execute("SELECT record_id,annotator,mode,payload,created_at FROM reviews ORDER BY id").fetchall()
    latest = {}
    for rid, who, mode, payload, at in rows:
        latest[(rid, who)] = dict(record_id=rid, annotator=who, mode=mode, payload=json.loads(payload), created_at=at)
    return list(latest.values())
