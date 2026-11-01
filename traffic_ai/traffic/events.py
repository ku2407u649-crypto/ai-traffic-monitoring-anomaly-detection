"""Event storage: SQLite database, CSV export and snapshot images."""
from __future__ import annotations
import csv
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2

from .engine import Event

COLUMNS = ["t", "frame", "type", "track_id", "cls", "detail", "severity", "snapshot"]


class EventStore:
    def __init__(self, out_dir: Path, save_snapshots: bool = True):
        self.out_dir = Path(out_dir)
        self.snap_dir = self.out_dir / "snapshots"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if save_snapshots:
            self.snap_dir.mkdir(exist_ok=True)
        self.save_snapshots = save_snapshots
        self.db_path = self.out_dir / "events.db"
        if self.db_path.exists():
            self.db_path.unlink()
        self.db = sqlite3.connect(self.db_path)
        self.db.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL, frame INTEGER, type TEXT, "
            "track_id INTEGER, cls TEXT, detail TEXT, severity TEXT, snapshot TEXT, created_at TEXT)")
        self.db.commit()
        self.rows: list[Event] = []

    def add(self, ev: Event, frame_bgr=None) -> None:
        if self.save_snapshots and frame_bgr is not None:
            name = f"{ev.type}_id{ev.track_id}_{ev.t:.1f}s.jpg"
            cv2.imwrite(str(self.snap_dir / name), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            ev.snapshot = f"snapshots/{name}"
        self.rows.append(ev)
        self.db.execute(
            "INSERT INTO events (t, frame, type, track_id, cls, detail, severity, snapshot, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (ev.t, ev.frame, ev.type, ev.track_id, ev.cls, ev.detail, ev.severity, ev.snapshot,
             datetime.now().isoformat(timespec="seconds")))
        self.db.commit()

    def export_csv(self) -> Path:
        p = self.out_dir / "events.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for ev in self.rows:
                d = asdict(ev)
                w.writerow({k: d[k] for k in COLUMNS})
        return p

    def close(self) -> None:
        self.db.close()
