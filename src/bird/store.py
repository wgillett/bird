"""SQLite store for visits and observations, plus crop files on disk."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2

from bird.detect import BBox
from bird.frames import Image

SCHEMA = """
CREATE TABLE IF NOT EXISTS visit (
    id INTEGER PRIMARY KEY,
    start REAL NOT NULL,
    end REAL,
    best_crop_path TEXT,
    species TEXT,
    confidence REAL,
    detector_id TEXT NOT NULL,
    classifier_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observation (
    id INTEGER PRIMARY KEY,
    visit_id INTEGER NOT NULL REFERENCES visit(id),
    timestamp REAL NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    w INTEGER NOT NULL,
    h INTEGER NOT NULL,
    det_score REAL NOT NULL,
    species TEXT NOT NULL,
    prob REAL NOT NULL,
    crop_path TEXT
);
CREATE INDEX IF NOT EXISTS observation_visit ON observation(visit_id);
"""


@dataclass(frozen=True)
class Visit:
    id: int
    start: float
    end: float | None
    best_crop_path: str | None
    species: str | None
    confidence: float | None
    detector_id: str
    classifier_id: str


@dataclass(frozen=True)
class Observation:
    id: int
    visit_id: int
    timestamp: float
    bbox: BBox
    det_score: float
    species: str
    prob: float
    crop_path: str | None


class Store:
    def __init__(self, db_path: Path, crops_dir: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.crops_dir = crops_dir
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(SCHEMA)
        self._crop_seq = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # Visits

    def open_visit(self, start: float, detector_id: str, classifier_id: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO visit (start, detector_id, classifier_id) VALUES (?, ?, ?)",
            (start, detector_id, classifier_id),
        )
        self._conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    def close_visit(
        self,
        visit_id: int,
        end: float,
        species: str,
        confidence: float,
        best_crop_path: str | None,
    ) -> None:
        self._conn.execute(
            "UPDATE visit SET end = ?, species = ?, confidence = ?, best_crop_path = ?"
            " WHERE id = ?",
            (end, species, confidence, best_crop_path, visit_id),
        )
        self._conn.commit()

    def visits(self) -> list[Visit]:
        rows = self._conn.execute(
            "SELECT id, start, end, best_crop_path, species, confidence, detector_id, classifier_id"
            " FROM visit ORDER BY start"
        ).fetchall()
        return [Visit(*row) for row in rows]

    # Observations

    def add_observation(
        self,
        visit_id: int,
        timestamp: float,
        bbox: BBox,
        det_score: float,
        species: str,
        prob: float,
        crop_path: str | None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO observation"
            " (visit_id, timestamp, x, y, w, h, det_score, species, prob, crop_path)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                visit_id,
                timestamp,
                bbox.x,
                bbox.y,
                bbox.w,
                bbox.h,
                det_score,
                species,
                prob,
                crop_path,
            ),
        )
        self._conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    def observations(self, visit_id: int) -> list[Observation]:
        rows = self._conn.execute(
            "SELECT id, visit_id, timestamp, x, y, w, h, det_score, species, prob, crop_path"
            " FROM observation WHERE visit_id = ? ORDER BY timestamp",
            (visit_id,),
        ).fetchall()
        return [
            Observation(
                id=r[0],
                visit_id=r[1],
                timestamp=r[2],
                bbox=BBox(r[3], r[4], r[5], r[6]),
                det_score=r[7],
                species=r[8],
                prob=r[9],
                crop_path=r[10],
            )
            for r in rows
        ]

    # Crops

    def save_crop(self, crop: Image, timestamp: float) -> Path:
        """Write a crop as JPEG under crops_dir/YYYY/MM/DD/ and return its path."""
        when = datetime.fromtimestamp(timestamp, tz=UTC)
        directory = self.crops_dir / when.strftime("%Y/%m/%d")
        directory.mkdir(parents=True, exist_ok=True)
        self._crop_seq += 1
        millis = int(timestamp * 1000) % 1000
        path = directory / f"{when.strftime('%H%M%S')}_{millis:03d}_{self._crop_seq:04d}.jpg"
        if not cv2.imwrite(str(path), crop):
            raise RuntimeError(f"Could not write crop to {path}")
        return path
