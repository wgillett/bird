"""Orchestrator: frame -> motion gate -> detector -> classifier -> store, grouped into visits."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from bird.classify import Classifier, Prediction
from bird.detect import Detection, Detector
from bird.frames import Frame, FrameSource
from bird.motion import Gate
from bird.store import Store

log = logging.getLogger(__name__)


@dataclass
class OpenVisit:
    id: int
    last_detection: float
    votes: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    best_score: float = -1.0
    best_crop_path: str | None = None

    def record(self, prediction: Prediction, det_score: float, crop_path: str | None) -> None:
        self.votes[prediction.species] += prediction.probability
        score = det_score * prediction.probability
        if score > self.best_score:
            self.best_score = score
            self.best_crop_path = crop_path

    def verdict(self) -> tuple[str, float]:
        total = sum(self.votes.values())
        if total <= 0:
            return "unknown", 0.0
        species = max(self.votes, key=lambda s: self.votes[s])
        return species, self.votes[species] / total


class Pipeline:
    def __init__(
        self,
        source: FrameSource,
        gate: Gate,
        detector: Detector,
        classifier: Classifier,
        store: Store,
        debounce_seconds: float = 10.0,
        min_det_score: float = 0.5,
        save_crops: bool = True,
    ) -> None:
        self.source = source
        self.gate = gate
        self.detector = detector
        self.classifier = classifier
        self.store = store
        self.debounce_seconds = debounce_seconds
        self.min_det_score = min_det_score
        self.save_crops = save_crops
        self._open: OpenVisit | None = None
        self.frames_seen = 0
        self.frames_gated = 0

    def process(self, frame: Frame) -> list[Detection]:
        """Run one frame through the cascade. Returns the detections that passed."""
        self.frames_seen += 1
        motion = self.gate.check(frame.image)
        if not motion.triggered:
            self._maybe_close(frame.timestamp)
            return []
        self.frames_gated += 1

        detections = [d for d in self.detector.detect(frame.image) if d.score >= self.min_det_score]
        if not detections:
            self._maybe_close(frame.timestamp)
            return []

        visit = self._ensure_open(frame.timestamp)
        for det in detections:
            crop = det.bbox.crop(frame.image)
            predictions = self.classifier.classify(crop)
            top = predictions[0] if predictions else Prediction("unknown", 0.0)
            crop_path = (
                str(self.store.save_crop(crop, frame.timestamp)) if self.save_crops else None
            )
            self.store.add_observation(
                visit_id=visit.id,
                timestamp=frame.timestamp,
                bbox=det.bbox,
                det_score=det.score,
                species=top.species,
                prob=top.probability,
                crop_path=crop_path,
            )
            visit.record(top, det.score, crop_path)
            log.info(
                "visit %d: %s (%.2f) det=%.2f bbox=%s",
                visit.id,
                top.species,
                top.probability,
                det.score,
                det.bbox,
            )
        visit.last_detection = frame.timestamp
        return detections

    def run(self, min_interval: float = 0.0) -> None:
        """Process frames until the source is exhausted or interrupted."""
        try:
            for frame in self.source.frames():
                started = time.monotonic()
                self.process(frame)
                if min_interval > 0:
                    remaining = min_interval - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            self.finish()

    def finish(self) -> None:
        """Close any open visit. Call when the frame stream ends."""
        if self._open is not None:
            self._close(self._open, self._open.last_detection)

    def _ensure_open(self, timestamp: float) -> OpenVisit:
        self._maybe_close(timestamp)
        if self._open is None:
            visit_id = self.store.open_visit(timestamp, self.detector.id, self.classifier.id)
            self._open = OpenVisit(id=visit_id, last_detection=timestamp)
            log.info("visit %d opened at %.3f", visit_id, timestamp)
        return self._open

    def _maybe_close(self, timestamp: float) -> None:
        if self._open is not None and timestamp - self._open.last_detection > self.debounce_seconds:
            self._close(self._open, self._open.last_detection)

    def _close(self, visit: OpenVisit, end: float) -> None:
        species, confidence = visit.verdict()
        self.store.close_visit(visit.id, end, species, confidence, visit.best_crop_path)
        log.info("visit %d closed: %s (%.2f)", visit.id, species, confidence)
        self._open = None
