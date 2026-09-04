from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from bird.classify import Prediction, StubClassifier
from bird.detect import BBox, Detection, StubDetector
from bird.frames import DirectorySource, Frame, Image
from bird.motion import AlwaysOpenGate
from bird.pipeline import Pipeline
from bird.store import Store
from tests.conftest import blank


class _TimedFrames:
    def __init__(self, timestamps: list[float]) -> None:
        self.timestamps = timestamps

    def frames(self) -> Iterator[Frame]:
        for t in self.timestamps:
            yield Frame(timestamp=t, image=blank(), source="test")


class _ToggleDetector:
    """Detects a bird only on frames whose timestamp is in `hits`."""

    id = "toggle"

    def __init__(self, hits: set[float]) -> None:
        self.hits = hits
        self.current = 0.0

    def detect(self, image: Image) -> list[Detection]:
        if self.current in self.hits:
            return [Detection(BBox(0, 0, 10, 10), 0.9)]
        return []


class _VotingClassifier:
    id = "voting"

    def __init__(self, answers: list[Prediction]) -> None:
        self.answers = answers
        self.calls = 0

    def classify(self, crop: Image) -> list[Prediction]:
        answer = self.answers[self.calls % len(self.answers)]
        self.calls += 1
        return [answer]


def test_replay_with_stubs_records_one_visit(frames_dir: Path, tmp_path: Path) -> None:
    store = Store(tmp_path / "bird.sqlite", tmp_path / "crops")
    pipeline = Pipeline(
        source=DirectorySource(frames_dir, interval=1.0, start=1000.0),
        gate=AlwaysOpenGate(),
        detector=StubDetector(),
        classifier=StubClassifier(),
        store=store,
        debounce_seconds=5.0,
    )
    pipeline.run()
    visits = store.visits()
    assert len(visits) == 1
    assert (visits[0].start, visits[0].end) == (1000.0, 1005.0)
    assert visits[0].species == "unknown"
    assert len(store.observations(visits[0].id)) == 6
    assert len(list((tmp_path / "crops").rglob("*.jpg"))) == 6


def test_debounce_splits_visits_and_votes(tmp_path: Path) -> None:
    # Detections at t=0,1,2 then a gap, then t=20,21. Debounce of 5s -> two visits.
    timestamps = [0.0, 1.0, 2.0, 3.0, 20.0, 21.0, 40.0]
    detector = _ToggleDetector(hits={0.0, 1.0, 2.0, 20.0, 21.0})
    classifier = _VotingClassifier(
        [Prediction("robin", 0.9), Prediction("jay", 0.4), Prediction("robin", 0.8)]
    )
    store = Store(tmp_path / "bird.sqlite", tmp_path / "crops")

    class _Source:
        def frames(self) -> Iterator[Frame]:
            for f in _TimedFrames(timestamps).frames():
                detector.current = f.timestamp
                yield f

    pipeline = Pipeline(
        source=_Source(),
        gate=AlwaysOpenGate(),
        detector=detector,
        classifier=classifier,
        store=store,
        debounce_seconds=5.0,
        save_crops=False,
    )
    pipeline.run()

    visits = store.visits()
    assert [(v.start, v.end) for v in visits] == [(0.0, 2.0), (20.0, 21.0)]
    # First visit: robin 0.9 + jay 0.4 + robin 0.8 -> robin with 1.7/2.1
    assert visits[0].species == "robin"
    assert visits[0].confidence is not None
    assert abs(visits[0].confidence - 1.7 / 2.1) < 1e-9
    assert visits[0].detector_id == "toggle"
    assert visits[0].best_crop_path is None
    assert len(store.observations(visits[0].id)) == 3
    assert len(store.observations(visits[1].id)) == 2
