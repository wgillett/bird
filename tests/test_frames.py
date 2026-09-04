from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from bird.frames import DirectorySource, Frame, LatestFrameSource
from tests.conftest import blank


def test_directory_source_yields_sorted_frames_with_timestamps(frames_dir: Path) -> None:
    source = DirectorySource(frames_dir, interval=2.0, start=100.0)
    frames = list(source.frames())
    assert len(frames) == 6
    assert [f.timestamp for f in frames] == [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
    assert frames[0].source.endswith("frame_000.png")
    assert frames[0].image.shape == (120, 160, 3)


def test_directory_source_ignores_non_images(frames_dir: Path) -> None:
    (frames_dir / "notes.txt").write_text("not an image")
    assert len(list(DirectorySource(frames_dir).frames())) == 6


class _Burst:
    """A source that emits many frames at once, then stops."""

    def __init__(self, n: int) -> None:
        self.n = n

    def frames(self) -> Iterator[Frame]:
        for i in range(self.n):
            yield Frame(timestamp=float(i), image=blank(), source="burst")


def test_latest_frame_source_drops_stale_frames() -> None:
    source = LatestFrameSource(_Burst(200))
    seen = list(source.frames())
    assert 1 <= len(seen) <= 200
    # Whatever we saw, the last frame delivered is the newest one available.
    assert seen[-1].timestamp == 199.0
    # Timestamps only ever move forward.
    assert all(a.timestamp < b.timestamp for a, b in zip(seen, seen[1:], strict=False))
