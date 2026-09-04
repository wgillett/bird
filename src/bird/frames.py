"""Frame sources: the camera, or a directory of images for replay and tests."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Frame:
    timestamp: float
    image: Image
    source: str


class FrameSource(Protocol):
    def frames(self) -> Iterator[Frame]: ...


class DirectorySource:
    """Yield the images in a directory, in sorted filename order, with synthetic timestamps."""

    def __init__(self, directory: Path, interval: float = 1.0, start: float = 0.0) -> None:
        self.directory = directory
        self.interval = interval
        self.start = start

    def paths(self) -> list[Path]:
        return sorted(p for p in self.directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    def frames(self) -> Iterator[Frame]:
        for i, path in enumerate(self.paths()):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read image {path}")
            yield Frame(
                timestamp=self.start + i * self.interval,
                image=np.asarray(image, dtype=np.uint8),
                source=str(path),
            )


class CameraSource:
    """Yield frames from a USB camera via OpenCV."""

    def __init__(
        self,
        index: int = 0,
        width: int | None = None,
        height: int | None = None,
        warmup: int = 10,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.warmup = warmup

    def frames(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.index}")
        try:
            if self.width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # Discard a few frames to let auto-exposure settle.
            for _ in range(self.warmup):
                cap.read()
            while True:
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError("Could not read frame from camera")
                yield Frame(
                    timestamp=time.time(),
                    image=np.asarray(image, dtype=np.uint8),
                    source=f"camera:{self.index}",
                )
        finally:
            cap.release()


class LatestFrameSource:
    """Run an inner source on a thread and yield only its newest frame.

    Frames produced while the consumer is busy are dropped, so a slow pipeline
    never falls behind a fast camera.
    """

    def __init__(self, inner: FrameSource) -> None:
        self.inner = inner
        self._latest: Frame | None = None
        self._seq = 0
        self._done = False
        self._error: BaseException | None = None
        self._cond = threading.Condition()

    def _capture(self) -> None:
        try:
            for frame in self.inner.frames():
                with self._cond:
                    self._latest = frame
                    self._seq += 1
                    self._cond.notify_all()
                    if self._done:
                        return
        except BaseException as exc:  # propagate to consumer
            self._error = exc
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def stop(self) -> None:
        with self._cond:
            self._done = True
            self._cond.notify_all()

    def frames(self) -> Iterator[Frame]:
        thread = threading.Thread(target=self._capture, name="capture", daemon=True)
        thread.start()
        seen = 0
        try:
            while True:
                with self._cond:
                    self._cond.wait_for(lambda: self._seq > seen or self._done)
                    if self._error is not None:
                        raise self._error
                    if self._seq == seen:
                        return
                    seen = self._seq
                    frame = self._latest
                assert frame is not None
                yield frame
        finally:
            self.stop()
