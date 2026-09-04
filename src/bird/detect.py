"""Detector stage: is there a bird, and where?"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bird.frames import Image


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int

    def crop(self, image: Image) -> Image:
        return image[self.y : self.y + self.h, self.x : self.x + self.w]


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    score: float
    label: str = "bird"


class Detector(Protocol):
    id: str

    def detect(self, image: Image) -> list[Detection]: ...


class StubDetector:
    """Reports the whole frame as one bird. Lets the pipeline run before a model is chosen."""

    id = "stub-detector"

    def detect(self, image: Image) -> list[Detection]:
        h, w = image.shape[:2]
        return [Detection(bbox=BBox(0, 0, w, h), score=1.0)]
