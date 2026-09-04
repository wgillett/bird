"""Motion gate: the cheapest stage, drops frames where nothing is happening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from bird.frames import Image


@dataclass(frozen=True)
class MotionResult:
    triggered: bool
    foreground_fraction: float


class Gate(Protocol):
    def check(self, image: Image) -> MotionResult: ...


class AlwaysOpenGate:
    """Passes every frame. Useful for tests and for replaying non-consecutive images."""

    def check(self, image: Image) -> MotionResult:
        return MotionResult(triggered=True, foreground_fraction=1.0)


class MotionGate:
    """Background subtraction (MOG2) on a downscaled frame.

    Triggers when the foreground covers at least ``min_area_fraction`` of the frame.
    The first ``warmup`` frames never trigger while the background model settles.
    """

    def __init__(
        self,
        min_area_fraction: float = 0.002,
        scale_width: int = 320,
        history: int = 200,
        var_threshold: float = 32.0,
        warmup: int = 5,
    ) -> None:
        self.min_area_fraction = min_area_fraction
        self.scale_width = scale_width
        self.warmup = warmup
        self._seen = 0
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )

    def check(self, image: Image) -> MotionResult:
        h, w = image.shape[:2]
        if w > self.scale_width:
            scale = self.scale_width / w
            image = np.asarray(
                cv2.resize(image, (self.scale_width, max(1, int(h * scale)))), dtype=np.uint8
            )
        mask = self._subtractor.apply(image)
        fraction = float(np.count_nonzero(mask)) / float(mask.size)
        self._seen += 1
        if self._seen <= self.warmup:
            return MotionResult(triggered=False, foreground_fraction=fraction)
        return MotionResult(
            triggered=fraction >= self.min_area_fraction, foreground_fraction=fraction
        )
