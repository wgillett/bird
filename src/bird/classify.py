"""Classifier stage: what kind of bird is in this crop?"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bird.frames import Image


@dataclass(frozen=True)
class Prediction:
    species: str
    probability: float


class Classifier(Protocol):
    id: str

    def classify(self, crop: Image) -> list[Prediction]:
        """Return predictions sorted by descending probability."""
        ...


class StubClassifier:
    """Always answers "unknown". Lets the pipeline run before a model is chosen."""

    id = "stub-classifier"

    def classify(self, crop: Image) -> list[Prediction]:
        return [Prediction(species="unknown", probability=1.0)]
