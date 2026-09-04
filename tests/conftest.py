from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from bird.frames import Image


def blank(width: int = 160, height: int = 120) -> Image:
    return np.full((height, width, 3), 64, dtype=np.uint8)


def with_square(x: int, y: int, size: int = 30, width: int = 160, height: int = 120) -> Image:
    """A blank frame with a bright square at (x, y). Moving it between frames creates motion."""
    image = blank(width, height)
    image[y : y + size, x : x + size] = (0, 200, 255)
    return image


def write_images(directory: Path, images: list[Image]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, image in enumerate(images):
        path = directory / f"frame_{i:03d}.png"
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


@pytest.fixture
def frames_dir(tmp_path: Path) -> Path:
    """Six frames: a square that moves across the scene."""
    images = [with_square(10 + 20 * i, 40) for i in range(6)]
    write_images(tmp_path / "frames", images)
    return tmp_path / "frames"
