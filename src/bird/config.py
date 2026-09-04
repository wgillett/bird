"""Typed settings. Override with BIRD_* environment variables or a .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BIRD_", env_file=".env", extra="ignore")

    camera_index: int = 0
    width: int | None = 1920
    height: int | None = 1080
    fps_target: float = 5.0
    """Maximum frames per second the processing loop will consume."""

    motion_min_area: float = 0.002
    """Fraction of the frame that must be foreground for the motion gate to trigger."""
    min_det_score: float = 0.5
    debounce_seconds: float = 10.0
    """Gap without detections after which a visit is closed."""

    data_dir: Path = Path("data")
    detector_model: Path | None = None
    classifier_model: Path | None = None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "bird.sqlite"

    @property
    def crops_dir(self) -> Path:
        return self.data_dir / "crops"
