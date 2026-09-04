"""Command-line entry points."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import click
import cv2

from bird.classify import StubClassifier
from bird.config import Settings
from bird.detect import StubDetector
from bird.frames import CameraSource, DirectorySource, LatestFrameSource
from bird.motion import AlwaysOpenGate, Gate, MotionGate
from bird.pipeline import Pipeline
from bird.store import Store


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
def main(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _build_pipeline(
    settings: Settings, source: LatestFrameSource | DirectorySource, gate: Gate
) -> Pipeline:
    store = Store(settings.db_path, settings.crops_dir)
    return Pipeline(
        source=source,
        gate=gate,
        detector=StubDetector(),
        classifier=StubClassifier(),
        store=store,
        debounce_seconds=settings.debounce_seconds,
        min_det_score=settings.min_det_score,
    )


def _print_visits(store: Store) -> None:
    visits = store.visits()
    click.echo(f"{len(visits)} visit(s)")
    for v in visits:
        start = datetime.fromtimestamp(v.start, tz=UTC).isoformat(timespec="seconds")
        duration = (v.end - v.start) if v.end is not None else 0.0
        n = len(store.observations(v.id))
        click.echo(
            f"  #{v.id} {start} {duration:6.1f}s {n:4d} obs  {v.species} ({v.confidence:.2f})"
            if v.confidence is not None
            else f"  #{v.id} {start} (open)"
        )


@main.command()
def run() -> None:
    """Run the pipeline continuously on the camera."""
    settings = Settings()
    camera = CameraSource(settings.camera_index, settings.width, settings.height)
    source = LatestFrameSource(camera)
    pipeline = _build_pipeline(settings, source, MotionGate(settings.motion_min_area))
    interval = 1.0 / settings.fps_target if settings.fps_target > 0 else 0.0
    click.echo(f"Running on camera {settings.camera_index}; Ctrl-C to stop.")
    try:
        pipeline.run(min_interval=interval)
    except KeyboardInterrupt:
        pass
    finally:
        _print_visits(pipeline.store)
        pipeline.store.close()


@main.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--interval", default=1.0, show_default=True, help="Seconds between frames.")
@click.option("--motion/--no-motion", default=True, show_default=True, help="Use the motion gate.")
def replay(directory: Path, interval: float, motion: bool) -> None:
    """Run the pipeline over a directory of images."""
    settings = Settings()
    source = DirectorySource(directory, interval=interval, start=datetime.now(tz=UTC).timestamp())
    gate: Gate = MotionGate(settings.motion_min_area) if motion else AlwaysOpenGate()
    pipeline = _build_pipeline(settings, source, gate)
    pipeline.run()
    click.echo(f"{pipeline.frames_seen} frames, {pipeline.frames_gated} passed the motion gate")
    _print_visits(pipeline.store)
    pipeline.store.close()


@main.command()
def visits() -> None:
    """List recorded visits."""
    settings = Settings()
    with Store(settings.db_path, settings.crops_dir) as store:
        _print_visits(store)


@main.command()
def snapshot() -> None:
    """Grab one frame from the camera and show it."""
    settings = Settings()
    source = CameraSource(settings.camera_index, settings.width, settings.height)
    frame = next(iter(source.frames()))
    cv2.imshow("Camera Frame", frame.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@main.command()
def focus() -> None:
    """Live focus-score overlay (Laplacian variance) to help set manual focus. Press q to quit."""
    settings = Settings()
    source = CameraSource(settings.camera_index, settings.width, settings.height, warmup=0)
    try:
        for frame in source.frames():
            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            cv2.putText(
                frame.image,
                f"Focus: {score:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Focus Helper", frame.image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
