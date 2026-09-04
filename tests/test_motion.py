from __future__ import annotations

from bird.motion import AlwaysOpenGate, MotionGate
from tests.conftest import blank, with_square


def test_static_scene_does_not_trigger() -> None:
    gate = MotionGate(warmup=5)
    results = [gate.check(blank()) for _ in range(30)]
    assert not any(r.triggered for r in results)


def test_moving_square_triggers_after_warmup() -> None:
    gate = MotionGate(warmup=5)
    for _ in range(30):
        gate.check(with_square(10, 40))
    moved = gate.check(with_square(100, 40))
    assert moved.triggered
    assert moved.foreground_fraction > 0.01


def test_always_open_gate() -> None:
    assert AlwaysOpenGate().check(blank()).triggered
