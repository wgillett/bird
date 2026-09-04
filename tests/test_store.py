from __future__ import annotations

from pathlib import Path

from bird.detect import BBox
from bird.store import Store
from tests.conftest import blank


def test_visit_round_trip(tmp_path: Path) -> None:
    with Store(tmp_path / "db" / "bird.sqlite", tmp_path / "crops") as store:
        visit_id = store.open_visit(1000.0, "det-1", "cls-1")
        crop_path = store.save_crop(blank(), 1000.0)
        store.add_observation(
            visit_id, 1000.0, BBox(1, 2, 30, 40), 0.9, "robin", 0.8, str(crop_path)
        )
        store.add_observation(visit_id, 1001.0, BBox(5, 6, 30, 40), 0.7, "robin", 0.6, None)
        store.close_visit(visit_id, 1001.0, "robin", 0.7, str(crop_path))

        visits = store.visits()
        assert len(visits) == 1
        v = visits[0]
        assert (v.start, v.end, v.species, v.confidence) == (1000.0, 1001.0, "robin", 0.7)
        assert (v.detector_id, v.classifier_id) == ("det-1", "cls-1")
        assert v.best_crop_path == str(crop_path)

        obs = store.observations(visit_id)
        assert [o.timestamp for o in obs] == [1000.0, 1001.0]
        assert obs[0].bbox == BBox(1, 2, 30, 40)
        assert obs[0].crop_path == str(crop_path)
        assert obs[1].crop_path is None


def test_save_crop_uses_dated_directory(tmp_path: Path) -> None:
    with Store(tmp_path / "bird.sqlite", tmp_path / "crops") as store:
        # 2024-03-05 12:00:00 UTC
        path = store.save_crop(blank(), 1709640000.0)
        assert path.exists()
        assert path.parent == tmp_path / "crops" / "2024" / "03" / "05"
        assert path.suffix == ".jpg"
