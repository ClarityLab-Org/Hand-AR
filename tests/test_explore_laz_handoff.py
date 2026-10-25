import sys

import pandas as pd


def test_explore_passes_loaded_dataframe_to_engine(monkeypatch, tmp_path):
    import explore
    from handarm.engine import explore_standalone
    from handarm.geometry import laz_loader

    laz_path = tmp_path / "scan.laz"
    laz_path.touch()
    point_cloud = pd.DataFrame({"x": [0], "y": [0], "z": [0]})
    forwarded = {}

    monkeypatch.setattr(
        sys, "argv", [
            "explore.py", str(laz_path), "--points", "100", "--no-cache", "--single-mesh"
        ]
    )
    monkeypatch.setattr(
        laz_loader, "load_laz_to_dataframe", lambda *args, **kwargs: point_cloud
    )
    monkeypatch.setattr(
        explore_standalone, "main", lambda **kwargs: forwarded.update(kwargs)
    )

    explore.main()

    assert forwarded["point_cloud"] is point_cloud
    assert forwarded["target_path"] == str(laz_path)
    assert forwarded["model_path"] is None
