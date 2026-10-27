import sys

from handarm.gui import launcher


def test_point_cloud_launch_does_not_add_default_model(monkeypatch, tmp_path):
    launched = {}
    monkeypatch.setattr(launcher, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda command, cwd: launched.update(command=command, cwd=cwd),
    )

    launcher._launch_local_asset("/data/campus.laz", "explore")

    assert launched["command"] == [
        sys.executable,
        str(tmp_path / "explore.py"),
        "/data/campus.laz",
    ]
    assert launched["cwd"] == str(tmp_path)
