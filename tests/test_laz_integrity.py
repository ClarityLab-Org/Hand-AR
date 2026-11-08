"""Tests for the optional cross-tool integrity wrapper."""

from scripts import verify_laz_integrity


def test_pdal_failure_is_reported_without_propagating(monkeypatch, tmp_path):
    path = tmp_path / "sample.laz"
    path.write_bytes(b"not-a-real-laz")
    monkeypatch.setattr(verify_laz_integrity.shutil, "which", lambda _: "/usr/bin/pdal")
    monkeypatch.setattr(
        verify_laz_integrity.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("pdal crashed")
        ),
    )
    ok, detail = verify_laz_integrity._pdal_check("/usr/bin/pdal", str(path), 1)
    assert not ok
    assert "pdal crashed" in detail
