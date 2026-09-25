"""Tests for LAS/LAZ loading, sampling, diagnostics, and caching."""

import os
import time
from pathlib import Path

import laspy
import numpy as np
import pytest

from handarm.geometry.laz_loader import (
    _warn_if_implausibly_small,
    describe_laz_failure,
    load_laz_to_dataframe,
)


def _write_las(path: Path, count: int, x_offset: int = 0) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    cloud = laspy.LasData(header)
    cloud.x = np.arange(count) + x_offset
    cloud.y = np.arange(count) * 2
    cloud.z = np.full(count, 10)
    cloud.red = np.full(count, 65535)
    cloud.green = np.zeros(count)
    cloud.blue = np.full(count, 32768)
    cloud.write(path)


def test_sampling_is_bounded_and_deterministic(tmp_path):
    path = tmp_path / "sample.las"
    _write_las(path, 100, 0)

    first = load_laz_to_dataframe(str(path), target_points=25, chunk_size=7, use_cache=False)
    second = load_laz_to_dataframe(str(path), target_points=25, chunk_size=7, use_cache=False)

    assert len(first) == 25
    assert first.equals(second)


def test_sampling_uses_all_points_when_target_is_larger(tmp_path):
    path = tmp_path / "sample.las"
    _write_las(path, 12)

    frame = load_laz_to_dataframe(str(path), target_points=100, chunk_size=3, use_cache=False)

    assert len(frame) == 12


def test_source_identity_invalidates_cache(tmp_path):
    path = tmp_path / "sample.las"
    _write_las(path, 20, 0)
    first = load_laz_to_dataframe(
        str(path), target_points=10, auto_center=False, use_cache=True
    )

    time.sleep(0.001)
    _write_las(path, 20, 1000)
    os.utime(path, ns=(time.time_ns(), time.time_ns()))
    second = load_laz_to_dataframe(
        str(path), target_points=10, auto_center=False, use_cache=True
    )

    assert not first.equals(second)
    assert len(list((tmp_path / "cache").glob("*.parquet"))) == 2


def test_truncated_file_fails_without_partial_result_or_cache(tmp_path):
    path = tmp_path / "broken.las"
    _write_las(path, 100)
    raw = path.read_bytes()
    path.write_bytes(raw[:-100])

    with pytest.raises(RuntimeError, match=r"point offset .*100"):
        load_laz_to_dataframe(str(path), target_points=10, chunk_size=10, use_cache=True)

    assert not list((tmp_path / "cache").glob("*.parquet"))


def test_describe_laz_failure_includes_cause():
    message = describe_laz_failure(
        "broken.laz",
        20,
        100,
        ValueError("short block"),
        point_offset=20,
    )

    assert "broken.laz" in message
    assert "20 of the expected 100" in message
    assert "ValueError: short block" in message


def test_implausibly_small_file_logs_warning(tmp_path, caplog):
    path = tmp_path / "tiny.las"
    path.write_bytes(b"x" * 100)

    with caplog.at_level("WARNING"):
        _warn_if_implausibly_small(str(path), point_count=1_000_000, point_size=36)

    assert "looks small" in caplog.text


def test_allow_partial_returns_marked_data_without_cache(tmp_path):
    path = tmp_path / "partial.las"
    _write_las(path, 100)
    path.write_bytes(path.read_bytes()[:-100])

    frame = load_laz_to_dataframe(
        str(path),
        target_points=10,
        chunk_size=10,
        use_cache=True,
        allow_partial=True,
    )

    assert frame.attrs["is_partial"] is True
    assert frame.attrs["points_decoded"] == 90
    assert not list((tmp_path / "cache").glob("*.parquet"))
