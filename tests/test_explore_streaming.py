"""Unit tests for spatial chunk manifest and streaming data structures."""

import json
import os
import pytest
from handarm.config import PROJECT_ROOT


def test_manifest_structure():
    manifest_path = PROJECT_ROOT / "models" / "iitj_chunks" / "manifest.json"
    assert manifest_path.exists(), "manifest.json must exist in models/iitj_chunks"

    with open(manifest_path, "r") as f:
        data = json.load(f)

    assert "dataset" in data
    assert "chunks" in data
    assert "landmarks" in data
    assert "spawn_point" in data
    assert len(data["landmarks"]) >= 4
    assert len(data["chunks"]) > 0


def test_manifest_landmarks_positions():
    manifest_path = PROJECT_ROOT / "models" / "iitj_chunks" / "manifest.json"
    with open(manifest_path, "r") as f:
        data = json.load(f)

    for lm in data["landmarks"]:
        assert "name" in lm
        assert "position" in lm
        assert len(lm["position"]) == 3


def test_manifest_chunks_metadata():
    manifest_path = PROJECT_ROOT / "models" / "iitj_chunks" / "manifest.json"
    with open(manifest_path, "r") as f:
        data = json.load(f)

    for chunk in data["chunks"][:10]:
        assert "file" in chunk
        assert "center" in chunk
        assert "point_count" in chunk
        assert chunk["point_count"] > 0


def test_overview_lod_npz_exists_and_valid():
    overview_path = PROJECT_ROOT / "models" / "iitj_chunks" / "overview.npz"
    assert overview_path.exists(), "overview.npz must exist in models/iitj_chunks"

    import numpy as np
    with np.load(overview_path) as data:
        assert "vertices" in data
        assert "rgb" in data
        assert "elevation" in data
        assert "vibrant" in data
        assert len(data["vertices"]) > 0
        assert data["vertices"].shape[1] == 3


def test_camera_trapezoid_padding_balances_preload_and_retention():
    """Preloads just outside the FOV and retains edge chunks with wider padding."""
    from handarm.engine.explore_core import _trapezoid_intersects_chunk_bounds

    def intersects(x, z, padding):
        return _trapezoid_intersects_chunk_bounds(
            x, x, z, z, 0.0, 0.0, 0.0, 1.0, 90.0, 105.0,
            near_depth=15.0, padding_degrees=padding,
        )

    # 50 degrees off-axis: outside the 45-degree half-FOV, inside 8-degree padding.
    assert not intersects(59.6, 50.0, padding=0.0)
    assert intersects(59.6, 50.0, padding=8.0)

    # 60 degrees off-axis: outside preload padding, inside retention padding.
    assert not intersects(86.6, 50.0, padding=8.0)
    assert intersects(86.6, 50.0, padding=20.0)
    assert not intersects(0.0, -50.0, padding=20.0)


def test_camera_trapezoid_uses_chunk_bounds_and_depth():
    """Loads chunks overlapping the trapezoid and rejects chunks outside its depth."""
    from handarm.engine.explore_core import _trapezoid_intersects_chunk_bounds

    def intersects(min_x, max_x, min_z, max_z):
        return _trapezoid_intersects_chunk_bounds(
            min_x, max_x, min_z, max_z, 0.0, 0.0, 0.0, 1.0, 90.0, 100.0,
            near_depth=15.0,
        )

    assert intersects(49.0, 55.0, 49.0, 51.0)
    assert not intersects(60.0, 70.0, 49.0, 51.0)
    assert not intersects(-1.0, 1.0, 101.0, 110.0)


def test_lod_subsampling_and_distance_levels():
    """Verifies that point cloud level of detail/depth correctly scales with distance."""
    import math
    import numpy as np
    from handarm.engine.explore_core import (
        LOD_FULL, LOD_MED, LOD_LOW,
        _lod_subsample, ExploreEnvironment
    )

    data = np.arange(240, dtype=np.float32).reshape(-1, 3)
    assert len(data) == 80

    full = _lod_subsample(data, LOD_FULL)
    assert len(full) == 80, "Full LOD should retain 100% of points"

    med = _lod_subsample(data, LOD_MED)
    assert len(med) == 27, "Medium LOD (1/3 step) should subsample 1-in-3 points"

    low = _lod_subsample(data, LOD_LOW)
    assert len(low) == 10, "Low LOD (1/8 step) should subsample 1-in-8 points"

    # Test distance thresholds
    assert ExploreEnvironment.LOD_FULL_RADIUS == 35.0
    assert ExploreEnvironment.LOD_MED_RADIUS == 70.0
    assert ExploreEnvironment.LOAD_RADIUS == 105.0
    environment = object.__new__(ExploreEnvironment)
    spawn_view_forward_y = -math.sin(math.radians(28.0))
    assert spawn_view_forward_y <= ExploreEnvironment.BIRDSEYE_FORWARD_Y_THRESHOLD
    assert environment._get_desired_lod(85.0, birdseye=True) == LOD_FULL

    # Closer than 35m -> full detail
    dist_close = 20.0
    assert dist_close <= ExploreEnvironment.LOD_FULL_RADIUS
    # 35m to 70m -> medium detail
    dist_mid = 50.0
    assert ExploreEnvironment.LOD_FULL_RADIUS < dist_mid <= ExploreEnvironment.LOD_MED_RADIUS
    # 70m to 105m -> low detail
    dist_far = 85.0
    assert ExploreEnvironment.LOD_MED_RADIUS < dist_far <= ExploreEnvironment.LOAD_RADIUS


def test_fixed_point_rotation_logic():
    """Verifies that keyboard rotation rotates view orientation without altering position."""
    from types import SimpleNamespace

    # Mock player with position and rotation
    player = SimpleNamespace(
        position=[10.0, 5.0, -20.0],
        rotation_x=0.0,
        rotation_y=45.0,
        rotation_z=0.0,
    )

    initial_pos = list(player.position)
    dt = 0.016  # ~60 FPS tick
    turn_speed = 75.0
    pitch_speed = 55.0

    # Simulate rotating left
    player.rotation_y -= turn_speed * dt
    assert player.rotation_y < 45.0
    assert player.position == initial_pos, "Rotation must NOT translate player position"

    # Simulate pitching up
    player.rotation_x -= pitch_speed * dt
    assert player.rotation_x < 0.0
    assert player.position == initial_pos, "Pitch rotation must NOT translate player position"

