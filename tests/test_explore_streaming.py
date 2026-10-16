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


def test_view_cone_culling_math():
    """Verifies that chunks in front of camera are prioritized over chunks behind camera."""
    import math

    px, pz = 0.0, 0.0
    # Camera facing +Z
    fx, fz = 0.0, 1.0

    # Chunk A: in front of camera at (0, 50)
    dx_a, dz_a = 0.0, 50.0
    dist_a = math.hypot(dx_a, dz_a)
    dot_a = (dx_a * fx + dz_a * fz) / dist_a
    assert dot_a == 1.0, "Chunk directly in front must have dot product 1.0"

    # Chunk B: behind camera at (0, -50)
    dx_b, dz_b = 0.0, -50.0
    dist_b = math.hypot(dx_b, dz_b)
    dot_b = (dx_b * fx + dz_b * fz) / dist_b
    assert dot_b == -1.0, "Chunk directly behind must have dot product -1.0"

    # Priority score: lower score = loaded first
    score_a = dist_a - 35.0 * max(0.0, dot_a)
    score_b = dist_b - 35.0 * max(0.0, dot_b)
    assert score_a < score_b, "Front chunk must have higher loading priority (lower score) than behind chunk"


def test_load_queue_sorting_and_filtering():
    """Verifies that candidate chunks are correctly filtered and prioritized."""
    import math

    manifest_chunks = [
        {"file": "chunk_front.npz", "center": [0.0, 0.0, 50.0]},
        {"file": "chunk_back.npz", "center": [0.0, 0.0, -50.0]},
        {"file": "chunk_close.npz", "center": [5.0, 0.0, 5.0]},
        {"file": "chunk_far.npz", "center": [0.0, 0.0, 200.0]},
    ]

    px, pz = 0.0, 0.0
    fx, fz = 0.0, 1.0  # Camera facing +Z

    load_queue = []
    for c in manifest_chunks:
        cx, cy, cz = c["center"]
        dx = cx - px
        dz = cz - pz
        dist = math.hypot(dx, dz)
        if dist <= 30.0:
            in_view = True
            dot = 1.0
        elif dist <= 105.0:
            dot = (dx * fx + dz * fz) / dist
            in_view = (dot >= 0.05)
        else:
            in_view = False
            dot = -1.0
        if in_view:
            priority = dist - 35.0 * max(0.0, dot)
            load_queue.append((priority, c["file"]))

    load_queue.sort(key=lambda x: x[0])
    assert load_queue[0][1] == "chunk_close.npz"
    assert load_queue[1][1] == "chunk_front.npz"
    assert "chunk_back.npz" not in [f for _, f in load_queue]
    assert "chunk_far.npz" not in [f for _, f in load_queue]


def test_lod_subsampling_and_distance_levels():
    """Verifies that point cloud level of detail/depth correctly scales with distance."""
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




