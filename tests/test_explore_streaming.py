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
