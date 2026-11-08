import pytest
import os
from handarm.scanning.local_scanner import scan_local
from handarm.scanning import sketchfab_client
import handarm.scanning.database as db

@pytest.fixture
def mock_db(tmp_path, monkeypatch):
    db_file = tmp_path / "database.json"
    monkeypatch.setattr(db, "INDEX_FILE", str(db_file))
    monkeypatch.setattr(sketchfab_client, "load_index", db.load_index)
    monkeypatch.setattr(sketchfab_client, "save_index", db.save_index)
    return db_file

def test_scan_local_finds_glb_files(tmp_path):
    """Test local scanner finds .glb files and defaults to inspect mode."""
    (tmp_path / "model1.glb").touch()
    (tmp_path / "model2.glb").touch()
    
    results = scan_local(str(tmp_path))
    assert len(results) == 2
    for r in results:
        assert r["source"] == "local"
        assert r["mode"] == "inspect"
        assert r["name"] in ["model1.glb", "model2.glb"]

def test_scan_local_csv_gets_explore_mode(tmp_path):
    """Test local scanner finds .csv and defaults to explore mode."""
    (tmp_path / "points.csv").touch()
    
    results = scan_local(str(tmp_path))
    assert len(results) == 1
    assert results[0]["name"] == "points.csv"
    assert results[0]["mode"] == "explore"

def test_scan_local_ignores_other_extensions(tmp_path):
    """Test local scanner ignores non-3D file extensions."""
    (tmp_path / "test.txt").touch()
    (tmp_path / "script.py").touch()
    
    results = scan_local(str(tmp_path))
    assert len(results) == 0

def test_unified_search_deduplicates(tmp_path, mock_db, monkeypatch):
    """Test unified search combines and deduplicates results."""
    def mock_scan(folder):
        return [{"name": "model1", "path": "/test/model1.glb", "source": "local"}]
    
    def mock_web(query):
        return [{"name": "model1", "path": "/test/model1.glb", "source": "web", "uid": "123"},
                {"name": "model2", "uid": "456", "source": "web"}]
    
    monkeypatch.setattr(sketchfab_client, "scan_local", mock_scan)
    monkeypatch.setattr(sketchfab_client, "search_web", mock_web)
    
    results = sketchfab_client.unified_search(str(tmp_path), query="car", use_cache=False)
    # /test/model1.glb should be deduplicated based on path
    assert len(results) == 2

def test_unified_search_uses_cache(tmp_path, mock_db, monkeypatch):
    """Test unified search returns cache and doesn't call scan_local when query matches."""
    cached_data = {
        "query": "car",
        "results": [{"name": "cached_model", "source": "local"}]
    }
    db.save_index(cached_data)
    
    scan_called = False
    def mock_scan(folder):
        nonlocal scan_called
        scan_called = True
        return []
    
    monkeypatch.setattr(sketchfab_client, "scan_local", mock_scan)
    
    results = sketchfab_client.unified_search(str(tmp_path), query="car", use_cache=True)
    assert len(results) == 1
    assert results[0]["name"] == "cached_model"
    assert scan_called is False

def test_search_web_returns_empty_without_token(monkeypatch):
    """Test search_web returns empty list when no token is available."""
    monkeypatch.setattr(sketchfab_client, "_token_available", False)
    
    results = sketchfab_client.search_web("test")
    assert results == []
