import pytest
import os
import json
import handarm.scanning.database as db

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Sets INDEX_FILE to a temporary file path."""
    db_file = tmp_path / "database.json"
    monkeypatch.setattr(db, "INDEX_FILE", str(db_file))
    return db_file

def test_save_and_load_roundtrip(temp_db):
    """Save valid data and load it back, verifying equality."""
    data = {"query": "test", "results": [{"name": "item1"}]}
    db.save_index(data)
    loaded = db.load_index()
    assert loaded == data

def test_load_nonexistent_returns_empty(temp_db):
    """Loading when file does not exist returns empty dict."""
    assert not temp_db.exists()
    assert db.load_index() == {}

def test_load_corrupted_returns_empty(temp_db, capsys):
    """Loading invalid JSON returns empty dict and prints warning."""
    temp_db.write_text("invalid json {")
    loaded = db.load_index()
    assert loaded == {}
    out, _ = capsys.readouterr()
    assert "Corrupted or empty database" in out

def test_atomic_write_uses_temp_file(tmp_path, monkeypatch):
    """Verify os.replace is used to atomically rename temp file."""
    db_file = tmp_path / "database.json"
    monkeypatch.setattr(db, "INDEX_FILE", str(db_file))
    
    replace_called = False
    original_replace = os.replace
    
    def mock_replace(src, dst):
        nonlocal replace_called
        assert src == str(db_file) + ".tmp"
        assert dst == str(db_file)
        replace_called = True
        original_replace(src, dst)
        
    monkeypatch.setattr(os, "replace", mock_replace)
    
    db.save_index({"test": 1})
    assert replace_called is True

def test_save_creates_file(temp_db):
    """Saving to new path ensures file is created."""
    assert not temp_db.exists()
    db.save_index({"test": 1})
    assert temp_db.exists()
