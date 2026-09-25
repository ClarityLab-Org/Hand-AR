"""Local filesystem scanner for 3D model and point cloud files."""

import os
from typing import Dict, List

from handarm.scanning.database import load_index


def scan_local(root_folder: str) -> List[Dict]:
    """Scans local directory for 3D model files (.glb, .obj, .csv) and assigns view modes.

    CSV files default to 'explore' mode, all others to 'inspect'.
    Preserves previously saved mode selections from database.json.

    Args:
        root_folder: Directory path containing local 3D assets.

    Returns:
        List of dicts describing each local model file with keys:
        name, path, source ('local'), mode ('inspect' or 'explore').
    """
    glb_files: List[Dict] = []
    existing_modes: Dict[str, str] = {}

    # Try to load existing modes from database.json
    # Check parent directory first (project root), then the folder itself
    project_dir = os.path.dirname(root_folder)
    db_path = os.path.join(project_dir, "database.json")

    if not os.path.exists(db_path):
        db_path = os.path.join(root_folder, "database.json")

    data = load_index(db_path)
    for item in data.get("results", []):
        if "name" in item and "mode" in item:
            existing_modes[item["name"]] = item["mode"]

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith((".glb", ".csv", ".obj")):
                full_path = os.path.join(root, file)
                default_mode = "explore" if file.lower().endswith(".csv") else "inspect"
                saved_mode = existing_modes.get(file, default_mode)

                glb_files.append({
                    "name": file,
                    "path": full_path,
                    "source": "local",
                    "mode": saved_mode,
                })

    return glb_files
