"""
Interactive 3D Model & Point Cloud Viewer for Hand-AR Project.

Usage:
    python3 trial2.py [model_path] [mode]
    mode: 'inspect' (default) or 'explore'
"""
import sys

from handarm import config
from handarm.engine.unified_viewer import main as run_viewer


def main() -> None:
    selected_asset = sys.argv[1] if len(sys.argv) > 1 else None
    requested_mode = sys.argv[2] if len(sys.argv) > 2 else "inspect"
    initial_mode = "pointcloud" if requested_mode == "explore" else "object"
    target_path = str(config.PROJECT_ROOT / "models" / "iitj_chunks")

    run_viewer(
        target_path=selected_asset if initial_mode == "pointcloud" else target_path,
        model_path=selected_asset if initial_mode == "object" else None,
        initial_mode=initial_mode,
    )

if __name__ == "__main__":
    main()