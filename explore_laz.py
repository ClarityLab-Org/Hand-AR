"""
Direct LAZ/LAS LiDAR Point Cloud Explorer for Hand-AR.

Accepts .laz or .las files directly into the Hand-AR interactive 3D navigation pipeline
with real-time MediaPipe hand tracking, perspective point splatting, and architectural shading.

Usage:
    python explore_laz.py [path/to/pointcloud.laz]
    python explore_laz.py campus.laz --points 600000
    python explore_laz.py campus.laz --no-cache

Controls:
    [W][A][S][D] / Hand Gestures : Move & steer through campus
    [Space] / [Left Shift]       : Fly up / Fly down
    [C]                          : Cycle color modes (Architectural, Drone RGB, Blueprint, Contours)
    [P]                          : Cycle point thickness (2px, 3px, 5px, 8px)
    [O]                          : Toggle 3D Perspective Splats (creates solid surfaces)
    [F]                          : Toggle Flight / Grounded mode
    [1] - [4]                    : Teleport to campus landmarks
    [R]                          : Reset player position
    [Q] / [Escape]               : Quit
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct LAZ/LAS LiDAR Point Cloud Explorer for Hand-AR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python explore_laz.py campus.laz
    python explore_laz.py /path/to/college_campus.laz --points 800000
    python explore_laz.py campus.laz --no-cache
        """
    )
    parser.add_argument(
        "laz_path",
        nargs="?",
        default=None,
        help="Path to the .laz or .las LiDAR point cloud file"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=500_000,
        metavar="N",
        help="Target number of points to stream (default: 500,000 for smooth 60 FPS)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable Parquet caching of downsampled points"
    )

    args, remaining = parser.parse_known_args()

    # Find candidate file if none provided
    laz_path = args.laz_path
    if laz_path is None:
        # Search for .laz/.las files in common project locations
        candidates = []
        for search_dir in ["iitj_data", "models", "."]:
            if os.path.exists(search_dir):
                for f in os.listdir(search_dir):
                    if f.lower().endswith((".laz", ".las")):
                        candidates.append(os.path.join(search_dir, f))

        if candidates:
            laz_path = candidates[0]
            print(f"ℹ Auto-detected LiDAR file: {laz_path}")
        else:
            print("❌ Error: No .laz or .las file provided.")
            print()
            parser.print_help()
            print("\nPlease provide a path to your LiDAR scan, e.g.:")
            print("  python explore_laz.py path/to/campus.laz")
            sys.exit(1)

    if not os.path.exists(laz_path):
        print(f"❌ Error: File not found: {laz_path}")
        sys.exit(1)

    ext = os.path.splitext(laz_path)[1].lower()
    if ext not in (".laz", ".las"):
        print(f"⚠️ Warning: Expected a .laz or .las file, but got: {ext}")

    # Verify laspy is installed, fallback to venv if missing
    try:
        import laspy
    except ImportError:
        venv_python = Path(__file__).resolve().parent / "DC_env" / "bin" / "python"
        if venv_python.is_file():
            print("⚠️ Found virtual environment. Re-running with venv python...")
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)
        else:
            print("❌ Error: laspy is required to stream .laz files directly.")
            print("   Please install it with: pip install 'laspy[lazrs]'")
            sys.exit(1)

    print("=" * 65)
    print("🏛️  Hand-AR Direct LAZ/LAS LiDAR Explorer")
    print(f"📦 Input File:     {laz_path}")
    print(f"🎯 Target Density: {args.points:,} points")
    print(f"💾 Caching:        {'Disabled' if args.no_cache else 'Enabled'}")
    print("=" * 65)

    # Pre-stream and cache if needed using our loader
    from handarm.geometry.laz_loader import load_laz_to_dataframe
    df = load_laz_to_dataframe(
        laz_path,
        target_points=args.points,
        auto_center=True,
        use_cache=not args.no_cache,
    )

    # Write temporary CSV as fallback for any component that only accepts CSV paths
    temp_csv = os.path.join(os.path.dirname(laz_path), f".{Path(laz_path).stem}_stream.csv")
    df.to_csv(temp_csv, index=False)
    sys.argv = [sys.argv[0], temp_csv]

    from handarm.engine.explore_standalone import main as explore_main
    try:
        explore_main(model_input=df)
    finally:
        if os.path.exists(temp_csv):
            try:
                os.remove(temp_csv)
            except OSError:
                pass


if __name__ == "__main__":
    main()
