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
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Explicitly allow viewing decoded prefix data; never caches it",
    )

    args = parser.parse_args()

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
            print(f" Auto-detected LiDAR file: {laz_path}")
        else:
            print(" Error: No .laz or .las file provided.")
            print()
            parser.print_help()
            print("\nPlease provide a path to your LiDAR scan, e.g.:")
            print("  python explore_laz.py path/to/campus.laz")
            sys.exit(1)

    if not os.path.exists(laz_path):
        print(f" Error: File not found: {laz_path}")
        sys.exit(1)

    ext = os.path.splitext(laz_path)[1].lower()
    if ext not in (".laz", ".las"):
        print(f" Warning: Expected a .laz or .las file, but got: {ext}")

    # Verify laspy is installed
    try:
        import laspy
    except ImportError:
        print(" Error: laspy is required to stream .laz files directly.")
        print("   Please install it with: pip install 'laspy[lazrs]'")
        sys.exit(1)

    print("=" * 65)
    print("  Hand-AR Direct LAZ/LAS LiDAR Explorer")
    print(f" Input File:     {laz_path}")
    print(f" Target Density: {args.points:,} points")
    print(f" Caching:        {'Disabled' if args.no_cache else 'Enabled'}")
    if args.allow_partial:
        print("  PARTIAL MODE ENABLED: incomplete data may be displayed")
    print("=" * 65)

    # Pre-stream and cache if needed using our loader
    from handarm.geometry.laz_loader import load_laz_to_dataframe
    df = load_laz_to_dataframe(
        laz_path,
        target_points=args.points,
        auto_center=True,
        use_cache=not args.no_cache,
        allow_partial=args.allow_partial,
    )

    from handarm.engine.explore_standalone import main as explore_main
    explore_main(df)


if __name__ == "__main__":
    main()
