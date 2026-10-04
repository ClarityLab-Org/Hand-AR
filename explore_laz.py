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

# Fixed maximum limit: render only 20,499,455 points and stop taking further points
MAX_POINTS = 20_499_455


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct LAZ/LAS LiDAR Point Cloud Explorer for Hand-AR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    python explore_laz.py campus.laz
    python explore_laz.py /path/to/college_campus.laz --points {MAX_POINTS}
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
        default=MAX_POINTS,
        metavar="N",
        help=f"Target number of points to render (default: {MAX_POINTS:,}, capped at {MAX_POINTS:,})"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=MAX_POINTS,
        metavar="N",
        help=f"Maximum points to read from LiDAR file before stopping (default: {MAX_POINTS:,})"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable Parquet caching of downsampled points"
    )
    parser.add_argument(
        "--single-mesh",
        action="store_true",
        default=False,
        help="Force loading all points as a single monolithic mesh (legacy mode, slower)"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Explicitly allow viewing decoded prefix data; never caches it",
    )

    args = parser.parse_args()

    # Enforce strict point cap: only render up to MAX_POINTS (20,499,455)
    # and stop taking further points after reaching this threshold
    target_points = min(args.points, MAX_POINTS) if args.points > 0 else MAX_POINTS
    max_points = min(args.max_points, MAX_POINTS) if args.max_points > 0 else MAX_POINTS

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

    # Determine if pre-partitioned dynamic spatial chunks exist
    chunks_dir = None
    if not args.single_mesh:
        candidate_chunk_dirs = [
            os.path.join("models", "iitj_chunks"),
            os.path.join(os.path.dirname(laz_path), f"{Path(laz_path).stem}_chunks"),
            os.path.join("models", f"{Path(laz_path).stem}_chunks"),
            os.path.join(os.path.dirname(laz_path), "chunks"),
        ]
        for cdir in candidate_chunk_dirs:
            if os.path.isdir(cdir) and os.path.exists(os.path.join(cdir, "manifest.json")):
                has_npz = any(f.endswith(".npz") and f != "overview.npz" for f in os.listdir(cdir))
                if has_npz:
                    chunks_dir = cdir
                    break

        if chunks_dir is None:
            # Auto-generate dynamic spatial chunks for instant subsequent launches and smooth 60 FPS
            print("=" * 65)
            print("  Building Dynamic Spatial Chunks for Smooth 60 FPS Streaming...")
            print(f" Source:  {laz_path}")
            target_chunks_dir = os.path.join("models", f"{Path(laz_path).stem}_chunks")
            if "iitj" in laz_path.lower():
                target_chunks_dir = os.path.join("models", "iitj_chunks")
            print(f" Target:  {target_chunks_dir}")
            print("=" * 65)
            from scripts.build_spatial_chunks import build_spatial_chunks
            build_spatial_chunks(laz_path=laz_path, output_dir=target_chunks_dir)
            chunks_dir = target_chunks_dir

    if chunks_dir is not None:
        print("=" * 65)
        print("  Hand-AR Dynamic View-Based LiDAR Streamer (Smooth 60 FPS)")
        print(f" Input File:       {laz_path}")
        print(f" Dynamic Chunks:   {chunks_dir}")
        print(f" Streaming Mode:   Load-As-You-View (Ultra-low FPS fluctuations)")
        print("=" * 65)
        from handarm.engine.explore_standalone import main as explore_main
        explore_main(target_path=chunks_dir)
        return

    # Fallback legacy single-mesh mode (when explicitly requested with --single-mesh)
    print("=" * 65)
    print("  Hand-AR Single-Mesh LiDAR Mode (--single-mesh requested)")
    print(f" Input File:     {laz_path}")
    print(f" Target Density: {target_points:,} points (capped at {MAX_POINTS:,})")
    print(f" Stop Limit:     {max_points:,} points")
    print(f" Caching:        {'Disabled' if args.no_cache else 'Enabled'}")
    if args.allow_partial:
        print("  PARTIAL MODE ENABLED: incomplete data may be displayed")
    print("=" * 65)

    from handarm.geometry.laz_loader import load_laz_to_dataframe
    df = load_laz_to_dataframe(
        laz_path,
        target_points=target_points,
        max_points=max_points,
        auto_center=True,
        use_cache=not args.no_cache,
        allow_partial=args.allow_partial,
    )

    if len(df) > MAX_POINTS:
        df = df.iloc[:MAX_POINTS]

    print(f" Rendering {len(df):,} points in Hand-AR Explorer...")

    from handarm.engine.explore_standalone import main as explore_main
    explore_main(df)


if __name__ == "__main__":
    main()
