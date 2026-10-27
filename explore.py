"""Explore CSV, LAZ/LAS, or chunked point clouds with Hand-AR."""

import argparse
import json
import os
from pathlib import Path

MAX_POINTS = 20_499_455


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore point clouds with Hand-AR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    python explore.py
    python explore.py pointcloud.csv
    python explore.py campus.laz --points {MAX_POINTS}
    python explore.py campus.laz --no-cache
        """,
    )
    parser.add_argument("input_path", nargs="?", help="CSV, LAZ, or LAS file")
    parser.add_argument(
        "--points", type=int, default=MAX_POINTS, metavar="N",
        help=f"Target points (default: {MAX_POINTS:,}, maximum: {MAX_POINTS:,})",
    )
    parser.add_argument(
        "--max-points", type=int, default=MAX_POINTS, metavar="N",
        help=f"Maximum points to read (default: {MAX_POINTS:,})",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable Parquet caching")
    parser.add_argument(
        "--single-mesh", action="store_true",
        help="Render the LAZ/LAS data as one mesh instead of chunk streaming",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow viewing decoded prefix data if the source is incomplete",
    )
    parser.add_argument(
        "--rebuild-chunks", "-f", action="store_true",
        help="Rebuild dynamic spatial chunks from the source file",
    )
    parser.add_argument("--chunks-dir", metavar="DIR", help="Use this point-cloud chunk directory")
    parser.add_argument("--model-path", help="Optional 3D model to display with the point cloud")
    args = parser.parse_args()

    input_path = args.input_path
    if input_path is None:
        candidates = [
            os.path.join(directory, name)
            for directory in ("iitj_data", "models", ".")
            if os.path.isdir(directory)
            for name in os.listdir(directory)
            if name.lower().endswith((".laz", ".las"))
        ]
        if candidates:
            candidates.sort(
                key=lambda path: (
                    0 if "new" in os.path.basename(path).lower()
                    else 1 if "pointcloud" in os.path.basename(path).lower()
                    else 2,
                    -os.path.getmtime(path),
                )
            )
            input_path = candidates[0]
            print(f"Auto-detected LiDAR file: {input_path}")
        else:
            from handarm.engine.unified_viewer import main as explore_main
            explore_main(model_path=args.model_path, initial_mode="pointcloud")
            return

    if not os.path.exists(input_path):
        parser.error(f"File not found: {input_path}")

    if os.path.splitext(input_path)[1].lower() not in (".laz", ".las"):
        from handarm.engine.unified_viewer import main as explore_main
        explore_main(
            target_path=input_path, model_path=args.model_path,
            initial_mode="pointcloud",
        )
        return

    try:
        import laspy  # noqa: F401
    except ImportError:
        parser.error("laspy is required; install it with: pip install 'laspy[lazrs]'")

    target_points = min(args.points, MAX_POINTS) if args.points > 0 else MAX_POINTS
    max_points = min(args.max_points, MAX_POINTS) if args.max_points > 0 else MAX_POINTS

    chunks_dir = args.chunks_dir
    if not args.single_mesh and chunks_dir is None and not args.rebuild_chunks:
        stem = Path(input_path).stem
        candidates = (
            os.path.join("models", f"{stem}_chunks"),
            os.path.join(os.path.dirname(input_path), f"{stem}_chunks"),
            os.path.join("models", "iitj_chunks"),
            os.path.join(os.path.dirname(input_path), "chunks"),
        )
        for candidate in candidates:
            manifest_file = os.path.join(candidate, "manifest.json")
            if not os.path.isdir(candidate) or not os.path.isfile(manifest_file):
                continue
            if not any(
                name.endswith(".npz") and name != "overview.npz"
                for name in os.listdir(candidate)
            ):
                continue
            try:
                with open(manifest_file, encoding="utf-8") as manifest:
                    source = json.load(manifest).get("source_file")
                if source and os.path.basename(source).lower() != os.path.basename(input_path).lower():
                    continue
                if (
                    os.path.getmtime(input_path) > os.path.getmtime(manifest_file)
                    and os.path.basename(candidate) != f"{stem}_chunks"
                ):
                    continue
            except (OSError, ValueError):
                continue
            chunks_dir = candidate
            break

    if not args.single_mesh and chunks_dir is None:
        chunks_dir = os.path.join("models", f"{Path(input_path).stem}_chunks")
        print(f"Building dynamic spatial chunks from {input_path} into {chunks_dir}...")
        from scripts.build_spatial_chunks import build_spatial_chunks
        build_spatial_chunks(laz_path=input_path, output_dir=chunks_dir)

    from handarm.engine.unified_viewer import main as explore_main
    if chunks_dir is not None:
        explore_main(
            target_path=chunks_dir, model_path=args.model_path,
            initial_mode="pointcloud",
        )
        return

    from handarm.geometry.laz_loader import load_laz_to_dataframe
    point_cloud = load_laz_to_dataframe(
        input_path,
        target_points=target_points,
        max_points=max_points,
        auto_center=True,
        use_cache=not args.no_cache,
        allow_partial=args.allow_partial,
    )
    if len(point_cloud) > MAX_POINTS:
        point_cloud = point_cloud.iloc[:MAX_POINTS]
    print(f"Rendering {len(point_cloud):,} points...")
    explore_main(
        point_cloud=point_cloud,
        target_path=input_path,
        model_path=args.model_path,
        initial_mode="pointcloud",
    )


if __name__ == "__main__":
    main()
