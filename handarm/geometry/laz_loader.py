"""
Direct .laz / .las LiDAR Point Cloud Loader and Streamer for Hand-AR.

Provides high-performance, memory-safe streaming of LASzip (.laz) and LAS (.las)
files directly into pandas DataFrames consumable by Ursina ExploreEnvironment.
Handles 16-bit RGB normalization, origin centering, elevation baselining, and caching.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def load_laz_to_dataframe(
    laz_path: str,
    target_points: int = 500_000,
    chunk_size: int = 500_000,
    auto_center: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Directly reads a .laz or .las LiDAR file and returns a centered, normalized
    pandas DataFrame with columns ['x', 'y', 'z', 'r', 'g', 'b'].

    Args:
        laz_path:      Path to the .laz or .las file.
        target_points: Target number of output points.
                       Set to -1 to load all points without subsampling.
        chunk_size:    Chunk size for streaming decompression to minimize RAM spikes.
        auto_center:   Center X/Z around (0,0) and baseline Z near 0 to avoid GPU float jitter.
        use_cache:     Cache downsampled result as Parquet for instant subsequent loads.

    Returns:
        pd.DataFrame with columns 'x', 'y', 'z', 'r', 'g', 'b'.
    """
    if not os.path.exists(laz_path):
        raise FileNotFoundError(f"LiDAR file not found: {laz_path}")

    # -- Cache check --
    cache_path: Optional[Path] = None
    if use_cache:
        file_hash_name = Path(laz_path).stem + f"_{target_points}pts_cache.parquet"
        cache_dir = Path(laz_path).parent / "cache"
        if not cache_dir.exists():
            cache_dir = Path(__file__).resolve().parent.parent.parent / "models_compressed"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / file_hash_name

            if (cache_path.exists()
                    and os.path.getmtime(str(cache_path)) >= os.path.getmtime(laz_path)):
                print(f"⚡ Loading cached point cloud: {cache_path}...")
                return pd.read_parquet(str(cache_path))
        except Exception:
            cache_path = None

    try:
        import laspy
    except ImportError:
        raise ImportError(
            "laspy with lazrs is required to read .laz/.las files directly.\n"
            "Please run:  pip install 'laspy[lazrs]'"
        )

    print(f"📦 Streaming LiDAR point cloud from: {laz_path}...")

    with laspy.open(laz_path) as reader:
        total_points = reader.header.point_count

        if target_points > 0 and total_points > target_points:
            step = max(1, total_points // target_points)
            print(f"🎯 Subsampling ~{target_points:,} points (1 in {step}) for real-time FPS...")
        else:
            step = 1
            print(f"🎯 Loading all {total_points:,} points...")

        xs, ys, zs = [], [], []
        rs, gs, bs = [], [], []

        has_color = (
            hasattr(reader.header.point_format, 'red')
            or 'red' in reader.header.point_format.dimension_names
        )

        read_count = 0
        try:
            for chunk in reader.chunk_iterator(chunk_size):
                xs.append(np.array(chunk.x[::step], dtype=np.float32))
                ys.append(np.array(chunk.y[::step], dtype=np.float32))
                zs.append(np.array(chunk.z[::step], dtype=np.float32))

                if has_color and hasattr(chunk, 'red') and len(chunk.red) > 0:
                    rr = np.array(chunk.red[::step], dtype=np.float32)
                    rg = np.array(chunk.green[::step], dtype=np.float32)
                    rb = np.array(chunk.blue[::step], dtype=np.float32)

                    max_val = max(float(np.max(rr)), float(np.max(rg)), float(np.max(rb)), 1.0)
                    scale = 255.0 / 65535.0 if max_val > 255 else 1.0

                    rs.append((rr * scale).astype(np.uint8))
                    gs.append((rg * scale).astype(np.uint8))
                    bs.append((rb * scale).astype(np.uint8))

                read_count += len(chunk)
                print(f"  ⏳ Read {read_count:,} / {total_points:,} points...", end="\r", flush=True)

        except Exception as e:
            print(f"\n  ℹ Stream boundary: {type(e).__name__}. Proceeding with collected data...")

        print()

    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    all_z = np.concatenate(zs)

    # Filter extreme noise: keep points within [ground - 5m, ground + 80m] elevation
    z_baseline = float(np.percentile(all_z, 5))
    valid_mask = (all_z >= z_baseline - 5.0) & (all_z <= z_baseline + 80.0)
    all_x = all_x[valid_mask]
    all_y = all_y[valid_mask]
    all_z = all_z[valid_mask]

    # Determine colors
    if rs:
        all_r = np.concatenate(rs)[valid_mask]
        all_g = np.concatenate(gs)[valid_mask]
        all_b = np.concatenate(bs)[valid_mask]
    else:
        # Default warm sandstone color when RGB is absent in the scan
        all_r = np.full(len(all_x), 210, dtype=np.uint8)
        all_g = np.full(len(all_x), 190, dtype=np.uint8)
        all_b = np.full(len(all_x), 160, dtype=np.uint8)

    # Center coordinates around the dataset origin
    if auto_center:
        all_x = all_x - float(np.mean(all_x))
        all_y = all_y - float(np.mean(all_y))
        all_z = all_z - float(np.percentile(all_z, 10))

    df = pd.DataFrame({
        'x': np.round(all_x, 3),
        'y': np.round(all_y, 3),
        'z': np.round(all_z, 3),
        'r': all_r,
        'g': all_g,
        'b': all_b,
    })

    print(f"✅ Extracted {len(df):,} LiDAR points!")

    # Cache to Parquet for instant reload next run
    if use_cache and cache_path is not None:
        try:
            df.to_parquet(str(cache_path), index=False)
            print(f"💾 Cached at: {cache_path}")
        except Exception:
            pass

    return df
