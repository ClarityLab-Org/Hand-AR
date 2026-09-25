"""
Direct .laz / .las LiDAR Point Cloud Loader and Streamer for Hand-AR.

Provides high-performance, memory-safe streaming of LASzip (.laz) and LAS (.las)
files directly into pandas DataFrames consumable by Ursina ExploreEnvironment.
Handles 16-bit RGB normalization, origin centering, elevation baselining, and caching.
"""

import os
import logging
from itertools import chain
from pathlib import Path
from typing import Iterator, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _warn_if_implausibly_small(path: str, point_count: int, point_size: int) -> None:
    """Warn when compressed size is far below a conservative lower bound."""
    file_size = os.path.getsize(path)
    minimum_expected = point_count * point_size * 0.01
    if file_size < minimum_expected:
        logger.warning(
            "Warning: file size (%s bytes) looks small for %s declared points; "
            "the file may be truncated. Continuing with full verification...",
            file_size,
            f"{point_count:,}",
        )


def _point_size(path: str) -> int:
    import laspy

    with laspy.open(path) as reader:
        return reader.header.point_format.size


def describe_laz_failure(
    path: str,
    points_decoded: int,
    point_count: int,
    cause: BaseException,
    *,
    chunk_index: Optional[int] = None,
    point_offset: Optional[int] = None,
) -> str:
    """Return a consistent diagnostic for an incomplete LAS/LAZ decode."""
    location = f"chunk {chunk_index}, point offset {point_offset:,}" if chunk_index is not None else (
        f"point offset {points_decoded:,}"
    )
    return (
        f"Could not read the complete LiDAR file {path!r}: decoding stopped at "
        f"{location}. Successfully decoded {points_decoded:,} of the expected "
        f"{point_count:,} points. Original error: {type(cause).__name__}: {cause}"
    )


def iter_laz_chunks(
    laz_path: str,
    chunk_size: int = 500_000,
) -> Iterator[tuple[int, int, object, int, bool]]:
    """Yield decoded chunks with their global offsets and source metadata."""
    import laspy

    with laspy.open(laz_path) as reader:
        point_count = reader.header.point_count
        has_color = (
            hasattr(reader.header.point_format, "red")
            or "red" in reader.header.point_format.dimension_names
        )
        point_offset = 0
        for chunk_index, chunk in enumerate(reader.chunk_iterator(chunk_size)):
            yield chunk_index, point_offset, chunk, point_count, has_color
            point_offset += len(chunk)


def load_laz_to_dataframe(
    laz_path: str,
    target_points: int = 500_000,
    chunk_size: int = 500_000,
    auto_center: bool = True,
    use_cache: bool = True,
    allow_partial: bool = False,
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
        allow_partial:   Opt in to returning decoded data if the source ends early.

    Returns:
        pd.DataFrame with columns 'x', 'y', 'z', 'r', 'g', 'b'.
    """
    if not os.path.exists(laz_path):
        raise FileNotFoundError(f"LiDAR file not found: {laz_path}")

    # -- Cache check --
    cache_path: Optional[Path] = None
    if use_cache:
        source = os.stat(laz_path)
        identity = f"{source.st_size}_{source.st_mtime_ns}"
        file_hash_name = Path(laz_path).stem + f"_{target_points}pts_{identity}_cache.parquet"
        cache_dir = Path(laz_path).parent / "cache"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / file_hash_name

            if (cache_path.exists()
                    and os.path.getmtime(str(cache_path)) >= os.path.getmtime(laz_path)):
                print(f" Loading cached point cloud: {cache_path}...")
                return pd.read_parquet(str(cache_path))
        except OSError:
            cache_dir = Path(__file__).resolve().parent.parent.parent / "models_compressed"
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir / file_hash_name
            except OSError:
                cache_path = None

    try:
        import laspy
    except ImportError:
        raise ImportError(
            "laspy with lazrs is required to read .laz/.las files directly.\n"
            "Please run:  pip install 'laspy[lazrs]'"
        )

    print(f" Streaming LiDAR point cloud from: {laz_path}...")

    total_points = None
    read_count = 0
    chunk_index = 0
    point_offset = 0
    partial = False
    try:
        chunks = iter_laz_chunks(laz_path, chunk_size)
        first_chunk = next(chunks, None)
        if first_chunk is None:
            raise RuntimeError(f"LiDAR file contains no readable points: {laz_path}")
        chunk_index, point_offset, chunk, total_points, has_color = first_chunk
        try:
            _warn_if_implausibly_small(laz_path, total_points, _point_size(laz_path))
        except OSError:
            pass
        sample_count = target_points if target_points > 0 else total_points
        sample_count = min(sample_count, total_points)
        if sample_count < total_points:
            rng = np.random.default_rng(0)
            sample_indices = np.sort(
                rng.choice(total_points, sample_count, replace=False)
            )
            print(
                f" Sampling {sample_count:,} points across the complete "
                f"{total_points:,}-point file for real-time FPS..."
            )
        else:
            sample_indices = np.arange(total_points)
            print(f" Loading all {total_points:,} points...")

        xs, ys, zs = [], [], []
        rs, gs, bs = [], [], []

        # One streaming pass both validates every compressed block and collects the sample.
        for chunk_index, point_offset, chunk, _, _ in chain((first_chunk,), chunks):
            end = point_offset + len(chunk)
            selected = sample_indices[
                (sample_indices >= point_offset) & (sample_indices < end)
            ] - point_offset
            if len(selected):
                xs.append(np.asarray(chunk.x)[selected].astype(np.float32))
                ys.append(np.asarray(chunk.y)[selected].astype(np.float32))
                zs.append(np.asarray(chunk.z)[selected].astype(np.float32))

                if has_color and hasattr(chunk, "red"):
                    rs.append(np.asarray(chunk.red)[selected])
                    gs.append(np.asarray(chunk.green)[selected])
                    bs.append(np.asarray(chunk.blue)[selected])

            read_count = end
            print(
                f"   Read {read_count:,} / {total_points:,} points...",
                end="\r",
                flush=True,
            )
        if read_count != total_points:
            raise RuntimeError(
                f"LiDAR stream ended at {read_count:,} of {total_points:,} points"
            )
    except Exception as e:
        if allow_partial and read_count and xs:
            print(
                f"\n  PARTIAL DATASET: loaded {read_count:,} of "
                f"{total_points:,} points; caching disabled."
            )
            partial = True
        else:
            partial = False
        if partial:
            pass
        elif isinstance(e, RuntimeError) and str(e).startswith("Could not read"):
            raise
        else:
            raise RuntimeError(
                describe_laz_failure(
                    laz_path,
                    read_count,
                    total_points or 0,
                    e,
                    point_offset=read_count,
                )
            ) from e

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
        all_r = np.concatenate(rs).astype(np.float32)
        all_g = np.concatenate(gs).astype(np.float32)
        all_b = np.concatenate(bs).astype(np.float32)
        max_val = max(float(np.max(all_r)), float(np.max(all_g)), float(np.max(all_b)), 1.0)
        scale = 255.0 / 65535.0 if max_val > 255 else 1.0
        all_r = (all_r * scale).astype(np.uint8)[valid_mask]
        all_g = (all_g * scale).astype(np.uint8)[valid_mask]
        all_b = (all_b * scale).astype(np.uint8)[valid_mask]
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
    if partial:
        df.attrs["is_partial"] = True
        df.attrs["points_decoded"] = read_count
        df.attrs["point_count"] = total_points

    print(f" Extracted {len(df):,} LiDAR points!")

    # Cache to Parquet for instant reload next run
    if use_cache and cache_path is not None and not partial:
        try:
            df.to_parquet(str(cache_path), index=False)
            print(f" Cached at: {cache_path}")
        except Exception as e:
            print(f"  Cache write failed ({type(e).__name__}: {e}); continuing without cache.")

    return df
