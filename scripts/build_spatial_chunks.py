"""
Spatial Grid Partitioner & Architectural LOD Pipeline for IIT Jodhpur LiDAR Point Cloud.
Generates authentic Jodhpur Sandstone/Terracotta palettes, building highlights, and spatial chunks.
"""

import json
import os
import sys
import time
import laspy
import numpy as np


def _create_architectural_palette(y_elev: np.ndarray, raw_r: np.ndarray, raw_g: np.ndarray, raw_b: np.ndarray) -> np.ndarray:
    """
    Creates an authentic IIT Jodhpur Architectural color palette:
    - Ground / Roads / Plazas (0 - 1.5m): Desert Sandstone & Pavement tones
    - Building Walls & Facades (1.5 - 7.5m): Iconic Jodhpur Red Sandstone & Terracotta
    - Rooftops / Solar Panels (> 7.5m): High-tech Solar Blue / Slate
    - Vegetation / Foliage: Desert Oasis Green accents
    """
    total = len(y_elev)
    colors = np.zeros((total, 4), dtype=np.float32)
    colors[:, 3] = 1.0  # Alpha

    # Normalize raw RGB
    nr = raw_r.astype(np.float32) / 255.0
    ng = raw_g.astype(np.float32) / 255.0
    nb = raw_b.astype(np.float32) / 255.0

    # Detect vegetation (green channel dominance)
    is_tree = (ng > nr * 0.95) & (ng > nb * 1.1) & (y_elev > 0.8) & (y_elev < 6.0)

    # 1. Ground level (0 - 1.5m) -> Desert Sand & Pathways
    ground_mask = y_elev < 1.5
    # Blend raw photo color with warm desert sand
    colors[ground_mask, 0] = np.clip(nr[ground_mask] * 0.7 + 0.25, 0.0, 1.0)
    colors[ground_mask, 1] = np.clip(ng[ground_mask] * 0.7 + 0.22, 0.0, 1.0)
    colors[ground_mask, 2] = np.clip(nb[ground_mask] * 0.7 + 0.16, 0.0, 1.0)

    # 2. Building Walls (1.5m - 7.5m) -> Jodhpur Red Sandstone / Terracotta
    wall_mask = (y_elev >= 1.5) & (y_elev < 7.5) & (~is_tree)
    # Authentic Red Sandstone: R ~ 0.75-0.85, G ~ 0.35-0.45, B ~ 0.25-0.35
    h_factor = (y_elev[wall_mask] - 1.5) / 6.0
    colors[wall_mask, 0] = np.clip(0.72 + 0.15 * h_factor + nr[wall_mask] * 0.1, 0.0, 1.0)
    colors[wall_mask, 1] = np.clip(0.36 + 0.08 * h_factor + ng[wall_mask] * 0.1, 0.0, 1.0)
    colors[wall_mask, 2] = np.clip(0.24 + 0.06 * h_factor + nb[wall_mask] * 0.1, 0.0, 1.0)

    # 3. Rooftops & Solar Canopies (>= 7.5m) -> Solar Blue / Slate
    roof_mask = (y_elev >= 7.5) & (~is_tree)
    colors[roof_mask, 0] = np.clip(0.22 + nr[roof_mask] * 0.1, 0.0, 1.0)
    colors[roof_mask, 1] = np.clip(0.42 + ng[roof_mask] * 0.15, 0.0, 1.0)
    colors[roof_mask, 2] = np.clip(0.75 + nb[roof_mask] * 0.2, 0.0, 1.0)

    # 4. Trees / Greenery
    if np.any(is_tree):
        colors[is_tree, 0] = 0.22
        colors[is_tree, 1] = 0.58
        colors[is_tree, 2] = 0.26

    return colors


def build_spatial_chunks(
    laz_path: str = "iitj_data/pointcloud.laz",
    output_dir: str = "models/iitj_chunks",
    cell_size: float = 35.0,
    chunk_stream_size: int = 1_000_000,
    overview_target: int = 120_000,
) -> None:
    """Streams a LiDAR file, centers coordinate space, and partitions into spatial grid chunks."""
    if not os.path.exists(laz_path):
        raise FileNotFoundError(f"Source file not found: {laz_path}")

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    print(f"🏛️ Starting IIT Jodhpur Campus Partitioner for {laz_path}...")
    print(f"📦 Output Directory: {output_dir}")

    raw_x_list, raw_y_list, raw_z_list = [], [], []
    raw_r_list, raw_g_list, raw_b_list = [], [], []

    total_read = 0

    with laspy.open(laz_path) as reader:
        has_color = hasattr(reader.header.point_format, 'red') or 'red' in reader.header.point_format.dimension_names

        try:
            for chunk in reader.chunk_iterator(chunk_stream_size):
                # Subsample 1 in 3 for maximum density on buildings
                step = 3
                cx = np.array(chunk.x[::step], dtype=np.float32)
                cy = np.array(chunk.y[::step], dtype=np.float32)
                cz = np.array(chunk.z[::step], dtype=np.float32)

                raw_x_list.append(cx)
                raw_y_list.append(cy)
                raw_z_list.append(cz)

                if has_color and hasattr(chunk, 'red') and len(chunk.red) > 0:
                    rr = np.array(chunk.red[::step], dtype=np.float32)
                    rg = np.array(chunk.green[::step], dtype=np.float32)
                    rb = np.array(chunk.blue[::step], dtype=np.float32)

                    max_val = max(float(np.max(rr)), float(np.max(rg)), float(np.max(rb)), 1.0)
                    scale = 255.0 / 65535.0 if max_val > 255 else 1.0

                    raw_r_list.append((rr * scale).astype(np.uint8))
                    raw_g_list.append((rg * scale).astype(np.uint8))
                    raw_b_list.append((rb * scale).astype(np.uint8))

                total_read += len(chunk)
                print(f"  ⏳ Read {total_read:,} raw LiDAR points...", end="\r", flush=True)

        except Exception as e:
            print(f"\n  ℹ Reached file boundary after {total_read:,} points.")

    print("\n🔄 Computing campus centering and architectural elevation...")
    raw_x = np.concatenate(raw_x_list)
    raw_y = np.concatenate(raw_y_list)
    raw_z = np.concatenate(raw_z_list)

    # In UTM: raw_x is Easting, raw_y is Northing, raw_z is Altitude
    # Center X and Z around (0, 0)
    ursina_x = raw_x - float(np.mean(raw_x))
    ursina_z = raw_y - float(np.mean(raw_y))

    # Calculate true ground elevation baseline (10th percentile)
    ground_baseline = float(np.percentile(raw_z, 10))
    ursina_y = np.maximum(0.0, raw_z - ground_baseline)

    # Filter out sky noise / flying artifacts > 30m above ground
    valid_mask = ursina_y < 25.0
    ursina_x = ursina_x[valid_mask]
    ursina_y = ursina_y[valid_mask]
    ursina_z = ursina_z[valid_mask]

    if raw_r_list:
        raw_r = np.concatenate(raw_r_list)[valid_mask]
        raw_g = np.concatenate(raw_g_list)[valid_mask]
        raw_b = np.concatenate(raw_b_list)[valid_mask]
    else:
        raw_r = np.full(len(ursina_x), 210, dtype=np.uint8)
        raw_g = np.full(len(ursina_x), 190, dtype=np.uint8)
        raw_b = np.full(len(ursina_x), 160, dtype=np.uint8)

    total_dense_points = len(ursina_x)
    print(f"📊 Cleaned Campus Points: {total_dense_points:,}")
    print(f"📐 Campus Dimensions: X = [{np.min(ursina_x):.1f}m, {np.max(ursina_x):.1f}m], "
          f"Z = [{np.min(ursina_z):.1f}m, {np.max(ursina_z):.1f}m], "
          f"Building Heights = [0.0m, {np.max(ursina_y):.1f}m]")

    # Compute color palettes
    norm_r = raw_r.astype(np.float32) / 255.0
    norm_g = raw_g.astype(np.float32) / 255.0
    norm_b = raw_b.astype(np.float32) / 255.0

    # 1. Palette 1: Authentic Jodhpur Sandstone & Terracotta Architectural Palette
    sandstone_colors = _create_architectural_palette(ursina_y, raw_r, raw_g, raw_b)

    # 2. Palette 2: Dynamic Elevation Heatmap
    y_max = max(float(np.max(ursina_y)), 1.0)
    y_norm = ursina_y / y_max
    elev_r = np.zeros(total_dense_points, dtype=np.float32)
    elev_g = np.zeros(total_dense_points, dtype=np.float32)
    elev_b = np.zeros(total_dense_points, dtype=np.float32)

    m1 = y_norm < 0.25
    f1 = y_norm[m1] / 0.25
    elev_r[m1], elev_g[m1], elev_b[m1] = 0.0, f1 * 0.8, 1.0 - f1 * 0.2

    m2 = (y_norm >= 0.25) & (y_norm < 0.5)
    f2 = (y_norm[m2] - 0.25) / 0.25
    elev_r[m2], elev_g[m2], elev_b[m2] = 0.0, 0.8 + f2 * 0.2, 0.8 - f2 * 0.8

    m3 = (y_norm >= 0.5) & (y_norm < 0.75)
    f3 = (y_norm[m3] - 0.5) / 0.25
    elev_r[m3], elev_g[m3], elev_b[m3] = f3, 1.0, 0.0

    m4 = y_norm >= 0.75
    f4 = (y_norm[m4] - 0.75) / 0.25
    elev_r[m4], elev_g[m4], elev_b[m4] = 1.0, np.maximum(0.0, 1.0 - f4 * 0.8), f4 * 0.2

    elevation_colors = np.column_stack([elev_r, elev_g, elev_b, np.ones(total_dense_points, dtype=np.float32)])

    # 3. Palette 3: True Drone RGB
    true_rgb = np.column_stack([norm_r, norm_g, norm_b, np.ones(total_dense_points, dtype=np.float32)])

    # Partition into grid
    grid_gx = np.floor(ursina_x / cell_size).astype(np.int32)
    grid_gz = np.floor(ursina_z / cell_size).astype(np.int32)
    cell_keys = grid_gx * 100000 + grid_gz
    sort_idx = np.argsort(cell_keys)

    sorted_keys = cell_keys[sort_idx]
    unique_sorted_keys, split_indices = np.unique(sorted_keys, return_index=True)
    cell_groups = np.split(sort_idx, split_indices[1:])

    manifest_chunks = []
    saved_count = 0

    for key, indices in zip(unique_sorted_keys, cell_groups):
        if len(indices) < 30:
            continue

        gx = int(grid_gx[indices[0]])
        gz = int(grid_gz[indices[0]])

        chunk_verts = np.column_stack([ursina_x[indices], ursina_y[indices], ursina_z[indices]]).astype(np.float32)
        chunk_sandstone = sandstone_colors[indices]
        chunk_elev = elevation_colors[indices]
        chunk_rgb = true_rgb[indices]

        center_x = float(np.mean(chunk_verts[:, 0]))
        center_y = float(np.mean(chunk_verts[:, 1]))
        center_z = float(np.mean(chunk_verts[:, 2]))

        chunk_filename = f"chunk_{gx}_{gz}.npz"
        chunk_path = os.path.join(output_dir, chunk_filename)

        np.savez_compressed(
            chunk_path,
            vertices=chunk_verts,
            rgb=chunk_sandstone,        # Default: Authentic Sandstone Architecture
            elevation=chunk_elev,      # Elevation Heatmap
            vibrant=chunk_rgb,         # True Drone RGB
            center=np.array([center_x, center_y, center_z], dtype=np.float32),
        )

        manifest_chunks.append({
            "gx": gx,
            "gz": gz,
            "file": chunk_filename,
            "center": [center_x, center_y, center_z],
            "point_count": int(len(indices)),
            "min_bounds": [float(np.min(chunk_verts[:, 0])), float(np.min(chunk_verts[:, 1])), float(np.min(chunk_verts[:, 2]))],
            "max_bounds": [float(np.max(chunk_verts[:, 0])), float(np.max(chunk_verts[:, 1])), float(np.max(chunk_verts[:, 2]))],
        })
        saved_count += 1

    # Overview LOD
    overview_step = max(1, total_dense_points // overview_target)
    ov_idx = np.arange(0, total_dense_points, overview_step)
    ov_verts = np.column_stack([ursina_x[ov_idx], ursina_y[ov_idx], ursina_z[ov_idx]]).astype(np.float32)

    overview_path = os.path.join(output_dir, "overview.npz")
    np.savez_compressed(
        overview_path,
        vertices=ov_verts,
        rgb=sandstone_colors[ov_idx],
        elevation=elevation_colors[ov_idx],
        vibrant=true_rgb[ov_idx],
    )
    print(f"🌍 Saved Overview LOD: {overview_path} ({len(ov_verts):,} points)")

    # Manifest with IITJ Landmark Waypoints
    landmarks = [
        {"name": "🏛️ Academic Complex & Lecture Halls", "position": [-45.0, 15.0, 30.0]},
        {"name": "🏫 Department Buildings & Labs", "position": [-65.0, 15.0, 20.0]},
        {"name": "🎓 Central Plaza & Courtyard", "position": [0.0, 12.0, -10.0]},
        {"name": "📍 Campus Main Entrance Avenue", "position": [55.0, 10.0, -35.0]},
    ]

    manifest = {
        "dataset": "IIT Jodhpur Campus LiDAR",
        "cell_size": cell_size,
        "total_points": total_dense_points,
        "overview_points": len(ov_verts),
        "chunk_count": saved_count,
        "spawn_point": [40.0, 32.0, -55.0],      # High aerial drone vantage point
        "spawn_rotation": [28.0, -45.0, 0.0],     # Look towards academic complex
        "bounds": {
            "min_x": float(np.min(ursina_x)), "max_x": float(np.max(ursina_x)),
            "min_z": float(np.min(ursina_z)), "max_z": float(np.max(ursina_z)),
            "min_y": float(np.min(ursina_y)), "max_y": float(np.max(ursina_y)),
        },
        "landmarks": landmarks,
        "chunks": manifest_chunks,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - start_time
    print(f"✅ IIT Jodhpur Architecture Partitioner complete in {elapsed:.1f}s! ({saved_count} chunks)")


if __name__ == "__main__":
    build_spatial_chunks()
