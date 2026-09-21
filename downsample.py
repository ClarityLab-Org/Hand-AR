"""
Fast, memory-safe streaming downsampler for LiDAR .laz files.
Converts iitj_data/pointcloud.laz into models/iitj_campus.csv for Hand-ArM2.
Handles partial / truncated LAZ files gracefully.
"""

import os
import sys
import laspy
import numpy as np
import pandas as pd


def convert_laz_to_csv(
    laz_path: str = "iitj_data/pointcloud.laz",
    output_path: str = "models/iitj_campus.csv",
    target_points: int = 400_000,
    chunk_size: int = 500_000,
) -> str:
    """Streams a .laz file in chunks, downsamples, normalizes coordinates & RGB, and saves to CSV."""
    if not os.path.exists(laz_path):
        raise FileNotFoundError(f"Source file not found: {laz_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"📦 Opening {laz_path}...")
    with laspy.open(laz_path) as reader:
        # Step size based on typical point density (20.5M points in file -> ~50 step gives ~410k points)
        step = 50

        print(f"🎯 Target output points: ~{target_points:,} (sampling 1 every {step} points)")

        sampled_x, sampled_y, sampled_z = [], [], []
        sampled_r, sampled_g, sampled_b = [], [], []

        processed = 0
        has_color = hasattr(reader.header.point_format, 'red') or 'red' in reader.header.point_format.dimension_names

        try:
            for i, chunk in enumerate(reader.chunk_iterator(chunk_size)):
                cx = np.array(chunk.x[::step], dtype=np.float32)
                cy = np.array(chunk.y[::step], dtype=np.float32)
                cz = np.array(chunk.z[::step], dtype=np.float32)

                sampled_x.append(cx)
                sampled_y.append(cy)
                sampled_z.append(cz)

                if has_color and hasattr(chunk, 'red') and len(chunk.red) > 0:
                    raw_r = np.array(chunk.red[::step], dtype=np.float32)
                    raw_g = np.array(chunk.green[::step], dtype=np.float32)
                    raw_b = np.array(chunk.blue[::step], dtype=np.float32)

                    max_val = max(float(np.max(raw_r)), float(np.max(raw_g)), float(np.max(raw_b)), 1.0)
                    scale = 255.0 / 65535.0 if max_val > 255 else 1.0

                    sampled_r.append((raw_r * scale).astype(np.uint8))
                    sampled_g.append((raw_g * scale).astype(np.uint8))
                    sampled_b.append((raw_b * scale).astype(np.uint8))

                processed += len(chunk)
                print(f"  ⏳ Read {processed:,} points (collected {sum(len(a) for a in sampled_x):,} subsampled points)...", end="\r", flush=True)

        except Exception as e:
            print(f"\n  ℹ Reached file boundary after {processed:,} points ({type(e).__name__}). Proceeding with extracted data...")

        print()

    print("🔄 Merging & centering coordinate space...")
    all_x = np.concatenate(sampled_x)
    all_y = np.concatenate(sampled_y)
    all_z = np.concatenate(sampled_z)

    # Subsample further if collected points exceed target
    if len(all_x) > target_points * 1.2:
        sub_idx = np.linspace(0, len(all_x) - 1, target_points, dtype=int)
        all_x = all_x[sub_idx]
        all_y = all_y[sub_idx]
        all_z = all_z[sub_idx]
        if sampled_r:
            all_r = np.concatenate(sampled_r)[sub_idx]
            all_g = np.concatenate(sampled_g)[sub_idx]
            all_b = np.concatenate(sampled_b)[sub_idx]
    else:
        if sampled_r:
            all_r = np.concatenate(sampled_r)
            all_g = np.concatenate(sampled_g)
            all_b = np.concatenate(sampled_b)

    # Center at origin
    all_x -= float(np.mean(all_x))
    all_y -= float(np.mean(all_y))
    all_z -= float(np.min(all_z))

    if not sampled_r:
        z_norm = ((all_z - np.min(all_z)) / (np.max(all_z) - np.min(all_z) + 1e-5) * 255).astype(np.uint8)
        all_r = z_norm
        all_g = z_norm
        all_b = z_norm

    print(f"💾 Writing {len(all_x):,} points to {output_path}...")
    df = pd.DataFrame({
        'x': np.round(all_x, 3),
        'y': np.round(all_y, 3),
        'z': np.round(all_z, 3),
        'r': all_r,
        'g': all_g,
        'b': all_b,
    })
    df.to_csv(output_path, index=False)
    print(f"✅ Successfully created: {output_path} ({os.path.getsize(output_path) / (1024*1024):.1f} MB, {len(df):,} points)")
    return output_path


if __name__ == "__main__":
    convert_laz_to_csv()

