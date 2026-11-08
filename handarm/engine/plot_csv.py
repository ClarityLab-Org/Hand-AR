"""
Standalone CSV Point Cloud Plotter Utility for Hand-ArM2 Project.

Renders downsampled point cloud CSV data with coordinate axes in Ursina.

Usage:
    python plotcsv.py [path/to/cloud.csv]
"""

import os
import sys

import pandas as pd
from ursina import *

from handarm.geometry.alignment import auto_align_up_axis


def main() -> None:
    """Main entry point for the point cloud plotter."""
    csv_path = "pointcloud_sample.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    app = Ursina()

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df.iloc[:10]
        df = auto_align_up_axis(df)

        vertices = [
            Vec3(x, y, z)
            for x, y, z in zip(df["x"], df["y"], df["z"])
        ]

        point_colors = [
            (r / 255.0, g / 255.0, b / 255.0, 1.0)
            for r, g, b in zip(df["r"], df["g"], df["b"])
        ]

        mesh = Mesh(
            vertices=vertices,
            colors=point_colors,
            mode='point',
            thickness=0.009,
        )

        Entity(model=mesh)

    # Render XYZ Axes (X=Red, Y=Green, Z=Blue)
    axis_length = 20
    Entity(model=Mesh(vertices=[Vec3(0, 0, 0), Vec3(axis_length, 0, 0)],
                      mode='line', thickness=3), color=color.red)
    Entity(model=Mesh(vertices=[Vec3(0, 0, 0), Vec3(0, axis_length, 0)],
                      mode='line', thickness=3), color=color.green)
    Entity(model=Mesh(vertices=[Vec3(0, 0, 0), Vec3(0, 0, axis_length)],
                      mode='line', thickness=3), color=color.blue)

    EditorCamera()

    app.run()


if __name__ == "__main__":
    main()
