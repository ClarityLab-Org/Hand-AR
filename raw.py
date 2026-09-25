import pandas as pd
import open3d as o3d
import numpy as np

df = pd.read_csv("models/iitj_campus.csv")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].values)
pcd.colors = o3d.utility.Vector3dVector(df[["r", "g", "b"]].values / 255.0)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.get_render_option().point_size = 2.0  # default is usually 1.0
vis.run()