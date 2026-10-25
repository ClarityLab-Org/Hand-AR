# Point Cloud Rendering and Scene Generation

This document explains how the project turns LiDAR point cloud data into the 3D scene you see in the explorer, and which functions are responsible for the visual rendering.

## 1. Main rendering flow

The visual pipeline starts in the entry script:

- `explore.py`
  - `main()` picks the `.laz` or `.las` file.
  - It calls `load_laz_to_dataframe(...)` to decode the LiDAR file into a pandas dataframe.
  - Then it opens the standalone explorer and passes the dataframe to `handarm.engine.explore_standalone.main(point_cloud=df, target_path=laz_path)`.

This is important because the old flow tried to read a raw CSV path, but the real LAZ loader returns a dataframe, not a CSV file. The renderer expects a dataframe object that already contains `x, y, z, r, g, b` columns.

## 2. LiDAR data loading and preparation

### File: `handarm/geometry/laz_loader.py`

#### `load_laz_to_dataframe(...)`
Purpose:
- Reads `.laz` / `.las` data directly with `laspy`
- Downsamples to a target point count for real-time rendering
- Normalizes color channels (`r/g/b`)
- Explores scale/centering for usable world coordinates
- Optionally caches output as a Parquet file

What it does:
- Validates that the file exists
- Checks cache directory and loads a cached parquet if it is newer than the source file
- Streams the LAS/LAZ in chunks using `reader.chunk_iterator(chunk_size)`
- Reads `x`, `y`, `z` coordinates and optional `red`, `green`, `blue` values
- Applies filtering and centering:
  - `auto_center=True` shifts the cloud to a stable origin
  - `z_baseline` and elevation filtering are used to remove large outliers
- Builds a dataframe with columns:
  - `x`, `y`, `z`, `r`, `g`, `b`
- Saves the dataframe to parquet if caching is enabled

This returned dataframe is the actual 3D data source used by the renderer.

## 3. Coordinate alignment

### File: `handarm/geometry/alignment.py`

#### `auto_align_up_axis(df)`
Purpose:
- Reorients the cloud so the ground plane sits on the X-Z plane and height is represented by Y.

How it works:
- Computes covariance of the 3D point cloud
- Uses eigenvalues to detect the smallest spread axis
- If needed, swaps axes so the dataset is aligned with Ursina’s expected world axes

This is a critical preprocessing step because the viewer expects a ground-plane coordinate system for walking and flying through the point cloud.

## 4. Scene creation and object rendering

### File: `handarm/engine/explore_standalone.py`

#### `main(point_cloud=None, target_path=None)`
Purpose:
- Starts the Ursina application
- Initializes the camera, UI, and hand tracker
- Creates the `ExploreEnvironment` for the point cloud scene

Important behavior:
- If a dataframe is passed in, the app does not force the engine to reopen a CSV.
- It calls:
  - `ExploreEnvironment(csv_path, point_cloud=point_cloud, ...)`

This is the exact bridge between preloaded point cloud data and the rendering engine.

## 5. Core point-cloud scene engine

### File: `handarm/engine/explore_core.py`

This is the actual 3D scene logic.

### `ExploreEnvironment.__init__(...)`
Purpose:
- Initializes the world scene and either:
  - loads a chunked dataset from a manifest directory, or
  - loads a single point cloud dataframe into a renderable mesh

Important parts:
- `scene.fog_color = color.color(0, 0, 0.07)`
- `scene.fog_density = (80, 650)`
- Sets up `self.loaded_chunks`, `self.chunk_manifest`, `self.landmarks`, and the player controller

#### Single-point-cloud path
When the dataset is not chunked, it does:
- `df = pd.read_csv(target_path)` or `df = point_cloud`
- `df = df.iloc[::downsample_step]`
- `df = auto_align_up_axis(df)`
- Builds `points_xz` and `points_y`
- Extracts RGB arrays `r/g/b`
- Produces palette variants:
  - `palette_rgb`
  - `palette_elevation`
  - `palette_contrast`
- Creates a `Mesh` from the point coordinates:

```python
vertices = [Vec3(x, y, z) for x, y, z in zip(df["x"], df["y"], df["z"])]
self.single_mesh = Mesh(
    vertices=vertices,
    colors=self.palette_rgb,
    mode='point',
    render_points_in_3d=False,
    thickness=self.current_thickness,
)
self.single_entity = Entity(model=self.single_mesh)
self.single_entity.set_render_mode_perspective(False)
self.single_entity.set_render_mode_thickness(self.current_thickness)
```

This is the direct render call that turns the point cloud into a visible object in Ursina.
Single-cloud meshes, overview meshes, and streamed chunks are children of
`ExploreEnvironment.world_root`. The root stays at the world origin because
player movement, landmark positions, and chunk LOD distances all use world-space
coordinates.

### `ExploreEnvironment.render_scene(...)`
Renders the initialized point cloud and an optional model beneath the same
`world_root`. Models are loaded from the supplied path, made double-sided, and
placed/scaled with `model_pos` and `model_scale`. Hand gestures move and rotate
the player/camera; both renderables remain visible to that same camera.

### `cycle_color_mode()`
Purpose:
- Switches the visual palette between:
  - Natural RGB
  - Elevation Heatmap
  - Vibrant RGB

What it does:
- Replaces mesh colors with the selected palette
- Regenerates the mesh and refreshes the texture stage

### `cycle_point_thickness()`
Purpose:
- Increases or changes point size to make the cloud denser or more visible

### `reset_player()`
Purpose:
- Returns the player to the spawn location so the scene resets cleanly

### `jump_to_landmark()`
Purpose:
- Teleports the camera/player to a landmark position

## 6. Chunked distance-based LOD rendering

### `LoadedChunk`
This class represents a section of the large point cloud, typically from a chunked spatial dataset.

#### `__init__(...)`
It receives chunk data such as:
- `vertices`
- `rgb`
- `elevation`
- `vibrant`

It applies LOD subsampling:
- `LOD_FULL`: no downsampling
- `LOD_MED`: every 3rd point
- `LOD_LOW`: every 8th point

Then it builds a `Mesh`:

```python
self.mesh = Mesh(
    vertices=vertices,
    colors=active_colors,
    mode='point',
    render_points_in_3d=False,
    thickness=thickness,
)
self.entity = Entity(model=self.mesh)
```

This is the chunk-based rendering model used for large campus-scale or city-scale point clouds.

### `set_color_mode(mode)`
Changes the active palette for a chunk without rebuilding the whole chunk from scratch.

### `set_thickness(thickness)`
Changes point size for a loaded chunk.

## 7. LOD and streaming logic

The `ExploreEnvironment` class also manages dynamic level-of-detail:

- `LOAD_RADIUS = 105.0`
- `LOD_FULL_RADIUS = 35.0`
- `LOD_MED_RADIUS = 70.0`
- `UNLOAD_RADIUS = 135.0`

This means:
- close chunks are loaded at full detail
- medium-distance chunks are more sparse
- far chunks are reduced or removed to keep the frame rate stable

The scene is not static: as the player moves, the system prioritizes nearby chunks and unloads distant ones.

## 8. How the scene is actually displayed

The final visual output is achieved using Ursina primitives:

- `Mesh` holds the point geometry
- `Entity(model=self.mesh)` makes that mesh part of the 3D scene
- `mode='point'` tells Ursina to render the cloud as points, not triangles
- `render_points_in_3d=False` keeps the point cloud in a screen-space style
- `thickness` controls point size

This makes it possible to render millions of LiDAR points efficiently without normal polygon meshes.

## 9. Rendering-related function summary

### Primary render functions

- `explore.py::main()`
  - Starts the LAZ-to-scene pipeline

- `handarm.geometry.laz_loader.load_laz_to_dataframe()`
  - Converts raw `.laz` data into renderable `DataFrame`

- `handarm.engine.explore_standalone.main()`
  - Creates the Ursina application and scene

- `handarm.engine.explore_core.ExploreEnvironment.__init__()`
  - Builds the actual point cloud scene object

- `handarm.engine.explore_core.ExploreEnvironment.render_scene()`
  - Adds an optional model alongside the point cloud under `world_root`

- `handarm.engine.explore_core.LoadedChunk.__init__()`
  - Creates chunk renderer objects for large datasets

- `handarm.engine.explore_core.cycle_color_mode()`
  - Switches point cloud coloring modes

- `handarm.engine.explore_core.cycle_point_thickness()`
  - Adjusts visual density/size of points

## 10. One-line summary

The 3D point cloud scene is rendered by converting LiDAR data into a dataframe, then building one or more Ursina `Mesh` objects with `mode='point'`, attaching them to `Entity` objects, and updating them dynamically by chunk, color, and point thickness as the user navigates the scene.
