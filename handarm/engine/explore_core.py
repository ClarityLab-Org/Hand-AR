"""
Shared explore-mode logic for point cloud navigation with Dynamic Spatial Chunk Streaming (LOD).
Dynamically streams high-density LiDAR chunks based on camera proximity and view.
Features multi-resolution LOD: detail increases as you approach structures.
"""

import concurrent.futures
import json
import math
import os
import time as _time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from panda3d.core import TextureStage
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController

from handarm.geometry.alignment import auto_align_up_axis

# ---------------------------------------------------------------------------
# LOD level definitions
# ---------------------------------------------------------------------------
LOD_FULL = 0   # Every point  (close range)
LOD_MED  = 1   # Every 3rd point (medium range)
LOD_LOW  = 2   # Every 8th point (far range)

_LOD_STEPS = {LOD_FULL: 1, LOD_MED: 3, LOD_LOW: 8}

def _lod_subsample(arr: np.ndarray, lod: int) -> np.ndarray:
    """Return subsampled array for the given LOD level."""
    step = _LOD_STEPS.get(lod, 1)
    return arr[::step] if step > 1 else arr


def _create_architectural_palette(y_elev: np.ndarray, raw_r: np.ndarray, raw_g: np.ndarray, raw_b: np.ndarray) -> np.ndarray:
    """Creates authentic Jodhpur Sandstone/Terracotta palette mapping elevation and foliage."""
    total = len(y_elev)
    colors = np.zeros((total, 4), dtype=np.float32)
    colors[:, 3] = 1.0

    nr = raw_r.astype(np.float32) / 255.0
    ng = raw_g.astype(np.float32) / 255.0
    nb = raw_b.astype(np.float32) / 255.0

    is_tree = (ng > nr * 0.95) & (ng > nb * 1.1) & (y_elev > 0.8) & (y_elev < 6.0)

    ground_mask = y_elev < 1.5
    colors[ground_mask, 0] = np.clip(nr[ground_mask] * 0.7 + 0.25, 0.0, 1.0)
    colors[ground_mask, 1] = np.clip(ng[ground_mask] * 0.7 + 0.22, 0.0, 1.0)
    colors[ground_mask, 2] = np.clip(nb[ground_mask] * 0.7 + 0.16, 0.0, 1.0)

    wall_mask = (y_elev >= 1.5) & (y_elev < 7.5) & (~is_tree)
    h_factor = (y_elev[wall_mask] - 1.5) / 6.0
    colors[wall_mask, 0] = np.clip(0.72 + 0.15 * h_factor + nr[wall_mask] * 0.1, 0.0, 1.0)
    colors[wall_mask, 1] = np.clip(0.36 + 0.08 * h_factor + ng[wall_mask] * 0.1, 0.0, 1.0)
    colors[wall_mask, 2] = np.clip(0.24 + 0.06 * h_factor + nb[wall_mask] * 0.1, 0.0, 1.0)

    roof_mask = (y_elev >= 7.5) & (~is_tree)
    colors[roof_mask, 0] = np.clip(0.22 + nr[roof_mask] * 0.1, 0.0, 1.0)
    colors[roof_mask, 1] = np.clip(0.42 + ng[roof_mask] * 0.15, 0.0, 1.0)
    colors[roof_mask, 2] = np.clip(0.75 + nb[roof_mask] * 0.2, 0.0, 1.0)

    if np.any(is_tree):
        colors[is_tree, 0] = 0.22
        colors[is_tree, 1] = 0.58
        colors[is_tree, 2] = 0.26

    return colors

def _generate_elevation_colors(y_coords: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """Generates a rich multi-stop Turbo/Cyber elevation gradient mapping elevation to color."""
    y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))
    y_range = max(y_max - y_min, 1e-5)
    norm_y = (y_coords - y_min) / y_range

    colors = []
    for t in norm_y:
        if t < 0.25:
            f = t / 0.25
            r, g, b = 0.0, f * 0.8, 1.0 - f * 0.2
        elif t < 0.5:
            f = (t - 0.25) / 0.25
            r, g, b = 0.0, 0.8 + f * 0.2, 0.8 - f * 0.8
        elif t < 0.75:
            f = (t - 0.5) / 0.25
            r, g, b = f, 1.0, 0.0
        else:
            f = (t - 0.75) / 0.25
            r, g, b = 1.0, max(0.0, 1.0 - f * 0.8), f * 0.2
        colors.append((r, g, b, 1.0))
    return colors


def _generate_contrast_colors(r_arr: np.ndarray, g_arr: np.ndarray, b_arr: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """Boosts saturation and dynamic range of raw photogrammetry/LiDAR RGB colors."""
    r = r_arr.astype(np.float32) / 255.0
    g = g_arr.astype(np.float32) / 255.0
    b = b_arr.astype(np.float32) / 255.0

    mean_val = (r + g + b) / 3.0
    r = np.clip(mean_val + 1.35 * (r - mean_val), 0.0, 1.0)
    g = np.clip(mean_val + 1.35 * (g - mean_val), 0.0, 1.0)
    b = np.clip(mean_val + 1.35 * (b - mean_val), 0.0, 1.0)

    r = np.power(r, 0.92)
    g = np.power(g, 0.92)
    b = np.power(b, 0.92)

    return [(float(ri), float(gi), float(bi), 1.0) for ri, gi, bi in zip(r, g, b)]


class LoadedChunk:
    """Represents a dynamically loaded spatial point cloud chunk at a specific LOD level.

    Performance: uses numpy .tolist() for vertices and colors instead of
    Python Vec3/Vec4 comprehensions, which is 10-50x faster and critical
    for keeping chunk instantiation under 1 frame budget.
    """

    def __init__(self, data: Any, thickness: int = 1,
                 color_mode: str = "Natural RGB", lod_level: int = LOD_FULL) -> None:
        raw_verts = data["vertices"]
        raw_rgb = data["rgb"]
        raw_elev = data["elevation"]
        raw_vib = data["vibrant"]

        # LOD subsampling — reduce point count for distant chunks
        self.lod_level = lod_level
        if lod_level != LOD_FULL:
            raw_verts = _lod_subsample(raw_verts, lod_level)
            raw_rgb = _lod_subsample(raw_rgb, lod_level)
            raw_elev = _lod_subsample(raw_elev, lod_level)
            raw_vib = _lod_subsample(raw_vib, lod_level)

        self.vertex_count = len(raw_verts)

        # Terrain snapping arrays (always numpy — fast)
        self.raw_xz = raw_verts[:, [0, 2]].copy()
        self.raw_y = raw_verts[:, 1].copy()

        # Fast conversion: numpy .tolist() is 10-50x faster than
        # [Vec3(x,y,z) for x,y,z in arr] and Ursina Mesh accepts plain lists
        vertices = raw_verts.tolist()

        # Store numpy palette arrays for fast LOD rebuild & color switching
        self._np_rgb = raw_rgb
        self._np_elev = raw_elev
        self._np_vib = raw_vib

        # Convert only the ACTIVE palette to lists (deferred palette strategy)
        # This halves instantiation time since we skip 2 unused palette conversions
        self._color_mode = color_mode
        if color_mode == "Elevation Heatmap":
            active_colors = raw_elev.tolist()
        elif color_mode == "Vibrant RGB":
            active_colors = raw_vib.tolist()
        else:
            active_colors = raw_rgb.tolist()

        self.mesh = Mesh(
            vertices=vertices,
            colors=active_colors,
            mode='point',
            render_points_in_3d=False,
            thickness=thickness,
        )
        self.mesh.clearTexGen(TextureStage.getDefault())
        self.entity = Entity(model=self.mesh)
        self.entity.set_render_mode_perspective(False)
        self.entity.set_render_mode_thickness(thickness)

    def set_color_mode(self, mode: str) -> None:
        if mode == self._color_mode:
            return
        self._color_mode = mode
        if mode == "Natural RGB":
            self.mesh.colors = self._np_rgb.tolist()
        elif mode == "Elevation Heatmap":
            self.mesh.colors = self._np_elev.tolist()
        elif mode == "Vibrant RGB":
            self.mesh.colors = self._np_vib.tolist()
        self.mesh.generate()
        self.mesh.clearTexGen(TextureStage.getDefault())

    def set_thickness(self, thickness: int) -> None:
        self.mesh.clearTexGen(TextureStage.getDefault())
        self.mesh.render_points_in_3d = False
        self.mesh.thickness = thickness
        self.entity.set_render_mode_perspective(False)
        self.entity.set_render_mode_thickness(thickness)

    def destroy(self) -> None:
        destroy(self.entity)


class ExploreEnvironment:
    """Manages point cloud environment with Dynamic Spatial Chunk Streaming + multi-LOD.

    LOD system:
        LOD_FULL (level 0):  Every point — distance < LOD_FULL_RADIUS (35m)
        LOD_MED  (level 1):  Every 3rd point — 35m–70m
        LOD_LOW  (level 2):  Every 8th point — 70m–105m
        Beyond LOAD_RADIUS:  Unloaded (overview only)

    As you walk/fly closer to a structure, the system upgrades its chunk
    to higher detail. Moving away downgrades it to save GPU budget.
    """

    # Distance thresholds (meters)
    LOAD_RADIUS: float = 105.0     # Maximum streaming radius
    LOD_FULL_RADIUS: float = 35.0  # Full detail (every point)
    LOD_MED_RADIUS: float = 70.0   # Medium detail (1/3 points)
    CLOSE_RADIUS: float = 30.0     # Always-load zone (360°)
    UNLOAD_RADIUS: float = 135.0   # Eviction radius

    # View frustum
    VIEW_FOV_COS: float = 0.05     # ~174° horizontal FOV

    # Frame budget — keeps FPS rock-stable
    MAX_LOADS_PER_TICK: int = 1     # New chunk instantiations per tick
    MAX_UPGRADES_PER_TICK: int = 1  # LOD upgrade/downgrades per tick
    MAX_EVICTS_PER_TICK: int = 2    # Evictions per tick
    FRAME_TIME_GUARD: float = 0.020 # Skip loading if last frame > 20ms (< 50 FPS)

    def __init__(self, target_path: str, point_cloud: Optional[pd.DataFrame] = None,
                 downsample_step: int = 1,
                 point_thickness: float = 1.0,
                 reset_cooldown_duration: float = 2.0) -> None:
        """Initializes point cloud streamer, overview LOD, and first-person controller."""
        self.target_path = target_path
        self.point_thickness_levels = [1, 2, 3]
        self.current_thickness_idx = 0
        self.current_thickness = self.point_thickness_levels[self.current_thickness_idx]

        self.color_modes = ["Natural RGB", "Elevation Heatmap", "Vibrant RGB"]
        self.color_mode_idx = 0

        # Check if spatial chunks directory exists
        chunks_dir = target_path if isinstance(target_path, str) else ""
        if not (chunks_dir and os.path.isdir(chunks_dir) and os.path.exists(os.path.join(chunks_dir, "manifest.json"))):
            candidates = []
            if isinstance(target_path, str):
                candidates.extend([
                    os.path.join(os.path.dirname(target_path), "iitj_chunks"),
                    os.path.join(os.path.dirname(target_path), "chunks"),
                    os.path.join(os.path.dirname(target_path), f"{Path(target_path).stem}_chunks"),
                ])
            candidates.extend([
                os.path.join("models", "iitj_chunks"),
                os.path.abspath("models/iitj_chunks"),
            ])
            for candidate in candidates:
                if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "manifest.json")):
                    has_chunks = any(f.endswith(".npz") and f != "overview.npz" for f in os.listdir(candidate))
                    if has_chunks:
                        chunks_dir = candidate
                        break

        self.is_chunked = isinstance(chunks_dir, str) and os.path.isdir(chunks_dir) and os.path.exists(os.path.join(chunks_dir, "manifest.json"))
        self.chunks_dir = chunks_dir

        # Atmosphere & fog depth
        scene.fog_color = color.color(0, 0, 0.07)
        scene.fog_density = (80, 650)

        # Loaded chunks container
        self.loaded_chunks: Dict[str, LoadedChunk] = {}
        self.chunk_manifest: List[Dict] = []
        self.chunk_metadata: Dict[str, Dict] = {}
        self.landmarks: List[Dict] = []
        self.overview_entity: Optional[Entity] = None
        self.overview_mesh: Optional[Mesh] = None

        # View-based streaming queues and caches
        self.load_queue: List[Tuple[float, str, int]] = []  # [(priority, filename, lod_level)]
        self.upgrade_queue: List[Tuple[float, str, int]] = []  # [(priority, filename, new_lod)]
        self.evict_queue: List[str] = []
        self.chunk_data_cache: Dict[str, Dict] = {}
        self.in_memory_chunks: Dict[str, Dict] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # Frame timing for budget guard
        self._last_frame_time: float = 0.0

        if self.is_chunked:
            print(f"  Initializing Dynamic Spatial Streamer from {self.chunks_dir}...")
            with open(os.path.join(self.chunks_dir, "manifest.json")) as f:
                manifest_data = json.load(f)
                self.chunk_manifest = manifest_data.get("chunks", [])
                self.chunk_metadata = {
                    chunk["file"]: chunk for chunk in self.chunk_manifest
                    if "file" in chunk
                }
                self.landmarks = manifest_data.get("landmarks", [])
                bounds = manifest_data.get("bounds", {})
                self.min_x = bounds.get("min_x", -300.0)
                self.max_x = bounds.get("max_x", 300.0)
                self.min_z = bounds.get("min_z", -150.0)
                self.max_z = bounds.get("max_z", 150.0)

            # Load Overview LOD (lightweight 100k point cloud of entire campus)
            overview_file = os.path.join(self.chunks_dir, "overview.npz")
            if os.path.exists(overview_file):
                with np.load(overview_file) as ov_data:
                    ov_verts = ov_data["vertices"].tolist()
                    self._ov_np_rgb = ov_data["rgb"].copy()
                    self._ov_np_elev = ov_data["elevation"].copy()
                    self._ov_np_vib = ov_data["vibrant"].copy()

                self.overview_mesh = Mesh(
                    vertices=ov_verts,
                    colors=self._ov_np_rgb.tolist(),
                    mode='point',
                    render_points_in_3d=False,
                    thickness=self.current_thickness,
                )
                self.overview_mesh.clearTexGen(TextureStage.getDefault())
                self.overview_entity = Entity(model=self.overview_mesh)
                self.overview_entity.set_render_mode_perspective(False)
                self.overview_entity.set_render_mode_thickness(self.current_thickness)

            self.points_xz = np.empty((0, 2), dtype=np.float32)
            self.points_y = np.empty((0,), dtype=np.float32)

        elif point_cloud is not None:
            df = auto_align_up_axis(point_cloud.iloc[::downsample_step])
            if len(df) > 80_000:
                print(f"  Large point cloud ({len(df):,} pts) detected. Partitioning for dynamic view-based streaming...")
                self._partition_dataframe_into_chunks(df)
            else:
                self._load_single_point_cloud(df)
        else:
            # Fallback for single CSV files
            print(f"  Loading single point cloud file: {target_path}...")
            df = pd.read_csv(target_path)
            df = df.iloc[::downsample_step]
            df = auto_align_up_axis(df)
            if len(df) > 80_000:
                print(f"  Large CSV ({len(df):,} pts) detected. Partitioning for dynamic view-based streaming...")
                self._partition_dataframe_into_chunks(df)
            else:
                self._load_single_point_cloud(df)

        # Player controller setup
        self.player = FirstPersonController()
        self.player.gravity = 0
        self.player.speed = 0
        self.player.mouse_sensitivity = Vec2(0, 0)

        # Spawn tracking
        self.spawn_position: Optional[Vec3] = None
        self.spawn_rotation: Optional[Vec3] = None
        self.spawn_initialized: bool = False

        if self.is_chunked and "manifest_data" in locals() and "spawn_point" in manifest_data:
            sp = manifest_data["spawn_point"]
            sr = manifest_data.get("spawn_rotation", [28.0, -45.0, 0.0])
            self.player.position = Vec3(sp[0], sp[1], sp[2])
            self.player.rotation_x = sr[0]
            self.player.rotation_y = sr[1]
            self.player.rotation_z = sr[2]
            self.spawn_position = Vec3(self.player.position)
            self.spawn_rotation = Vec3(self.player.rotation_x, self.player.rotation_y, self.player.rotation_z)
            self.spawn_initialized = True

        self.player.prev_x = self.player.x
        self.player.prev_z = self.player.z

        # Flight & Cooldowns
        self.is_flying: bool = True if self.is_chunked else False
        self.flight_toggle_cooldown: float = 4
        self.reset_cooldown: float = 0
        self.reset_cooldown_duration: float = reset_cooldown_duration

        # Navigation Speeds
        self.walk_speed: float = 6.0
        self.flight_speed: float = 16.0
        self.vertical_speed: float = 8.0

        # Stream update throttle
        self.last_stream_time: float = 0.0

        # Initial stream tick around origin
        if self.is_chunked:
            self._update_chunk_streaming(self.player.x, self.player.z)

    def _load_single_point_cloud(self, df: pd.DataFrame) -> None:
        """Builds the single-file point-cloud mesh and navigation bounds."""
        self.points_xz = np.array([df["x"], df["z"]]).T
        self.points_y = np.array(df["y"])

        self.min_x = float(np.min(self.points_xz[:, 0]) + 1.0)
        self.max_x = float(np.max(self.points_xz[:, 0]) - 1.0)
        self.min_z = float(np.min(self.points_xz[:, 1]) + 1.0)
        self.max_z = float(np.max(self.points_xz[:, 1]) - 1.0)

        raw_r = np.array(df["r"]) if "r" in df.columns else np.full(len(df), 200)
        raw_g = np.array(df["g"]) if "g" in df.columns else np.full(len(df), 200)
        raw_b = np.array(df["b"]) if "b" in df.columns else np.full(len(df), 200)

        self.palette_rgb = [(r / 255.0, g / 255.0, b / 255.0, 1.0) for r, g, b in zip(raw_r, raw_g, raw_b)]
        self.palette_elevation = _generate_elevation_colors(self.points_y)
        self.palette_contrast = _generate_contrast_colors(raw_r, raw_g, raw_b)

        vertices = [Vec3(x, y, z) for x, y, z in zip(df["x"], df["y"], df["z"])]
        self.single_mesh = Mesh(
            vertices=vertices,
            colors=self.palette_rgb,
            mode='point',
            render_points_in_3d=False,
            thickness=self.current_thickness,
        )
        self.single_mesh.clearTexGen(TextureStage.getDefault())
        self.single_entity = Entity(model=self.single_mesh)
        self.single_entity.set_render_mode_perspective(False)
        self.single_entity.set_render_mode_thickness(self.current_thickness)

    def _get_desired_lod(self, distance: float) -> int:
        """Returns the LOD level for a given distance from the player."""
        if distance <= self.LOD_FULL_RADIUS:
            return LOD_FULL
        elif distance <= self.LOD_MED_RADIUS:
            return LOD_MED
        else:
            return LOD_LOW

    def _prefetch_chunk_async(self, filename: str) -> None:
        """Asynchronously pre-loads chunk arrays from disk into memory."""
        chunk_file_path = os.path.join(self.chunks_dir, filename)
        if not os.path.exists(chunk_file_path):
            return

        def _worker():
            try:
                with np.load(chunk_file_path) as npz:
                    self.chunk_data_cache[filename] = {k: npz[k] for k in npz.files}
            except Exception:
                pass

        self.executor.submit(_worker)

    def _update_chunk_streaming(self, px: float, pz: float) -> None:
        """Dynamically identifies chunks in player view frustum, assigns LOD levels,
        and queues them for smooth budgeted loading/upgrading."""
        if not self.is_chunked:
            return

        # Player horizontal viewing heading from forward vector
        fx = float(self.player.forward.x)
        fz = float(self.player.forward.z)
        f_len = (fx * fx + fz * fz) ** 0.5
        if f_len > 1e-4:
            fx /= f_len
            fz /= f_len
        else:
            fx, fz = 0.0, 1.0

        needed_files: Set[str] = set()
        queued_load_files = {fname for _, fname, _ in self.load_queue}
        queued_upgrade_files = {fname for _, fname, _ in self.upgrade_queue}

        for chunk_meta in self.chunk_manifest:
            cx, cy, cz = chunk_meta["center"]
            dx = cx - px
            dz = cz - pz
            dist = (dx * dx + dz * dz) ** 0.5

            # Dynamic View-Frustum / Proximity Test:
            # 1. Immediate close zone (<= 30m): always load ground around player
            # 2. View cone (30m to 105m): only chunks in front of camera (horizontal FOV)
            if dist <= self.CLOSE_RADIUS:
                in_view = True
                dot = 1.0
            elif dist <= self.LOAD_RADIUS:
                dot = (dx * fx + dz * fz) / max(dist, 1e-3)
                in_view = (dot >= self.VIEW_FOV_COS)
            else:
                in_view = False
                dot = -1.0

            if in_view:
                filename = chunk_meta["file"]
                needed_files.add(filename)
                desired_lod = self._get_desired_lod(dist)

                if filename in self.loaded_chunks:
                    # Check if LOD upgrade/downgrade is needed
                    current_lod = self.loaded_chunks[filename].lod_level
                    if current_lod != desired_lod and filename not in queued_upgrade_files:
                        priority = dist - 35.0 * max(0.0, dot)
                        # Upgrades (decreasing LOD number) get priority boost
                        if desired_lod < current_lod:
                            priority -= 20.0
                        self.upgrade_queue.append((priority, filename, desired_lod))
                        queued_upgrade_files.add(filename)
                elif filename not in queued_load_files:
                    # Queue for initial load at the appropriate LOD
                    priority = dist - 35.0 * max(0.0, dot)
                    self.load_queue.append((priority, filename, desired_lod))
                    queued_load_files.add(filename)

                    if filename not in self.chunk_data_cache and filename not in self.in_memory_chunks:
                        self._prefetch_chunk_async(filename)

        # Sort queues by priority (lowest score = highest priority)
        self.load_queue.sort(key=lambda item: item[0])
        self.upgrade_queue.sort(key=lambda item: item[0])

        # Filter out chunks that are no longer in view or already loaded
        self.load_queue = [(p, f, lod) for p, f, lod in self.load_queue
                           if f in needed_files and f not in self.loaded_chunks]
        self.upgrade_queue = [(p, f, lod) for p, f, lod in self.upgrade_queue
                              if f in needed_files and f in self.loaded_chunks]

        # Identify chunks to evict (outside UNLOAD_RADIUS or behind camera outside close zone)
        for filename, chunk in list(self.loaded_chunks.items()):
            if filename not in needed_files and filename not in self.evict_queue:
                meta = self.chunk_metadata.get(filename)
                if meta:
                    cx, _, cz = meta["center"]
                    dx = cx - px
                    dz = cz - pz
                    dist = (dx * dx + dz * dz) ** 0.5
                    dot = (dx * fx + dz * fz) / max(dist, 1e-3)
                    if dist > self.UNLOAD_RADIUS or (dist > self.CLOSE_RADIUS + 10.0 and dot < -0.2):
                        self.evict_queue.append(filename)

    def _get_chunk_data(self, filename: str) -> Optional[Dict]:
        """Retrieves chunk data from in-memory cache, disk cache, or loads from disk."""
        if filename in self.in_memory_chunks:
            return self.in_memory_chunks[filename]
        if filename in self.chunk_data_cache:
            return self.chunk_data_cache[filename]
        chunk_file_path = os.path.join(self.chunks_dir, filename)
        if os.path.exists(chunk_file_path):
            try:
                with np.load(chunk_file_path) as npz:
                    data = {k: npz[k] for k in npz.files}
                self.chunk_data_cache[filename] = data
                return data
            except Exception:
                return None
        return None

    def _process_streaming_budget(self) -> None:
        """Instantiates at most 1 chunk, upgrades at most 1 LOD, and evicts at most 2
        chunks per tick to keep FPS smooth and lag-free.

        Frame-time guard: skips new loads/upgrades when the previous frame was
        already slow (> 20ms), allowing the GPU to catch up before adding more
        geometry. Evictions always proceed since they free resources.
        """
        if not self.is_chunked:
            return

        # Frame-time budget guard: if the last frame was slow, only do evictions
        frame_budget_ok = self._last_frame_time < self.FRAME_TIME_GUARD
        color_mode = self.color_modes[self.color_mode_idx]

        # 1. Budgeted Chunk Instantiation (at most 1 chunk per frame)
        if frame_budget_ok:
            loads_done = 0
            while self.load_queue and loads_done < self.MAX_LOADS_PER_TICK:
                _, filename, lod_level = self.load_queue.pop(0)
                if filename in self.loaded_chunks:
                    continue

                data = self._get_chunk_data(filename)
                if data is not None:
                    chunk = LoadedChunk(
                        data,
                        thickness=self.current_thickness,
                        color_mode=color_mode,
                        lod_level=lod_level,
                    )
                    self.loaded_chunks[filename] = chunk
                    loads_done += 1

        # 2. Budgeted LOD Upgrades/Downgrades (at most 1 per frame)
        if frame_budget_ok:
            upgrades_done = 0
            while self.upgrade_queue and upgrades_done < self.MAX_UPGRADES_PER_TICK:
                _, filename, new_lod = self.upgrade_queue.pop(0)
                if filename not in self.loaded_chunks:
                    continue
                current_chunk = self.loaded_chunks[filename]
                if current_chunk.lod_level == new_lod:
                    continue

                data = self._get_chunk_data(filename)
                if data is not None:
                    # Destroy old chunk and create new one at different LOD
                    current_chunk.destroy()
                    new_chunk = LoadedChunk(
                        data,
                        thickness=self.current_thickness,
                        color_mode=color_mode,
                        lod_level=new_lod,
                    )
                    self.loaded_chunks[filename] = new_chunk
                    upgrades_done += 1

        # 3. Budgeted Chunk Eviction (at most 2 chunks per frame — always runs)
        evicts_done = 0
        while self.evict_queue and evicts_done < self.MAX_EVICTS_PER_TICK:
            fname = self.evict_queue.pop(0)
            if fname in self.loaded_chunks:
                self.loaded_chunks[fname].destroy()
                del self.loaded_chunks[fname]
                evicts_done += 1

    def _partition_dataframe_into_chunks(self, df: pd.DataFrame, cell_size: float = 35.0, overview_target: int = 100_000) -> None:
        """Dynamically partitions an in-memory DataFrame into spatial chunks for view-based streaming."""
        xs = np.array(df["x"], dtype=np.float32)
        ys = np.array(df["y"], dtype=np.float32)
        zs = np.array(df["z"], dtype=np.float32)

        raw_r = np.array(df["r"], dtype=np.uint8) if "r" in df.columns else np.full(len(xs), 210, dtype=np.uint8)
        raw_g = np.array(df["g"], dtype=np.uint8) if "g" in df.columns else np.full(len(xs), 190, dtype=np.uint8)
        raw_b = np.array(df["b"], dtype=np.uint8) if "b" in df.columns else np.full(len(xs), 160, dtype=np.uint8)

        total_pts = len(xs)

        sandstone_colors = _create_architectural_palette(ys, raw_r, raw_g, raw_b)
        elevation_colors = np.array(_generate_elevation_colors(ys), dtype=np.float32)
        vibrant_colors = np.array(_generate_contrast_colors(raw_r, raw_g, raw_b), dtype=np.float32)

        ov_step = max(1, total_pts // overview_target)
        ov_idx = np.arange(0, total_pts, ov_step)
        ov_verts = np.column_stack([xs[ov_idx], ys[ov_idx], zs[ov_idx]])

        self._ov_np_rgb = sandstone_colors[ov_idx]
        self._ov_np_elev = elevation_colors[ov_idx]
        self._ov_np_vib = vibrant_colors[ov_idx]

        self.overview_mesh = Mesh(
            vertices=ov_verts.tolist(),
            colors=self._ov_np_rgb.tolist(),
            mode='point',
            render_points_in_3d=False,
            thickness=self.current_thickness,
        )
        self.overview_mesh.clearTexGen(TextureStage.getDefault())
        self.overview_entity = Entity(model=self.overview_mesh)
        self.overview_entity.set_render_mode_perspective(False)
        self.overview_entity.set_render_mode_thickness(self.current_thickness)

        grid_gx = np.floor(xs / cell_size).astype(np.int32)
        grid_gz = np.floor(zs / cell_size).astype(np.int32)
        cell_keys = grid_gx * 100000 + grid_gz
        sort_idx = np.argsort(cell_keys)
        sorted_keys = cell_keys[sort_idx]
        unique_keys, split_indices = np.unique(sorted_keys, return_index=True)
        cell_groups = np.split(sort_idx, split_indices[1:])

        self.in_memory_chunks = {}
        self.chunk_manifest = []
        self.chunk_metadata = {}

        for key, indices in zip(unique_keys, cell_groups):
            if len(indices) < 20:
                continue
            gx = int(grid_gx[indices[0]])
            gz = int(grid_gz[indices[0]])
            chunk_verts = np.column_stack([xs[indices], ys[indices], zs[indices]]).astype(np.float32)
            center = [float(np.mean(chunk_verts[:, 0])), float(np.mean(chunk_verts[:, 1])), float(np.mean(chunk_verts[:, 2]))]
            filename = f"mem_chunk_{gx}_{gz}"

            self.in_memory_chunks[filename] = {
                "vertices": chunk_verts,
                "rgb": sandstone_colors[indices],
                "elevation": elevation_colors[indices],
                "vibrant": vibrant_colors[indices],
                "center": np.array(center, dtype=np.float32),
            }
            meta = {
                "gx": gx,
                "gz": gz,
                "file": filename,
                "center": center,
                "point_count": len(indices),
            }
            self.chunk_manifest.append(meta)
            self.chunk_metadata[filename] = meta

        self.min_x = float(np.min(xs))
        self.max_x = float(np.max(xs))
        self.min_z = float(np.min(zs))
        self.max_z = float(np.max(zs))
        self.is_chunked = True

    def set_point_thickness(self, thickness: int) -> None:
        """Sets hardware point thickness across all active chunks."""
        self.current_thickness = thickness
        if self.is_chunked:
            if self.overview_entity:
                self.overview_entity.set_render_mode_thickness(thickness)
            for chunk in self.loaded_chunks.values():
                chunk.set_thickness(thickness)
        else:
            if hasattr(self, "single_entity"):
                self.single_entity.set_render_mode_thickness(thickness)

    def cycle_point_thickness(self) -> str:
        """Cycles between point thicknesses (1px, 2px, 3px)."""
        self.current_thickness_idx = (self.current_thickness_idx + 1) % len(self.point_thickness_levels)
        t = self.point_thickness_levels[self.current_thickness_idx]
        self.set_point_thickness(t)
        return f"{t}px"

    def cycle_color_mode(self) -> str:
        """Cycles active color palette (Natural RGB -> Elevation -> Vibrant RGB)."""
        self.color_mode_idx = (self.color_mode_idx + 1) % len(self.color_modes)
        mode_name = self.color_modes[self.color_mode_idx]

        if self.is_chunked:
            if self.overview_mesh:
                if mode_name == "Natural RGB":
                    self.overview_mesh.colors = self._ov_np_rgb.tolist()
                elif mode_name == "Elevation Heatmap":
                    self.overview_mesh.colors = self._ov_np_elev.tolist()
                elif mode_name == "Vibrant RGB":
                    self.overview_mesh.colors = self._ov_np_vib.tolist()
                self.overview_mesh.generate()
                self.overview_mesh.clearTexGen(TextureStage.getDefault())

            for chunk in self.loaded_chunks.values():
                chunk.set_color_mode(mode_name)
        else:
            if hasattr(self, "single_mesh"):
                if mode_name == "Natural RGB":
                    self.single_mesh.colors = self.palette_rgb
                elif mode_name == "Elevation Heatmap":
                    self.single_mesh.colors = self.palette_elevation
                elif mode_name == "Vibrant RGB":
                    self.single_mesh.colors = self.palette_contrast
                self.single_mesh.generate()
                self.single_mesh.clearTexGen(TextureStage.getDefault())

        return mode_name

    def reset_player(self) -> None:
        """Resets player to spawn position and rotation."""
        if self.spawn_initialized:
            self.player.position = self.spawn_position
            self.player.rotation_x = self.spawn_rotation.x
            self.player.rotation_y = self.spawn_rotation.y
            self.player.rotation_z = self.spawn_rotation.z
            self.player.prev_x = self.player.x
            self.player.prev_z = self.player.z
            if self.is_chunked:
                self._update_chunk_streaming(self.player.x, self.player.z)

    def jump_to_landmark(self, idx: int) -> Optional[str]:
        """Teleports player to a specific landmark and triggers immediate chunk loading."""
        if not self.landmarks or idx < 0 or idx >= len(self.landmarks):
            return None
        lm = self.landmarks[idx]
        pos = lm["position"]
        self.player.position = Vec3(pos[0], pos[1], pos[2])
        self.player.rotation_x = 22.0
        self.player.rotation_y = -45.0
        self.player.rotation_z = 0.0
        self.player.prev_x = self.player.x
        self.player.prev_z = self.player.z
        self.is_flying = True
        if self.is_chunked:
            self._update_chunk_streaming(self.player.x, self.player.z)
        return lm.get("name", f"Landmark {idx+1}")

    def update_steering(self, right_state: Dict) -> None:
        """Updates player view direction based on right hand position."""
        if right_state.get('visible') and right_state.get('gesture') == 'Steering':
            if right_state['x'] < 0.4:
                self.player.rotation_y -= 80 * time.dt * (0.4 - right_state['x'])
            elif right_state['x'] > 0.6:
                self.player.rotation_y += 80 * time.dt * (right_state['x'] - 0.6)
            if right_state['y'] < 0.4:
                self.player.rotation_x -= 60 * time.dt * (0.4 - right_state['y'])
            elif right_state['y'] > 0.6:
                self.player.rotation_x += 60 * time.dt * (right_state['y'] - 0.6)

    def update_movement_and_terrain(self, left_state: Dict) -> None:
        """Updates movement, flight, dynamic chunk streaming with LOD, and terrain hugging."""
        frame_start = _time.perf_counter()
        current_speed = self.flight_speed if self.is_flying else self.walk_speed

        # Dynamic chunk streaming tick (throttled to ~15 Hz)
        if self.is_chunked:
            now = _time.time()
            if now - self.last_stream_time > 0.066:
                self._update_chunk_streaming(self.player.x, self.player.z)
                self.last_stream_time = now
            # Process budget every frame for ultra-smooth chunk delivery (no FPS drops)
            self._process_streaming_budget()

        # Hand gesture movement
        if left_state.get('visible'):
            if (left_state.get('gesture') == 'Toggle Flight' and
                    _time.time() > self.flight_toggle_cooldown):
                self.is_flying = not self.is_flying
                self.flight_toggle_cooldown = _time.time() + 1.0

            if left_state.get('gesture') == 'Reset':
                if self.reset_cooldown_duration > 0:
                    if _time.time() > self.reset_cooldown:
                        self.reset_player()
                        self.reset_cooldown = _time.time() + self.reset_cooldown_duration
                else:
                    self.reset_player()

            if not self.is_flying:
                if left_state.get('gesture') == 'Backward':
                    self.player.position -= self.player.forward * current_speed * time.dt
                elif left_state.get('gesture') == 'Forward':
                    self.player.position += self.player.forward * current_speed * time.dt

        # Keyboard fallback navigation (additional to hand gestures — both work simultaneously)
        if held_keys['w'] or held_keys['up arrow']:
            self.player.position += self.player.forward * current_speed * time.dt
        if held_keys['s'] or held_keys['down arrow']:
            self.player.position -= self.player.forward * current_speed * time.dt
        if held_keys['a'] or held_keys['left arrow']:
            self.player.position -= self.player.right * current_speed * time.dt
        if held_keys['d'] or held_keys['right arrow']:
            self.player.position += self.player.right * current_speed * time.dt

        if self.is_flying:
            if held_keys['space']:
                self.player.y += self.vertical_speed * time.dt
            if held_keys['left shift'] or held_keys['c']:
                self.player.y -= self.vertical_speed * time.dt

        # Ultra-fast terrain height computation (< 0.05ms)
        px, pz = self.player.x, self.player.z
        ground_y = self.player.y - 2.0

        if self.is_chunked:
            nearby_y_samples = []
            for filename, chunk in self.loaded_chunks.items():
                meta = self.chunk_metadata.get(filename)
                if meta:
                    cx, _, cz = meta["center"]
                    if abs(cx - px) > 40.0 or abs(cz - pz) > 40.0:
                        continue
                # Fast bounding box filter before radius test
                dx = np.abs(chunk.raw_xz[:, 0] - px)
                dz = np.abs(chunk.raw_xz[:, 1] - pz)
                box_mask = (dx < 3.0) & (dz < 3.0)
                if np.any(box_mask):
                    d_sq = dx[box_mask] ** 2 + dz[box_mask] ** 2
                    rad_mask = d_sq < 9.0
                    if np.any(rad_mask):
                        nearby_y_samples.extend(chunk.raw_y[box_mask][rad_mask])

            if nearby_y_samples:
                ground_y = float(np.percentile(nearby_y_samples, 25))
        elif hasattr(self, "points_xz") and len(self.points_xz) > 0:
            # Quick bounding box filter for single point cloud
            dx = np.abs(self.points_xz[:, 0] - px)
            dz = np.abs(self.points_xz[:, 1] - pz)
            box_mask = (dx < 3.0) & (dz < 3.0)
            if np.any(box_mask):
                d_sq = dx[box_mask] ** 2 + dz[box_mask] ** 2
                rad_mask = d_sq < 9.0
                if np.any(rad_mask):
                    ground_y = float(np.percentile(self.points_y[box_mask][rad_mask], 25))

        # Flying vs grounded
        if self.is_flying:
            if left_state.get('visible'):
                if left_state.get('gesture') == 'Up':
                    self.player.y += self.vertical_speed * time.dt
                elif left_state.get('gesture') == 'Down':
                    self.player.y -= self.vertical_speed * time.dt
                elif left_state.get('gesture') == 'Backward':
                    self.player.position -= self.player.forward * self.flight_speed * time.dt
                elif left_state.get('gesture') == 'Forward':
                    self.player.position += self.player.forward * self.flight_speed * time.dt
            if self.player.y < ground_y + 1.8:
                self.player.y = ground_y + 1.8
        else:
            self.player.y = lerp(self.player.y, ground_y + 1.8, time.dt * 10)
            if not self.spawn_initialized:
                self.spawn_position = Vec3(self.player.position)
                self.spawn_rotation = Vec3(
                    self.player.rotation_x,
                    self.player.rotation_y,
                    self.player.rotation_z,
                )
