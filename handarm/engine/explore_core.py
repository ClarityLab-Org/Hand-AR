"""
Shared explore-mode logic for point cloud navigation with Dynamic Spatial Chunk Streaming (LOD).
Dynamically streams high-density LiDAR chunks based on camera proximity and view.
"""

import json
import os
import time as _time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from panda3d.core import TextureStage
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController

from handarm.geometry.alignment import auto_align_up_axis


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
    """Represents a dynamically loaded high-density spatial point cloud chunk."""

    def __init__(self, data: np.lib.npyio.NpzFile, thickness: int = 1, color_mode: str = "Natural RGB") -> None:
        raw_verts = data["vertices"]
        self.vertices = [Vec3(x, y, z) for x, y, z in raw_verts]
        self.raw_xz = raw_verts[:, [0, 2]]
        self.raw_y = raw_verts[:, 1]

        self.palette_rgb = [Vec4(r, g, b, a) for r, g, b, a in data["rgb"]]
        self.palette_elevation = [Vec4(r, g, b, a) for r, g, b, a in data["elevation"]]
        self.palette_contrast = [Vec4(r, g, b, a) for r, g, b, a in data["vibrant"]]

        initial_colors = self.palette_rgb
        if color_mode == "Elevation Heatmap":
            initial_colors = self.palette_elevation
        elif color_mode == "Vibrant RGB":
            initial_colors = self.palette_contrast

        self.mesh = Mesh(
            vertices=self.vertices,
            colors=initial_colors,
            mode='point',
            render_points_in_3d=False,
            thickness=thickness,
        )
        self.mesh.clearTexGen(TextureStage.getDefault())
        self.entity = Entity(model=self.mesh)
        self.entity.set_render_mode_perspective(False)
        self.entity.set_render_mode_thickness(thickness)

    def set_color_mode(self, mode: str) -> None:
        if mode == "Natural RGB":
            self.mesh.colors = self.palette_rgb
        elif mode == "Elevation Heatmap":
            self.mesh.colors = self.palette_elevation
        elif mode == "Vibrant RGB":
            self.mesh.colors = self.palette_contrast
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
    """Manages point cloud environment with Dynamic Spatial Chunk Streaming."""

    LOAD_RADIUS: float = 85.0     # Radius (meters) to stream high-density building chunks
    UNLOAD_RADIUS: float = 120.0  # Radius (meters) to evict out-of-range chunks

    def __init__(self, target_path: str, downsample_step: int = 1,
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
        chunks_dir = target_path
        if not (os.path.isdir(chunks_dir) and os.path.exists(os.path.join(chunks_dir, "manifest.json"))):
            default_chunks_dir = os.path.join(os.path.dirname(target_path), "iitj_chunks")
            if os.path.exists(os.path.join(default_chunks_dir, "manifest.json")):
                chunks_dir = default_chunks_dir

        self.is_chunked = os.path.isdir(chunks_dir) and os.path.exists(os.path.join(chunks_dir, "manifest.json"))
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

        if self.is_chunked:
            print(f"⚡ Initializing Dynamic Spatial Streamer from {self.chunks_dir}...")
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
                    ov_verts = [Vec3(x, y, z) for x, y, z in ov_data["vertices"]]
                    self.ov_palette_rgb = [Vec4(r, g, b, a) for r, g, b, a in ov_data["rgb"]]
                    self.ov_palette_elevation = [Vec4(r, g, b, a) for r, g, b, a in ov_data["elevation"]]
                    self.ov_palette_contrast = [Vec4(r, g, b, a) for r, g, b, a in ov_data["vibrant"]]

                self.overview_mesh = Mesh(
                    vertices=ov_verts,
                    colors=self.ov_palette_rgb,
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

        else:
            # Fallback for single CSV files
            print(f"📄 Loading single point cloud file: {target_path}...")
            df = pd.read_csv(target_path)
            df = df.iloc[::downsample_step]
            df = auto_align_up_axis(df)

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

        # Player controller setup
        self.player = FirstPersonController()
        self.player.gravity = 0
        self.player.speed = 0
        self.player.mouse_sensitivity = Vec2(0, 0)
        self.player.prev_x = self.player.x
        self.player.prev_z = self.player.z

        # Spawn tracking
        self.spawn_position: Optional[Vec3] = None
        self.spawn_rotation: Optional[Vec3] = None
        self.spawn_initialized: bool = False

        # Flight & Cooldowns
        self.is_flying: bool = False
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

    def _update_chunk_streaming(self, px: float, pz: float) -> None:
        """Loads nearby chunks and evicts distant chunks dynamically."""
        if not self.is_chunked:
            return

        needed_files: Set[str] = set()

        for chunk_meta in self.chunk_manifest:
            cx, cy, cz = chunk_meta["center"]
            dist_sq = (cx - px) ** 2 + (cz - pz) ** 2

            if dist_sq <= self.LOAD_RADIUS ** 2:
                filename = chunk_meta["file"]
                needed_files.add(filename)

                if filename not in self.loaded_chunks:
                    chunk_file_path = os.path.join(self.chunks_dir, filename)
                    if os.path.exists(chunk_file_path):
                        with np.load(chunk_file_path) as data:
                            chunk = LoadedChunk(
                                data,
                                thickness=self.current_thickness,
                                color_mode=self.color_modes[self.color_mode_idx],
                            )
                        self.loaded_chunks[filename] = chunk

        # Evict chunks outside UNLOAD_RADIUS
        to_remove = []
        for filename, chunk in self.loaded_chunks.items():
            if filename not in needed_files:
                meta = self.chunk_metadata.get(filename)
                if meta:
                    cx, _, cz = meta["center"]
                    dist_sq = (cx - px) ** 2 + (cz - pz) ** 2
                    if dist_sq > self.UNLOAD_RADIUS ** 2:
                        chunk.destroy()
                        to_remove.append(filename)

        for filename in to_remove:
            del self.loaded_chunks[filename]

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
                    self.overview_mesh.colors = self.ov_palette_rgb
                elif mode_name == "Elevation Heatmap":
                    self.overview_mesh.colors = self.ov_palette_elevation
                elif mode_name == "Vibrant RGB":
                    self.overview_mesh.colors = self.ov_palette_contrast
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
        """Updates movement, flight, dynamic chunk streaming, and terrain hugging."""
        current_speed = self.flight_speed if self.is_flying else self.walk_speed

        # Dynamic chunk streaming tick (throttled to 10 Hz)
        if self.is_chunked and (_time.time() - self.last_stream_time > 0.1):
            self._update_chunk_streaming(self.player.x, self.player.z)
            self.last_stream_time = _time.time()

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

        # Keyboard fallback navigation
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

        # Terrain height computation from loaded chunks or point cloud
        px, pz = self.player.x, self.player.z
        ground_y = self.player.y - 2.0

        if self.is_chunked:
            nearby_y_samples = []
            for chunk in self.loaded_chunks.values():
                d_sq = (chunk.raw_xz[:, 0] - px) ** 2 + (chunk.raw_xz[:, 1] - pz) ** 2
                mask = d_sq < 9.0
                if np.any(mask):
                    nearby_y_samples.extend(chunk.raw_y[mask])

            if nearby_y_samples:
                ground_y = float(np.percentile(nearby_y_samples, 25))
        else:
            dists_sq = ((self.points_xz[:, 0] - px) ** 2 + (self.points_xz[:, 1] - pz) ** 2)
            ground_mask = dists_sq < 9.0
            if np.any(ground_mask):
                ground_y = float(np.percentile(self.points_y[ground_mask], 25))

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
                self.spawn_initialized = True

        # Boundary clamping
        self.player.x = clamp(self.player.x, self.min_x, self.max_x)
        self.player.z = clamp(self.player.z, self.min_z, self.max_z)
