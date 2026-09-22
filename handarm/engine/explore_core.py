"""Shared explore-mode logic for point cloud navigation."""

import numpy as np
import pandas as pd
import time as _time
from typing import Dict, List, Optional, Tuple
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


class ExploreEnvironment:
    """Manages point cloud environment and first-person player navigation."""

    def __init__(self, csv_path: str, downsample_step: int = 1,
                 point_thickness: float = 1.0,
                 reset_cooldown_duration: float = 2.0) -> None:
        """Load point cloud, create mesh entity, and set up player controller."""
        df = pd.read_csv(csv_path)
        df = df.iloc[::downsample_step]
        df = auto_align_up_axis(df)

        self.points_xz = np.array([df["x"], df["z"]]).T
        self.points_y = np.array(df["y"])

        self.min_x = float(np.min(self.points_xz[:, 0]) + 1.0)
        self.max_x = float(np.max(self.points_xz[:, 0]) - 1.0)
        self.min_z = float(np.min(self.points_xz[:, 1]) + 1.0)
        self.max_z = float(np.max(self.points_xz[:, 1]) - 1.0)

        self.vertices = [Vec3(x, y, z) for x, y, z in zip(df["x"], df["y"], df["z"])]

        # Precompute color palettes
        raw_r = np.array(df["r"]) if "r" in df.columns else np.full(len(df), 200)
        raw_g = np.array(df["g"]) if "g" in df.columns else np.full(len(df), 200)
        raw_b = np.array(df["b"]) if "b" in df.columns else np.full(len(df), 200)

        self.palette_rgb = [(r / 255.0, g / 255.0, b / 255.0, 1.0)
                            for r, g, b in zip(raw_r, raw_g, raw_b)]
        self.palette_elevation = _generate_elevation_colors(self.points_y)
        self.palette_contrast = _generate_contrast_colors(raw_r, raw_g, raw_b)

        self.color_modes = ["Natural RGB", "Elevation Heatmap", "Vibrant RGB"]
        self.color_mode_idx = 0

        # Construct point mesh
        self.mesh = Mesh(
            vertices=self.vertices,
            colors=self.palette_rgb,
            mode='point',
            render_points_in_3d=False,
            thickness=1,
        )
        self.mesh.clearTexGen(TextureStage.getDefault())
        self.point_entity = Entity(model=self.mesh)
        self.point_thickness_levels = [1, 2, 3]
        self.current_thickness_idx = 0
        self.set_point_thickness(self.point_thickness_levels[self.current_thickness_idx])

        # Player controller
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

        # Flight state
        self.is_flying: bool = False
        self.flight_toggle_cooldown: float = 4
        self.reset_cooldown: float = 0
        self.reset_cooldown_duration: float = reset_cooldown_duration

    def set_point_thickness(self, thickness: int) -> None:
        """Configures hardware point size in screen pixels."""
        self.mesh.clearTexGen(TextureStage.getDefault())
        self.mesh.render_points_in_3d = False
        self.mesh.thickness = thickness
        self.point_entity.set_render_mode_perspective(False)
        self.point_entity.set_render_mode_thickness(thickness)

    def cycle_point_thickness(self) -> str:
        """Cycles between point sizes (1px, 2px, 3px)."""
        self.current_thickness_idx = (self.current_thickness_idx + 1) % len(self.point_thickness_levels)
        t = self.point_thickness_levels[self.current_thickness_idx]
        self.set_point_thickness(t)
        return f"{t}px"

    def cycle_color_mode(self) -> str:
        """Cycles point cloud color palette (Natural RGB -> Elevation -> Vibrant RGB)."""
        self.color_mode_idx = (self.color_mode_idx + 1) % len(self.color_modes)
        mode_name = self.color_modes[self.color_mode_idx]

        if mode_name == "Natural RGB":
            self.mesh.colors = self.palette_rgb
        elif mode_name == "Elevation Heatmap":
            self.mesh.colors = self.palette_elevation
        elif mode_name == "Vibrant RGB":
            self.mesh.colors = self.palette_contrast

        self.mesh.generate()
        self.mesh.clearTexGen(TextureStage.getDefault())
        t = self.point_thickness_levels[self.current_thickness_idx]
        self.set_point_thickness(t)
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
        """Updates movement, flight toggling, terrain following, and collision."""
        # Flight toggle + reset
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
                    self.player.position -= self.player.forward * 2.0 * time.dt
                elif left_state.get('gesture') == 'Forward':
                    self.player.position += self.player.forward * 2.0 * time.dt

        # Terrain height computation
        px, pz = self.player.x, self.player.z
        dists_sq = ((self.points_xz[:, 0] - px) ** 2 +
                    (self.points_xz[:, 1] - pz) ** 2)

        ground_mask = dists_sq < 9.0
        ground_y = self.player.y - 2.0

        if np.any(ground_mask):
            ground_y = float(np.percentile(self.points_y[ground_mask], 25))

        # Flying vs grounded
        if self.is_flying:
            if left_state.get('visible'):
                if left_state.get('gesture') == 'Up':
                    self.player.y += 2 * time.dt
                elif left_state.get('gesture') == 'Down':
                    self.player.y -= 2 * time.dt
                elif left_state.get('gesture') == 'Backward':
                    self.player.position -= self.player.forward * 2.0 * time.dt
                elif left_state.get('gesture') == 'Forward':
                    self.player.position += self.player.forward * 2.0 * time.dt
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

        # Collision detection
        collision_mask = dists_sq < 0.09
        if np.any(collision_mask):
            close_points_y = self.points_y[collision_mask]
            local_foot_level = float(np.min(close_points_y))
            if self.player.y < local_foot_level + 1.0:
                if np.any((close_points_y > local_foot_level + 0.15) &
                          (close_points_y < local_foot_level + 0.5)):
                    self.player.x = self.player.prev_x
                    self.player.z = self.player.prev_z
                else:
                    self.player.prev_x = self.player.x
                    self.player.prev_z = self.player.z
        else:
            self.player.prev_x = self.player.x
            self.player.prev_z = self.player.z

