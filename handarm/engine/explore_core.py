"""Shared explore-mode logic for point cloud navigation."""

import numpy as np
import pandas as pd
import time as _time
from typing import Dict, Optional, Tuple
from panda3d.core import TextureStage
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController

from handarm.geometry.alignment import auto_align_up_axis


class ExploreEnvironment:
    """Manages point cloud environment and first-person player navigation."""

    def __init__(self, csv_path: str, downsample_step: int = 1,
                 point_thickness: float = 1.0,
                 reset_cooldown_duration: float = 2.0) -> None:
        """Load point cloud, create mesh entity, and set up player controller."""
        # Load and process point cloud
        df = pd.read_csv(csv_path)
        df = df.iloc[::downsample_step]
        df = auto_align_up_axis(df)

        self.points_xz = np.array([df["x"], df["z"]]).T
        self.points_y = np.array(df["y"])

        self.min_x = float(np.min(self.points_xz[:, 0]) + 1.0)
        self.max_x = float(np.max(self.points_xz[:, 0]) - 1.0)
        self.min_z = float(np.min(self.points_xz[:, 1]) + 1.0)
        self.max_z = float(np.max(self.points_xz[:, 1]) - 1.0)

        vertices = [Vec3(x, y, z) for x, y, z in zip(df["x"], df["y"], df["z"])]
        point_colors = [(r / 255.0, g / 255.0, b / 255.0, 1.0)
                        for r, g, b in zip(df["r"], df["g"], df["b"])]
        mesh = Mesh(
            vertices=vertices,
            colors=point_colors,
            mode='point',
            render_points_in_3d=False,
            thickness=1,
        )
        mesh.clearTexGen(TextureStage.getDefault())
        self.point_entity = Entity(model=mesh)
        self.point_entity.set_render_mode_thickness(1)
        self.point_entity.set_render_mode_perspective(False)

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
