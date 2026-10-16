"""
Standalone Explore Mode for Point Cloud Roaming in Ursina.

Enables hand gesture control for first-person walking and flying over terrain.
Uses the explore.py standalone gesture variant (simple index_up, 0.05 thumbs_up
margin, Down-before-Up priority). No hand-registration init overlay.

Usage:
    python explore.py [path/to/cloud.csv]
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import psutil
import sys
import time as _time
from collections import deque
from typing import Deque, Optional, Tuple

import pandas as pd
from panda3d.core import Texture as P3DTexture
from ursina import *

from handarm import config
from handarm.engine.explore_core import ExploreEnvironment
from handarm.tracking.gestures import (
    compute_explore_features,
    classify_right_explore,
    classify_left_explore_standalone,
    gesture_label,
)
from handarm.tracking.hand_tracker import HandTracker


class NavigationHUD:
    """Stable top overlay for gesture state, point loading, and performance."""

    UPDATE_INTERVAL = 3.0

    def __init__(self, max_points: int) -> None:
        self.max_points = max_points
        self.started_at = _time.monotonic()
        self.last_metrics_update = self.started_at
        self.frame_times: Deque[float] = deque()
        self.cpu_samples: Deque[float] = deque()
        self.ram_samples: Deque[float] = deque()

        Entity(
            parent=camera.ui,
            model="quad",
            scale=(2.0, 0.16),
            position=(0, 0.43),
            color=color.color(0, 0, 0, 0.72),
            z=0.1,
        )
        self.points_text = Text(
            parent=camera.ui,
            text="📊 Loaded Points: 0",
            position=(-0.05, 0.49),
            origin=(0, 0),
            scale=0.8,
            color=color.azure,
            z=0.09,
        )
        self.gesture_text = Text(
            parent=camera.ui,
            text="🖐️ Open Palm: Neutral / Idle",
            position=(-0.96, 0.49),
            origin=(-0.5, 0),
            scale=0.68,
            color=color.yellow,
            z=0.09,
        )
        self.fps_text = Text(
            parent=camera.ui,
            text="FPS: --",
            position=(0.78, 0.455),
            origin=(-0.5, 0),
            scale=0.75,
            color=color.lime,
            z=0.09,
        )
        self.cpu_text = Text(
            parent=camera.ui,
            text="CPU: --%",
            position=(0.78, 0.415),
            origin=(-0.5, 0),
            scale=0.75,
            color=color.orange,
            z=0.09,
        )
        self.ram_text = Text(
            parent=camera.ui,
            text="RAM: --%",
            position=(0.78, 0.375),
            origin=(-0.5, 0),
            scale=0.75,
            color=color.cyan,
            z=0.09,
        )

    @staticmethod
    def _point_count(env: ExploreEnvironment) -> Tuple[int, int]:
        """Return only points represented by currently rendered geometry."""
        loaded_chunks = getattr(env, "loaded_chunks", {})
        loaded = sum(int(getattr(chunk, "vertex_count", 0)) for chunk in loaded_chunks.values())
        overview_colors = getattr(env, "_ov_np_rgb", None)
        overview_count = len(overview_colors) if overview_colors is not None else 0

        if loaded or overview_count:
            loaded += overview_count
            return loaded, loaded

        single_mesh = getattr(env, "single_mesh", None)
        if single_mesh is not None:
            return len(getattr(single_mesh, "vertices", ())), len(getattr(single_mesh, "vertices", ()))

        return loaded, 0

    def update(self, env: ExploreEnvironment, gesture: str, dt: float) -> None:
        now = _time.monotonic()
        self.frame_times.append(now)
        self.cpu_samples.append(psutil.cpu_percent(interval=None))
        self.ram_samples.append(psutil.virtual_memory().percent)
        cutoff = now - self.UPDATE_INTERVAL
        while self.frame_times and self.frame_times[0] < cutoff:
            self.frame_times.popleft()
        while self.cpu_samples and len(self.cpu_samples) > len(self.frame_times):
            self.cpu_samples.popleft()
        while self.ram_samples and len(self.ram_samples) > len(self.frame_times):
            self.ram_samples.popleft()

        loaded, _ = self._point_count(env)
        self.points_text.text = f"📊 Loaded Points: {loaded:,}"
        self.gesture_text.text = gesture

        if (now - self.last_metrics_update >= self.UPDATE_INTERVAL
                and len(self.frame_times) > 1):
            elapsed = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / elapsed if elapsed else 0.0
            cpu = sum(self.cpu_samples) / len(self.cpu_samples)
            ram = sum(self.ram_samples) / len(self.ram_samples)
            self.fps_text.text = f"FPS: {fps:.1f}"
            self.cpu_text.text = f"CPU: {cpu:.1f}%"
            self.ram_text.text = f"RAM: {ram:.1f}%"
            self.last_metrics_update = now


def _navigation_gesture(gesture: str, features: Optional[dict] = None) -> str:
    """Convert classifier output into the user-facing navigation vocabulary."""
    if not features:
        return gesture_label("None")
    landmarks = features.get("landmarks")
    if landmarks:
        pinch = np.hypot(
            landmarks[4].x - landmarks[8].x,
            landmarks[4].y - landmarks[8].y,
        )
        if pinch < 0.05:
            return gesture_label("Pinch")
    return gesture_label(gesture, landmarks)


def main(point_cloud: Optional[pd.DataFrame] = None, target_path: Optional[str] = None) -> None:
    """Main entry point for standalone explore mode."""

    # File loading
    target = target_path or (sys.argv[1] if len(sys.argv) > 1 else "pointcloud_sample.csv")

    if point_cloud is None and not os.path.exists(target):
        print(f"ERROR: Model or chunks not found: {target}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Gesture classifier (explore.py standalone variant)
    # -----------------------------------------------------------------------
    def _classify_standalone(landmarks, hand_type):
        features = compute_explore_features(landmarks)
        if hand_type == 'Right':
            gesture = classify_right_explore(features)
        else:
            gesture = classify_left_explore_standalone(features)
        return {
            'gesture': gesture,
            'navigation': _navigation_gesture(gesture, features),
        }

    # -----------------------------------------------------------------------
    # Frame annotator (standalone: draws landmarks + always shows gesture text)
    # -----------------------------------------------------------------------
    _mp_hands = mp.solutions.hands
    _mp_drawing = mp.solutions.drawing_utils

    def _annotate_standalone(frame_rgb, results, state):
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                _mp_drawing.draw_landmarks(
                    frame_rgb, hand_lms, _mp_hands.HAND_CONNECTIONS,
                )
        cv2.putText(frame_rgb, f"LEFT : {state['Left']['gesture']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame_rgb, f"RIGHT: {state['Right']['gesture']}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # Standalone keeps RGB, resizes to 320x240 (matching original explore.py)
        return cv2.resize(frame_rgb, (320, 240))

    # -----------------------------------------------------------------------
    # Ursina app
    # -----------------------------------------------------------------------
    app = Ursina()
    window.fps_counter.enabled = False
    window.entity_counter.enabled = False
    window.collider_counter.enabled = False
    window.color = color.color(0, 0, 0.08)
    tracker = HandTracker(
        classify_hand=_classify_standalone,
        annotate_frame=_annotate_standalone,
        detection_confidence=0.7,
        camera_index=config.CAMERA_INDEX,
    )

    # Camera feed UI (direct texture, matching original explore.py)
    FEED_W, FEED_H = 320, 240
    _p3d_tex = P3DTexture('camera_feed')
    _p3d_tex.setup2dTexture(FEED_W, FEED_H, P3DTexture.TUnsignedByte,
                            P3DTexture.FRgb)
    _p3d_tex.setMagfilter(P3DTexture.FTLinear)
    _p3d_tex.setMinfilter(P3DTexture.FTLinear)

    Entity(parent=camera.ui, model='quad', scale=(0.445, 0.315),
           position=(-0.67, -0.335), color=color.color(0, 0, 0.35), z=0.01)
    feed_view = Entity(parent=camera.ui, model='quad', scale=(0.44, 0.31),
                       position=(-0.67, -0.335))
    feed_view.model.setTexture(_p3d_tex)
    Text("LIVE", position=(-0.67, -0.168), scale=0.9, color=color.red)

    # Legend
    Text(
        text="<yellow>HAND CONTROLS\n\n"
             "<orange>RIGHT HAND (ALWAYS ACTIVE)\n"
             "<white>🖐️ Open Palm  : Enable View Control\n"
             "<white>✊ Closed Fist : Freeze View\n"
             "<white>🤏 Pinch      : Zoom In/Out\n"
             "<white>👈 Point Left : Look Left\n"
             "<white>👉 Point Right: Look Right\n"
             "<white>^  Move Up   : Look Up\n"
             "<white>v  Move Down : Look Down\n\n"
             "<cyan>LEFT HAND (WALK MODE)\n"
             "<white>✌️ Peace       : Toggle Flight Mode\n"
             "<white>Move Palm Up : Move Forward\n"
             "<white>Move Palm Dn : Move Backward\n"
             "<white>✊ Closed Fist : Stop\n\n"
             "<lime>LEFT HAND (FLIGHT MODE)\n"
             "<white>👍 Thumbs Up  : Position Reset\n"
             "<white>✌️ Peace       : Toggle Walk Mode\n"
             "<white>🖐️ Open Palm  : Fly Up/Down\n"
             "<white>Move Palm Up : Fly Forward\n"
             "<white>Move Palm Dn : Fly Backward\n"
             "<white>✊ Closed Fist : Stop",
        position=window.top_left + Vec2(0.02, -0.02),
        origin=(-0.5, 0.5), scale=0.8, background=True,
    )
    # Point cloud environment (standalone: ::6 downsample, thickness 4, no reset cooldown)
    print(f"Loading Point Cloud Environment from {target}...")
    env = ExploreEnvironment(
        target,
        point_cloud=point_cloud,
        downsample_step=1,
        point_thickness=1.0,
        reset_cooldown_duration=0.0,
    )

    hud = NavigationHUD(max_points=point_cloud.attrs.get("point_count", len(point_cloud))
                        if point_cloud is not None else 0)

    right_status = Text(text="Right: Not Detected", position=(0.3, -0.30),
                        scale=1.1, color=color.orange)
    left_status = Text(text="Left: Not Detected", position=(0.3, -0.35),
                       scale=1.1, color=color.cyan)
    flight_status = Text(text="Mode: Flying" if env.is_flying else "Mode: Grounded",
                         position=(0.3, -0.40), scale=1.1,
                         color=color.cyan if env.is_flying else color.green)
    color_status = Text(text="Color: Natural RGB [C]", position=(0.3, -0.45),
                        scale=1.1, color=color.yellow)
    size_status = Text(text="Point Size: 1px [P]", position=(0.3, -0.50),
                       scale=1.1, color=color.white)
    chunk_status = Text(text="Streamed: --", position=(0.3, -0.55),
                        scale=1.0, color=color.lime)
    landmark_status = Text(text="Jump: [1] Academic [2] Labs [3] Plaza [4] Gate",
                           position=(0.3, -0.60), scale=0.9, color=color.azure)
    rotate_status = Text(text="Rotate: [Arrows] / [J][L] | 360° [T]",
                         position=(0.3, -0.65), scale=0.9, color=color.azure)

    # -----------------------------------------------------------------------
    # Update loop
    # -----------------------------------------------------------------------
    def update() -> None:
        # Camera feed
        frame = tracker.get_frame()
        if frame is not None:
            # Convert RGB (from MediaPipe tracker) to BGR for Panda3D's FRgb texture buffer
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            small = np.flipud(bgr)
            small = np.ascontiguousarray(small, dtype=np.uint8)
            buf = _p3d_tex.modifyRamImage()
            memoryview(buf).cast('B')[:] = small.tobytes()
            _p3d_tex.setRamImage(buf)

        state = tracker.get_state()
        left, right = state['Left'], state['Right']
        active_hand = right if right['visible'] else left
        hud.update(
            env,
            active_hand.get("navigation", "🖐️ Open Palm - Neutral / Idle"),
            time.dt,
        )

        left_status.text = (f"Left: {gesture_label(left['gesture'])}"
                            if left['visible'] else "Left: Not Detected")
        right_status.text = (f"Right: {gesture_label(right['gesture'])}"
                             if right['visible'] else "Right: Not Detected")

        env.update_steering(right)
        env.update_movement_and_terrain(left)

        flight_status.text = ("Mode: Flying" if env.is_flying
                              else "Mode: Grounded")
        flight_status.color = color.red if env.is_flying else color.green

        if env.is_chunked:
            pts_count = sum(c.vertex_count for c in env.loaded_chunks.values())
            # Count chunks at each LOD level
            lod_counts = [0, 0, 0]
            for c in env.loaded_chunks.values():
                if c.lod_level < 3:
                    lod_counts[c.lod_level] += 1
            chunk_status.text = (
                f"Streamed: {len(env.loaded_chunks)} chunks ({pts_count/1e6:.2f}M pts) "
                f"LOD: {lod_counts[0]}hi {lod_counts[1]}med {lod_counts[2]}lo"
            )

    # -----------------------------------------------------------------------
    # Input handler
    # -----------------------------------------------------------------------
    def input(key: str) -> None:
        if key == 'escape' or key == 'q':
            mouse.locked = False
            application.quit()
        elif key == 'r':
            env.reset_player()
            rotate_status.text = "Rotate: [Arrows] / [J][L] | 360° [T]"
            rotate_status.color = color.azure
        elif key == 't':
            is_active = env.toggle_auto_rotate()
            rotate_status.text = "Rotate: 360° Turntable ON [T]" if is_active else "Rotate: [Arrows] / [J][L] | 360° [T]"
            rotate_status.color = color.lime if is_active else color.azure
        elif key == 'c':
            new_mode = env.cycle_color_mode()
            color_status.text = f"Color: {new_mode} [C]"
        elif key == 'p':
            new_size = env.cycle_point_thickness()
            size_status.text = f"Point Size: {new_size} [P]"
        elif key == 'f':
            env.is_flying = not env.is_flying
        elif key in ('1', '2', '3', '4'):
            name = env.jump_to_landmark(int(key) - 1)
            if name:
                landmark_status.text = f"At: {name} [{key}]"

    # Register with Ursina
    _main_mod = sys.modules['__main__']
    _main_mod.update = update
    _main_mod.input = input

    app.run()
    tracker.stop()


if __name__ == "__main__":
    main()
