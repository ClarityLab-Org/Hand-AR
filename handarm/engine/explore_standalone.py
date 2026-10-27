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
from direct.task import Task
from panda3d.core import Texture as P3DTexture, WindowProperties
from ursina import *

from handarm import config
from handarm.engine.explore_core import ExploreEnvironment
from handarm.engine.hud_components import (
    _create_gesture_guide_texture,
    _load_ui_fonts,
    _render_telemetry_hud,
)
from handarm.engine.ui_common import WINDOW_SIZE, safe_set_text
from handarm.tracking.gestures import (
    compute_explore_features,
    compute_thumbs_up,
    classify_both_hands,
    classify_right_explore,
    classify_left_explore_standalone,
    gesture_label,
)
from handarm.tracking.hand_tracker import HandTracker



class NavigationHUD:
    """Stable top overlay for gesture state, point loading, and performance telemetry."""

    UPDATE_INTERVAL = 0.4

    def __init__(self, max_points: int, system_monitor=None) -> None:
        self.max_points = max_points
        self.system_monitor = system_monitor
        self.started_at = _time.monotonic()
        self.last_metrics_update = self.started_at
        self.frame_times: Deque[float] = deque()
        self.fps_history: Deque[float] = deque([60.0] * 6, maxlen=24)
        self.cpu_history: Deque[float] = deque([15.0] * 6, maxlen=24)
        self.ram_history: Deque[float] = deque([50.0] * 6, maxlen=24)

        self.font_label, self.font_val = _load_ui_fonts()

        # Clean, high-contrast top bar stats
        self.gesture_text = Text(
            parent=camera.ui,
            text="GESTURE: Neutral / Hover",
            position=(-0.82, 0.47),
            origin=(-0.5, 0),
            scale=0.9,
            color=color.yellow,
            z=0.09,
        )
        self.points_text = Text(
            parent=camera.ui,
            text="Loaded: 0 / Total: 0",
            position=(0.0, 0.47),
            origin=(0, 0),
            scale=0.95,
            color=color.white,
            z=0.09,
        )
        self._points_text_value = "Loaded: 0 / Total: 0"
        self._gesture_text_value = "GESTURE: Neutral / Hover"

        # High-performance telemetry panel with live graphs for FPS, CPU, RAM
        # Positioned cleanly away from window control buttons
        initial_img = _render_telemetry_hud(
            fps=60.0,
            cpu=15.0,
            ram=50.0,
            fps_history=list(self.fps_history),
            cpu_history=list(self.cpu_history),
            ram_history=list(self.ram_history),
            font_label=self.font_label,
            font_val=self.font_val,
        )
        self.perf_tex = Texture(initial_img)
        self.perf_tex.filtering = True
        self.perf_panel = Entity(
            parent=camera.ui,
            model="quad",
            texture=self.perf_tex,
            position=(0.52, 0.455),
            scale=(0.50, 0.07),
            z=0.08,
        )

    @staticmethod
    def _point_count(env: ExploreEnvironment) -> Tuple[int, int]:
        """Return points represented by current geometry and the dataset total."""
        loaded_chunks = getattr(env, "loaded_chunks", {})
        loaded = sum(int(getattr(chunk, "vertex_count", 0)) for chunk in loaded_chunks.values())
        overview_colors = getattr(env, "_ov_np_rgb", None)
        overview_count = len(overview_colors) if overview_colors is not None else 0
        total = int(getattr(env, "total_points", 0))

        if loaded or overview_count:
            loaded += overview_count
            return loaded, total or loaded

        single_mesh = getattr(env, "single_mesh", None)
        if single_mesh is not None:
            loaded = len(getattr(single_mesh, "vertices", ()))
            return loaded, total or loaded

        return loaded, total

    def update(self, env: ExploreEnvironment, gesture: str, dt: float) -> None:
        now = _time.monotonic()
        self.frame_times.append(now)
        cutoff = now - 1.0
        while self.frame_times and self.frame_times[0] < cutoff:
            self.frame_times.popleft()

        loaded, total = self._point_count(env)
        points_text = f"Loaded: {loaded:,} / Total: {total:,}"
        if points_text != self._points_text_value:
            safe_set_text(self.points_text, points_text)
            self._points_text_value = points_text
        clean_gesture = gesture.replace("🖐️", "").replace("🖐", "").replace("👌", "").replace("✌️", "").replace("✌", "").replace("👍", "").strip()
        gesture_text = f"GESTURE: {clean_gesture or gesture}"
        if gesture_text != self._gesture_text_value:
            safe_set_text(self.gesture_text, gesture_text)
            self._gesture_text_value = gesture_text

        if now - self.last_metrics_update >= self.UPDATE_INTERVAL:
            elapsed = self.frame_times[-1] - self.frame_times[0] if len(self.frame_times) > 1 else 0.0
            fps = (len(self.frame_times) - 1) / elapsed if elapsed > 0 else (1.0 / dt if dt > 0 else 60.0)
            if self.system_monitor is None:
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
            else:
                _, cpu, ram = self.system_monitor.sample(fps)

            self.fps_history.append(fps)
            self.cpu_history.append(cpu)
            self.ram_history.append(ram)

            new_img = _render_telemetry_hud(
                fps=fps,
                cpu=cpu,
                ram=ram,
                fps_history=list(self.fps_history),
                cpu_history=list(self.cpu_history),
                ram_history=list(self.ram_history),
                font_label=self.font_label,
                font_val=self.font_val,
            )
            try:
                if getattr(self.perf_tex, 'parent', None) is not None:
                    self.perf_tex._cached_image = new_img
                    self.perf_tex.apply()
            except Exception:
                pass
            self.last_metrics_update = now


def _navigation_gesture(gesture: str, features: Optional[dict] = None) -> str:
    """Convert classifier output into the user-facing navigation vocabulary."""
    return gesture_label(gesture if features else "None")


def main(point_cloud: Optional[pd.DataFrame] = None,
         target_path: Optional[str] = None,
         model_path: Optional[str] = None,
         model_scale: float = 1.0,
         model_pos: Optional[Tuple[float, float, float]] = None,
         downsample_step: int = 1) -> None:
    """Main entry point for standalone explore mode."""

    # File loading
    target = target_path or (sys.argv[1] if len(sys.argv) > 1 else "pointcloud_sample.csv")

    if point_cloud is None and not os.path.exists(target):
        print(f"ERROR: Model or chunks not found: {target}")
        sys.exit(1)

    _pinch_distances = {'Left': None, 'Right': None}
    emoji_font = _load_emoji_font(36)

    # -----------------------------------------------------------------------
    # Gesture classifier (explore.py standalone variant)
    # -----------------------------------------------------------------------
    def _classify_standalone(landmarks, hand_type):
        features = compute_explore_features(landmarks)
        if hand_type == 'Right':
            previous_pinch = _pinch_distances[hand_type]
            gesture = classify_right_explore(features, previous_pinch)
            _pinch_distances[hand_type] = features['pinch_distance']
        else:
            gesture = classify_left_explore_standalone(features)
        return {
            'gesture': gesture,
            'navigation': _navigation_gesture(gesture, features),
            'open_palm': features['open_palm'],
            'thumbs_up': compute_thumbs_up(features, margin=0.02),
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
        # Clean camera preview with landmarks, resized to 320x240
        return cv2.resize(frame_rgb, (320, 240))

    # -----------------------------------------------------------------------
    # Ursina app
    # -----------------------------------------------------------------------
    windowed_size = WINDOW_SIZE
    app = Ursina(size=windowed_size, development_mode=False)
    fullscreen = False

    def request_window_state(is_fullscreen: bool) -> None:
        window.fullscreen = is_fullscreen
        props = WindowProperties()
        props.set_fullscreen(is_fullscreen)
        if is_fullscreen:
            props.set_origin(window.main_monitor.x, window.main_monitor.y)
        else:
            props.set_size(*windowed_size)
            x = int(window.main_monitor.x + (window.main_monitor.width - windowed_size[0]) / 2)
            y = int(window.main_monitor.y + (window.main_monitor.height - windowed_size[1]) / 2)
            props.set_origin(x, y)
        application.base.win.request_properties(props)
        if not is_fullscreen:
            window.setOrigin(x, y)
            application.base.win.request_properties(window)

    request_window_state(fullscreen)
    base.graphicsEngine.renderFrame()
    base.graphicsEngine.renderFrame()

    def _report_window_size(task):
        print(
            f"[Window] requested {windowed_size[0]}x{windowed_size[1]}; "
            f"actual {base.win.get_x_size()}x{base.win.get_y_size()} "
            f"at {tuple(window.getOrigin())}"
        )
        return Task.done

    base.taskMgr.doMethodLater(0.5, _report_window_size, "report-window-size")
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

    # Use clean, bold modern sans-serif font for all Ursina UI text (not thin serif Symbola)
    for font_cand in (
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.exists(font_cand):
            Text.default_font = font_cand
            break

    Entity(parent=camera.ui, model='quad', scale=(0.445, 0.315),
           position=(-0.65, -0.32), color=color.color(0, 0, 0.35), z=0.01)
    feed_view = Entity(parent=camera.ui, model='quad', scale=(0.44, 0.31),
                       position=(-0.65, -0.32))
    feed_view.model.setTexture(_p3d_tex)
    Text("LIVE", position=(-0.65, -0.14), scale=0.85, color=color.red)

    # Crisp, large-format gesture HUD card with prominent emojis and high-contrast typography
    gesture_guide = Entity(
        parent=camera.ui,
        model='quad',
        texture=_create_gesture_guide_texture(),
        scale=(0.45, 0.43),
        position=(-0.64, 0.135),
        z=0.05,
    )

    # Point cloud environment (thickness and speeds loaded from config/.env)
    print(f"Loading Point Cloud Environment from {target}...")
    print(f"  [Config] Point Thickness / Size: {config.POINT_THICKNESS}px")
    print(f"  [Config] Speeds: Flight {config.FLIGHT_SPEED} m/s, Walk {config.WALK_SPEED} m/s, Turn {config.TURN_SPEED}°/s")
    env = ExploreEnvironment(
        target,
        point_cloud=point_cloud,
        downsample_step=downsample_step,
        point_thickness=config.POINT_THICKNESS,
        reset_cooldown_duration=0.0,
    )
    env.render_scene(
        point_cloud=point_cloud,
        model_path=model_path,
        model_scale=model_scale,
        model_pos=model_pos or (0.0, 0.0, 0.0),
        downsample_step=downsample_step,
    )

    hud = NavigationHUD(max_points=point_cloud.attrs.get("point_count", len(point_cloud))
                        if point_cloud is not None else 0)

    # High-contrast bottom status bar with dark backing pill for maximum legibility
    Entity(parent=camera.ui, model='quad', scale=(1.25, 0.05),
           position=(0.14, -0.47), color=color.color(0, 0, 0, 0.75), z=0.1)

    landmark_status = Text(text="Jump: [1] Academic [2] Labs [3] Plaza [4] Gate",
                           position=(-0.38, -0.47), scale=0.8, color=color.white, z=0.09)
    rotate_status = Text(text="Rotate: [Arrows] / [J][L] | 360° [T]",
                         position=(0.24, -0.47), scale=0.8, color=color.yellow, z=0.09)

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
        both_gesture = "None"
        if left['visible'] and right['visible']:
            both_gesture = classify_both_hands(left, right)

        active_hand = right if right['visible'] else left
        active_gesture = both_gesture if both_gesture != "None" else active_hand.get("gesture", "None")
        hud.update(env, gesture_label(active_gesture), time.dt)

        # A valid two-hand gesture has exclusive control for this frame. Do
        # not let the right-hand palm also trigger rotation or zoom.
        steering_state = right if both_gesture == "None" else {"visible": False}
        env.update_steering(steering_state)
        movement_state = left
        if both_gesture == "Both hand Move Up":
            movement_state = dict(left, gesture="Move Up")
        elif both_gesture == "Both hand Move Down":
            movement_state = dict(left, gesture="Move Down")
        env.update_movement_and_terrain(movement_state)

    # -----------------------------------------------------------------------
    # Input handler
    # -----------------------------------------------------------------------
    def input(key: str) -> None:
        nonlocal fullscreen
        if key == 'escape' or key == 'q':
            mouse.locked = False
            application.quit()
        elif key == 'f11':
            fullscreen = not fullscreen
            request_window_state(fullscreen)
        elif key == 'h':
            gesture_guide.enabled = not gesture_guide.enabled
        elif key == 'r':
            env.reset_player()
            safe_set_text(rotate_status, "Rotate: [Arrows] / [J][L] | 360° [T]")
            rotate_status.color = color.yellow
        elif key == 't':
            is_active = env.toggle_auto_rotate()
            safe_set_text(
                rotate_status,
                "Rotate: 360° Turntable ON [T]" if is_active
                else "Rotate: [Arrows] / [J][L] | 360° [T]",
            )
            rotate_status.color = color.lime if is_active else color.yellow
        elif key == 'c':
            new_mode = env.cycle_color_mode()
            print(f"Color mode: {new_mode}")
        elif key == 'p':
            new_size = env.cycle_point_thickness()
            print(f"[explore] Point thickness: {new_size}")
        elif key == 'f':
            env.is_flying = not env.is_flying
        elif key in ('1', '2', '3', '4'):
            name = env.jump_to_landmark(int(key) - 1)
            if name:
                safe_set_text(landmark_status, f"At: {name} [{key}]")

    # Register with Ursina
    _main_mod = sys.modules['__main__']
    _main_mod.update = update
    _main_mod.input = input

    app.run()
    tracker.stop()


if __name__ == "__main__":
    main()
