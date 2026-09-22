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

from panda3d.core import Texture as P3DTexture
from ursina import *

from handarm import config
from handarm.engine.explore_core import ExploreEnvironment
from handarm.engine.ui_common import PerformanceUI
from handarm.tracking.gestures import (
    compute_explore_features,
    classify_right_explore,
    classify_left_explore_standalone,
)
from handarm.tracking.hand_tracker import HandTracker


def main() -> None:
    """Main entry point for standalone explore mode."""

    # File loading
    csv_path = "pointcloud_sample.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"ERROR: Model file not found: {csv_path}")
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
        return {'gesture': gesture}

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
    window.color = color.color(0, 0, 0.08)

    tracker = HandTracker(
        classify_hand=_classify_standalone,
        annotate_frame=_annotate_standalone,
        detection_confidence=0.7,
        camera_index=config.CAMERA_INDEX,
    )

    # Performance UI
    perf_ui = PerformanceUI()

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
             "<white>[Open Palm]  : Enable View Control\n"
             "<white>[Fist]       : Freeze View\n"
             "<white><- Move Left : Look Left\n"
             "<white>-> Move Right: Look Right\n"
             "<white>^  Move Up   : Look Up\n"
             "<white>v  Move Down : Look Down\n\n"
             "<cyan>LEFT HAND (WALK MODE)\n"
             "<white>[Peace]      : Toggle Flight Mode\n"
             "<white>Move Palm Up : Move Forward\n"
             "<white>Move Palm Dn : Move Backward\n"
             "<white>[Fist]       : Stop\n\n"
             "<lime>LEFT HAND (FLIGHT MODE)\n"
             "<white>[Thumbs Up]  : Position Reset\n"
             "<white>[Peace]      : Toggle Walk Mode\n"
             "<white>[Index Up]   : Fly Up\n"
             "<white>[Index Down] : Fly Down\n"
             "<white>Move Palm Up : Fly Forward\n"
             "<white>Move Palm Dn : Fly Backward\n"
             "<white>[Fist]       : Stop",
        position=window.top_left + Vec2(0.02, -0.02),
        origin=(-0.5, 0.5), scale=0.8, background=True,
    )
    right_status = Text(text="Right: Not Detected", position=(0.3, -0.30),
                        scale=1.1, color=color.orange)
    left_status = Text(text="Left: Not Detected", position=(0.3, -0.35),
                       scale=1.1, color=color.cyan)
    flight_status = Text(text="Mode: Grounded", position=(0.3, -0.40),
                         scale=1.1, color=color.green)
    color_status = Text(text="Color: Natural RGB [C]", position=(0.3, -0.45),
                        scale=1.1, color=color.yellow)
    size_status = Text(text="Point Size: 1px [P]", position=(0.3, -0.50),
                       scale=1.1, color=color.white)

    # Point cloud environment (standalone: ::6 downsample, thickness 4, no reset cooldown)
    print("Loading Point Cloud Environment...")
    env = ExploreEnvironment(
        csv_path,
        downsample_step=1,
        point_thickness=1.0,
        reset_cooldown_duration=0.0,
    )

    # -----------------------------------------------------------------------
    # Update loop
    # -----------------------------------------------------------------------
    def update() -> None:
        fps = int(1 / time.dt) if time.dt > 0 else 0
        perf_ui.update(fps, psutil.cpu_percent(interval=None),
                       psutil.virtual_memory().percent)

        # Camera feed
        frame = tracker.get_frame()
        if frame is not None:
            small = np.flipud(frame)
            small = np.ascontiguousarray(small, dtype=np.uint8)
            buf = _p3d_tex.modifyRamImage()
            memoryview(buf).cast('B')[:] = small.tobytes()
            _p3d_tex.setRamImage(buf)

        state = tracker.get_state()
        left, right = state['Left'], state['Right']

        left_status.text = (f"Left: {left['gesture']}"
                            if left['visible'] else "Left: Not Detected")
        right_status.text = (f"Right: {right['gesture']}"
                             if right['visible'] else "Right: Not Detected")

        env.update_steering(right)
        env.update_movement_and_terrain(left)

        flight_status.text = ("Mode: Flying" if env.is_flying
                              else "Mode: Grounded")
        flight_status.color = color.red if env.is_flying else color.green

    # -----------------------------------------------------------------------
    # Input handler
    # -----------------------------------------------------------------------
    def input(key: str) -> None:
        if key == 'escape' or key == 'q':
            mouse.locked = False
            application.quit()
        elif key == 'r':
            env.reset_player()
        elif key == 'c':
            new_mode = env.cycle_color_mode()
            color_status.text = f"Color: {new_mode} [C]"
        elif key == 'p':
            new_size = env.cycle_point_thickness()
            size_status.text = f"Point Size: {new_size} [P]"
        elif key == 'f':
            env.is_flying = not env.is_flying

    # Register with Ursina
    _main_mod = sys.modules['__main__']
    _main_mod.update = update
    _main_mod.input = input

    app.run()
    tracker.stop()


if __name__ == "__main__":
    main()
