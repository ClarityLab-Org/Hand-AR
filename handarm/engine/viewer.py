"""
Interactive 3D Model & Point Cloud Viewer for Hand-ArM2 Project.

Supports Inspect Mode (object rotation/zoom/translation via gestures)
and Explore Mode (first-person point cloud navigation).

Usage:
    python trial2.py [model_path] [mode]
    mode: 'inspect' (default) or 'explore'
"""

import cv2
import datetime
import math
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import psutil
import sys
import time as _time

from panda3d.core import Texture as P3DTexture, Filename
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController

from handarm import config
from handarm.geometry.alignment import auto_align_up_axis
from handarm.engine.ui_common import (
    CameraFeedUI, PerformanceUI, InitOverlay, draw_skeletons,
)
from handarm.tracking.gestures import (
    is_open_palm_relaxed, is_peace, is_pinch, pinch_distance,
    compute_explore_features, classify_right_explore,
    classify_left_explore_trial2,
)
from handarm.tracking.hand_tracker import HandTracker
from handarm.engine.explore_core import ExploreEnvironment


def main() -> None:
    """Main entry point for the 3D viewer."""

    # -----------------------------------------------------------------------
    # Argument parsing
    # -----------------------------------------------------------------------
    selected_model = sys.argv[1] if len(sys.argv) > 1 else None
    viewer_mode = sys.argv[2] if len(sys.argv) > 2 else "inspect"

    if selected_model:
        model_path = selected_model
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            sys.exit(1)
    else:
        model_path = os.path.join(config.MODELS_DIR, "Old_Rusty_Car.glb")
        if not os.path.exists(model_path):
            print(f"ERROR: Default model not found: {model_path}")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Ursina app
    # -----------------------------------------------------------------------
    app = Ursina()
    window.color = color.color(0, 0, 0.08)
    os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Mutable state container (avoids globals — captured by closure)
    # -----------------------------------------------------------------------
    class S:
        pass
    s = S()

    # -----------------------------------------------------------------------
    # Mode-specific setup
    # -----------------------------------------------------------------------
    car = None          # 3D model entity (inspect mode)
    player = None       # FirstPersonController (explore mode)
    env = None          # ExploreEnvironment (explore mode)
    tracker = None      # HandTracker (explore mode)
    cap = None          # cv2.VideoCapture (inspect mode)
    hands_det = None    # MediaPipe Hands (inspect mode)
    gesture_text = None
    rot_slider = trans_slider = zoom_slider = None

    if viewer_mode == "inspect":
        print("--- SETUP: INSPECT MODE ---")
        car = Entity()

        if model_path.lower().endswith(".csv"):
            df = pd.read_csv(model_path)
            df = auto_align_up_axis(df)
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
            car.model = mesh
            car.set_render_mode_thickness(1)
            car.set_render_mode_perspective(False)
        else:
            panda_path = Filename.from_os_specific(model_path)
            loaded = base.loader.load_model(panda_path)
            if loaded is None:
                sys.exit(1)
            loaded.reparent_to(car)

        car.scale = 1
        try:
            min_b, max_b = car.model.get_tight_bounds()
        except Exception:
            min_b, max_b = car.get_tight_bounds()
        center = (min_b + max_b) / 2
        car.origin = center
        car.position = Vec3(0, 0, 0)
        size = max(max_b.x - min_b.x, max_b.y - min_b.y, max_b.z - min_b.z)
        if size > 0:
            car.scale = 3 / size

        camera.position = Vec3(0, 0, -12)
        camera.look_at(car.position)

        s.model_locked = False
        s.camera_locked = False
        s.view_mode = "free"

        def toggle_model_lock() -> None:
            s.model_locked = not s.model_locked
            model_btn.text = f"Model Lock: {'ON' if s.model_locked else 'OFF'}"
            model_btn.color = (color.color(0, 0, 0.30) if s.model_locked
                               else color.color(0, 0, 0.20))

        def set_view(mode: str) -> None:
            s.view_mode = mode
            s.camera_locked = True
            if mode == "front":
                camera.position = Vec3(0, 0, -12)
            elif mode == "side":
                camera.position = Vec3(12, 0, 0)
            elif mode == "top":
                camera.position = Vec3(0, 12, 0)
            elif mode == "iso":
                camera.position = Vec3(8, 6, -8)
            camera.look_at(car.position)

        BTN = dict(color=color.color(0, 0, 0.20),
                   highlight_color=color.color(0, 0, 0.35),
                   text_color=color.white)
        model_btn = Button("Model Lock: OFF", scale=(0.22, 0.055),
                           position=(-0.67, 0.42), **BTN,
                           on_click=toggle_model_lock)
        Text("View Presets", parent=camera.ui, position=(-0.67, 0.275),
             origin=(0, 0), scale=0.8, color=color.gray)
        vbtn = dict(scale=(0.10, 0.050), **BTN)
        Button("Front", position=(-0.725, 0.22),
               on_click=lambda: set_view("front"), **vbtn)
        Button("Side", position=(-0.615, 0.22),
               on_click=lambda: set_view("side"), **vbtn)
        Button("Top", position=(-0.725, 0.16),
               on_click=lambda: set_view("top"), **vbtn)
        Button("ISO", position=(-0.615, 0.16),
               on_click=lambda: set_view("iso"), **vbtn)

        s.rot_sens, s.trans_sens, s.zoom_sens = 2, 15, 40
        Text("Sensitivity", parent=camera.ui, position=(0.55, 0.05),
             scale=0.9, color=color.white)
        Text("Rotation", parent=camera.ui, position=(0.55, -0.02), scale=0.7)
        rot_slider = Slider(min=0.5, max=5, default=s.rot_sens, step=0.1,
                            position=(0.55, -0.06), scale=0.3)
        Text("Translation", parent=camera.ui, position=(0.55, -0.14), scale=0.7)
        trans_slider = Slider(min=5, max=30, default=s.trans_sens, step=1,
                              position=(0.55, -0.18), scale=0.3)
        Text("Zoom", parent=camera.ui, position=(0.55, -0.26), scale=0.7)
        zoom_slider = Slider(min=10, max=80, default=s.zoom_sens, step=1,
                             position=(0.55, -0.30), scale=0.3)

        Text("RIGHT HAND\n  Index finger  ->  Rotate\n  Pinch closer  ->  Zoom in\n"
             "  Pinch apart   ->  Zoom out\n  Peace sign    ->  Screenshot\n\n"
             "LEFT HAND\n  Open palm     ->  Pause\n  Wrist move    ->  Translate",
             parent=camera.ui, position=(-0.785, 0.095), scale=0.72,
             color=color.color(0, 0, 0.75))
        gesture_text = Text("Gesture: None", parent=camera.ui,
                            position=(0, 0.46), origin=(0, 0),
                            scale=1.1, color=color.white)

        # MediaPipe + camera
        mp_hands = mp.solutions.hands
        hands_det = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        cap = cv2.VideoCapture(config.CAMERA_INDEX)

        # Motion state
        s.smooth_rx = s.smooth_ry = 0.0
        s.smooth_tx = s.smooth_ty = 0.0
        s.smooth_zoom = 0.0
        s.last_rx = s.last_ry = None
        s.last_lx = s.last_ly = None
        s.last_zoom = None
        s.PAUSE_FRAMES = s.UNPAUSE_FRAMES = 0
        s.paused = False
        s.screenshot_cooldown = s.screenshot_timer = 0

    elif viewer_mode == "explore":
        print("--- SETUP: EXPLORE MODE ---")
        env = ExploreEnvironment(
            model_path,
            downsample_step=1,
            point_thickness=1.5,
            reset_cooldown_duration=2.0,
        )
        player = env.player

        # Gesture classifier for trial2.py explore variant
        def _classify_explore(landmarks, hand_type):
            features = compute_explore_features(landmarks)
            if hand_type == 'Right':
                gesture = classify_right_explore(features)
            else:
                gesture = classify_left_explore_trial2(features)
            return {'gesture': gesture}

        # Frame annotator — uses init overlay's registered_hands
        def _annotate_explore(frame_rgb, results, state):
            draw_skeletons(frame_rgb, results, init_overlay.registered_hands)
            if init_overlay.state == "locked":
                cv2.putText(frame_rgb, f"LEFT : {state['Left']['gesture']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"RIGHT: {state['Right']['gesture']}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # UI text elements for explore mode (Clean ASCII icons for Panda3D font)
        legend = Text(
            text="<yellow>HAND CONTROLS\n\n<orange>RIGHT HAND (ALWAYS ACTIVE)\n"
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
        chunk_status = Text(text="Streamed: --", position=(0.3, -0.55),
                            scale=1.0, color=color.lime)
        landmark_status = Text(text="Jump: [1] Academic [2] Labs [3] Plaza [4] Gate",
                               position=(0.3, -0.60), scale=0.9, color=color.azure)

    # -----------------------------------------------------------------------
    # Shared UI
    # -----------------------------------------------------------------------
    perf_ui = PerformanceUI()
    cam_feed = CameraFeedUI(config.FEED_WIDTH, config.FEED_HEIGHT)
    init_overlay = InitOverlay()

    screenshot_text = Text("Screenshot Saved!", parent=camera.ui,
                           position=(0, 0.35), origin=(0, 0),
                           scale=1.2, color=color.cyan, enabled=False)

    # Create tracker AFTER init_overlay exists (closure captures it)
    if viewer_mode == "explore":
        tracker = HandTracker(
            classify_hand=_classify_explore,
            annotate_frame=_annotate_explore,
            detection_confidence=0.7,
            camera_index=config.CAMERA_INDEX,
        )

    # -----------------------------------------------------------------------
    # Motion constants (inspect mode)
    # -----------------------------------------------------------------------
    ALPHA_ROT = 0.12
    ALPHA_TRANS = 0.10
    ALPHA_ZOOM = 0.10
    ROT_DECAY = 0.82
    TRANS_DECAY = 0.80
    ZOOM_DECAY = 0.78
    MAX_ROT_DELTA = 60
    MAX_TRANS_DELTA = 0.12
    MAX_ZOOM_DELTA = 0.06
    PAUSE_ENGAGE_FRAMES = 8
    PAUSE_RELEASE_FRAMES = 6

    # -----------------------------------------------------------------------
    # Helper: full motion reset
    # -----------------------------------------------------------------------
    def _full_motion_reset() -> None:
        s.last_rx = s.last_ry = s.last_lx = s.last_ly = s.last_zoom = None
        s.smooth_rx = s.smooth_ry = s.smooth_tx = s.smooth_ty = s.smooth_zoom = 0.0
        s.paused = False
        s.PAUSE_FRAMES = 0
        s.UNPAUSE_FRAMES = 0

    # -----------------------------------------------------------------------
    # Main update loop
    # -----------------------------------------------------------------------
    def update() -> None:
        fps = int(1 / time.dt) if time.dt > 0 else 0
        perf_ui.update(fps, psutil.cpu_percent(interval=None),
                       psutil.virtual_memory().percent)

        if viewer_mode == "inspect":
            s.rot_sens = rot_slider.value
            s.trans_sens = trans_slider.value
            s.zoom_sens = zoom_slider.value

            ok, frame = cap.read()
            if not ok:
                return
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands_det.process(rgb)
            display_frame = frame.copy()

            # Collect visible hands this frame
            seen_this_frame = {}
            if res.multi_hand_landmarks and res.multi_handedness:
                for i, h in enumerate(res.multi_handedness):
                    label = h.classification[0].label
                    seen_this_frame[label] = res.multi_hand_landmarks[i]

            h_px, w_px, _ = frame.shape

            # Init overlay processing
            if not init_overlay.process(seen_this_frame):
                if init_overlay.state == "waiting" and not seen_this_frame:
                    pass
                if viewer_mode == "inspect":
                    _full_motion_reset()
                draw_skeletons(display_frame, res, init_overlay.registered_hands)
                cam_feed.upload_frame(display_frame)
                gesture_text.text = f"Init: {init_overlay.state.upper()}"
                gesture_text.color = color.orange
                return

            # Locked — process gestures
            left_lms = (seen_this_frame.get("Left")
                        if "Left" in init_overlay.registered_hands else None)
            right_lms = (seen_this_frame.get("Right")
                         if "Right" in init_overlay.registered_hands else None)

            # Check for hand loss with grace period
            if not any(lbl in seen_this_frame for lbl in init_overlay.registered_hands):
                init_overlay.lost_frames += 1
                if init_overlay.lost_frames >= InitOverlay.LOST_GRACE_FRAMES:
                    init_overlay.reset(motion_reset_fn=_full_motion_reset)
                    cam_feed.upload_frame(display_frame)
                    gesture_text.text = "Gesture: None"
                    gesture_text.color = color.white
                    return
            else:
                init_overlay.lost_frames = 0

            if init_overlay.lost_frames > 0:
                gesture_text.text = (
                    f"Hand lost! Re-init in "
                    f"{InitOverlay.LOST_GRACE_FRAMES - init_overlay.lost_frames}..."
                )
                gesture_text.color = color.red

            # Pause logic
            left_is_open = (left_lms is not None and
                            is_open_palm_relaxed(left_lms.landmark))
            if not s.paused:
                if left_is_open:
                    s.PAUSE_FRAMES += 1
                    s.UNPAUSE_FRAMES = 0
                else:
                    s.PAUSE_FRAMES = 0
                if s.PAUSE_FRAMES >= PAUSE_ENGAGE_FRAMES:
                    s.paused = True
                    s.UNPAUSE_FRAMES = 0
                    s.last_rx = s.last_ry = s.last_lx = s.last_ly = s.last_zoom = None
                    s.smooth_rx = s.smooth_ry = 0.0
                    s.smooth_tx = s.smooth_ty = 0.0
                    s.smooth_zoom = 0.0
            else:
                if not left_is_open:
                    s.UNPAUSE_FRAMES += 1
                    s.PAUSE_FRAMES = 0
                else:
                    s.UNPAUSE_FRAMES = 0
                if s.UNPAUSE_FRAMES >= PAUSE_RELEASE_FRAMES:
                    s.paused = False
                    s.PAUSE_FRAMES = s.UNPAUSE_FRAMES = 0
                    s.last_rx = s.last_ry = s.last_lx = s.last_ly = s.last_zoom = None
                    s.smooth_rx = s.smooth_ry = 0.0
                    s.smooth_tx = s.smooth_ty = 0.0
                    s.smooth_zoom = 0.0

            current_gest = "None"
            if s.paused:
                current_gest = "Paused"
                if (right_lms and is_peace(right_lms.landmark) and
                        s.screenshot_cooldown == 0):
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    app.win.saveScreenshot(Filename.fromOsSpecific(
                        os.path.abspath(
                            os.path.join(config.SCREENSHOT_DIR,
                                         f"shot_{ts}.png")
                        )
                    ))
                    screenshot_text.enabled = True
                    s.screenshot_timer = 2.0
                    s.screenshot_cooldown = 25
            else:
                # Rotation
                if (right_lms and not s.model_locked and
                        not is_pinch(right_lms.landmark)):
                    current_gest = "Rotating"
                    ix = int(right_lms.landmark[8].x * w_px)
                    iy = int(right_lms.landmark[8].y * h_px)
                    if s.last_rx is not None:
                        dx = max(-MAX_ROT_DELTA,
                                 min(MAX_ROT_DELTA, ix - s.last_rx))
                        dy = max(-MAX_ROT_DELTA,
                                 min(MAX_ROT_DELTA, iy - s.last_ry))
                        s.smooth_rx = (s.smooth_rx * (1 - ALPHA_ROT) +
                                       (dy * s.rot_sens) * ALPHA_ROT)
                        s.smooth_ry = (s.smooth_ry * (1 - ALPHA_ROT) +
                                       (-dx * s.rot_sens) * ALPHA_ROT)
                        car.rotation_x += s.smooth_rx
                        car.rotation_y += s.smooth_ry
                    else:
                        s.smooth_rx = s.smooth_ry = 0.0
                    s.last_rx, s.last_ry = ix, iy
                else:
                    s.smooth_rx *= ROT_DECAY
                    s.smooth_ry *= ROT_DECAY
                    s.last_rx = s.last_ry = None

                # Zoom
                if right_lms and not s.camera_locked:
                    current_gest += " + Zoom" if current_gest != "None" else "Zooming"
                    pd_dist = pinch_distance(right_lms.landmark)
                    if s.last_zoom is not None:
                        delta = pd_dist - s.last_zoom
                        if abs(delta) < 0.003:
                            delta = 0.0
                        delta = max(-MAX_ZOOM_DELTA,
                                    min(MAX_ZOOM_DELTA, delta))
                        s.smooth_zoom = (s.smooth_zoom * (1 - ALPHA_ZOOM) +
                                         (delta * s.zoom_sens * 2) * ALPHA_ZOOM)
                        camera.z = clamp(camera.z - s.smooth_zoom, -35, -3)
                    else:
                        s.smooth_zoom = 0.0
                    s.last_zoom = pd_dist
                else:
                    s.smooth_zoom *= ZOOM_DECAY
                    s.last_zoom = None

                # Translation
                if left_lms and not s.model_locked:
                    current_gest = "Moving"
                    lx = int(left_lms.landmark[0].x * w_px)
                    ly = int(left_lms.landmark[0].y * h_px)
                    if s.last_lx is not None:
                        dx = (lx - s.last_lx) / w_px
                        dy = -(ly - s.last_ly) / h_px
                        dx = max(-MAX_TRANS_DELTA, min(MAX_TRANS_DELTA, dx))
                        dy = max(-MAX_TRANS_DELTA, min(MAX_TRANS_DELTA, dy))
                        s.smooth_tx = (s.smooth_tx * (1 - ALPHA_TRANS) +
                                       dx * s.trans_sens * ALPHA_TRANS)
                        s.smooth_ty = (s.smooth_ty * (1 - ALPHA_TRANS) +
                                       dy * s.trans_sens * ALPHA_TRANS)
                        car.position += Vec3(s.smooth_tx, s.smooth_ty, 0)
                    else:
                        s.smooth_tx = s.smooth_ty = 0.0
                    s.last_lx, s.last_ly = lx, ly
                else:
                    s.smooth_tx *= TRANS_DECAY
                    s.smooth_ty *= TRANS_DECAY
                    s.last_lx = s.last_ly = None

            if init_overlay.lost_frames == 0:
                gesture_text.text = f"Gesture: {current_gest}"
                gesture_text.color = color.red if s.paused else color.white
            draw_skeletons(display_frame, res, init_overlay.registered_hands)
            if s.paused:
                cv2.rectangle(display_frame, (0, h_px // 2 - 20),
                              (w_px, h_px // 2 + 20), (18, 18, 28), -1)
                cv2.putText(display_frame, "-- PAUSED --",
                            (w_px // 2 - 95, h_px // 2 + 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 210, 255), 2)
            cam_feed.upload_frame(display_frame)

            if s.screenshot_cooldown > 0:
                s.screenshot_cooldown -= 1
            if s.screenshot_timer > 0:
                s.screenshot_timer -= time.dt
            if s.screenshot_timer <= 0 and screenshot_text.enabled:
                screenshot_text.enabled = False

        elif viewer_mode == "explore":
            frame = tracker.get_frame()
            if frame is not None:
                cam_feed.upload_frame(frame)

            state = tracker.get_state()
            left, right = state['Left'], state['Right']

            # Init overlay
            seen_this_frame = {}
            if left['visible']:
                seen_this_frame['Left'] = True
            if right['visible']:
                seen_this_frame['Right'] = True

            if not init_overlay.process(seen_this_frame):
                return

            # Status text updates
            left_status.text = (f"Left: {left['gesture']}"
                                if left['visible'] else "Left: Not Detected")
            right_status.text = (f"Right: {right['gesture']}"
                                 if right['visible'] else "Right: Not Detected")

            # Steering + movement
            env.update_steering(right)
            env.update_movement_and_terrain(left)

            # Flight status display
            flight_status.text = ("Mode: Flying" if env.is_flying
                                  else "Mode: Grounded")
            flight_status.color = color.red if env.is_flying else color.green

            if env.is_chunked:
                pts_count = sum(len(c.vertices) for c in env.loaded_chunks.values())
                chunk_status.text = f"Streamed: {len(env.loaded_chunks)} chunks ({pts_count/1e6:.1f}M pts)"

    # -----------------------------------------------------------------------
    # Input handler
    # -----------------------------------------------------------------------
    def input(key: str) -> None:
        if key == 'escape' or key == 'q':
            mouse.locked = False
            application.quit()
        elif key == 'r':
            if viewer_mode == "explore" and env:
                env.reset_player()
        elif key == 'c':
            if viewer_mode == "explore" and env:
                new_mode = env.cycle_color_mode()
                color_status.text = f"Color: {new_mode} [C]"
        elif key == 'p':
            if viewer_mode == "explore" and env:
                new_size = env.cycle_point_thickness()
                size_status.text = f"Point Size: {new_size} [P]"
        elif key == 'f':
            if viewer_mode == "explore" and env:
                env.is_flying = not env.is_flying
        elif key in ('1', '2', '3', '4'):
            if viewer_mode == "explore" and env:
                name = env.jump_to_landmark(int(key) - 1)
                if name:
                    landmark_status.text = f"At: {name} [{key}]"

    # -----------------------------------------------------------------------
    # Register with Ursina's __main__ discovery
    # -----------------------------------------------------------------------
    _main_mod = sys.modules['__main__']
    _main_mod.update = update
    _main_mod.input = input

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    app.run()

    # Cleanup
    if viewer_mode == "inspect" and cap:
        cap.release()
    elif viewer_mode == "explore" and tracker:
        tracker.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
