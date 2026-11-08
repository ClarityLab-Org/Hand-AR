"""Object inspection mode for the unified viewer."""

import datetime
import math
import os

import cv2
from panda3d.core import Filename, loadPrcFileData
from ursina import *

from handarm import config
from handarm.engine.hud_components import (
    TelemetryGraphUI,
    _create_gesture_guide_texture,
    create_panel_texture,
)
from handarm.engine.ui_common import InitOverlay, draw_skeletons, safe_set_text


class ObjectGestureMap:
    """Object-mode gesture interpretation, intentionally separate from navigation."""

    ACTIONS = {
        "right": {"Peace": "screenshot", "Pinch": "zoom", "Index": "rotate"},
        "left": {"Open Palm": "pause", "Wrist Move": "translate"},
    }

    @staticmethod
    def _coord(point, axis):
        value = getattr(point, ("x", "y", "z")[axis], None)
        return value if value is not None else point[axis]

    @staticmethod
    def _open_palm(landmarks):
        return sum(
            ObjectGestureMap._coord(landmarks[tip], 1)
            < ObjectGestureMap._coord(landmarks[base], 1) - 0.01
            for tip, base in zip((8, 12, 16), (5, 9, 13))
        ) >= 2

    @staticmethod
    def _peace(landmarks):
        return (
            ObjectGestureMap._coord(landmarks[8], 1) < ObjectGestureMap._coord(landmarks[6], 1)
            and ObjectGestureMap._coord(landmarks[12], 1) < ObjectGestureMap._coord(landmarks[10], 1)
            and ObjectGestureMap._coord(landmarks[16], 1) > ObjectGestureMap._coord(landmarks[14], 1)
        )

    @staticmethod
    def _pinch(landmarks):
        return math.dist(
            (ObjectGestureMap._coord(landmarks[4], 0), ObjectGestureMap._coord(landmarks[4], 1)),
            (ObjectGestureMap._coord(landmarks[8], 0), ObjectGestureMap._coord(landmarks[8], 1)),
        ) < 0.05

    def classify(self, landmarks, hand_type):
        if hand_type == "Right":
            gesture = (
                "Peace" if self._peace(landmarks)
                else "Pinch" if self._pinch(landmarks)
                else "Index"
            )
        else:
            gesture = "Open Palm" if self._open_palm(landmarks) else "Wrist Move"
        return {"gesture": gesture}


class ObjectMode:
    stabilize_gestures = False
    include_landmarks = True

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

    def __init__(self, model_path, tracker, app, camera_feed, system_monitor):
        self.model_path = model_path
        self.tracker = tracker
        self.app = app
        self.camera_feed = camera_feed
        self.system_monitor = system_monitor
        self.gestures = ObjectGestureMap()
        self.model_root = Entity(parent=scene, enabled=False)
        self.car = self._load_model()
        self.model_root.enabled = False
        self.camera_locked = False
        self.model_locked = False
        self.view_mode = "free"
        self.camera_state = None
        self.rot_sens, self.trans_sens, self.zoom_sens = 2, 15, 40
        self.last_rx = self.last_ry = self.last_lx = self.last_ly = self.last_zoom = None
        self.smooth_rx = self.smooth_ry = self.smooth_tx = self.smooth_ty = self.smooth_zoom = 0.0
        self.paused = False
        self.pause_frames = self.unpause_frames = 0
        self.screenshot_cooldown = self.screenshot_timer = 0
        self.last_frame_id = -1
        self._gesture_label = "WAITING"
        self.gesture_guide = Entity(
            parent=camera.ui,
            model="quad",
            texture=_create_gesture_guide_texture(mode="object"),
            scale=(0.45, 0.43),
            position=(-0.64, 0.10),
            z=0.05,
        )
        card_texture = create_panel_texture()
        button_texture = create_panel_texture(512, 160)
        self.model_btn = Button(
            "Model Lock: OFF",
            parent=camera.ui,
            texture=button_texture,
            scale=(0.34, 0.055),
            position=(0.69, 0.35),
            color=color.white,
            highlight_color=color.color(0, 0.2, 0.2),
            text_color=color.cyan,
            on_click=self.toggle_model_lock,
        )
        self.view_panel = Entity(
            parent=camera.ui, model="quad", texture=card_texture,
            scale=(0.36, 0.16), position=(0.69, 0.235), z=0.02,
        )
        self.view_title = Text(
            "VIEW PRESETS", parent=camera.ui, position=(0.69, 0.285),
            origin=(0, 0), scale=0.72, color=color.cyan,
        )
        view_button_args = dict(
            parent=camera.ui,
            texture=button_texture,
            scale=(0.15, 0.045),
            color=color.white,
            highlight_color=color.color(0, 0.2, 0.2),
            text_color=color.white,
        )
        self.view_buttons = [
            Button("⊙ Front", position=(0.60, 0.245), on_click=lambda: self.set_view("front"), **view_button_args),
            Button("↔ Side", position=(0.78, 0.245), on_click=lambda: self.set_view("side"), **view_button_args),
            Button("↑ Top", position=(0.60, 0.195), on_click=lambda: self.set_view("top"), **view_button_args),
            Button("◇ ISO", position=(0.78, 0.195), on_click=lambda: self.set_view("iso"), **view_button_args),
        ]
        self.sensitivity_panel = Entity(
            parent=camera.ui, model="quad", texture=card_texture,
            scale=(0.36, 0.35), position=(0.69, -0.075), z=0.02,
        )
        self.sensitivity_title = Text(
            "SENSITIVITY", parent=camera.ui, position=(0.69, 0.075),
            origin=(0, 0), scale=0.72, color=color.cyan,
        )
        self.rot_title = Text("Rotation", parent=camera.ui, position=(0.56, 0.015), scale=0.62, color=color.white)
        self.rot_slider = Slider(
            min=0.5, max=5, default=self.rot_sens, step=0.1,
            position=(0.69, -0.005), scale=0.24, bar_color=color.color(0, 0, 0.3),
        )
        self.trans_title = Text("Translation", parent=camera.ui, position=(0.56, -0.075), scale=0.62, color=color.white)
        self.trans_slider = Slider(
            min=5, max=30, default=self.trans_sens, step=1,
            position=(0.69, -0.095), scale=0.24, bar_color=color.color(0, 0, 0.3),
        )
        self.zoom_title = Text("Zoom", parent=camera.ui, position=(0.56, -0.165), scale=0.62, color=color.white)
        self.zoom_slider = Slider(
            min=10, max=80, default=self.zoom_sens, step=1,
            position=(0.69, -0.185), scale=0.24, bar_color=color.color(0, 0, 0.3),
        )
        for slider in (self.rot_slider, self.trans_slider, self.zoom_slider):
            slider.bg.color = color.color(0, 0, 0.3)
            slider.knob.color = color.cyan
            slider.knob.text_entity.color = color.white
            slider.label.color = color.white

        self.gesture_text = Text(
            "GESTURE: WAITING", parent=camera.ui, position=(-0.82, 0.47),
            origin=(-0.5, 0), scale=0.9, color=color.yellow, z=0.09,
        )
        vertex_count = sum(
            geom_node.node().get_geom(index).get_vertex_data().get_num_rows()
            for geom_node in self.car.find_all_matches("**/+GeomNode")
            for index in range(geom_node.node().get_num_geoms())
        )
        if not vertex_count and getattr(self.car.model, "vertices", None) is not None:
            vertex_count = len(self.car.model.vertices)
        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        self.model_info = Text(
            f"Model: {model_name} | Vertices: {vertex_count:,}",
            parent=camera.ui, position=(0, 0.47), origin=(0, 0),
            scale=0.95, color=color.white, z=0.09,
        )
        self.status_bar = Entity(
            parent=camera.ui, model="quad", scale=(1.25, 0.05),
            position=(0.14, -0.47), color=color.color(0, 0, 0, 0.75), z=0.1,
        )
        self.view_status = Text(
            "View: [1] Front  [2] Side  [3] Top  [4] ISO",
            parent=camera.ui, position=(-0.38, -0.47),
            scale=0.8, color=color.white, z=0.09,
        )
        self.gesture_status = Text(
            "Rotate: Right hand | Translate: Left hand",
            parent=camera.ui, position=(0.24, -0.47),
            scale=0.8, color=color.yellow, z=0.09,
        )
        self.init_overlay = InitOverlay()
        self.init_overlay._panel.texture = create_panel_texture(512, 128)
        self.init_overlay._panel.color = color.white
        self.init_overlay._panel.position = (0, -0.31)
        self.init_overlay._status_text.position = (0, -0.28)
        self.init_overlay._status_text.scale = 0.95
        self.init_overlay._status_text.color = color.yellow
        self.init_overlay._sub_text.position = (0, -0.33)
        self.init_overlay._sub_text.scale = 0.68
        self.init_overlay._sub_text.color = color.white
        self.init_overlay._bar_bg.position = (0, -0.37)
        self.init_overlay._bar_fill.position = (-0.21, -0.37)
        self.performance = TelemetryGraphUI()
        self.screenshot_panel = Entity(
            parent=camera.ui, model="quad", texture=card_texture,
            scale=(0.30, 0.08), position=(0, -0.31), z=0.02, enabled=False,
        )
        self.screenshot_text = Text(
            "Screenshot Saved!", parent=camera.ui, position=(0, -0.31),
            origin=(0, 0), scale=0.9, color=color.cyan, enabled=False,
        )
        self.screenshot_text.enabled = False
        self.ui_entities = [
            self.gesture_guide, self.model_btn, self.view_panel,
            self.view_title, *self.view_buttons,
            self.sensitivity_panel,
            self.sensitivity_title, self.rot_title, self.rot_slider,
            self.trans_title, self.trans_slider, self.zoom_title,
            self.zoom_slider, self.gesture_text, self.model_info,
            self.status_bar, self.view_status, self.gesture_status,
            self.performance.panel, self.screenshot_panel, self.screenshot_text,
            self.init_overlay._panel, self.init_overlay._status_text,
            self.init_overlay._sub_text, self.init_overlay._bar_bg,
            self.init_overlay._bar_fill,
        ]

    def _load_model(self):
        car = Entity(parent=self.model_root)
        if self.model_path.lower().endswith(".csv"):
            import pandas as pd
            from handarm.geometry.alignment import auto_align_up_axis

            frame = auto_align_up_axis(pd.read_csv(self.model_path))
            colors = [
                (r / 255.0, g / 255.0, b / 255.0, 1.0)
                for r, g, b in zip(frame["r"], frame["g"], frame["b"])
            ]
            mesh = Mesh(
                vertices=[Vec3(x, y, z) for x, y, z in zip(frame["x"], frame["y"], frame["z"])],
                colors=colors, mode="point", render_points_in_3d=False, thickness=1,
            )
            mesh.clearTexGen(TextureStage.getDefault())
            car.model = mesh
            car.set_render_mode_thickness(1)
            car.set_render_mode_perspective(False)
        else:
            loadPrcFileData("", "gltf-skip-animations 1")
            loaded = base.loader.load_model(Filename.from_os_specific(self.model_path))
            if loaded is None:
                raise RuntimeError(f"Could not load model: {self.model_path}")
            loaded.reparent_to(car)
        car.scale = 1
        try:
            min_b, max_b = car.model.get_tight_bounds()
        except Exception:
            min_b, max_b = car.get_tight_bounds()
        car.origin = (min_b + max_b) / 2
        car.position = Vec3(0, 0, 0)
        size = max(max_b.x - min_b.x, max_b.y - min_b.y, max_b.z - min_b.z)
        if size > 0:
            car.scale = 3 / size
        return car

    def classify_hand(self, landmarks, hand_type):
        return self.gestures.classify(landmarks, hand_type)

    def annotate_frame(self, frame_rgb, results, state):
        draw_skeletons(frame_rgb, results, self.init_overlay.registered_hands)
        if self.init_overlay.state == "locked":
            cv2.putText(frame_rgb, f"LEFT : {state['Left']['gesture']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_rgb, f"RIGHT: {state['Right']['gesture']}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame_rgb

    def activate(self):
        self.model_root.enabled = True
        self.camera_feed.set_layout(
            feed_position=(-0.65, -0.32),
            border_position=(-0.65, -0.32),
            live_position=(-0.65, -0.14),
            live_scale=0.85,
        )
        camera.parent = scene
        if self.camera_state is None:
            camera.position = Vec3(0, 0, -12)
            camera.look_at(self.car.position)
        else:
            camera.position, camera.rotation, camera.fov = self.camera_state
        mouse.locked = False
        for entity in self.ui_entities:
            entity.enabled = True
        self.screenshot_panel.enabled = False
        self.screenshot_text.enabled = False

    def deactivate(self):
        self.camera_state = (Vec3(camera.position), Vec3(camera.rotation), camera.fov)
        self.model_root.enabled = False
        for entity in self.ui_entities:
            entity.enabled = False

    def set_view(self, mode):
        self.view_mode = mode
        self.camera_locked = True
        self.camera_position = {
            "front": Vec3(0, 0, -12),
            "side": Vec3(12, 0, 0),
            "top": Vec3(0, 12, 0),
            "iso": Vec3(8, 6, -8),
        }[mode]
        camera.position = self.camera_position
        camera.look_at(self.car.position)

    def toggle_model_lock(self):
        self.model_locked = not self.model_locked
        safe_set_text(self.model_btn, f"Model Lock: {'ON' if self.model_locked else 'OFF'}")
        self.model_btn.text_entity.color = color.lime if self.model_locked else color.cyan

    def _set_gesture_status(self, label):
        if label != self._gesture_label:
            safe_set_text(self.gesture_text, f"GESTURE: {label}")
            self._gesture_label = label

    def _reset_motion(self):
        self.last_rx = self.last_ry = self.last_lx = self.last_ly = self.last_zoom = None
        self.smooth_rx = self.smooth_ry = self.smooth_tx = self.smooth_ty = self.smooth_zoom = 0.0
        self.paused = False
        self.pause_frames = self.unpause_frames = 0

    def update(self, dt):
        fps = int(1 / dt) if dt > 0 else 0
        metrics = self.system_monitor.sample(fps)
        if self.system_monitor.updated:
            self.performance.update(*metrics)
        self.rot_sens = self.rot_slider.value
        self.trans_sens = self.trans_slider.value
        self.zoom_sens = self.zoom_slider.value
        state, frame, frame_id = self.tracker.get_snapshot()
        if frame is None or frame_id == self.last_frame_id:
            if self.screenshot_timer > 0:
                self.screenshot_timer -= dt
            if self.screenshot_timer <= 0:
                self.screenshot_text.enabled = False
                self.screenshot_panel.enabled = False
            return
        self.last_frame_id = frame_id
        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        left, right = state["Left"], state["Right"]
        seen = {hand: True for hand in ("Left", "Right") if state[hand]["visible"]}
        if not self.init_overlay.process(seen):
            self._reset_motion()
            self.camera_feed.upload_frame(display_frame)
            self._set_gesture_status(self.init_overlay.state.upper())
            return

        left_lms = left.get("landmarks") if "Left" in self.init_overlay.registered_hands else None
        right_lms = right.get("landmarks") if "Right" in self.init_overlay.registered_hands else None
        if not any(hand in seen for hand in self.init_overlay.registered_hands):
            self.init_overlay.lost_frames += 1
            if self.init_overlay.lost_frames >= self.init_overlay.LOST_GRACE_FRAMES:
                self.init_overlay.reset(motion_reset_fn=self._reset_motion)
                self.camera_feed.upload_frame(display_frame)
                self._set_gesture_status("None")
                return
        else:
            self.init_overlay.lost_frames = 0
        h_px, w_px, _ = display_frame.shape
        left_open = left_lms is not None and ObjectGestureMap._open_palm(left_lms)
        if not self.paused:
            self.pause_frames = self.pause_frames + 1 if left_open else 0
            self.unpause_frames = 0
            if self.pause_frames >= self.PAUSE_ENGAGE_FRAMES:
                self.paused = True
                self._reset_motion()
                self.paused = True
        else:
            self.unpause_frames = self.unpause_frames + 1 if not left_open else 0
            if self.unpause_frames >= self.PAUSE_RELEASE_FRAMES:
                self.paused = False
                self.pause_frames = self.unpause_frames = 0
                self._reset_motion()

        right_peace = right_lms is not None and ObjectGestureMap._peace(right_lms)
        if right_peace and self.screenshot_cooldown == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.app.win.saveScreenshot(Filename.fromOsSpecific(os.path.abspath(
                os.path.join(config.SCREENSHOT_DIR, f"shot_{timestamp}.png")
            )))
            self.screenshot_text.enabled = True
            self.screenshot_panel.enabled = True
            self.screenshot_timer = 2.0
            self.screenshot_cooldown = 25

        gesture = "Paused" if self.paused else "None"
        if not self.paused:
            if (
                right_lms and not right_peace and not self.model_locked
                and not ObjectGestureMap._pinch(right_lms)
            ):
                gesture = "Rotating"
                ix, iy = int(right_lms[8][0] * w_px), int(right_lms[8][1] * h_px)
                if self.last_rx is not None:
                    dx = max(-self.MAX_ROT_DELTA, min(self.MAX_ROT_DELTA, ix - self.last_rx))
                    dy = max(-self.MAX_ROT_DELTA, min(self.MAX_ROT_DELTA, iy - self.last_ry))
                    self.smooth_rx = self.smooth_rx * (1 - self.ALPHA_ROT) + dy * self.rot_sens * self.ALPHA_ROT
                    self.smooth_ry = self.smooth_ry * (1 - self.ALPHA_ROT) - dx * self.rot_sens * self.ALPHA_ROT
                    self.car.rotation_x += self.smooth_rx
                    self.car.rotation_y += self.smooth_ry
                else:
                    self.smooth_rx = self.smooth_ry = 0.0
                self.last_rx, self.last_ry = ix, iy
            else:
                self.smooth_rx *= self.ROT_DECAY
                self.smooth_ry *= self.ROT_DECAY
                self.last_rx = self.last_ry = None

            if right_lms and not right_peace and not self.camera_locked:
                gesture += " + Zoom" if gesture != "None" else "Zooming"
                distance = math.dist(right_lms[4][:2], right_lms[8][:2])
                if self.last_zoom is not None:
                    delta = distance - self.last_zoom
                    if abs(delta) < 0.003:
                        delta = 0.0
                    delta = max(-self.MAX_ZOOM_DELTA, min(self.MAX_ZOOM_DELTA, delta))
                    self.smooth_zoom = self.smooth_zoom * (1 - self.ALPHA_ZOOM) + delta * self.zoom_sens * 2 * self.ALPHA_ZOOM
                    camera.z = clamp(camera.z - self.smooth_zoom, -35, -3)
                else:
                    self.smooth_zoom = 0.0
                self.last_zoom = distance
            else:
                self.smooth_zoom *= self.ZOOM_DECAY
                self.last_zoom = None

            if left_lms and not self.model_locked:
                gesture = "Moving"
                lx, ly = int(left_lms[0][0] * w_px), int(left_lms[0][1] * h_px)
                if self.last_lx is not None:
                    dx = max(-self.MAX_TRANS_DELTA, min(self.MAX_TRANS_DELTA, (lx - self.last_lx) / w_px))
                    dy = max(-self.MAX_TRANS_DELTA, min(self.MAX_TRANS_DELTA, -(ly - self.last_ly) / h_px))
                    self.smooth_tx = self.smooth_tx * (1 - self.ALPHA_TRANS) + dx * self.trans_sens * self.ALPHA_TRANS
                    self.smooth_ty = self.smooth_ty * (1 - self.ALPHA_TRANS) + dy * self.trans_sens * self.ALPHA_TRANS
                    self.car.position += Vec3(self.smooth_tx, self.smooth_ty, 0)
                else:
                    self.smooth_tx = self.smooth_ty = 0.0
                self.last_lx, self.last_ly = lx, ly
            else:
                self.smooth_tx *= self.TRANS_DECAY
                self.smooth_ty *= self.TRANS_DECAY
                self.last_lx = self.last_ly = None

        if self.init_overlay.lost_frames == 0:
            self._set_gesture_status(gesture)
        if self.paused:
            cv2.rectangle(display_frame, (0, h_px // 2 - 20), (w_px, h_px // 2 + 20), (18, 18, 28), -1)
            cv2.putText(display_frame, "-- PAUSED --", (w_px // 2 - 95, h_px // 2 + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 210, 255), 2)
        self.camera_feed.upload_frame(display_frame)
        self.screenshot_cooldown = max(0, self.screenshot_cooldown - 1)
        self.screenshot_timer = max(0.0, self.screenshot_timer - dt)
        if self.screenshot_timer <= 0:
            self.screenshot_text.enabled = False
            self.screenshot_panel.enabled = False

    def input(self, key):
        if key == "h":
            self.gesture_guide.enabled = not self.gesture_guide.enabled
            return
        views = {"1": "front", "2": "side", "3": "top", "4": "iso"}
        if key in views:
            self.set_view(views[key])
