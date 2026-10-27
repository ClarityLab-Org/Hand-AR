"""Point-cloud navigation mode for the unified viewer."""

import cv2
import mediapipe as mp
from ursina import *

from handarm import config
from handarm.engine.explore_core import ExploreEnvironment
from handarm.engine.explore_standalone import (
    NavigationHUD,
    _create_gesture_guide_texture,
)
from handarm.engine.ui_common import safe_set_text
from handarm.tracking.gestures import (
    classify_both_hands,
    classify_left_explore_standalone,
    classify_right_explore,
    compute_explore_features,
    compute_thumbs_up,
    gesture_label,
)


class PointCloudMode:
    stabilize_gestures = True
    include_landmarks = False
    GESTURE_MAP = {
        "right": classify_right_explore,
        "left": classify_left_explore_standalone,
    }

    def __init__(
        self, target_path, camera_feed, point_cloud=None, downsample_step=1,
        tracker=None, system_monitor=None,
    ):
        self.target_path = target_path
        self.camera_feed = camera_feed
        self.point_cloud = point_cloud
        self.downsample_step = downsample_step
        self.tracker = tracker
        self.system_monitor = system_monitor
        self._pinch_distances = {"Left": None, "Right": None}

        self.env = ExploreEnvironment(
            target_path,
            point_cloud=point_cloud,
            downsample_step=downsample_step,
            point_thickness=config.POINT_THICKNESS,
            reset_cooldown_duration=0.0,
        )
        self.env.render_scene(
            point_cloud=point_cloud,
            downsample_step=downsample_step,
        )
        max_points = (
            point_cloud.attrs.get("point_count", len(point_cloud))
            if point_cloud is not None else 0
        )
        self.hud = NavigationHUD(
            max_points=max_points, system_monitor=system_monitor,
        )
        self.gesture_guide = Entity(
            parent=camera.ui,
            model="quad",
            texture=_create_gesture_guide_texture(),
            scale=(0.45, 0.43),
            position=(-0.64, 0.135),
            z=0.05,
        )
        self.landmark_status = Text(
            text="Jump: [1] Academic [2] Labs [3] Plaza [4] Gate",
            position=(-0.38, -0.47), scale=0.8, color=color.white, z=0.09,
        )
        self.rotate_status = Text(
            text="Rotate: [Arrows] / [J][L] | 360° [T]",
            position=(0.24, -0.47), scale=0.8, color=color.yellow, z=0.09,
        )
        self.status_bar = Entity(
            parent=camera.ui, model="quad", scale=(1.25, 0.05),
            position=(0.14, -0.47), color=color.color(0, 0, 0, 0.75), z=0.1,
        )
        self.ui_entities = [
            self.gesture_guide, self.landmark_status, self.rotate_status,
            self.status_bar, self.hud.gesture_text, self.hud.points_text,
            self.hud.perf_panel,
        ]

    def classify_hand(self, landmarks, hand_type):
        features = compute_explore_features(landmarks)
        if hand_type == "Right":
            previous = self._pinch_distances[hand_type]
            gesture = self.GESTURE_MAP["right"](features, previous)
            self._pinch_distances[hand_type] = features["pinch_distance"]
        else:
            gesture = self.GESTURE_MAP["left"](features)
        return {
            "gesture": gesture,
            "navigation": gesture_label(gesture if features else "None"),
            "open_palm": features["open_palm"],
            "thumbs_up": compute_thumbs_up(features, margin=0.02),
        }

    @staticmethod
    def annotate_frame(frame_rgb, results, state):
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_rgb, hand_lms, mp.solutions.hands.HAND_CONNECTIONS,
                )
        return cv2.resize(frame_rgb, (320, 240))

    def activate(self):
        self._pinch_distances = {"Left": None, "Right": None}
        self.camera_feed.set_layout(
            feed_position=(-0.65, -0.32),
            border_position=(-0.65, -0.32),
            live_position=(-0.65, -0.14),
            live_scale=0.85,
        )
        self.env.world_root.enabled = True
        self.env.player.enabled = True
        self.env.player.cursor.enabled = True
        camera.parent = self.env.player.camera_pivot
        camera.position = Vec3(0, 0, 0)
        camera.rotation = Vec3(0, 0, 0)
        camera.fov = self.env.zoom_target_fov
        for entity in self.ui_entities:
            entity.enabled = True

    def deactivate(self):
        self.env.world_root.enabled = False
        self.env.player.enabled = False
        self.env.player.cursor.enabled = False
        for entity in self.ui_entities:
            entity.enabled = False

    def update(self, dt):
        frame = self.tracker.get_frame()
        if frame is not None:
            self.camera_feed.upload_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        state = self.tracker.get_state()
        left, right = state["Left"], state["Right"]
        both_gesture = (
            classify_both_hands(left, right)
            if left["visible"] and right["visible"] else "None"
        )
        active_hand = right if right["visible"] else left
        active_gesture = (
            both_gesture if both_gesture != "None"
            else active_hand.get("gesture", "None")
        )
        self.hud.update(self.env, gesture_label(active_gesture), dt)
        self.env.update_steering(
            right if both_gesture == "None" else {"visible": False}
        )
        movement_state = left
        if both_gesture == "Both hand Move Up":
            movement_state = dict(left, gesture="Move Up")
        elif both_gesture == "Both hand Move Down":
            movement_state = dict(left, gesture="Move Down")
        self.env.update_movement_and_terrain(movement_state)

    def input(self, key):
        if key == "h":
            self.gesture_guide.enabled = not self.gesture_guide.enabled
        elif key == "r":
            self.env.reset_player()
            safe_set_text(
                self.rotate_status, "Rotate: [Arrows] / [J][L] | 360° [T]",
            )
            self.rotate_status.color = color.yellow
        elif key == "t":
            active = self.env.toggle_auto_rotate()
            safe_set_text(
                self.rotate_status,
                "Rotate: 360° Turntable ON [T]" if active
                else "Rotate: [Arrows] / [J][L] | 360° [T]",
            )
            self.rotate_status.color = color.lime if active else color.yellow
        elif key == "c":
            self.env.cycle_color_mode()
        elif key == "p":
            self.env.cycle_point_thickness()
        elif key == "f":
            self.env.is_flying = not self.env.is_flying
        elif key in ("1", "2", "3", "4"):
            name = self.env.jump_to_landmark(int(key) - 1)
            if name:
                safe_set_text(self.landmark_status, f"At: {name} [{key}]")
