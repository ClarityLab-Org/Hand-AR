"""
Shared UI components for the Ursina-based 3D viewer.

Provides camera feed rendering, performance monitoring overlay,
hand-registration initialization overlay, and skeleton drawing.
"""

import cv2
import mediapipe as mp
import numpy as np
from panda3d.core import Texture as P3DTexture
from ursina import Entity, Text, camera, color

from handarm import config


# ---------------------------------------------------------------------------
# MediaPipe drawing setup
# ---------------------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

LEFT_STYLE = mp_drawing.DrawingSpec(color=(0, 220, 0), thickness=2, circle_radius=3)
RIGHT_STYLE = mp_drawing.DrawingSpec(color=(50, 180, 255), thickness=2, circle_radius=3)
LEFT_CONN = mp_drawing.DrawingSpec(color=(0, 180, 0), thickness=2)
RIGHT_CONN = mp_drawing.DrawingSpec(color=(0, 130, 230), thickness=2)


def draw_skeletons(
    display_frame: np.ndarray,
    results,
    registered_hands: dict = None,
) -> None:
    """Renders MediaPipe hand skeleton landmarks and connection lines onto a frame.

    Args:
        display_frame: Video frame to draw on (modified in-place).
        results: MediaPipe hands detection results object.
        registered_hands: Dict of actively registered hand labels.
            Hands not in this dict are rendered dimmed.
    """
    if not (results and results.multi_hand_landmarks and results.multi_handedness):
        return
    h_px, w_px, _ = display_frame.shape
    for i, hand_lms in enumerate(results.multi_hand_landmarks):
        label = results.multi_handedness[i].classification[0].label
        if (registered_hands is not None and
                label not in registered_hands and len(registered_hands) > 0):
            ns = mp_drawing.DrawingSpec(color=(60, 30, 30), thickness=1, circle_radius=2)
            cs = mp_drawing.DrawingSpec(color=(50, 20, 20), thickness=1)
            tag_col, tag_label = (80, 30, 30), f"{label.upper()} (ignored)"
        else:
            ns = LEFT_STYLE if label == "Left" else RIGHT_STYLE
            cs = LEFT_CONN if label == "Left" else RIGHT_CONN
            tag_col = (0, 220, 0) if label == "Left" else (50, 180, 255)
            tag_label = label.upper()
        wx = int(hand_lms.landmark[0].x * w_px)
        wy = int(hand_lms.landmark[0].y * h_px)
        cv2.putText(display_frame, tag_label, (wx - 20, wy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, tag_col, 2)
        mp_drawing.draw_landmarks(
            display_frame, hand_lms, mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=ns, connection_drawing_spec=cs,
        )


# ---------------------------------------------------------------------------
# Camera feed texture
# ---------------------------------------------------------------------------

class CameraFeedUI:
    """Manages a live camera feed texture displayed in the Ursina UI overlay.

    Creates a Panda3D texture and two UI quads (border + feed), plus a 'LIVE'
    label positioned at the bottom-left of the viewport.
    """

    def __init__(self, feed_w: int = None, feed_h: int = None) -> None:
        self.feed_w: int = feed_w or config.FEED_WIDTH
        self.feed_h: int = feed_h or config.FEED_HEIGHT

        # Panda3D texture for direct RAM writes
        self._p3d_tex = P3DTexture('camera_feed')
        self._p3d_tex.setup2dTexture(
            self.feed_w, self.feed_h,
            P3DTexture.TUnsignedByte, P3DTexture.FRgb,
        )
        self._p3d_tex.setMagfilter(P3DTexture.FTLinear)
        self._p3d_tex.setMinfilter(P3DTexture.FTLinear)
        buf = self._p3d_tex.modifyRamImage()
        memoryview(buf).cast('B')[:] = b'\x00' * (self.feed_w * self.feed_h * 3)

        # UI entities
        Entity(
            parent=camera.ui, model='quad',
            scale=(0.445, 0.315), position=(-0.67, -0.335),
            color=color.color(0, 0, 0.35), origin=(0, 0), z=0.01,
        )
        self._feed_view = Entity(
            parent=camera.ui, model='quad',
            scale=(0.44, 0.31), position=(-0.67, -0.335), origin=(0, 0),
        )
        self._feed_view.model.setTexture(self._p3d_tex)
        Text(
            "LIVE", parent=camera.ui,
            position=(-0.67, -0.168), origin=(0, 0),
            scale=0.9, color=color.red,
        )

    def upload_frame(self, frame: np.ndarray) -> None:
        """Resizes and uploads a video frame to the UI texture.

        Args:
            frame: Video frame (BGR or RGB, any size) — resized to feed_w×feed_h.
        """
        small = cv2.resize(frame, (self.feed_w, self.feed_h),
                           interpolation=cv2.INTER_LINEAR)
        small = np.flipud(small)
        small = np.ascontiguousarray(small, dtype=np.uint8)
        buf = self._p3d_tex.modifyRamImage()
        memoryview(buf).cast('B')[:] = small.tobytes()
        self._p3d_tex.setRamImage(buf)


# ---------------------------------------------------------------------------
# Performance monitoring overlay
# ---------------------------------------------------------------------------

class PerformanceUI:
    """Displays FPS, CPU%, and RAM% in the top-right corner of the viewport."""

    def __init__(self) -> None:
        self.fps_text = Text(
            "FPS: --", parent=camera.ui,
            position=(0.68, 0.46), origin=(0, 0),
            scale=0.85, color=color.lime,
        )
        self.cpu_text = Text(
            "CPU: --", parent=camera.ui,
            position=(0.68, 0.41), origin=(0, 0),
            scale=0.85, color=color.orange,
        )
        self.ram_text = Text(
            "RAM: --", parent=camera.ui,
            position=(0.68, 0.36), origin=(0, 0),
            scale=0.85, color=color.cyan,
        )

    def update(self, fps: int, cpu: float, ram: float) -> None:
        """Updates the performance text values."""
        self.fps_text.text = f"FPS: {fps}"
        self.cpu_text.text = f"CPU: {cpu:.0f}%"
        self.ram_text.text = f"RAM: {ram:.0f}%"


# ---------------------------------------------------------------------------
# Hand-registration initialization overlay
# ---------------------------------------------------------------------------

class InitOverlay:
    """Scanning overlay shown while the hand-registration state machine runs.

    Displays a panel with status text, sub-text, and a progress bar.
    """

    HOLD_FRAMES: int = 45
    LOST_GRACE_FRAMES: int = 10

    def __init__(self) -> None:
        self.state: str = "waiting"
        self.frames: int = 0
        self.lost_frames: int = 0
        self.registered_hands: dict = {}

        self._panel = Entity(
            parent=camera.ui, model='quad',
            scale=(0.60, 0.13), position=(0, 0.0),
            color=color.color(0, 0, 0.12, 0.88), origin=(0, 0),
            z=0.02, enabled=True,
        )
        self._status_text = Text(
            "Show your hand(s) to begin", parent=camera.ui,
            position=(0, 0.03), origin=(0, 0),
            scale=1.05, color=color.yellow, enabled=True,
        )
        self._sub_text = Text(
            "Hold still while scanning...", parent=camera.ui,
            position=(0, -0.02), origin=(0, 0),
            scale=0.75, color=color.color(0, 0, 0.80), enabled=True,
        )
        self._bar_bg = Entity(
            parent=camera.ui, model='quad',
            scale=(0.42, 0.018), position=(0, -0.06),
            color=color.color(0, 0, 0.25), origin=(0, 0),
            z=0.02, enabled=True,
        )
        self._bar_fill = Entity(
            parent=camera.ui, model='quad',
            scale=(0.0, 0.014), position=(-0.21, -0.06),
            color=color.cyan, origin=(-0.5, 0),
            z=0.01, enabled=True,
        )

    def set_visible(self, enabled: bool) -> None:
        """Shows or hides all overlay elements."""
        self._panel.enabled = enabled
        self._status_text.enabled = enabled
        self._sub_text.enabled = enabled
        self._bar_bg.enabled = enabled
        self._bar_fill.enabled = enabled

    def update_bar(self, fraction: float) -> None:
        """Updates progress bar fill (0.0 to 1.0)."""
        self._bar_fill.scale_x = 0.42 * max(0.0, min(1.0, fraction))

    def reset(self, motion_reset_fn=None) -> None:
        """Resets the state machine to 'waiting'. Optionally calls a motion reset."""
        self.state = "waiting"
        self.frames = 0
        self.lost_frames = 0
        self.registered_hands = {}
        if motion_reset_fn:
            motion_reset_fn()
        self.update_bar(0)
        self.set_visible(True)
        self._status_text.text = "Show your hand(s) to begin"
        self._status_text.color = color.yellow
        self._sub_text.text = "Hold still while scanning..."

    def process(self, seen_this_frame: dict) -> bool:
        """Runs one frame of the registration state machine.

        Args:
            seen_this_frame: Dict mapping hand label -> landmarks for hands
                visible in this frame.

        Returns:
            True if the state is 'locked' (gestures should be processed),
            False if still initializing (caller should skip gesture processing).
        """
        if self.state == "locked":
            # Check for hand loss
            if not any(lbl in seen_this_frame for lbl in self.registered_hands):
                self.lost_frames += 1
                if self.lost_frames >= self.LOST_GRACE_FRAMES:
                    self.reset()
                    return False
            else:
                self.lost_frames = 0
            return True

        # Not locked yet — run scanning logic
        if not seen_this_frame:
            self.frames = 0
            self.update_bar(0)
            if self.state != "waiting":
                self.state = "waiting"
                self._status_text.text = "Show your hand(s) to begin"
                self._status_text.color = color.yellow
                self._sub_text.text = "Hold still while scanning..."
        else:
            self.state = "scanning"
            self.frames += 1
            self.update_bar(self.frames / self.HOLD_FRAMES)
            self._status_text.text = (
                "Scanning..." if self.frames % 10 < 5 else "Hold steady..."
            )
            self._status_text.color = color.orange
            if self.frames >= self.HOLD_FRAMES:
                self.registered_hands = {label: True for label in seen_this_frame}
                self.state = "locked"
                self.lost_frames = 0
                self.update_bar(1.0)
                self.set_visible(False)

        return False

    @property
    def status_text(self):
        """Direct access to the status Text entity for external updates."""
        return self._status_text
