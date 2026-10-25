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
from typing import Deque, List, Optional, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from direct.task import Task
from panda3d.core import Texture as P3DTexture, WindowProperties
from ursina import *

from handarm import config
from handarm.engine.explore_core import ExploreEnvironment
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


def _load_emoji_font(size: int):
    """Load a Unicode font that PIL can rasterize into the camera texture."""
    candidates = (
        "/usr/share/fonts/gdouros-symbola/Symbola.ttf",
        "DejaVuSans.ttf",
        "NotoSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf",
        "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _create_gesture_guide_texture() -> Texture:
    """Renders a sharp, high-DPI HUD card with large prominent Unicode emojis and bold controls."""
    width, height = 860, 830
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Deep dark solid backdrop with glowing neon cyan border
    draw.rounded_rectangle(
        (4, 4, width - 4, height - 4),
        radius=22,
        fill=(8, 14, 24, 248),
        outline=(0, 225, 255, 235),
        width=4,
    )

    # Search best fonts: bold sans-serif for text, large modern emoji font
    font_paths_bold = (
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    )
    font_paths_emoji = (
        "/usr/share/fonts/google-noto-emoji-fonts/NotoEmoji-Regular.ttf",
        "/usr/share/fonts/gdouros-symbola/Symbola.ttf",
    )

    font_title = font_sec = font_item = font_desc = None
    for p in font_paths_bold:
        if os.path.exists(p):
            try:
                font_title = ImageFont.truetype(p, 40)
                font_sec = ImageFont.truetype(p, 34)
                font_item = ImageFont.truetype(p, 28)
                font_desc = ImageFont.truetype(p, 26)
                break
            except Exception:
                continue

    font_emoji_title = font_emoji_sec = font_emoji_item = None
    for p in font_paths_emoji:
        if os.path.exists(p):
            try:
                font_emoji_title = ImageFont.truetype(p, 48)
                font_emoji_sec = ImageFont.truetype(p, 42)
                font_emoji_item = ImageFont.truetype(p, 42)
                break
            except Exception:
                continue

    if font_title is None:
        font_title = font_sec = font_item = font_desc = ImageFont.load_default()
    if font_emoji_item is None:
        font_emoji_title = font_emoji_sec = font_emoji_item = font_item

    # Load custom palm hand icon if available
    hand_icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "palm_hand_cropped.png")
    custom_palm = None
    custom_palm_left = None
    if os.path.exists(hand_icon_path):
        try:
            raw_hand = Image.open(hand_icon_path).convert("RGBA")
            h_target = 48
            w_target = max(1, int(h_target * raw_hand.width / raw_hand.height))
            custom_palm = raw_hand.resize((w_target, h_target), Image.Resampling.LANCZOS)
            custom_palm_left = custom_palm.transpose(Image.FLIP_LEFT_RIGHT)
        except Exception:
            custom_palm = None

    # Title header
    draw.text((36, 18), "🎮", font=font_emoji_title, fill=(255, 220, 50, 255))
    draw.text((100, 22), "HAND GESTURES", font=font_title, fill=(255, 220, 50, 255))
    draw.line((36, 78, width - 36, 78), fill=(0, 225, 255, 150), width=2)

    # Right Hand Section (With custom high-res palm icon)
    if custom_palm is not None:
        img.paste(custom_palm, (40, 92), custom_palm)
    else:
        draw.text((36, 94), "✋", font=font_emoji_sec, fill=(255, 160, 50, 255))
    draw.text((94, 98), "RIGHT HAND", font=font_sec, fill=(255, 160, 50, 255))
    right_items = [
        ("🤏", "Pinch", ":  Zoom In"),
        ("✌", "Peace Sign", ":  Zoom Out"),
        ("👍", "Thumbs up", ":  Reset view"),
        ("👈", "Palm Left", ":  Rotate left"),
        ("👉", "Palm Right", ":  Rotate right"),
        ("☝", "Palm up", ":  Rotate up"),
        ("👇", "Palm down", ":  Rotate down"),
    ]
    y = 152
    for icon, name, desc in right_items:
        draw.text((42, y - 6), icon, font=font_emoji_item, fill=(255, 255, 255, 255))
        draw.text((105, y), name, font=font_item, fill=(0, 225, 255, 255))
        draw.text((330, y), desc, font=font_desc, fill=(255, 255, 255, 255))
        y += 48

    # Left Hand Section (Only requested gestures)
    y += 8
    draw.line((36, y, width - 36, y), fill=(0, 225, 255, 80), width=1)
    y += 14
    if custom_palm_left is not None:
        img.paste(custom_palm_left, (40, y - 2), custom_palm_left)
    else:
        draw.text((36, y - 6), "✋", font=font_emoji_sec, fill=(0, 225, 255, 255))
    draw.text((94, y + 4), "LEFT HAND", font=font_sec, fill=(0, 225, 255, 255))
    y += 52
    left_items = [
        ("⬆", "Move Forward", ":  Hand Above Center"),
        ("⬇", "Move Backward", ":  Hand Below Center"),
        ("⬅", "Move Left", ":  Hand Left Center"),
        ("➡", "Move Right", ":  Hand Right Center"),
    ]
    for icon, name, desc in left_items:
        draw.text((42, y - 6), icon, font=font_emoji_item, fill=(255, 255, 255, 255))
        draw.text((105, y), name, font=font_item, fill=(0, 225, 255, 255))
        draw.text((400, y), desc, font=font_desc, fill=(255, 255, 255, 255))
        y += 48

    draw.line((36, y + 2, width - 36, y + 2), fill=(0, 225, 255, 80), width=1)
    draw.text((42, y + 14), "👐", font=font_emoji_item, fill=(255, 255, 255, 255))
    draw.text((105, y + 18), "BOTH HAND", font=font_item, fill=(255, 220, 50, 255))
    draw.text((330, y + 18), ":  Palms = Move Down | Palm + 👍 = Move Up",
              font=font_desc, fill=(255, 255, 255, 255))

    # Footer controls
    y += 10
    draw.line((36, y, width - 36, y), fill=(0, 220, 255, 120), width=2)
    y += 16
    draw.text((36, y), "[H] Hide/Show Guide  •  [C] Color Palette", font=font_desc, fill=(160, 215, 255, 255))

    tex = Texture(img)
    tex.filtering = True
    return tex


def _load_ui_fonts() -> Tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    """Load bold sans-serif fonts for telemetry HUD."""
    font_paths = (
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, 14), ImageFont.truetype(p, 16)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=14), ImageFont.load_default(size=16)
    except TypeError:
        f = ImageFont.load_default()
        return f, f


def _render_telemetry_hud(
    fps: float,
    cpu: float,
    ram: float,
    fps_history: List[float],
    cpu_history: List[float],
    ram_history: List[float],
    font_label: ImageFont.ImageFont,
    font_val: ImageFont.ImageFont,
) -> Image.Image:
    """Render high-contrast telemetry card with sparkline graphs for FPS, CPU, RAM."""
    W, H = 600, 84
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Outer container with rounded corners and border
    draw.rounded_rectangle(
        (0, 0, W - 1, H - 1),
        radius=10,
        fill=(12, 16, 25, 225),
        outline=(42, 58, 82, 220),
        width=2,
    )

    col_w = W // 3
    metrics = [
        ("FPS", f"{fps:.1f}", (57, 255, 20), fps_history, 144.0),
        ("CPU", f"{cpu:.1f}%", (255, 167, 38), cpu_history, 100.0),
        ("RAM", f"{ram:.1f}%", (41, 182, 246), ram_history, 100.0),
    ]

    for i, (label, val_str, color_rgb, history, max_val) in enumerate(metrics):
        x_off = i * col_w
        if i > 0:
            draw.line((x_off, 10, x_off, H - 10), fill=(40, 54, 76, 180), width=1)

        # Label and Value
        draw.text((x_off + 12, 8), label, font=font_label, fill=(160, 180, 205, 255))
        draw.text((x_off + 62, 7), val_str, font=font_val, fill=color_rgb + (255,))

        # Sparkline area
        gx0 = x_off + 10
        gy0 = 34
        gw = col_w - 20
        gh = 40

        # Graph background box
        draw.rounded_rectangle(
            (gx0, gy0, gx0 + gw, gy0 + gh),
            radius=4,
            fill=(8, 11, 18, 210),
            outline=(32, 44, 64, 160),
            width=1,
        )

        # 50% dashed / subtle reference line
        mid_y = gy0 + gh // 2
        for dx in range(0, gw, 8):
            draw.line((gx0 + dx, mid_y, min(gx0 + dx + 4, gx0 + gw), mid_y), fill=(35, 48, 70, 140), width=1)

        # Plot points
        pts = []
        n = len(history)
        step = gw / max(1, n - 1) if n > 1 else gw
        for idx, v in enumerate(history):
            norm = max(0.0, min(1.0, v / max_val))
            px = gx0 + idx * step
            py = (gy0 + gh - 2) - norm * (gh - 4)
            pts.append((px, py))

        if len(pts) >= 2:
            poly = [(gx0, gy0 + gh - 1)] + pts + [(gx0 + gw, gy0 + gh - 1)]
            fill_col = color_rgb + (50,)
            draw.polygon(poly, fill=fill_col)
            draw.line(pts, fill=color_rgb + (240,), width=2)
            last_x, last_y = pts[-1]
            draw.ellipse(
                (last_x - 3, last_y - 3, last_x + 3, last_y + 3),
                fill=(255, 255, 255, 255),
                outline=color_rgb,
                width=1,
            )
        elif pts:
            last_x, last_y = pts[-1]
            draw.ellipse(
                (last_x - 3, last_y - 3, last_x + 3, last_y + 3),
                fill=(255, 255, 255, 255),
                outline=color_rgb,
                width=1,
            )

    return img


class NavigationHUD:
    """Stable top overlay for gesture state, point loading, and performance telemetry."""

    UPDATE_INTERVAL = 0.4

    def __init__(self, max_points: int) -> None:
        self.max_points = max_points
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
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent

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
        scale=(0.48, 0.46),
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
