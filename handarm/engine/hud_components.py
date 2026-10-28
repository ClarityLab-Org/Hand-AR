"""Reusable HUD textures for the viewer modes."""

import os
from collections import deque
from functools import lru_cache
from typing import Deque, List, Tuple

from PIL import Image, ImageDraw, ImageFont
from ursina import Entity, Texture, camera, color


@lru_cache(maxsize=8)
def _load_emoji_font(size: int):
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


def _custom_hand_icons():
    hand_icon_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "assets",
        "palm_hand_cropped.png",
    )
    if not os.path.exists(hand_icon_path):
        return None, None
    try:
        raw_hand = Image.open(hand_icon_path).convert("RGBA")
        h_target = 48
        w_target = max(1, int(h_target * raw_hand.width / raw_hand.height))
        custom_palm = raw_hand.resize((w_target, h_target), Image.Resampling.LANCZOS)
        return custom_palm, custom_palm.transpose(Image.FLIP_LEFT_RIGHT)
    except (OSError, ValueError):
        return None, None


def _create_object_gesture_guide_texture() -> Texture:
    width, height = 860, 830
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (4, 4, width - 4, height - 4),
        radius=22,
        fill=(8, 14, 24, 248),
        outline=(0, 225, 255, 235),
        width=4,
    )
    font_paths = (
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    )
    font_title = font_section = font_item = font_desc = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font_title = ImageFont.truetype(path, 40)
                font_section = ImageFont.truetype(path, 34)
                font_item = ImageFont.truetype(path, 28)
                font_desc = ImageFont.truetype(path, 26)
                break
            except OSError:
                continue
    if font_title is None:
        font_title = font_section = font_item = font_desc = ImageFont.load_default()
    emoji_paths = (
        "/usr/share/fonts/google-noto-emoji-fonts/NotoEmoji-Regular.ttf",
        "/usr/share/fonts/gdouros-symbola/Symbola.ttf",
    )
    font_title_icon = font_icon = None
    for path in emoji_paths:
        if os.path.exists(path):
            try:
                font_title_icon = ImageFont.truetype(path, 48)
                font_icon = ImageFont.truetype(path, 42)
                break
            except OSError:
                continue
    if font_icon is None:
        font_title_icon = font_icon = font_item
    custom_palm, custom_palm_left = _custom_hand_icons()

    draw.text((36, 18), "🎮", font=font_title_icon, fill=(255, 220, 50, 255))
    draw.text((100, 22), "HAND GESTURES", font=font_title, fill=(255, 220, 50, 255))
    draw.line((36, 78, width - 36, 78), fill=(0, 225, 255, 150), width=2)

    if custom_palm is not None:
        img.paste(custom_palm, (40, 92), custom_palm)
    else:
        draw.text((36, 94), "✋", font=font_icon, fill=(255, 160, 50, 255))
    draw.text((94, 98), "RIGHT HAND", font=font_section, fill=(255, 160, 50, 255))

    def draw_item(y, icon, name, desc):
        if icon == "open_palm" and custom_palm is not None:
            img.paste(custom_palm, (42, y - 6), custom_palm)
        else:
            draw.text((42, y - 6), icon, font=font_icon, fill=(255, 255, 255, 255))
        draw.text((105, y), name, font=font_item, fill=(0, 225, 255, 255))
        draw.text((450, y), desc, font=font_desc, fill=(255, 255, 255, 255))

    for row, item in enumerate((
        ("☝", "Index finger", ":  Rotate"),
        ("🤏", "Pinch closer", ":  Zoom in"),
        ("🤏", "Pinch apart", ":  Zoom out"),
        ("✌", "Peace sign", ":  Screenshot"),
    )):
        draw_item(152 + row * 50, *item)

    draw.line((36, 374, width - 36, 374), fill=(0, 225, 255, 80), width=1)
    if custom_palm_left is not None:
        img.paste(custom_palm_left, (40, 390), custom_palm_left)
    else:
        draw.text((36, 392), "✋", font=font_icon, fill=(255, 160, 50, 255))
    draw.text((94, 396), "LEFT HAND", font=font_section, fill=(255, 160, 50, 255))
    draw_item(450, "open_palm", "Open palm", ":  Pause")
    draw_item(505, "👋", "Wrist move", ":  Translate")
    draw.line((36, 570, width - 36, 570), fill=(0, 220, 255, 120), width=2)
    draw.text(
        (36, 590), "[H] Hide/Show Guide",
        font=font_desc, fill=(160, 215, 255, 255),
    )
    tex = Texture(img)
    tex.filtering = True
    return tex


def _create_gesture_guide_texture(mode: str = "pointcloud") -> Texture:
    """Render the original Point Cloud guide without altering its appearance."""
    if mode == "object":
        return _create_object_gesture_guide_texture()
    if mode != "pointcloud":
        raise ValueError(f"Unknown gesture guide mode: {mode}")
    width, height = 860, 830
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (4, 4, width - 4, height - 4),
        radius=22,
        fill=(8, 14, 24, 248),
        outline=(0, 225, 255, 235),
        width=4,
    )
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
    for path in font_paths_bold:
        if os.path.exists(path):
            try:
                font_title = ImageFont.truetype(path, 40)
                font_sec = ImageFont.truetype(path, 34)
                font_item = ImageFont.truetype(path, 28)
                font_desc = ImageFont.truetype(path, 26)
                break
            except Exception:
                continue

    font_emoji_title = font_emoji_sec = font_emoji_item = None
    for path in font_paths_emoji:
        if os.path.exists(path):
            try:
                font_emoji_title = ImageFont.truetype(path, 48)
                font_emoji_sec = ImageFont.truetype(path, 42)
                font_emoji_item = ImageFont.truetype(path, 42)
                break
            except Exception:
                continue

    if font_title is None:
        font_title = font_sec = font_item = font_desc = ImageFont.load_default()
    if font_emoji_item is None:
        font_emoji_title = font_emoji_sec = font_emoji_item = font_item

    custom_palm, custom_palm_left = _custom_hand_icons()
    draw.text((36, 18), "🎮", font=font_emoji_title, fill=(255, 220, 50, 255))
    draw.text((100, 22), "HAND GESTURES", font=font_title, fill=(255, 220, 50, 255))
    draw.line((36, 78, width - 36, 78), fill=(0, 225, 255, 150), width=2)
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
        y += 42
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
        ("🫴", "Move Forward", ":  Hand Above Center"),
        ("🫳", "Move Backward", ":  Hand Below Center"),
        ("🤜", "Move Left", ":  Hand Left Center"),
        ("🤛", "Move Right", ":  Hand Right Center"),
    ]
    for icon, name, desc in left_items:
        draw.text((42, y - 6), icon, font=font_emoji_item, fill=(255, 255, 255, 255))
        draw.text((105, y), name, font=font_item, fill=(0, 225, 255, 255))
        draw.text((400, y), desc, font=font_desc, fill=(255, 255, 255, 255))
        y += 42
    draw.line((36, y + 2, width - 36, y + 2), fill=(0, 225, 255, 80), width=1)
    draw.text((42, y + 14), "👐", font=font_emoji_item, fill=(255, 255, 255, 255))
    draw.text((105, y + 18), "BOTH HAND", font=font_item, fill=(255, 220, 50, 255))
    draw.text(
        (105, y + 48), ":  Palms = Move Down | Palm + 👍 = Move Up",
        font=font_desc, fill=(255, 255, 255, 255),
    )
    y += 82
    draw.line((36, y, width - 36, y), fill=(0, 220, 255, 120), width=2)
    draw.text(
        (36, y + 16), "[H] Hide/Show Guide  •  [C] Color Palette",
        font=font_desc, fill=(160, 215, 255, 255),
    )
    tex = Texture(img)
    tex.filtering = True
    return tex


@lru_cache(maxsize=1)
def _load_ui_fonts() -> Tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    font_paths = (
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, 14), ImageFont.truetype(path, 16)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=14), ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
        return font, font


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
    width, height = 600, 84
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (0, 0, width - 1, height - 1),
        radius=10,
        fill=(12, 16, 25, 225),
        outline=(42, 58, 82, 220),
        width=2,
    )
    col_w = width // 3
    metrics = [
        ("FPS", f"{fps:.1f}", (57, 255, 20), fps_history, 144.0),
        ("CPU", f"{cpu:.1f}%", (255, 167, 38), cpu_history, 100.0),
        ("RAM", f"{ram:.1f}%", (41, 182, 246), ram_history, 100.0),
    ]
    for i, (label, val_str, color_rgb, history, max_val) in enumerate(metrics):
        x_off = i * col_w
        if i > 0:
            draw.line((x_off, 10, x_off, height - 10), fill=(40, 54, 76, 180), width=1)
        draw.text((x_off + 12, 8), label, font=font_label, fill=(160, 180, 205, 255))
        draw.text((x_off + 62, 7), val_str, font=font_val, fill=color_rgb + (255,))
        gx0, gy0, gw, gh = x_off + 10, 34, col_w - 20, 40
        draw.rounded_rectangle(
            (gx0, gy0, gx0 + gw, gy0 + gh),
            radius=4,
            fill=(8, 11, 18, 210),
            outline=(32, 44, 64, 160),
            width=1,
        )
        mid_y = gy0 + gh // 2
        for dx in range(0, gw, 8):
            draw.line(
                (gx0 + dx, mid_y, min(gx0 + dx + 4, gx0 + gw), mid_y),
                fill=(35, 48, 70, 140),
                width=1,
            )
        step = gw / max(1, len(history) - 1) if len(history) > 1 else gw
        pts = [
            (
                gx0 + idx * step,
                (gy0 + gh - 2) - max(0.0, min(1.0, value / max_val)) * (gh - 4),
            )
            for idx, value in enumerate(history)
        ]
        if len(pts) >= 2:
            draw.polygon(
                [(gx0, gy0 + gh - 1)] + pts + [(gx0 + gw, gy0 + gh - 1)],
                fill=color_rgb + (50,),
            )
            draw.line(pts, fill=color_rgb + (240,), width=2)
        if pts:
            last_x, last_y = pts[-1]
            draw.ellipse(
                (last_x - 3, last_y - 3, last_x + 3, last_y + 3),
                fill=(255, 255, 255, 255),
                outline=color_rgb,
                width=1,
            )
    return img


class TelemetryGraphUI:
    """Cached graph texture using the Point Cloud telemetry renderer."""

    def __init__(self) -> None:
        self.font_label, self.font_val = _load_ui_fonts()
        self.fps_history: Deque[float] = deque([60.0] * 6, maxlen=24)
        self.cpu_history: Deque[float] = deque([15.0] * 6, maxlen=24)
        self.ram_history: Deque[float] = deque([50.0] * 6, maxlen=24)
        self.texture = Texture(_render_telemetry_hud(
            60.0, 15.0, 50.0,
            list(self.fps_history), list(self.cpu_history), list(self.ram_history),
            self.font_label, self.font_val,
        ))
        self.texture.filtering = True
        self.panel = Entity(
            parent=camera.ui,
            model="quad",
            texture=self.texture,
            position=(0.52, 0.455),
            scale=(0.50, 0.07),
            z=0.08,
        )

    def update(self, fps: float, cpu: float, ram: float) -> None:
        self.fps_history.append(fps)
        self.cpu_history.append(cpu)
        self.ram_history.append(ram)
        image = _render_telemetry_hud(
            fps, cpu, ram,
            list(self.fps_history), list(self.cpu_history), list(self.ram_history),
            self.font_label, self.font_val,
        )
        self.texture._cached_image = image
        self.texture.apply()


@lru_cache(maxsize=4)
def create_panel_texture(width: int = 512, height: int = 256) -> Texture:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle(
        (3, 3, width - 4, height - 4),
        radius=max(8, height // 12),
        fill=(8, 14, 24, 248),
        outline=(0, 225, 255, 235),
        width=max(2, height // 64),
    )
    tex = Texture(img)
    tex.filtering = True
    return tex
