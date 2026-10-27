"""Single Ursina application and shared camera pipeline for both viewer modes."""

import os

import cv2
from direct.task import Task
from panda3d.core import WindowProperties
from ursina import *

from handarm import config
from handarm.engine.ui_common import CameraFeedUI, SystemMonitor, WINDOW_SIZE
from handarm.modes.manager import ModeManager
from handarm.modes.object_mode import ObjectMode
from handarm.modes.pointcloud_mode import PointCloudMode
from handarm.tracking.hand_tracker import HandTracker


def main(
    target_path=None,
    point_cloud=None,
    model_path=None,
    initial_mode="pointcloud",
    downsample_step=1,
):
    """Run both view modes in one Ursina app with a single camera and tracker."""
    if target_path is None:
        target_path = str(config.PROJECT_ROOT / "models" / "iitj_chunks")
    if model_path is None:
        model_path = os.path.join(config.MODELS_DIR, "Old_Rusty_Car.glb")

    app = Ursina(size=WINDOW_SIZE, development_mode=False)
    window.color = color.color(0, 0, 0.08)
    os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)

    windowed_size = WINDOW_SIZE
    fullscreen = False

    def request_window_state(is_fullscreen):
        nonlocal fullscreen
        fullscreen = is_fullscreen
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

    request_window_state(False)
    base.graphicsEngine.renderFrame()
    base.graphicsEngine.renderFrame()

    def report_window_size(task):
        print(
            f"[Window] requested {windowed_size[0]}x{windowed_size[1]}; "
            f"actual {base.win.get_x_size()}x{base.win.get_y_size()} "
            f"at {tuple(window.getOrigin())}"
        )
        return Task.done

    base.taskMgr.doMethodLater(0.5, report_window_size, "report-window-size")
    window.fps_counter.enabled = False
    window.entity_counter.enabled = False
    window.collider_counter.enabled = False

    for font_path in (
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.exists(font_path):
            Text.default_font = font_path
            break

    camera_feed = CameraFeedUI(
        320,
        240,
    )
    system_monitor = SystemMonitor()
    tracker = None

    def make_pointcloud_mode():
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Point-cloud source not found: {target_path}")
        return PointCloudMode(
            target_path, camera_feed, point_cloud, downsample_step, tracker,
            system_monitor,
        )

    def make_object_mode():
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Object model not found: {model_path}")
        return ObjectMode(model_path, tracker, app, camera_feed, system_monitor)

    manager = ModeManager(
        {"pointcloud": make_pointcloud_mode, "object": make_object_mode},
    )
    manager.start(initial_mode)
    tracker = HandTracker(
        classify_hand=manager.classify_hand,
        annotate_frame=manager.annotate_frame,
        detection_confidence=0.7,
        tracking_confidence=0.5,
        camera_index=config.CAMERA_INDEX,
        include_landmarks=manager.active.include_landmarks,
    )
    manager.set_tracker(tracker)

    def update():
        manager.update(time.dt)

    def input(key):
        if key in ("escape", "q"):
            mouse.locked = False
            application.quit()
        elif key == "f11":
            request_window_state(not fullscreen)
        else:
            manager.input(key)

    import sys
    main_module = sys.modules["__main__"]
    main_module.update = update
    main_module.input = input

    try:
        app.run()
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
