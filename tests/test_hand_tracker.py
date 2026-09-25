"""Focused lifecycle and failure tests for the threaded hand tracker."""

import time

import pytest

pytest.importorskip("cv2")
pytest.importorskip("mediapipe")
tracker_module = pytest.importorskip("handarm.tracking.hand_tracker")


class FakeCapture:
    def __init__(self, read_result):
        self.read_result = read_result
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return self.read_result

    def release(self):
        self.released = True


class FailingHands:
    def process(self, _frame):
        raise RuntimeError("test MediaPipe failure")

    def close(self):
        pass


class FakeHandsModule:
    def __init__(self, hands):
        self.hands = hands

    def Hands(self, **_kwargs):
        return self.hands


def _wait_for_status(tracker, predicate):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        if predicate(tracker.get_status()):
            return
        time.sleep(0.01)
    pytest.fail(f"status did not reach expected value: {tracker.get_status()}")


def test_camera_read_failure_is_reported_and_shutdown_is_clean(monkeypatch):
    capture = FakeCapture((False, None))
    monkeypatch.setattr(tracker_module.cv2, "VideoCapture", lambda _index: capture)
    monkeypatch.setattr(
        tracker_module.mp,
        "solutions",
        type(
            "Solutions",
            (),
            {"hands": FakeHandsModule(FailingHands()), "drawing_utils": object()},
        ),
        raising=False,
    )

    tracker = tracker_module.HandTracker(lambda *_args: {"gesture": "None"})
    try:
        _wait_for_status(
            tracker,
            lambda status: status["error"] == "Camera read failed",
        )
        assert tracker.get_status()["running"] is True
    finally:
        tracker.stop()

    assert capture.released
    assert tracker.get_status()["running"] is False


def test_mediapipe_exception_is_reported_without_killing_thread(monkeypatch):
    capture = FakeCapture((True, object()))
    monkeypatch.setattr(tracker_module.cv2, "VideoCapture", lambda _index: capture)
    monkeypatch.setattr(
        tracker_module.mp,
        "solutions",
        type(
            "Solutions",
            (),
            {"hands": FakeHandsModule(FailingHands()), "drawing_utils": object()},
        ),
        raising=False,
    )
    monkeypatch.setattr(tracker_module.cv2, "flip", lambda image, _axis: image)
    monkeypatch.setattr(tracker_module.cv2, "cvtColor", lambda image, _code: image)

    tracker = tracker_module.HandTracker(lambda *_args: {"gesture": "None"})
    try:
        _wait_for_status(
            tracker,
            lambda status: status["error"]
            == "MediaPipe processing failed: test MediaPipe failure",
        )
        assert tracker._thread.is_alive()
    finally:
        tracker.stop()
