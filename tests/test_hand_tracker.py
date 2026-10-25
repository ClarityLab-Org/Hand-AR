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


def test_hand_locking_behavior():
    """Verifies that hand locking locks onto initial hands, rejects intruders,
    and only allows a new hand when an existing hand is removed."""
    class FakeLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class FakeHandLandmarks:
        def __init__(self, x, y):
            self.landmark = [FakeLandmark(x, y)] * 21

    class FakeClassification:
        def __init__(self, label):
            self.label = label

    class FakeClassificationList:
        def __init__(self, label):
            self.classification = [FakeClassification(label)]

    class FakeResults:
        def __init__(self, hands_data):
            self.multi_hand_landmarks = [FakeHandLandmarks(x, y) for x, y, _ in hands_data]
            self.multi_handedness = [FakeClassificationList(label) for _, _, label in hands_data]

    tracker = tracker_module.HandTracker.__new__(tracker_module.HandTracker)
    tracker._start_time = time.monotonic()
    tracker._lock_window_seconds = 5.0
    tracker._lock_grace_frames = 15
    tracker._max_jump_distance = 0.45
    tracker._locked_slots = {
        'Left': {'locked': False, 'last_pos': None, 'lost_frames': 0},
        'Right': {'locked': False, 'last_pos': None, 'lost_frames': 0},
    }

    # 1. When both hands appear within initial window, both are recognized and locked
    res_both = FakeResults([(0.20, 0.50, 'Left'), (0.80, 0.50, 'Right')])
    matched, lms, _ = tracker._match_and_lock_hands(res_both)
    assert set(matched.keys()) == {'Left', 'Right'}
    assert tracker._locked_slots['Left']['locked'] is True
    assert tracker._locked_slots['Right']['locked'] is True
    assert len(lms) == 2

    # 2. When an intruder hand appears later, it is rejected and not recognized
    res_with_intruder = FakeResults([
        (0.21, 0.51, 'Left'),
        (0.79, 0.49, 'Right'),
        (0.45, 0.20, 'Right'),  # Intruder hand
    ])
    matched_intruder, lms_intruder, _ = tracker._match_and_lock_hands(res_with_intruder)
    assert len(matched_intruder) == 2
    assert len(lms_intruder) == 2
    # Verify the locked Right slot matched the original user hand, not the intruder
    assert matched_intruder['Right']['pos'] == (0.79, 0.49)

    # 3. Only when any one hand is removed, then a new hand will be detected to take its place
    # Remove Left hand for 15 frames
    for _ in range(15):
        res_no_left = FakeResults([(0.80, 0.50, 'Right')])
        tracker._match_and_lock_hands(res_no_left)

    assert tracker._locked_slots['Left']['locked'] is False, "Left slot must unlock after removal"
    assert tracker._locked_slots['Right']['locked'] is True, "Right hand must remain locked"

    # Now a new Left hand appears and takes its place
    res_new_left = FakeResults([(0.80, 0.50, 'Right'), (0.35, 0.55, 'Left')])
    matched_new, lms_new, _ = tracker._match_and_lock_hands(res_new_left)
    assert 'Left' in matched_new
    assert matched_new['Left']['pos'] == (0.35, 0.55)
    assert tracker._locked_slots['Left']['locked'] is True
    assert tracker._locked_slots['Right']['locked'] is True
