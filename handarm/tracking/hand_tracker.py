"""Thread-safe hand tracker using MediaPipe."""

import copy
import math
import threading
import time
from typing import Any, Callable, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    """Threaded MediaPipe hand tracker with configurable gesture classification.

    Runs camera capture and MediaPipe hand detection in a background daemon thread.
    State is accessed thread-safely via get_state() and get_frame().

    Args:
        classify_hand: Function(landmarks, hand_type) -> dict with at minimum
            {'gesture': str} and any additional feature booleans.
        annotate_frame: Optional function(frame_rgb, results, state) -> frame_rgb
            for drawing on the camera frame. Called in the background thread.
        detection_confidence: MediaPipe min_detection_confidence.
        max_hands: Maximum number of hands to detect.
        camera_index: OpenCV VideoCapture device index.
    """

    def __init__(
        self,
        classify_hand: Callable,
        annotate_frame: Optional[Callable] = None,
        detection_confidence: float = 0.7,
        max_hands: int = 2,
        camera_index: int = 0,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
        )
        self._cap = cv2.VideoCapture(camera_index)
        self._classify_hand = classify_hand
        self._annotate_frame = annotate_frame

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._state: Dict[str, Dict[str, Any]] = {
            'Left': {'visible': False, 'x': 0.5, 'y': 0.5, 'gesture': 'None'},
            'Right': {'visible': False, 'x': 0.5, 'y': 0.5, 'gesture': 'None'},
        }
        self._gesture_candidates = {'Left': None, 'Right': None}
        self._gesture_candidate_counts = {'Left': 0, 'Right': 0}
        self._owner_label: Optional[str] = None
        self._owner_candidate_label: Optional[str] = None
        self._owner_candidate_anchor = None
        self._owner_candidate_scale: Optional[float] = None
        self._owner_candidate_since: Optional[float] = None
        self._owner_last_seen = time.monotonic()
        self._owner_anchor = None
        self._owner_scale: Optional[float] = None
        self._latest_frame: Optional[np.ndarray] = None
        camera_connected = bool(getattr(self._cap, 'isOpened', lambda: True)())
        self._status: Dict[str, Any] = {
            'running': True,
            'camera_connected': camera_connected,
            'error': None if camera_connected else 'Camera is not available',
            'last_error': None if camera_connected else 'Camera is not available',
            'owner_status': 'Acquiring...',
            'owner_hand': None,
        }
        self._shutdown_lock = threading.Lock()
        self._resources_closed = False

        self.running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def get_state(self) -> Dict[str, Dict[str, Any]]:
        """Returns a thread-safe deep copy of current hand state."""
        with self._lock:
            return copy.deepcopy(self._state)

    def get_frame(self) -> Optional[np.ndarray]:
        """Returns a thread-safe copy of the latest annotated camera frame."""
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def get_status(self) -> Dict[str, Any]:
        """Returns a thread-safe copy of camera and worker health information."""
        with self._lock:
            return copy.deepcopy(self._status)

    def _set_status(self, **updates: Any) -> None:
        with self._lock:
            self._status.update(updates)

    @staticmethod
    def _hand_signature(hand_lms: Any) -> tuple:
        landmarks = hand_lms.landmark
        xs = [point.x for point in landmarks]
        ys = [point.y for point in landmarks]
        scale = max(max(xs) - min(xs), max(ys) - min(ys), 0.01)
        return (landmarks[0].x, landmarks[0].y, scale)

    def _update_owner(self, detections: Dict[str, tuple], now: float) -> Optional[str]:
        """Acquire one stable hand, then retain it by label and continuity."""
        if self._owner_label is not None:
            if self._owner_label in detections:
                wrist_x, wrist_y, scale = detections[self._owner_label]
                anchor_x, anchor_y = self._owner_anchor or (wrist_x, wrist_y)
                base_scale = self._owner_scale or scale
                continuous = (
                    math.dist((wrist_x, wrist_y), (anchor_x, anchor_y))
                    <= self.OWNER_TRACK_POSITION_TOLERANCE
                    and abs(scale - base_scale) / max(base_scale, 0.01)
                    <= self.OWNER_TRACK_SCALE_TOLERANCE
                )
                if continuous:
                    self._owner_anchor = (wrist_x, wrist_y)
                    self._owner_scale = scale
                    self._owner_last_seen = now
                    self._set_status(owner_status='Locked', owner_hand=self._owner_label)
                    return self._owner_label
            if now - self._owner_last_seen > self.OWNER_LOSS_TIMEOUT_SECONDS:
                self._owner_label = None
                self._owner_candidate_label = None
                self._owner_candidate_since = None
                self._owner_candidate_anchor = None
                self._owner_candidate_scale = None
                self._owner_anchor = None
                self._owner_scale = None
                self._set_status(owner_status='Lost', owner_hand=None)
            else:
                self._set_status(owner_status='Lost', owner_hand=None)
            return None

        if not detections:
            self._owner_candidate_label = None
            self._owner_candidate_since = None
            self._set_status(owner_status='Acquiring...', owner_hand=None)
            return None

        # Keep the first stable handedness candidate; a second hand cannot take
        # control during acquisition unless the candidate disappears.
        label = self._owner_candidate_label
        if label not in detections:
            label = next(iter(detections))
            wrist_x, wrist_y, scale = detections[label]
            self._owner_candidate_label = label
            self._owner_candidate_anchor = (wrist_x, wrist_y)
            self._owner_candidate_scale = scale
            self._owner_candidate_since = now
        else:
            wrist_x, wrist_y, scale = detections[label]
            anchor_x, anchor_y = self._owner_candidate_anchor
            base_scale = self._owner_candidate_scale or scale
            stable = (
                math.dist((wrist_x, wrist_y), (anchor_x, anchor_y))
                <= self.OWNER_POSITION_TOLERANCE
                and abs(scale - base_scale) / max(base_scale, 0.01)
                <= self.OWNER_SCALE_TOLERANCE
            )
            if not stable:
                self._owner_candidate_anchor = (wrist_x, wrist_y)
                self._owner_candidate_scale = scale
                self._owner_candidate_since = now

        candidate_since = self._owner_candidate_since or now
        if now - candidate_since >= self.OWNER_LOCK_SECONDS:
            self._owner_label = self._owner_candidate_label
            self._owner_last_seen = now
            self._owner_anchor = detections[self._owner_label][:2]
            self._owner_scale = detections[self._owner_label][2]
            self._set_status(owner_status='Locked', owner_hand=self._owner_label)
            return self._owner_label
        self._set_status(owner_status='Acquiring...', owner_hand=None)
        return None

    def _update_loop(self) -> None:
        """Background thread: continuously captures frames and classifies gestures."""
        try:
            while not self._stop_event.is_set():
                try:
                    success, img = self._cap.read()
                    if not success:
                        self._set_status(
                            camera_connected=False,
                            error='Camera read failed',
                            last_error='Camera read failed',
                        )
                        self._stop_event.wait(0.05)
                        continue

                    self._set_status(camera_connected=True, error=None)
                    img = cv2.flip(img, 1)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    try:
                        results = self._hands.process(img_rgb)
                    except Exception as exc:
                        message = f'MediaPipe processing failed: {exc}'
                        self._set_status(error=message, last_error=message)
                        self._stop_event.wait(0.05)
                        continue

                    new_state: Dict[str, Dict[str, Any]] = {
                        'Left': {'visible': False, 'x': 0.5, 'y': 0.5, 'gesture': 'None'},
                        'Right': {'visible': False, 'x': 0.5, 'y': 0.5, 'gesture': 'None'},
                    }

                    visible_hands = set()
                    detections = {}
                    if results.multi_hand_landmarks:
                        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                            hand_type = results.multi_handedness[idx].classification[0].label
                            visible_hands.add(hand_type)
                            wrist_x, wrist_y, scale = self._hand_signature(hand_lms)
                            detections[hand_type] = (wrist_x, wrist_y, scale)
                    owner_label = self._update_owner(detections, time.monotonic())
                    classified_hands: Dict[str, Dict[str, Any]] = {}
                    if results.multi_hand_landmarks:
                        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                            hand_type = results.multi_handedness[idx].classification[0].label
                            # Inspect both hands only to recognize the explicit
                            # paired-palm command; normal gestures remain owner-only.
                            if owner_label != hand_type and len(detections) != 2:
                                continue
                            hand_data = self._classify_hand(hand_lms.landmark, hand_type)
                            classified_hands[hand_type] = hand_data
                            if owner_label != hand_type:
                                continue
                            gesture = hand_data.get('gesture', 'None')
                            if gesture == self._gesture_candidates[hand_type]:
                                self._gesture_candidate_counts[hand_type] += 1
                            else:
                                self._gesture_candidates[hand_type] = gesture
                                self._gesture_candidate_counts[hand_type] = 1
                            previous = self._state[hand_type].get('gesture', 'None')
                            if self._gesture_candidate_counts[hand_type] < 2:
                                hand_data['gesture'] = previous
                                hand_data['navigation'] = self._state[hand_type].get(
                                    'navigation', '🖐️ Neutral / Idle'
                                )
                            hand_data['visible'] = True
                            hand_data['x'] = hand_lms.landmark[0].x
                            hand_data['y'] = hand_lms.landmark[0].y
                            new_state[hand_type] = hand_data

                    paired_palms = (
                        owner_label is not None
                        and len(classified_hands) == 2
                        and all(data.get('open_palm', False)
                                for data in classified_hands.values())
                    )
                    if paired_palms:
                        right_lms = next(
                            hand_lms for idx, hand_lms in enumerate(results.multi_hand_landmarks)
                            if results.multi_handedness[idx].classification[0].label == 'Right'
                        )
                        new_state['Right'] = {
                            'visible': True,
                            'x': right_lms.landmark[0].x,
                            'y': right_lms.landmark[0].y,
                            'gesture': 'Move Up',
                            'navigation': '👐 Move Up',
                        }
                        if owner_label == 'Left':
                            new_state['Left']['gesture'] = 'None'

                    for hand_type in ('Left', 'Right'):
                        if hand_type not in visible_hands:
                            self._gesture_candidates[hand_type] = None
                            self._gesture_candidate_counts[hand_type] = 0

                    if self._annotate_frame is not None:
                        img_rgb = self._annotate_frame(img_rgb, results, new_state)

                    with self._lock:
                        self._state = new_state
                        self._latest_frame = img_rgb
                    self._stop_event.wait(0.01)
                except Exception as exc:
                    # Keep the daemon alive and make unexpected frame errors observable.
                    message = f'Hand tracking failed: {exc}'
                    self._set_status(error=message, last_error=message)
                    self._stop_event.wait(0.05)
        finally:
            self._set_status(running=False)

    def stop(self) -> None:
        """Stops background thread and releases camera."""
        with self._shutdown_lock:
            self.running = False
            self._stop_event.set()
            if self._thread.is_alive() and threading.current_thread() is not self._thread:
                self._thread.join(timeout=1.0)
            if not self._resources_closed:
                try:
                    self._cap.release()
                finally:
                    try:
                        self._hands.close()
                    finally:
                        self._resources_closed = True
            self._set_status(running=False)
    OWNER_LOCK_SECONDS = 2.5
    OWNER_LOSS_TIMEOUT_SECONDS = 2.0
    OWNER_POSITION_TOLERANCE = 0.12
    OWNER_SCALE_TOLERANCE = 0.35
    OWNER_TRACK_POSITION_TOLERANCE = 0.30
    OWNER_TRACK_SCALE_TOLERANCE = 0.60
