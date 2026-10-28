"""Thread-safe hand tracker using MediaPipe."""

import copy
import threading
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
        lock_window_seconds: Initial time window in seconds during which the
            system locks onto the user's hands.
    """

    def __init__(
        self,
        classify_hand: Callable,
        annotate_frame: Optional[Callable] = None,
        detection_confidence: float = 0.7,
        max_hands: int = 2,
        camera_index: int = 0,
        tracking_confidence: Optional[float] = None,
        include_landmarks: bool = False,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        hands_options = {
            "max_num_hands": max_hands,
            "min_detection_confidence": detection_confidence,
        }
        if tracking_confidence is not None:
            hands_options["min_tracking_confidence"] = tracking_confidence
        self._hands = self._mp_hands.Hands(
            **hands_options,
        )
        self._cap = cv2.VideoCapture(camera_index)
        self._classify_hand = classify_hand
        self._annotate_frame = annotate_frame
        self._include_landmarks = include_landmarks
        self._stabilize_gestures = True

        # Handlocking configuration and spatial tracking state
        self._lock_window_seconds: float = lock_window_seconds
        self._start_time: float = time.monotonic()
        self._lock_grace_frames: int = 15  # ~0.5s before declaring a lost hand removed
        self._max_jump_distance: float = 0.45  # Max normalized movement per frame
        self._locked_slots: Dict[str, Dict[str, Any]] = {
            'Left': {'locked': False, 'last_pos': None, 'lost_frames': 0},
            'Right': {'locked': False, 'last_pos': None, 'lost_frames': 0},
        }

        self._lock = threading.Lock()
        self._classifier_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._state: Dict[str, Dict[str, Any]] = {
            'Left': {'visible': False, 'x': 0.5, 'y': 0.5, 'gesture': 'None'},
            'Right': {'visible': False, 'x': 0.5, 'y': 0.5, 'gesture': 'None'},
        }
        self._gesture_candidates = {'Left': None, 'Right': None}
        self._gesture_candidate_counts = {'Left': 0, 'Right': 0}
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id = 0
        camera_connected = bool(getattr(self._cap, 'isOpened', lambda: True)())
        self._status: Dict[str, Any] = {
            'running': True,
            'camera_connected': camera_connected,
            'error': None if camera_connected else 'Camera is not available',
            'last_error': None if camera_connected else 'Camera is not available',
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

    def get_snapshot(self):
        """Return one consistent hand state, annotated frame, and frame number."""
        with self._lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
            return copy.deepcopy(self._state), frame, self._frame_id

    def set_classifier(
        self, classify_hand: Callable, stabilize_gestures: bool,
        include_landmarks: Optional[bool] = None,
    ) -> None:
        """Switch the active mode's classifier without restarting MediaPipe or the camera."""
        with self._classifier_lock:
            self._classify_hand = classify_hand
            self._stabilize_gestures = stabilize_gestures
            if include_landmarks is not None:
                self._include_landmarks = include_landmarks
            self._gesture_candidates = {'Left': None, 'Right': None}
            self._gesture_candidate_counts = {'Left': 0, 'Right': 0}

    def get_status(self) -> Dict[str, Any]:
        """Returns a thread-safe copy of camera and worker health information."""
        with self._lock:
            status = copy.deepcopy(self._status)
            status['hand_locking'] = {
                'Left': self._locked_slots['Left']['locked'],
                'Right': self._locked_slots['Right']['locked'],
                'all_locked': (
                    self._locked_slots['Left']['locked']
                    and self._locked_slots['Right']['locked']
                ),
                'in_initial_window': (
                    time.monotonic() - self._start_time <= self._lock_window_seconds
                ),
            }
            return status

    def is_locked(self, hand_type: Optional[str] = None) -> bool:
        """Returns True if the specified hand (or both hands if None) are locked."""
        with self._lock:
            if hand_type:
                return bool(self._locked_slots.get(hand_type, {}).get('locked', False))
            return bool(
                self._locked_slots.get('Left', {}).get('locked', False)
                and self._locked_slots.get('Right', {}).get('locked', False)
            )

    def reset_hand_lock(self) -> None:
        """Resets the hand lock state and restarts the 5-second initial lock window."""
        with self._lock:
            self._start_time = time.monotonic()
            for slot in ('Left', 'Right'):
                self._locked_slots[slot] = {
                    'locked': False,
                    'last_pos': None,
                    'lost_frames': 0,
                }

    def _set_status(self, **updates: Any) -> None:
        with self._lock:
            self._status.update(updates)

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

                    matched_slots, filtered_lms, filtered_handedness = self._match_and_lock_hands(results)

                    new_state: Dict[str, Dict[str, Any]] = {
                        'Left': {
                            'visible': False,
                            'x': 0.5,
                            'y': 0.5,
                            'gesture': 'None',
                            'locked': self._locked_slots['Left']['locked'],
                        },
                        'Right': {
                            'visible': False,
                            'x': 0.5,
                            'y': 0.5,
                            'gesture': 'None',
                            'locked': self._locked_slots['Right']['locked'],
                        },
                    }

                    visible_hands = set()
                    if results.multi_hand_landmarks:
                        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                            hand_type = results.multi_handedness[idx].classification[0].label
                            visible_hands.add(hand_type)
                            with self._classifier_lock:
                                classify_hand = self._classify_hand
                                stabilize_gestures = self._stabilize_gestures
                                include_landmarks = self._include_landmarks
                            hand_data = classify_hand(hand_lms.landmark, hand_type)
                            if include_landmarks:
                                hand_data['landmarks'] = tuple(
                                    (point.x, point.y, point.z)
                                    for point in hand_lms.landmark
                                )
                            gesture = hand_data.get('gesture', 'None')
                            if stabilize_gestures:
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

                    for hand_type in ('Left', 'Right'):
                        if hand_type not in matched_slots:
                            self._gesture_candidates[hand_type] = None
                            self._gesture_candidate_counts[hand_type] = 0

                    if self._annotate_frame is not None:
                        # Expose only the matched/locked hands to annotation to ignore intruder hands
                        if results is not None:
                            try:
                                results.multi_hand_landmarks = filtered_lms if filtered_lms else None
                                results.multi_handedness = filtered_handedness if filtered_handedness else None
                            except Exception:
                                pass
                        img_rgb = self._annotate_frame(img_rgb, results, new_state)

                    with self._lock:
                        self._state = new_state
                        self._latest_frame = img_rgb
                        self._frame_id += 1
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
