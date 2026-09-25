"""Thread-safe hand tracker using MediaPipe."""

import copy
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
        self._state: Dict[str, Dict[str, Any]] = {
            'Left': {'visible': False, 'x': 0.5, 'y': 0.5, 'gesture': 'None'},
            'Right': {'visible': False, 'x': 0.5, 'y': 0.5, 'gesture': 'None'},
        }
        self._latest_frame: Optional[np.ndarray] = None

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

    def _update_loop(self) -> None:
        """Background thread: continuously captures frames and classifies gestures."""
        while self.running:
            success, img = self._cap.read()
            if not success:
                continue

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self._hands.process(img_rgb)

            new_state: Dict[str, Dict[str, Any]] = {
                'Left': {'visible': False, 'x': 0.5, 'y': 0.5, 'gesture': 'None'},
                'Right': {'visible': False, 'x': 0.5, 'y': 0.5, 'gesture': 'None'},
            }

            if results.multi_hand_landmarks:
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    hand_type = results.multi_handedness[idx].classification[0].label
                    hand_data = self._classify_hand(hand_lms.landmark, hand_type)
                    hand_data['visible'] = True
                    hand_data['x'] = hand_lms.landmark[0].x
                    hand_data['y'] = hand_lms.landmark[0].y
                    new_state[hand_type] = hand_data

            # Annotate frame (optional callback)
            if self._annotate_frame is not None:
                img_rgb = self._annotate_frame(img_rgb, results, new_state)

            with self._lock:
                self._state = new_state
                self._latest_frame = img_rgb

            time.sleep(0.01)

    def stop(self) -> None:
        """Stops background thread and releases camera."""
        self.running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cap.release()
        self._hands.close()
