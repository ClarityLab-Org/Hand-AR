"""Thread-safe hand tracker using MediaPipe."""

import copy
import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        lock_window_seconds: float = 5.0,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._max_hands = max_hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=max(4, max_hands),
            min_detection_confidence=detection_confidence,
        )
        self._cap = cv2.VideoCapture(camera_index)
        self._classify_hand = classify_hand
        self._annotate_frame = annotate_frame

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

    def _match_and_lock_hands(
        self, results: Any
    ) -> Tuple[Dict[str, Any], List[Any], List[Any]]:
        """Matches detected hands to locked Left and Right slots.

        Rules:
        1. Within initial 5-second window, detects hands. When both hands are recognized,
           both slots are locked onto.
        2. Once locked, the system tracks the locked hands based on spatial proximity.
           Any other/intruder hands that appear later are ignored.
        3. Only when any one hand is removed (absent for grace period), its slot is unlocked
           so that a new hand can be detected to take its place. The other hand remains locked.
        """
        now = time.monotonic()
        in_initial_window = (now - self._start_time <= self._lock_window_seconds)

        candidates = []
        if (
            results is not None
            and getattr(results, 'multi_hand_landmarks', None)
            and getattr(results, 'multi_handedness', None)
        ):
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                try:
                    hand_type = results.multi_handedness[idx].classification[0].label
                except (IndexError, AttributeError):
                    hand_type = 'Right' if idx == 0 else 'Left'
                wrist = hand_lms.landmark[0]
                pos = (float(wrist.x), float(wrist.y))
                candidates.append({
                    'idx': idx,
                    'hand_type': hand_type,
                    'landmarks': hand_lms,
                    'pos': pos,
                    'handedness': results.multi_handedness[idx],
                })

        matched_slots: Dict[str, Any] = {}
        used_candidate_indices = set()

        # Step 1: Match already locked slots based on spatial proximity to last_pos
        for slot in ('Left', 'Right'):
            slot_info = self._locked_slots[slot]
            if slot_info['locked'] and slot_info['last_pos'] is not None:
                best_cand = None
                best_dist = float('inf')
                # Primary match: candidate with matching handedness
                for cand in candidates:
                    if cand['idx'] in used_candidate_indices:
                        continue
                    if cand['hand_type'] == slot:
                        d = math.hypot(
                            cand['pos'][0] - slot_info['last_pos'][0],
                            cand['pos'][1] - slot_info['last_pos'][1],
                        )
                        if d < best_dist and d <= self._max_jump_distance:
                            best_dist = d
                            best_cand = cand

                # Secondary match: hand close to last_pos despite potential label flip
                if best_cand is None:
                    for cand in candidates:
                        if cand['idx'] in used_candidate_indices:
                            continue
                        d = math.hypot(
                            cand['pos'][0] - slot_info['last_pos'][0],
                            cand['pos'][1] - slot_info['last_pos'][1],
                        )
                        if d < 0.15 and d < best_dist:
                            best_dist = d
                            best_cand = cand

                if best_cand is not None:
                    matched_slots[slot] = best_cand
                    used_candidate_indices.add(best_cand['idx'])
                    slot_info['last_pos'] = best_cand['pos']
                    slot_info['lost_frames'] = 0

        # Step 2: For unlocked slots, detect candidate hands
        # (This handles initial acquisition OR a new hand replacing a removed hand)
        for slot in ('Left', 'Right'):
            slot_info = self._locked_slots[slot]
            if not slot_info['locked'] and slot not in matched_slots:
                best_cand = None
                best_dist = float('inf')
                for cand in candidates:
                    if cand['idx'] in used_candidate_indices:
                        continue
                    if cand['hand_type'] == slot:
                        if slot_info['last_pos'] is not None:
                            d = math.hypot(
                                cand['pos'][0] - slot_info['last_pos'][0],
                                cand['pos'][1] - slot_info['last_pos'][1],
                            )
                        else:
                            d = 0.0
                        if d < best_dist:
                            best_dist = d
                            best_cand = cand

                if best_cand is not None:
                    matched_slots[slot] = best_cand
                    used_candidate_indices.add(best_cand['idx'])
                    slot_info['last_pos'] = best_cand['pos']
                    slot_info['lost_frames'] = 0

        # Step 3: Check locking conditions
        # If both hands are recognized, immediately lock both!
        if 'Left' in matched_slots and 'Right' in matched_slots:
            self._locked_slots['Left']['locked'] = True
            self._locked_slots['Right']['locked'] = True
        elif not in_initial_window:
            # Beyond initial 5-second window, any active hand is locked
            for slot in matched_slots:
                self._locked_slots[slot]['locked'] = True

        # Step 4: Handle lost frames and hand removal
        # "Only when any one hand is removed, then a new hand will be detected to take its place"
        for slot in ('Left', 'Right'):
            if slot not in matched_slots:
                self._locked_slots[slot]['lost_frames'] += 1
                if self._locked_slots[slot]['lost_frames'] >= self._lock_grace_frames:
                    # Hand confirmed removed! Unlock this slot so a new hand can take its place
                    self._locked_slots[slot]['locked'] = False
                    self._locked_slots[slot]['last_pos'] = None
                    self._locked_slots[slot]['lost_frames'] = 0

        # Step 5: Gather filtered landmarks and handedness
        # (Any extra/intruder hand not matched to Left or Right is omitted and ignored)
        filtered_lms = []
        filtered_handedness = []
        for slot in ('Left', 'Right'):
            if slot in matched_slots:
                filtered_lms.append(matched_slots[slot]['landmarks'])
                filtered_handedness.append(matched_slots[slot]['handedness'])

        return matched_slots, filtered_lms, filtered_handedness

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

                    for hand_type, cand in matched_slots.items():
                        hand_lms = cand['landmarks']
                        hand_data = self._classify_hand(hand_lms.landmark, hand_type)
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
                        hand_data['locked'] = self._locked_slots[hand_type]['locked']
                        hand_data['x'] = cand['pos'][0]
                        hand_data['y'] = cand['pos'][1]
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
