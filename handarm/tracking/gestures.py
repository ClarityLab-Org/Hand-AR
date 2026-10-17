"""Pure gesture classification functions for Hand-ArM2.

No Ursina, Panda3D, or OpenCV dependencies — these operate on
landmark coordinate data only, making them independently testable.
"""

import math
from typing import Any, Dict, List

GESTURE_LABELS = {
    "None": "🖐️ Neutral / Hover",
    "Rotate Up": "☝️ Rotate Up",
    "Rotate Down": "👇 Rotate Down",
    "Rotate Left": "👈 Rotate Left",
    "Rotate Right": "👉 Rotate Right",
    "Move Forward": "🫴 Move Forward",
    "Move Backward": "🫳 Move Backward",
    "Move Left": "🤜 Move Left",
    "Move Right": "🤛 Move Right",
    "Reset View": "👍 Reset View",
    "Zoom In": "🤏 Zoom In",
    "Zoom Out": "✌️ Zoom Out",
}


def gesture_label(gesture: str, landmarks: List[Any] = None) -> str:
    """Return the emoji label used by the explore HUD for a classified gesture."""
    return GESTURE_LABELS.get(gesture, f"🖐️ {gesture}")


# ---------------------------------------------------------------------------
# Inspect-mode gestures (used by trial2.py in inspect mode)
# ---------------------------------------------------------------------------

def is_open_palm_relaxed(landmarks: List[Any]) -> bool:
    """Checks if hand landmarks match a relaxed open palm gesture.

    A relaxed open palm requires at least 2 of 3 fingertips (index, middle, ring)
    to be above their corresponding base joints by more than 0.01.

    Args:
        landmarks: List of 21 MediaPipe hand landmarks with .x, .y attributes.
    """
    tips, bases = [8, 12, 16], [5, 9, 13]
    return sum(landmarks[t].y < landmarks[b].y - 0.01 for t, b in zip(tips, bases)) >= 2


def is_peace(landmarks: List[Any]) -> bool:
    """Checks if hand landmarks match a peace sign gesture.

    Peace = index finger up AND middle finger up AND ring finger down.
    """
    return (landmarks[8].y < landmarks[6].y and
            landmarks[12].y < landmarks[10].y and
            landmarks[16].y > landmarks[14].y)


def pinch_distance(landmarks: List[Any]) -> float:
    """Calculates Euclidean distance between thumb tip (4) and index tip (8)."""
    return math.dist(
        (landmarks[4].x, landmarks[4].y),
        (landmarks[8].x, landmarks[8].y)
    )


def is_pinch(landmarks: List[Any]) -> bool:
    """Returns True if thumb and index fingertips are pinched (distance < 0.05)."""
    return pinch_distance(landmarks) < 0.05

# ---------------------------------------------------------------------------
# Explore-mode gesture feature computation (shared)
# ---------------------------------------------------------------------------

def compute_explore_features(landmarks: List[Any]) -> Dict[str, Any]:
    """Computes all raw gesture features from landmarks for explore mode.
    
    Returns a dict with all boolean features and distance measurements
    needed by the classification functions.
    """
    # Compute finger-up/down booleans
    ring_up = landmarks[16].y < landmarks[14].y
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y
    pinky_down = landmarks[20].y > landmarks[18].y

    wrist = landmarks[0]
    wrist_y = wrist.y

    # Distance from each fingertip to wrist
    thumb_dist = ((landmarks[4].x - wrist.x)**2 + (landmarks[4].y - wrist.y)**2) ** 0.5
    index_dist = ((landmarks[8].x - wrist.x)**2 + (landmarks[8].y - wrist.y)**2) ** 0.5
    middle_dist = ((landmarks[12].x - wrist.x)**2 + (landmarks[12].y - wrist.y)**2) ** 0.5
    ring_dist = ((landmarks[16].x - wrist.x)**2 + (landmarks[16].y - wrist.y)**2) ** 0.5
    pinky_dist = ((landmarks[20].x - wrist.x)**2 + (landmarks[20].y - wrist.y)**2) ** 0.5

    peace = index_up and middle_up and ring_down and pinky_down
    # Allow one imperfectly detected fingertip; MediaPipe landmarks can jitter
    # briefly while the hand is moving.
    open_palm = sum((index_up, middle_up, ring_up, pinky_up)) >= 3
    fist = not index_up and not middle_up and not ring_up and not pinky_up
    palm_down = (landmarks[8].y > wrist_y and landmarks[12].y > wrist_y and
                 landmarks[16].y > wrist_y and landmarks[20].y > wrist_y)
    # Simple index-up: only index extended (explore.py standalone variant)
    index_up_simple = index_up and not middle_up and not ring_up and not pinky_up

    return {
        'peace': peace,
        'open_palm': open_palm,
        'fist': fist,
        'palm_down': palm_down,
        'index_up_simple': index_up_simple,
        'wrist_y': wrist_y,
        'wrist_x': wrist.x,
        'pinch_distance': math.dist(
            (landmarks[4].x, landmarks[4].y),
            (landmarks[8].x, landmarks[8].y),
        ),
        'thumb_dist': thumb_dist,
        'index_dist': index_dist,
        'middle_dist': middle_dist,
        'ring_dist': ring_dist,
        'pinky_dist': pinky_dist,
        'landmarks': landmarks,
    }


def compute_index_up_distance(features: Dict) -> bool:
    """Distance-based index-up detection (trial2.py explore variant).
    Index must be significantly farther from wrist than other fingers AND above wrist."""
    return (features['index_dist'] > features['middle_dist'] + 0.05 and
            features['index_dist'] > features['ring_dist'] + 0.05 and
            features['index_dist'] > features['pinky_dist'] + 0.05 and
            features['landmarks'][8].y < features['wrist_y'])


def compute_index_down(features: Dict) -> bool:
    """Index-down detection (shared). Index significantly farther from wrist than others."""
    return (features['index_dist'] > features['middle_dist'] + 0.05 and
            features['index_dist'] > features['ring_dist'] + 0.05 and
            features['index_dist'] > features['pinky_dist'] + 0.05)


def compute_thumbs_up(features: Dict, margin: float = 0.08) -> bool:
    """Thumbs-up detection with configurable margin.
    Thumb must be farther from wrist than all other fingers by the margin."""
    return (features['thumb_dist'] > features['index_dist'] + margin and
            features['thumb_dist'] > features['middle_dist'] + margin and
            features['thumb_dist'] > features['ring_dist'] + margin and
            features['thumb_dist'] > features['pinky_dist'] + margin)


# ---------------------------------------------------------------------------
# Right hand classification (shared — identical in both variants)
# ---------------------------------------------------------------------------

def classify_right_explore(
    features: Dict,
    previous_pinch_distance: float = None,
) -> str:
    """Classify simple right-hand view controls from position and posture."""
    if compute_thumbs_up(features, margin=0.02):
        return 'Reset View'
    if features['pinch_distance'] < 0.065:
        return 'Zoom In'
    if features['peace']:
        return 'Zoom Out'
    return _directional_gesture(features, "Rotate")


# ---------------------------------------------------------------------------
# Left hand classification — TWO variants
# ---------------------------------------------------------------------------

def classify_left_explore_trial2(features: Dict) -> str:
    """Left-hand translation classifier used by the trial2 variant."""
    return _directional_gesture(features, "Move")


def classify_left_explore_standalone(features: Dict) -> str:
    """Classify four-direction movement and vertical flight."""
    return _directional_gesture(features, "Move")


def _directional_gesture(features: Dict, prefix: str) -> str:
    """Convert hand position into non-overlapping cardinal zones."""
    x = features['wrist_x']
    y = features['wrist_y']
    # The center cross is neutral. Forward/backward use a narrow vertical
    # corridor; left/right use a separate horizontal corridor. Corner
    # positions stay neutral instead of stealing a neighboring direction.
    center = 0.5
    half_zone = 0.20
    if prefix == "Move":
        # Give the full top and bottom bands to forward/backward. This
        # prevents a hand drifting sideways while raised from becoming a
        # left/right command.
        if y < center - 0.15:
            return "Move Forward"
        if y > center + 0.15:
            return "Move Backward"
        if center - half_zone <= x <= center + half_zone:
            return "None"
        if center - half_zone <= y <= center + half_zone:
            if x < center - 0.10:
                return "Move Left"
            if x > center + 0.10:
                return "Move Right"
        return 'None'
    # Use full-height top/bottom bands for vertical rotation. MediaPipe's
    # wrist x-coordinate often shifts while the hand is raised or lowered.
    if y < center - 0.15:
        return f"{prefix} Up"
    if y > center + 0.15:
        return f"{prefix} Down"
    if center - half_zone <= y <= center + half_zone:
        if x < center - 0.10:
            return f"{prefix} Left"
        if x > center + 0.10:
            return f"{prefix} Right"
    return 'None'
