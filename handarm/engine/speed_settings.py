"""Persistent per-gesture speed multipliers for explore mode."""

import json
from pathlib import Path
from typing import Dict

from handarm.tracking.gestures import GESTURE_LABELS


GESTURE_SPEED_DEFAULTS: Dict[str, float] = {
    gesture: 1.0 for gesture in GESTURE_LABELS if gesture != "None"
}
SETTINGS_PATH = Path.home() / ".hand_ar_gesture_speeds.json"


def load_gesture_speeds() -> Dict[str, float]:
    """Load saved multipliers, ignoring malformed or unknown entries."""
    values = dict(GESTURE_SPEED_DEFAULTS)
    try:
        with SETTINGS_PATH.open(encoding="utf-8") as handle:
            saved = json.load(handle)
    except FileNotFoundError:
        return values
    except (OSError, ValueError) as exc:
        print(f"Unable to load gesture speeds: {exc}")
        return values
    if not isinstance(saved, dict):
        return values
    for gesture in values:
        try:
            value = float(saved.get(gesture, values[gesture]))
        except (TypeError, ValueError):
            continue
        values[gesture] = max(0.25, min(2.0, value))
    return values


def save_gesture_speeds(values: Dict[str, float]) -> None:
    """Persist only known, bounded speed multipliers."""
    payload = {
        gesture: max(0.25, min(2.0, float(values.get(gesture, 1.0))))
        for gesture in GESTURE_SPEED_DEFAULTS
    }
    try:
        with SETTINGS_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except OSError as exc:
        print(f"Unable to save gesture speeds: {exc}")


def reset_gesture_speeds() -> Dict[str, float]:
    """Restore defaults and persist them."""
    values = dict(GESTURE_SPEED_DEFAULTS)
    save_gesture_speeds(values)
    return values
