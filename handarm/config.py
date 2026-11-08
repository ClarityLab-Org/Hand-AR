"""
Configuration module for Hand-ArM2 project.
Loads settings from .env file or uses defaults.
All defaults match the previously hardcoded values in the codebase.
"""

import os
from pathlib import Path
from typing import List

# Project root is the parent of the handarm/ package directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Try to load .env file
_env_file: Path = PROJECT_ROOT / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file, override=True)
    except ImportError:
        print(" python-dotenv not installed. Install with: pip install python-dotenv")
        print("  For now, using defaults. .env file will be ignored.")
else:
    print(f" No .env file found. Using defaults. Copy .env.example to .env to customize.")

# ---------------------------------------------------------------------------
# Camera Settings
# ---------------------------------------------------------------------------
CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
MIN_DETECTION_CONFIDENCE: float = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.8"))
MIN_TRACKING_CONFIDENCE: float = float(os.getenv("MIN_TRACKING_CONFIDENCE", "0.8"))

# ---------------------------------------------------------------------------
# Camera Feed Resolution
# ---------------------------------------------------------------------------
FEED_WIDTH: int = int(os.getenv("FEED_WIDTH", "320"))
FEED_HEIGHT: int = int(os.getenv("FEED_HEIGHT", "240"))

# ---------------------------------------------------------------------------
# Gesture Smoothing (currently informational — per-axis alphas are in engine)
# ---------------------------------------------------------------------------
SMOOTHING_ALPHA: float = float(os.getenv("SMOOTHING_ALPHA", "0.25"))

# ---------------------------------------------------------------------------
# Paths (always relative to project root, not CWD)
# ---------------------------------------------------------------------------
SCREENSHOT_DIR: str = os.getenv("SCREENSHOT_DIR", str(PROJECT_ROOT / "screenshots"))
MODELS_DIR: str = str(PROJECT_ROOT / "models")
MODELS_COMPRESSED_DIR: str = str(PROJECT_ROOT / "models_compressed")

# ---------------------------------------------------------------------------
# Sketchfab API
# ---------------------------------------------------------------------------
SKETCHFAB_TOKEN: str = os.getenv("SKETCHFAB_TOKEN", "")

# ---------------------------------------------------------------------------
# Point Cloud Rendering
# ---------------------------------------------------------------------------
def _load_point_size_from_env() -> float:
    """Robustly reads POINT_THICKNESS or POINT_SIZE from environment."""
    raw_pt = os.getenv("POINT_THICKNESS")
    raw_ps = os.getenv("POINT_SIZE")

    val_pt = None
    val_ps = None
    if raw_pt is not None:
        try:
            val_pt = float(raw_pt.strip())
        except ValueError:
            pass
    if raw_ps is not None:
        try:
            val_ps = float(raw_ps.strip())
        except ValueError:
            pass

    if val_pt is not None and val_ps is not None:
        # If both are set and differ, respect whichever was edited from 1.0 or pick val_pt
        if val_pt != val_ps:
            return val_pt if val_pt != 1.0 else val_ps
        return val_pt
    if val_pt is not None:
        return val_pt
    if val_ps is not None:
        return val_ps
    return 2.0


POINT_SIZE: float = _load_point_size_from_env()
POINT_THICKNESS: float = POINT_SIZE

# ---------------------------------------------------------------------------
# Navigation Speeds
# ---------------------------------------------------------------------------
WALK_SPEED: float = float(os.getenv("WALK_SPEED", "6.0"))
FLIGHT_SPEED: float = float(os.getenv("FLIGHT_SPEED", "16.0"))
VERTICAL_SPEED: float = float(os.getenv("VERTICAL_SPEED", "8.0"))
TURN_SPEED: float = float(os.getenv("TURN_SPEED", "75.0"))
PITCH_SPEED: float = float(os.getenv("PITCH_SPEED", "55.0"))

# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------
DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")


def validate_config() -> List[str]:
    """
    Validates loaded configuration values against allowed ranges and constraints.

    Returns:
        List of string error messages describing configuration validation failures.
        Empty list means all values are valid.
    """
    errors: List[str] = []

    if CAMERA_INDEX < 0:
        errors.append("CAMERA_INDEX must be >= 0")

    if not (0 < MIN_DETECTION_CONFIDENCE <= 1):
        errors.append("MIN_DETECTION_CONFIDENCE must be between 0 and 1")

    if not (0 < MIN_TRACKING_CONFIDENCE <= 1):
        errors.append("MIN_TRACKING_CONFIDENCE must be between 0 and 1")

    if FEED_WIDTH < 100 or FEED_HEIGHT < 100:
        errors.append("Feed resolution minimum is 100x100")

    if not (0 <= SMOOTHING_ALPHA <= 1):
        errors.append("SMOOTHING_ALPHA must be between 0 and 1")

    if not (1.0 <= POINT_SIZE <= 10.0):
        errors.append("POINT_SIZE must be between 1.0 and 10.0")

    if WALK_SPEED <= 0:
        errors.append("WALK_SPEED must be > 0")

    if FLIGHT_SPEED <= 0:
        errors.append("FLIGHT_SPEED must be > 0")

    if VERTICAL_SPEED <= 0:
        errors.append("VERTICAL_SPEED must be > 0")

    if TURN_SPEED <= 0:
        errors.append("TURN_SPEED must be > 0")

    if PITCH_SPEED <= 0:
        errors.append("PITCH_SPEED must be > 0")

    return errors


# Check for validation errors on import
_config_errors = validate_config()
if _config_errors:
    print("  Configuration errors found:")
    for _error in _config_errors:
        print(f"  - {_error}")
    print("\nPlease check your .env file or use .env.example as reference.")

if DEBUG:
    print("=" * 50)
    print("DEBUG MODE ENABLED")
    print("Configuration:")
    print(f"  Camera Index: {CAMERA_INDEX}")
    print(f"  Detection Confidence: {MIN_DETECTION_CONFIDENCE}")
    print(f"  Feed Resolution: {FEED_WIDTH}x{FEED_HEIGHT}")
    print(f"  Models Directory: {MODELS_DIR}")
    print(f"  Sketchfab Token: {'set' if SKETCHFAB_TOKEN else 'not set'}")
    print("=" * 50)
