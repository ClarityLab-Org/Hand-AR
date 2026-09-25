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
        load_dotenv(_env_file)
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
