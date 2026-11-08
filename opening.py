"""
Opening GUI Module for Hand-AR Project.
Thin wrapper for backward compatibility — delegates to handarm.gui.launcher.

Usage:
    python3 opening.py
"""
from handarm.gui.launcher import main

if __name__ == "__main__":
    main()