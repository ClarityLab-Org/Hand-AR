"""
Standalone CSV Point Cloud Plotter Utility.
Thin wrapper for backward compatibility — delegates to handarm.engine.plot_csv.

Usage:
    python3 plotcsv.py [path/to/cloud.csv]
"""
from handarm.engine.plot_csv import main

if __name__ == "__main__":
    main()