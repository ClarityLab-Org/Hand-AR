"""Cross-check LAS/LAZ integrity with laspy and optional PDAL."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from handarm.geometry.laz_loader import iter_laz_chunks


def _laspy_check(path: str) -> tuple[bool, str]:
    decoded = 0
    try:
        for _, offset, chunk, expected, _ in iter_laz_chunks(path):
            decoded = offset + len(chunk)
        return decoded == expected, f"{decoded:,}/{expected:,} points decoded"
    except Exception as exc:
        return False, f"failed at {decoded:,} points: {type(exc).__name__}: {exc}"


def _pdal_check(pdal: str, path: str, timeout: float) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [pdal, "info", "--stats", path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0:
            return True, "completed successfully"
        return False, (
            f"exit {result.returncode}: "
            f"{(result.stderr or result.stdout).strip()[-500:]}"
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout:g}s"
    except Exception as exc:
        return False, f"could not run: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    laspy_ok, laspy_detail = _laspy_check(args.path)
    print(f"laspy/lazrs: {'OK' if laspy_ok else 'FAILED'} ({laspy_detail})")

    pdal = shutil.which("pdal")
    if not pdal:
        print("PDAL: unavailable")
        if laspy_ok:
            print("INCONCLUSIVE: laspy succeeded, but PDAL is unavailable")
        else:
            print("INCONCLUSIVE: laspy failed and PDAL is unavailable")
        return 2

    pdal_ok, detail = _pdal_check(pdal, args.path, args.timeout)

    print(f"PDAL: {'OK' if pdal_ok else 'FAILED/CRASHED'} ({detail})")
    if not laspy_ok and not pdal_ok:
        print("CONFIRMED CORRUPT: both integrity checks failed")
        return 1
    if laspy_ok and pdal_ok:
        print("OK: both integrity checks succeeded")
        return 0
    print("INCONCLUSIVE: tools disagreed")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
