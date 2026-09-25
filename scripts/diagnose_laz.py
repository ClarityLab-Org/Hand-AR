"""Diagnose LAS/LAZ metadata and full-stream readability."""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import laspy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from handarm.geometry.laz_loader import iter_laz_chunks


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a .las or .laz file")
    parser.add_argument("--sha256", action="store_true", help="Print the file checksum")
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print(f"error: file not found: {args.path}", file=sys.stderr)
        return 2

    print(f"file: {args.path}")
    print(f"size: {os.path.getsize(args.path):,} bytes")
    print(f"laspy: {laspy.__version__}")
    try:
        import lazrs
        print(f"lazrs: {getattr(lazrs, '__version__', 'installed')}")
    except ImportError:
        print("lazrs: not installed")

    if args.sha256:
        print(f"sha256: {_sha256(args.path)}")

    with laspy.open(args.path) as reader:
        point_count = reader.header.point_count
    print(f"header point count: {point_count:,}")

    decoded = 0
    last_chunk_index = -1
    try:
        for chunk_index, point_offset, chunk, _, _ in iter_laz_chunks(args.path):
            last_chunk_index = chunk_index
            decoded = point_offset + len(chunk)
            print(
                f"decoded chunk {chunk_index}: "
                f"{decoded:,}/{point_count:,} points",
                end="\r",
                flush=True,
            )
    except Exception as exc:
        print()
        print(
            f"FAILED at chunk {last_chunk_index + 1}, point offset "
            f"{decoded:,} of {point_count:,}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"\nOK: decoded all {decoded:,} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
