#!/usr/bin/env python3
"""Fail closed if published demo media or its recorded source has drifted."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs" / "demo"


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> int:
    capture_path = DEMO / "capture.json"
    if not capture_path.exists():
        print("demo not published yet; source checks are covered by tests")
        return 0
    capture = json.loads(capture_path.read_text())
    source = capture["source_commit"]
    subprocess.run(["git", "merge-base", "--is-ancestor", source, "HEAD"], cwd=ROOT, check=True)
    if run("git", "rev-parse", f"{source}^{{tree}}") != capture["source_tree"]:
        raise SystemExit("recorded source tree does not match capture metadata")
    for path, expected_blob in capture["source_blobs"].items():
        if run("git", "rev-parse", f"{source}:{path}") != expected_blob:
            raise SystemExit(f"recorded source blob mismatch: {path}")
        if run("git", "hash-object", path) != expected_blob:
            raise SystemExit(f"runtime input changed since recording: {path}")
    for name, expected in capture["artifacts"].items():
        path = DEMO / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise SystemExit(f"artifact size mismatch: {name}")
        if sha256(path) != expected["sha256"]:
            raise SystemExit(f"artifact digest mismatch: {name}")
        if path.suffix in {".gif", ".mp4"}:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
                check=True,
            )
    if run("git", "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("verification must leave a clean tracked checkout")
    print(f"verified {len(capture['artifacts'])} artifacts at source {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
