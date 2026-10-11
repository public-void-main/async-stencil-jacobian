"""Capture immutable context needed to reproduce an experiment run."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_metadata(spec_path: Path, manifest_path: Path, output: Path) -> dict[str, object]:
    metadata: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": _sha256(spec_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "note": "GPU, CUDA, driver, compile flags and block size are recorded by the CUDA/profiling owner in results/toolchain.txt.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
