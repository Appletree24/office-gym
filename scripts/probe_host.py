#!/usr/bin/env python3
"""Read-only host probe; no credentials, downloads, or system modification."""
import argparse
import datetime
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

def probe():
    packages = {}
    for name in ("torch", "transformers", "trl", "peft", "verl", "vllm", "sglang", "mcp"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = {"available": False, "rows": [], "error": None}
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            process = subprocess.run(
                [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15, check=False)
            gpu.update(available=process.returncode == 0, rows=process.stdout.strip().splitlines(),
                       error=process.stderr.strip() or None, returncode=process.returncode)
        except (OSError, subprocess.TimeoutExpired) as exc:
            gpu["error"] = str(exc)
    disk = shutil.disk_usage(Path.cwd())
    return {"schema_version": 1, "kind": "host_probe_not_experiment",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python": sys.version, "platform": platform.platform(),
            "disk_bytes": {"total": disk.total, "free": disk.free},
            "packages": packages, "gpu": gpu,
            "limitations": ["No model inference, training, CUDA compatibility or GPU memory fit was verified."]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(probe(), ensure_ascii=False, indent=2) + "\n")
    print(f"Host probe written: {args.out}")

if __name__ == "__main__":
    main()

