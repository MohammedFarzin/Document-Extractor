"""Helper utilities for preparing the open-source pipeline environment."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict


def run(command: str) -> None:
    print(f"$ {command}")
    subprocess.run(command, shell=True, check=True)


def install_requirements(requirements: str) -> None:
    run(f"pip install -r {requirements}")


def verify_torch() -> Dict[str, str]:
    code = "import torch; print({'cuda': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()})"
    completed = subprocess.run(["python", "-c", code], capture_output=True, text=True, check=True)
    print(completed.stdout.strip())
    return json.loads(completed.stdout.strip().replace("'", '"'))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap helper for the open-source pipeline")
    parser.add_argument(
        "--requirements",
        default="requirements_opensource.txt",
        help="Requirements file to install",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip install step",
    )
    args = parser.parse_args()

    if not args.skip_install:
        install_requirements(args.requirements)

    try:
        verify_torch()
    except Exception as exc:  # pragma: no cover
        print(f"⚠️  Torch verification failed: {exc}")

    print("✅ Environment setup complete")


if __name__ == "__main__":  # pragma: no cover
    main()
