#!/usr/bin/env python3
"""Check runtime dependencies and local data prerequisites."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REQUIRED_MODULES = {
    "numpy": "numpy",
    "openai": "openai",
    "pandas": "pandas",
    "polars": "polars",
    "pyarrow": "pyarrow",
    "dotenv": "python-dotenv",
    "sklearn": "scikit-learn",
    "tqdm": "tqdm",
}
EXPECTED_DATA = ("laws_db.parquet", "court_db.parquet", "train.parquet", "test.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Fail when dependencies or data are missing.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    failures = 0
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or newer is required.")
        failures += 1
    else:
        print(f"Python: {sys.version.split()[0]} OK")

    missing_modules = [
        package for module, package in REQUIRED_MODULES.items() if importlib.util.find_spec(module) is None
    ]
    if missing_modules:
        print("Missing Python packages: " + ", ".join(missing_modules))
        failures += len(missing_modules)
    else:
        print("Python dependencies: OK")

    data_dir = root / "data"
    missing_data = [name for name in EXPECTED_DATA if not (data_dir / name).exists()]
    if missing_data:
        print("Data files not present (expected for a clean checkout): " + ", ".join(missing_data))
        if args.strict:
            failures += len(missing_data)
    else:
        print("Research data: present")

    if failures and args.strict:
        print("Environment check failed in strict mode.")
        return 1
    print("Environment check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
