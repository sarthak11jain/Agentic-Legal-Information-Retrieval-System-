#!/usr/bin/env python3
"""Public entry point for the main law-side retrieval pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from legal_rag.runner import run_law_retrieval  # noqa: E402


if __name__ == "__main__":
    run_law_retrieval()
