#!/usr/bin/env python3
"""Public entry point for the final LLM citation-filtering stage."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from legal_rag.runner import run_llm_filter  # noqa: E402


if __name__ == "__main__":
    run_llm_filter()
