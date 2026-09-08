"""Stable entry points for the current research scripts.

The original implementation is intentionally kept under ``code/`` while the
public API is being refactored. These runners provide stable commands without
duplicating or changing the ranking algorithms.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def run_legacy_script(relative_path: str, argv: Sequence[str] | None = None) -> None:
    """Execute one legacy research script with repository-relative imports."""

    script = REPOSITORY_ROOT / relative_path
    if not script.is_file():
        raise FileNotFoundError(f"Research entry point does not exist: {script}")

    script_directory = str(script.parent)
    if script_directory not in sys.path:
        sys.path.insert(0, script_directory)

    sys.argv = [str(script), *(list(argv) if argv is not None else sys.argv[1:])]
    runpy.run_path(str(script), run_name="__main__")


def run_law_retrieval(argv: Sequence[str] | None = None) -> None:
    """Run the main law-side retrieval pipeline."""

    run_legacy_script(
        "code/dev/law-only/lawside-retrieval/law_document_side_retrieval.py",
        argv,
    )


def run_llm_filter(argv: Sequence[str] | None = None) -> None:
    """Run the final LLM citation-filtering stage."""

    run_legacy_script(
        "code/dev/law-only/lawside-retrieval/trials/llm_law_submission_reranker.py",
        argv,
    )
