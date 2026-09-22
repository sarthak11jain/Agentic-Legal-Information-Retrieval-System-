"""Console commands for the supported data-free project experience."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

from .config import PipelineConfig, load_config
from .models import CitationDocument, Query
from .pipeline import run_pipeline


def _load_demo_input() -> tuple[Query, tuple[CitationDocument, ...]]:
    asset = resources.files("legal_rag.assets").joinpath("synthetic_corpus.json")
    payload = json.loads(asset.read_text(encoding="utf-8"))
    query = Query(
        query_id=payload["query"]["query_id"],
        text=payload["query"]["text"],
        gold_citations=tuple(payload["query"]["gold_citations"]),
    )
    documents = tuple(CitationDocument.from_mapping(item) for item in payload["documents"])
    return query, documents


def _load_default_config() -> PipelineConfig:
    asset = resources.files("legal_rag.assets").joinpath("default.toml")
    with resources.as_file(asset) as path:
        return load_config(Path(path))


def demo_main(argv: Sequence[str] | None = None) -> int:
    """Run the synthetic end-to-end retrieval example and print JSON."""

    parser = argparse.ArgumentParser(description=demo_main.__doc__)
    parser.add_argument("--config", type=Path, help="Optional TOML configuration override.")
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else _load_default_config()
    query, documents = _load_demo_input()
    print(json.dumps(run_pipeline(query, documents, config).to_dict(), indent=2, sort_keys=True))
    return 0


def check_main(argv: Sequence[str] | None = None) -> int:
    """Report whether the supported, data-free package can run."""

    parser = argparse.ArgumentParser(description=check_main.__doc__)
    parser.parse_args(argv)
    payload = {
        "python": sys.version.split()[0],
        "minimum_python": "3.10",
        "archive_dependencies_required": False,
        "competition_data_required": False,
        "network_required": False,
        "status": "ok" if sys.version_info >= (3, 10) else "unsupported_python",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(demo_main())
