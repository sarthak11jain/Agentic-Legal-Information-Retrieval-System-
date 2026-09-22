"""Configuration for the supported deterministic retrieval pipeline."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[import-not-found]
else:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Validated settings for the data-free reference pipeline."""

    rrf_k: int = 60
    seed_k: int = 2
    result_limit: int = 3
    lexical_weight: float = 0.50
    rrf_weight: float = 0.30
    graph_weight: float = 0.20

    def validate(self) -> None:
        """Raise ``ValueError`` when a configuration cannot rank safely."""

        if self.rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if self.seed_k <= 0 or self.result_limit <= 0:
            raise ValueError("seed_k and result_limit must be positive")
        weights = (self.lexical_weight, self.rrf_weight, self.graph_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("score weights must be non-negative and have a positive total")


def load_config(path: Path) -> PipelineConfig:
    """Load a supported pipeline configuration from a TOML file."""

    with path.open("rb") as handle:
        raw: Mapping[str, Any] = tomllib.load(handle)
    weights = raw.get("weights", {})
    if not isinstance(weights, Mapping):
        raise ValueError("[weights] must be a TOML table")
    config = PipelineConfig(
        rrf_k=int(raw.get("rrf_k", 60)),
        seed_k=int(raw.get("seed_k", 2)),
        result_limit=int(raw.get("result_limit", 3)),
        lexical_weight=float(weights.get("lexical", 0.50)),
        rrf_weight=float(weights.get("rrf", 0.30)),
        graph_weight=float(weights.get("graph", 0.20)),
    )
    config.validate()
    return config
