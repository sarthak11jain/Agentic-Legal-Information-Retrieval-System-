"""Validated defaults for the documented retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from .reranking import RerankWeights


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Research defaults used by the documented best-performing cascade."""

    rrf_k: int = 60
    subquery_top_k: int = 200
    domain_top_k: int = 10
    graph_mode: str = "both"
    graph_seed_k: int = 200
    graph_boost: float = 0.10
    court_anchor_ranked_k: int = 5
    court_anchor_top_targets: int = 50
    court_anchor_min_count: int = 2
    court_anchor_boost: float = 0.08
    rerank_top_k: int = 50
    weights: RerankWeights = RerankWeights()

    def validate(self) -> None:
        """Raise a helpful error if a pipeline configuration is unsafe."""

        if self.rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        for name in (
            "subquery_top_k",
            "domain_top_k",
            "graph_seed_k",
            "court_anchor_ranked_k",
            "court_anchor_top_targets",
            "court_anchor_min_count",
            "rerank_top_k",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.graph_mode not in {"incoming", "outgoing", "both"}:
            raise ValueError("graph_mode must be 'incoming', 'outgoing', or 'both'")
        if self.graph_boost < 0 or self.court_anchor_boost < 0:
            raise ValueError("boost values must be non-negative")
