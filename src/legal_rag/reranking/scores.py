"""Score-combination utilities for multi-stage reranking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class RerankWeights:
    """Weights for TF-IDF, dense similarity, and cross-encoder signals."""

    tfidf: float = 0.40
    cosine: float = 0.20
    cross_encoder: float = 0.40

    def __post_init__(self) -> None:
        values = (self.tfidf, self.cosine, self.cross_encoder)
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("reranking weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one reranking weight must be positive")


def combine_weighted_scores(
    tfidf_scores: Mapping[str, float],
    cosine_scores: Mapping[str, float],
    cross_encoder_scores: Mapping[str, float],
    *,
    weights: RerankWeights | None = None,
) -> dict[str, float]:
    """Combine already-normalized score maps for a shared candidate set."""

    weights = weights or RerankWeights()
    candidates = set(tfidf_scores) | set(cosine_scores) | set(cross_encoder_scores)
    return {
        citation: (
            weights.tfidf * tfidf_scores.get(citation, 0.0)
            + weights.cosine * cosine_scores.get(citation, 0.0)
            + weights.cross_encoder * cross_encoder_scores.get(citation, 0.0)
        )
        for citation in candidates
    }
