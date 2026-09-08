"""Small, dependency-free ranking utilities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence


def minmax_normalize(scores: Mapping[str, float]) -> dict[str, float]:
    """Normalize scores into ``[0, 1]`` using the observed min and max.

    A constant non-empty score map becomes all ones, matching the behavior
    used by the original research pipeline when no relative ordering exists.
    """

    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high <= low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def fuse_rankings(
    rankings: Iterable[Sequence[str]],
    *,
    rrf_k: int = 60,
) -> dict[str, float]:
    """Fuse ranked candidate sequences with Reciprocal Rank Fusion.

    Ranks are one-based. Duplicate candidates within one ranking contribute
    only their first occurrence, which prevents malformed input from
    artificially increasing a candidate's score.
    """

    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[str] = set()
        rank = 0
        for candidate in ranking:
            if candidate in seen:
                continue
            seen.add(candidate)
            rank += 1
            scores[candidate] += 1.0 / (rrf_k + rank)
    return dict(scores)
