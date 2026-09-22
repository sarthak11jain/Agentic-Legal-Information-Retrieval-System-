from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def split_refs(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def norm_space(value: object) -> str:
    return " ".join(str(value).split())


def iter_law_references(path: Path, *, batch_size: int) -> object:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for streaming court law references") from error

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["law_references"]):
        for value in batch.column("law_references").to_pylist():
            yield value


def build_court_law_exact_anchor_index(
    court_path: Path,
    *,
    valid_citations: set[str],
    min_count: int,
    batch_size: int = 2048,
) -> dict[str, dict[str, float]]:
    """Map exact law anchors to other laws co-cited in the same court documents."""
    if not court_path.exists():
        return {}
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    valid = {norm_space(citation) for citation in valid_citations}
    anchor_doc_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()

    for law_references in iter_law_references(court_path, batch_size=batch_size):
        refs: list[str] = []
        seen: set[str] = set()
        for raw_ref in split_refs(law_references):
            ref = norm_space(raw_ref)
            if ref in valid and ref not in seen:
                seen.add(ref)
                refs.append(ref)

        if len(refs) < 2:
            continue

        for anchor in refs:
            anchor_doc_counts[anchor] += 1
            for target in refs:
                if target != anchor:
                    pair_counts[(anchor, target)] += 1

    index: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for (anchor, target), count in pair_counts.items():
        if count >= min_count and anchor_doc_counts[anchor]:
            index[anchor][target] = count / anchor_doc_counts[anchor]

    return dict(index)


def score_exact_anchor_neighbors(
    *,
    exact_anchors: list[str],
    index: dict[str, dict[str, float]],
    boost: float,
    top_targets_per_anchor: int,
) -> dict[str, float]:
    if not exact_anchors or not index or boost <= 0 or top_targets_per_anchor <= 0:
        return {}

    scores: defaultdict[str, float] = defaultdict(float)
    for anchor in exact_anchors:
        neighbors = index.get(anchor, {})
        ranked_neighbors = sorted(
            neighbors,
            key=lambda citation: (neighbors[citation], citation),
            reverse=True,
        )[:top_targets_per_anchor]
        for target in ranked_neighbors:
            scores[target] += boost * neighbors[target]

    return dict(scores)
