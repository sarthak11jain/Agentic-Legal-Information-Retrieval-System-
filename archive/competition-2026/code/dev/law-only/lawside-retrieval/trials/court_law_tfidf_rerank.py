from __future__ import annotations

import math
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


def build_court_law_tfidf_index(
    court_path: Path,
    *,
    valid_citations: set[str],
    min_count: int,
    batch_size: int = 2048,
) -> dict[str, dict[str, float]]:
    """Build a normalized TF-IDF law co-citation index from court documents."""
    if not court_path.exists():
        return {}
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    valid = {norm_space(citation) for citation in valid_citations}
    doc_freq: Counter[str] = Counter()
    anchor_doc_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    doc_count = 0

    for law_references in iter_law_references(court_path, batch_size=batch_size):
        refs: list[str] = []
        seen: set[str] = set()
        for raw_ref in split_refs(law_references):
            ref = norm_space(raw_ref)
            if ref in valid and ref not in seen:
                seen.add(ref)
                refs.append(ref)

        if not refs:
            continue

        doc_count += 1
        doc_freq.update(refs)
        if len(refs) < 2:
            continue

        for anchor in refs:
            anchor_doc_counts[anchor] += 1
            for target in refs:
                if target != anchor:
                    pair_counts[(anchor, target)] += 1

    raw_index: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for (anchor, target), count in pair_counts.items():
        if count < min_count or not anchor_doc_counts[anchor]:
            continue
        tf = count / anchor_doc_counts[anchor]
        idf = 1.0 + math.log((1.0 + doc_count) / (1.0 + doc_freq[target]))
        raw_index[anchor][target] = tf * idf

    index: dict[str, dict[str, float]] = {}
    for anchor, targets in raw_index.items():
        max_weight = max(targets.values(), default=0.0)
        if max_weight <= 0:
            continue
        index[anchor] = {target: weight / max_weight for target, weight in targets.items()}

    return index


def rerank_court_law_tfidf_candidates(
    candidates: list[str],
    *,
    index: dict[str, dict[str, float]],
    boost: float,
    candidate_k: int,
    top_targets_per_anchor: int,
    rrf_k: float,
) -> list[str]:
    if not candidates or not index or boost <= 0 or candidate_k <= 0 or top_targets_per_anchor <= 0:
        return candidates

    scores = score_court_law_tfidf_candidates(
        candidates,
        index=index,
        boost=boost,
        candidate_k=candidate_k,
        top_targets_per_anchor=top_targets_per_anchor,
        rrf_k=rrf_k,
    )
    pool = list(dict.fromkeys(candidates[:candidate_k]))
    allowed = set(pool)
    reranked = sorted(pool, key=lambda citation: (scores.get(citation, 0.0), citation), reverse=True)
    if len(candidates) > candidate_k:
        reranked.extend(citation for citation in candidates[candidate_k:] if citation not in allowed)
    return reranked


def score_court_law_tfidf_candidates(
    candidates: list[str],
    *,
    index: dict[str, dict[str, float]],
    boost: float,
    candidate_k: int,
    top_targets_per_anchor: int,
    rrf_k: float,
) -> dict[str, float]:
    if not candidates or candidate_k <= 0:
        return {}

    pool = list(dict.fromkeys(candidates[:candidate_k]))
    allowed = set(pool)
    scores = {
        citation: 1.0 / (rrf_k + rank)
        for rank, citation in enumerate(pool, start=1)
    }
    if not index or boost <= 0 or top_targets_per_anchor <= 0:
        return scores

    top_seed_score = 1.0 / (rrf_k + 1)

    for rank, anchor in enumerate(pool, start=1):
        seed_weight = (1.0 / (rrf_k + rank)) / top_seed_score
        targets = index.get(anchor, {})
        ranked_targets = sorted(
            targets,
            key=lambda citation: (targets[citation], citation),
            reverse=True,
        )[:top_targets_per_anchor]
        for target in ranked_targets:
            if target in allowed and target != anchor:
                scores[target] = scores.get(target, 0.0) + boost * seed_weight * targets[target]

    return scores
