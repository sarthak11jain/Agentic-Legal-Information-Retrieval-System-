from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path
from typing import Callable

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_rag.reranking import RerankWeights, combine_weighted_scores


UTILS_DIR = Path(__file__).resolve().parents[3] / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from call_reranker import call_reranker  # noqa: E402


ScoreFn = Callable[[str, list[str]], list[dict[str, float]]]


def norm_space(value: object) -> str:
    return " ".join(str(value).split())


def load_env_file(env_path: Path | None) -> None:
    if not env_path or not env_path.exists():
        return

    for line in env_path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_procedural_books(path: Path) -> set[str]:
    if not path.exists():
        return set()

    text = path.read_text(errors="ignore")
    fenced = re.search(r"```text\s*(.*?)```", text, flags=re.S)
    content = fenced.group(1) if fenced else text
    books: set[str] = set()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.lower().startswith("batch"):
            continue
        if line.lower().startswith("proc "):
            line = line[5:].strip()
        if line:
            books.add(line)
    return books


def load_score_cache(score_path: Path, *, model: str) -> dict[tuple[str, str], float]:
    if not score_path.exists():
        return {}

    scores = pd.read_csv(score_path)
    if "model" in scores.columns:
        scores = scores[scores["model"].astype(str) == model]
    if not {"query_id", "citation", "score"}.issubset(scores.columns):
        return {}

    cache: dict[tuple[str, str], float] = {}
    for row in scores.itertuples(index=False):
        score = getattr(row, "score")
        if pd.isna(score):
            continue
        cache[(str(row.query_id), norm_space(row.citation))] = float(score)
    return cache


def format_law_document(citation: str, document: str, *, max_chars: int) -> str:
    text = f"{citation}\n{norm_space(document)}".strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def minmax_scores(scores: dict[str, float], citations: list[str]) -> dict[str, float]:
    values = {
        citation: float(scores.get(citation, 0.0))
        for citation in citations
        if math.isfinite(float(scores.get(citation, 0.0)))
    }
    if not values:
        return {citation: 0.0 for citation in citations}

    lo = min(values.values())
    hi = max(values.values())
    if hi <= lo:
        return {citation: 1.0 for citation in citations}
    return {citation: (values.get(citation, 0.0) - lo) / (hi - lo) for citation in citations}


def rerank_substantive_slots(
    *,
    query_id: str,
    query_text: str,
    candidates: list[str],
    citation_to_document: dict[str, str],
    citation_to_source_book: dict[str, str],
    procedural_books: set[str],
    score_cache: dict[tuple[str, str], float],
    top_k: int,
    model: str,
    score_fn: ScoreFn | None = None,
    api_key: str | None = None,
    latency: str | None = None,
    timeout_s: int = 120,
    max_document_chars: int = 4000,
) -> tuple[list[str], list[dict[str, object]], int]:
    """Rerank non-procedural candidates while freezing procedural positions."""
    if top_k <= 0 or not candidates:
        return candidates, [], 0

    pool = list(dict.fromkeys(candidates[:top_k]))
    tail = [citation for citation in candidates[top_k:] if citation not in set(pool)]
    score_rows: list[dict[str, object]] = []
    rerank_items: list[tuple[int, str, str]] = []
    frozen_positions: set[int] = set()

    for index, citation in enumerate(pool):
        source_book = citation_to_source_book.get(citation, "")
        is_procedural = source_book in procedural_books
        document = citation_to_document.get(citation, "")
        sent_to_reranker = bool(document and not is_procedural)
        if sent_to_reranker:
            rerank_items.append(
                (
                    index,
                    citation,
                    format_law_document(citation, document, max_chars=max_document_chars),
                )
            )
        else:
            frozen_positions.add(index)

        cached_score = score_cache.get((query_id, citation))
        score_rows.append(
            {
                "query_id": query_id,
                "original_rank": index + 1,
                "citation": citation,
                "source_book": source_book,
                "is_procedural": is_procedural,
                "sent_to_reranker": sent_to_reranker,
                "score": cached_score if cached_score is not None else math.nan,
                "model": model,
            }
        )

    missing_scores = [
        (index, citation, document)
        for index, citation, document in rerank_items
        if (query_id, citation) not in score_cache
    ]
    api_calls = 0
    if missing_scores:
        score_fn = score_fn or (
            lambda query, documents: call_reranker(
                query,
                documents,
                api_key=api_key,
                model=model,
                top_n=len(documents),
                latency=latency,
                timeout_s=timeout_s,
                backend=os.getenv("RERANK_BACKEND", "zeroentropy"),
            )
        )
        documents = [document for _, _, document in missing_scores]
        results = score_fn(query_text, documents)
        api_calls = 1
        for result in results:
            result_index = int(result["index"])
            if 0 <= result_index < len(missing_scores):
                _, citation, _ = missing_scores[result_index]
                score_cache[(query_id, citation)] = float(result["relevance_score"])

    for row in score_rows:
        if row["sent_to_reranker"]:
            row["score"] = score_cache.get((query_id, str(row["citation"])), math.nan)

    movable_positions = [
        index
        for index, citation in enumerate(pool)
        if index not in frozen_positions and (query_id, citation) in score_cache
    ]
    movable_citations = [pool[index] for index in movable_positions]
    movable_citations.sort(
        key=lambda citation: (score_cache.get((query_id, citation), -math.inf), -pool.index(citation)),
        reverse=True,
    )

    reranked_pool = pool[:]
    for position, citation in zip(movable_positions, movable_citations):
        reranked_pool[position] = citation

    return reranked_pool + tail, score_rows, api_calls


def rerank_substantive_slots_weighted(
    *,
    query_id: str,
    query_text: str,
    candidates: list[str],
    citation_to_document: dict[str, str],
    citation_to_source_book: dict[str, str],
    procedural_books: set[str],
    score_cache: dict[tuple[str, str], float],
    top_k: int,
    model: str,
    tfidf_scores: dict[str, float],
    cosine_scores: dict[str, float],
    tfidf_weight: float,
    cosine_weight: float,
    cross_encoder_weight: float,
    score_fn: ScoreFn | None = None,
    api_key: str | None = None,
    latency: str | None = None,
    timeout_s: int = 120,
    max_document_chars: int = 4000,
) -> tuple[list[str], list[dict[str, object]], int]:
    """Blend TF-IDF, cosine, and cross-encoder scores for substantive slots."""
    if top_k <= 0 or not candidates:
        return candidates, [], 0

    pool = list(dict.fromkeys(candidates[:top_k]))
    tail = [citation for citation in candidates[top_k:] if citation not in set(pool)]
    score_rows: list[dict[str, object]] = []
    rerank_items: list[tuple[int, str, str]] = []
    frozen_positions: set[int] = set()

    for index, citation in enumerate(pool):
        source_book = citation_to_source_book.get(citation, "")
        is_procedural = source_book in procedural_books
        document = citation_to_document.get(citation, "")
        sent_to_reranker = bool(document and not is_procedural)
        if sent_to_reranker:
            rerank_items.append(
                (
                    index,
                    citation,
                    format_law_document(citation, document, max_chars=max_document_chars),
                )
            )
        else:
            frozen_positions.add(index)

        cached_score = score_cache.get((query_id, citation))
        score_rows.append(
            {
                "query_id": query_id,
                "original_rank": index + 1,
                "citation": citation,
                "source_book": source_book,
                "is_procedural": is_procedural,
                "sent_to_reranker": sent_to_reranker,
                "tfidf_score": tfidf_scores.get(citation, math.nan),
                "cosine_score": cosine_scores.get(citation, math.nan),
                "score": cached_score if cached_score is not None else math.nan,
                "combined_score": math.nan,
                "model": model,
            }
        )

    missing_scores = [
        (index, citation, document)
        for index, citation, document in rerank_items
        if (query_id, citation) not in score_cache
    ]
    api_calls = 0
    if missing_scores:
        score_fn = score_fn or (
            lambda query, documents: call_reranker(
                query,
                documents,
                api_key=api_key,
                model=model,
                top_n=len(documents),
                latency=latency,
                timeout_s=timeout_s,
                backend=os.getenv("RERANK_BACKEND", "zeroentropy"),
            )
        )
        documents = [document for _, _, document in missing_scores]
        results = score_fn(query_text, documents)
        api_calls = 1
        for result in results:
            result_index = int(result["index"])
            if 0 <= result_index < len(missing_scores):
                _, citation, _ = missing_scores[result_index]
                score_cache[(query_id, citation)] = float(result["relevance_score"])

    movable_positions = [
        index
        for index, citation in enumerate(pool)
        if index not in frozen_positions and (query_id, citation) in score_cache
    ]
    movable_citations = [pool[index] for index in movable_positions]
    tfidf_norm = minmax_scores(tfidf_scores, movable_citations)
    cosine_norm = minmax_scores(cosine_scores, movable_citations)
    cross_norm = minmax_scores(
        {citation: score_cache.get((query_id, citation), 0.0) for citation in movable_citations},
        movable_citations,
    )

    combined_scores = combine_weighted_scores(
        tfidf_norm,
        cosine_norm,
        cross_norm,
        weights=RerankWeights(
            tfidf=tfidf_weight,
            cosine=cosine_weight,
            cross_encoder=cross_encoder_weight,
        ),
    )
    movable_citations.sort(
        key=lambda citation: (combined_scores.get(citation, -math.inf), -pool.index(citation)),
        reverse=True,
    )

    reranked_pool = pool[:]
    for position, citation in zip(movable_positions, movable_citations):
        reranked_pool[position] = citation

    for row in score_rows:
        citation = str(row["citation"])
        if row["sent_to_reranker"]:
            row["score"] = score_cache.get((query_id, citation), math.nan)
            row["combined_score"] = combined_scores.get(citation, math.nan)

    return reranked_pool + tail, score_rows, api_calls
