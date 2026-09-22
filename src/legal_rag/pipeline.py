"""Deterministic graph-augmented retrieval over supplied citation documents."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .config import PipelineConfig
from .evaluation import macro_f1
from .graph import expand_neighbors
from .models import CitationDocument, PipelineResult, Query, RankedCitation
from .retrieval import fuse_rankings, minmax_normalize

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(value: str) -> set[str]:
    return set(_TOKEN.findall(value.lower()))


def _overlap(query_terms: set[str], document: CitationDocument) -> float:
    document_terms = _tokens(document.text) | {keyword.lower() for keyword in document.keywords}
    return float(len(query_terms & document_terms))


def run_pipeline(
    query: Query,
    documents: Iterable[CitationDocument],
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Rank citations using lexical, RRF, and one-hop graph signals.

    The supported reference implementation intentionally uses deterministic local
    signals. It demonstrates pipeline composition without claiming to run the
    archived competition models or reproduce benchmark performance.
    """

    active_config = config or PipelineConfig()
    active_config.validate()
    corpus = tuple(documents)
    if not corpus:
        return PipelineResult(query=query, ranked_citations=())

    query_terms = _tokens(query.text)
    lexical_raw = {
        document.citation: _overlap(query_terms, document) for document in corpus
    }
    keyword_raw = {
        document.citation: float(
            len(query_terms & {keyword.lower() for keyword in document.keywords})
        )
        for document in corpus
    }
    citation_order = {document.citation: index for index, document in enumerate(corpus)}
    rankings = [
        sorted(
            lexical_raw,
            key=lambda citation: (-lexical_raw[citation], citation_order[citation]),
        ),
        sorted(
            keyword_raw,
            key=lambda citation: (-keyword_raw[citation], citation_order[citation]),
        ),
    ]
    rrf_raw = fuse_rankings(rankings, rrf_k=active_config.rrf_k)
    seed_citations = set(rankings[0][: active_config.seed_k])
    graph_neighbors = {
        "outgoing": {document.citation: set(document.neighbors) for document in corpus},
        "incoming": {},
    }
    expanded = expand_neighbors(seed_citations, graph_neighbors)

    lexical_scores = minmax_normalize(lexical_raw)
    rrf_scores = minmax_normalize(rrf_raw)
    graph_scores = {document.citation: float(document.citation in expanded) for document in corpus}
    ranked = sorted(
        (
            RankedCitation(
                citation=document.citation,
                lexical_score=lexical_scores[document.citation],
                rrf_score=rrf_scores[document.citation],
                graph_score=graph_scores[document.citation],
                score=(
                    active_config.lexical_weight * lexical_scores[document.citation]
                    + active_config.rrf_weight * rrf_scores[document.citation]
                    + active_config.graph_weight * graph_scores[document.citation]
                ),
            )
            for document in corpus
        ),
        key=lambda item: (-item.score, citation_order[item.citation]),
    )[: active_config.result_limit]
    score = None
    if query.gold_citations:
        score = macro_f1([[item.citation for item in ranked]], [query.gold_citations])
    return PipelineResult(query=query, ranked_citations=tuple(ranked), macro_f1=score)
