"""Typed public models for citation retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Query:
    """A legal-information request and optional evaluation labels."""

    query_id: str
    text: str
    gold_citations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CitationDocument:
    """A citation with safe, local retrieval metadata."""

    citation: str
    text: str
    keywords: tuple[str, ...] = ()
    neighbors: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CitationDocument:
        return cls(
            citation=str(raw["citation"]),
            text=str(raw["text"]),
            keywords=tuple(str(value) for value in raw.get("keywords", [])),
            neighbors=tuple(str(value) for value in raw.get("neighbors", [])),
        )


@dataclass(frozen=True, slots=True)
class RankedCitation:
    """One ranked citation and the score signals that produced it."""

    citation: str
    score: float
    lexical_score: float
    rrf_score: float
    graph_score: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "citation": self.citation,
            "score": round(self.score, 6),
            "lexical_score": round(self.lexical_score, 6),
            "rrf_score": round(self.rrf_score, 6),
            "graph_score": round(self.graph_score, 6),
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """A deterministic pipeline result suitable for JSON serialization."""

    query: Query
    ranked_citations: tuple[RankedCitation, ...]
    macro_f1: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query_id": self.query.query_id,
            "query": self.query.text,
            "ranked_citations": [item.to_dict() for item in self.ranked_citations],
            "demo_only": True,
            "network_required": False,
        }
        if self.macro_f1 is not None:
            payload["macro_f1"] = round(self.macro_f1, 6)
        return payload
