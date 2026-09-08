#!/usr/bin/env python3
"""Run a data-free demonstration of the retrieval composition primitives."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.graph import expand_neighbors
from legal_rag.retrieval import fuse_rankings


def main() -> int:
    query_path = ROOT / "examples" / "sample_query.json"
    query = json.loads(query_path.read_text(encoding="utf-8"))

    # These are tiny illustrative rankings. The real system creates them from
    # embeddings over the competition corpus.
    subquery_rankings = [
        ["Art. 5 BV", "Art. 29 BV", "Art. 9 BV"],
        ["Art. 29 BV", "Art. 5 BV", "Art. 9 BV"],
    ]
    graph = {
        "outgoing": {"Art. 29 BV": {"Art. 5 BV"}},
        "incoming": {"Art. 29 BV": {"Art. 9 BV"}},
    }

    fused_scores = fuse_rankings(subquery_rankings)
    ranked = sorted(fused_scores, key=lambda citation: (-fused_scores[citation], citation))
    graph_neighbors = sorted(expand_neighbors({ranked[0]}, graph))

    result = {
        "query_id": query["query_id"],
        "query": query["query"],
        "demo_only": True,
        "subquery_rankings": subquery_rankings,
        "rrf_scores": fused_scores,
        "ranked_candidates": ranked,
        "one_hop_graph_neighbors_of_top_candidate": graph_neighbors,
        "note": "Illustrative in-memory smoke demo; not a competition evaluation.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
