# Code migration map

The repository uses a staged migration so the validated research implementation remains available while reusable components move behind the `legal_rag` package boundary.

## Public package components

| Public module | Current responsibility | Integrated into |
|---|---|---|
| `legal_rag.retrieval` | RRF and score normalization | Main law-side retrieval |
| `legal_rag.graph` | Citation-neighbor lookup and expansion | Main law-side retrieval |
| `legal_rag.reranking` | Weighted TF-IDF/cosine/cross-encoder combination | Cross-encoder reranking |
| `legal_rag.llm` | Decision parsing and confirmation policy | LLM citation filtering |
| `legal_rag.evaluation` | Citation-level macro/micro F1 | Public tests and future evaluator |
| `legal_rag.config` | Validated documented defaults | Configuration and tests |

## Legacy research components

The following files still contain the data loading, model calls, and end-to-end orchestration. They remain under `code/` until each stage has equivalent tests and a stable public API:

- `code/dev/law-only/lawside-retrieval/law_document_side_retrieval.py`
- `code/dev/law-only/lawside-retrieval/trials/court_law_tfidf_rerank.py`
- `code/dev/law-only/lawside-retrieval/trials/substantive_cross_encoder_rerank.py`
- `code/dev/law-only/lawside-retrieval/trials/llm_law_submission_reranker.py`
- `code/dev/court-graph-retrieval/court_document_side_retrieval.py`
- `code/prepare/`
- `code/dev/utils/`

This boundary is intentional: the public primitives are tested first, while the original experiment scripts remain available for historical reproducibility and comparison.
