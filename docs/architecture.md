# Architecture

## Design objective

The system is designed for citation-level retrieval: given a complex legal scenario, return the Swiss legal sources needed to support a complete answer. The design prioritizes recall early in the pipeline and precision late in the pipeline.

## Component responsibilities

| Component | Responsibility |
|---|---|
| Query generation | Translate and decompose a long legal scenario into focused German legal subqueries |
| Dense retrieval | Retrieve semantically related law articles for each subquery |
| RRF fusion | Combine multiple subquery rankings into one candidate list |
| Hint parsing | Extract explicit article and legal-book signals from query text |
| Law graph | Expand candidates through incoming and outgoing law references |
| Court graph | Expand candidates through laws co-cited in court documents |
| TF-IDF reranking | Use court co-citation patterns as a conservative ordering signal |
| Cross-encoder | Score the semantic relationship between the query and candidate article |
| LLM filter | Remove only clear lower-ranked citation false positives, with confirmation |

## Retrieval philosophy

The pipeline uses a cascade rather than a single model. Dense retrieval provides broad semantic recall. Citation and co-citation graphs add structured legal relationships. Rerankers then use richer evidence on a smaller candidate pool, while the LLM is restricted to a conservative filtering role.

## Main implementation locations

- `code/dev/law-only/lawside-retrieval/law_document_side_retrieval.py` — law-side retrieval and candidate expansion
- `code/dev/law-only/lawside-retrieval/trials/court_law_exact_anchor.py` — court-law anchor expansion
- `code/dev/law-only/lawside-retrieval/trials/court_law_tfidf_rerank.py` — court-law TF-IDF reranking
- `code/dev/law-only/lawside-retrieval/trials/substantive_cross_encoder_rerank.py` — weighted cross-encoder reranking
- `code/dev/law-only/lawside-retrieval/trials/llm_law_submission_reranker.py` — LLM citation filtering
- `code/dev/gen-sub/` — query decomposition and subquery embedding generation
- `code/dev/utils/` — model and API clients

The first reusable public primitives live under `src/legal_rag/`. The current
research scripts delegate score normalization, citation-neighbor lookup, and
multi-signal reranking to these modules while the larger data-loading and
model-calling components are migrated incrementally.

The LLM stage keeps model calls separate from deterministic decision handling.
Model output is parsed into explicit `KEEP`/`REMOVE` decisions, and a
candidate is removed only when both the first pass and the confirmation pass
return `REMOVE`.
