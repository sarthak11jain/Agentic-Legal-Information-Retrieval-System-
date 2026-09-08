# Graph-Augmented Legal Information Retrieval System

An agentic, graph-augmented retrieval pipeline for identifying the Swiss legal sources relevant to complex legal questions.

This project was developed for the [Kaggle LLM Agentic Legal Information Retrieval competition](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval), where the task is to retrieve citation-level legal sources from a multilingual Swiss legal corpus.

[![CI](https://github.com/sarthak11jain/Agentic-Legal-Information-Retrieval-System-/actions/workflows/ci.yml/badge.svg)](https://github.com/sarthak11jain/Agentic-Legal-Information-Retrieval-System-/actions/workflows/ci.yml)

## Results

- **Final submission macro F1:** 0.23
- **Kaggle result:** Top 50
- **Task:** Citation-level retrieval on a hidden test set
- **Corpus scale:** Approximately 2.5 million Swiss court-decision considerations, together with a large federal-law article collection used for citation retrieval

The score and ranking above refer to the final competition submission. Additional validation and ablation results are documented in [`docs/experiment-log.md`](docs/experiment-log.md).

## Why this system

Legal retrieval is not only a semantic-similarity problem. A relevant article may be expressed differently from the query, may be referenced indirectly, or may be connected to the answer through related legal provisions and court decisions.

The system therefore combines:

- Query decomposition and German legal query generation
- Dense embedding retrieval over law articles
- Reciprocal Rank Fusion across multiple subqueries
- Exact citation and legal-book hint extraction
- Law-reference graph expansion
- Court-law co-citation expansion
- Conservative TF-IDF reranking
- Cross-encoder reranking
- LLM-based citation filtering with confirmation passes

## System architecture

```text
English legal question
          │
          ▼
German translation and legal query decomposition
          │
          ▼
Subquery embeddings ───────────────┐
          │                        │
          ▼                        │
Dense law-article retrieval        │
          │                        │
          ▼                        │
RRF candidate fusion               │
          │                        │
          ├── Citation and book hints
          ├── Law-reference graph expansion
          └── Court-law co-citation expansion
          │
          ▼
Court-law TF-IDF reranking
          │
          ▼
Cross-encoder reranking
          │
          ▼
LLM citation elimination and confirmation
          │
          ▼
Ranked Swiss legal citations
```

## Technical approach

### 1. Query decomposition

The original legal scenario is translated and decomposed into focused German legal subqueries. The decomposition targets explicit legal questions, statutory definitions, legal characterization, remedies, procedural pathways, and general legal principles.

### 2. Dense retrieval and RRF

Each subquery is embedded and searched against law-article embeddings. The top candidates from all subqueries are merged using Reciprocal Rank Fusion so that articles supported by multiple legal perspectives receive stronger rankings.

### 3. Citation-graph expansion

The pipeline models relationships between legal sources through article references, book-level relationships, and court-document co-citations. Retrieved articles can therefore expand to connected authorities that may not be close to the original query in embedding space.

### 4. Multi-stage reranking

The candidate pool is refined using court-law co-citation signals, TF-IDF-style weighting, and a cross-encoder. Procedural-law positions can be protected during semantic reranking because general-purpose rerankers may undervalue procedural authorities.

### 5. LLM citation filtering

The final stage evaluates lower-ranked candidates against the German query, the candidate article text, and court-usage context. A second confirmation pass is used before removing a citation, making the filter conservative rather than allowing the LLM to replace the retrieval system entirely.

## Repository structure

```text
code/
├── prepare/                 # Corpus preparation, translation, and embeddings
├── prompts/                 # LLM prompt builders
├── src/legal_rag/            # Stable, testable public package primitives
│   ├── retrieval/            # RRF and score normalization
│   ├── graph/                # Citation-neighbor expansion
│   ├── reranking/            # Multi-signal score combination
│   ├── evaluation/           # Citation-level metrics
│   └── llm/                  # Deterministic LLM-decision parsing and safety policy
├── dev/
│   ├── gen-sub/             # Query decomposition and subquery embeddings
│   ├── law-only/            # Law retrieval, graph expansion, reranking
│   └── utils/               # Embedding, reranker, LLM, and model-serving clients
└── archives/                # Historical preprocessing utilities

scripts/
├── run_law_retrieval.py      # Public law-retrieval entry point
└── run_llm_filter.py         # Public LLM-filter entry point

tests/
└── test_core_primitives.py   # Retrieval, graph, and metric tests

configs/
└── default.toml              # Documented research defaults

examples/
└── sample_query.json          # Input for the data-free smoke demo

docs/
├── pipeline.md              # Detailed six-stage pipeline description
├── experiment-log.md        # Experiment history and ablations
├── architecture.md         # System design and component responsibilities
├── migration-map.md         # Public package vs. legacy research code
├── reproduction.md          # Reproduction levels and execution guide
├── evaluation.md            # Metrics and result interpretation
├── data.md                  # Data policy, schemas, and setup requirements
└── procedural-law-books.md  # Procedural-law slot configuration
```

## Reproducibility

The source code and methods are public. Competition data, hidden labels, test queries, and derived labels are intentionally not committed to this repository. See [`docs/data.md`](docs/data.md) for the data boundary and required local file layout.

The full research pipeline requires precomputed corpus files, embeddings, and model/API configuration. A data-free smoke demo is available for inspecting the public package mechanics.

## Running the research pipeline

The current research entry points and configuration are documented in [`docs/reproduction.md`](docs/reproduction.md). The most detailed explanation of the final retrieval stages is [`docs/pipeline.md`](docs/pipeline.md).

The stable public entry points are:

```bash
python scripts/run_law_retrieval.py --help
python scripts/run_llm_filter.py --help
```

They currently delegate to the validated research implementation under `code/` while the internal modules are being refactored into the `legal_rag` package.

## Quick smoke demo

The repository includes a data-free demonstration of query-ranking fusion and
one-hop citation-graph expansion:

```bash
python scripts/run_demo.py
```

This uses a tiny in-memory example and does not represent a benchmark score or
legal advice. The full architecture is also available as
[`docs/architecture.mmd`](docs/architecture.mmd).

## Limitations and responsible use

This is a legal information retrieval system, not a legal-advice system. It ranks potentially relevant sources and does not determine the correct legal interpretation of a matter. Results can be incomplete, noisy, or sensitive to the available corpus, language, model, and evaluation protocol.

## License

The source code is released under the [MIT License](LICENSE). Competition data, hidden labels, test queries, and derived labels are not included and remain subject to the Kaggle competition rules and their original licenses.

## Competition

[LLM Agentic Legal Information Retrieval on Kaggle](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)
