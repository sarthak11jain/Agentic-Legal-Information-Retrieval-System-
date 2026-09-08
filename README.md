<div align="center">

# Graph-Augmented Legal Information Retrieval System

**An agentic retrieval pipeline for discovering citation-level Swiss legal sources for complex legal questions.**

[![CI](https://github.com/sarthak11jain/Agentic-Legal-Information-Retrieval-System-/actions/workflows/ci.yml/badge.svg)](https://github.com/sarthak11jain/Agentic-Legal-Information-Retrieval-System-/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Competition](https://img.shields.io/badge/Kaggle-Top%2050-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)

</div>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#results">Results</a> ·
  <a href="#system-architecture">Architecture</a> ·
  <a href="#reproduce">Reproduce</a> ·
  <a href="#repository-map">Repository</a>
</p>

> **Research project.** This repository contains the public implementation and technical documentation for a Kaggle legal information retrieval system. It is designed for source discovery and experimentation, not legal advice.

## Overview

The system retrieves Swiss legal citations relevant to a natural-language legal scenario. It treats retrieval as a structured reasoning problem: a relevant authority may be expressed in another language, referenced indirectly, or connected to the query through a chain of legal citations.

The pipeline combines multilingual query decomposition, dense retrieval, reciprocal-rank fusion, legal citation graphs, court-law co-citation signals, cross-encoder reranking, and conservative LLM-based filtering.

### Why graph augmentation?

Pure semantic similarity is useful for finding candidates, but it does not fully represent how legal sources relate to one another. Citation edges provide an additional signal: a retrieved article can lead to connected provisions and court decisions that are not among the nearest embedding neighbors.

## Results

| Metric | Final submission |
| --- | ---: |
| Macro F1 | **0.23** |
| Kaggle leaderboard | **Top 50** |
| Retrieval target | Citation-level legal sources |
| Corpus scale | Approximately **2.5M court-decision considerations**, together with a large federal-law article collection |

The score and ranking above refer to the final competition submission. Additional validation and ablation results are recorded in [`docs/experiment-log.md`](docs/experiment-log.md).

<div align="center">

**Final competition submission · Macro F1 0.23 · Top-50 Kaggle result**

</div>

## System architecture

```mermaid
flowchart TD
    Q["English legal question"] --> T["German translation"]
    T --> D["Query decomposition"]
    D --> E["Subquery embeddings"]
    E --> R["Dense law-article retrieval"]
    R --> F["Reciprocal Rank Fusion"]
    F --> H{"Structured retrieval signals"}
    H --> C["Citation and legal-book hints"]
    H --> G["Law-reference graph expansion"]
    H --> X["Court-law co-citation expansion"]
    C --> P["Candidate pool"]
    G --> P
    X --> P
    P --> W["Court-law TF-IDF reranking"]
    W --> CE["Cross-encoder reranking"]
    CE --> L["LLM relevance filtering"]
    L --> V["Confirmation pass"]
    V --> O["Ranked Swiss legal citations"]

    classDef input fill:#eef2ff,stroke:#6366f1,color:#1e293b;
    classDef retrieval fill:#dcfce7,stroke:#22c55e,color:#14532d;
    classDef graph fill:#fef3c7,stroke:#f59e0b,color:#78350f;
    classDef rerank fill:#fae8ff,stroke:#d946ef,color:#701a75;
    classDef output fill:#cffafe,stroke:#06b6d4,color:#164e63;
    class Q input;
    class T,D,E,R,F retrieval;
    class H,C,G,X graph;
    class P,W,CE,L,V rerank;
    class O output;
```

### Retrieval stages

| Stage | Role | Main signal |
| --- | --- | --- |
| Query understanding | Translate and decompose the scenario into focused German legal subqueries | Legal intent coverage |
| Candidate generation | Search law-article embeddings for every subquery | Dense semantic similarity |
| Candidate fusion | Merge subquery rankings | Reciprocal Rank Fusion |
| Graph expansion | Follow article references and court-law co-citation relationships | Structural connectivity |
| Reranking | Refine the candidate pool and protect important procedural authorities | TF-IDF, co-citation, cross-encoder scores |
| Final filtering | Remove weak citations only when an LLM and confirmation pass agree | LLM relevance judgment |

## Technical approach

### 1. Query decomposition

The original scenario is translated and decomposed into focused German legal subqueries covering explicit legal questions, statutory definitions, legal characterization, remedies, procedural pathways, and general legal principles.

### 2. Dense retrieval and reciprocal-rank fusion

Each subquery is embedded and searched against law-article embeddings. Reciprocal Rank Fusion rewards candidates supported by multiple legal perspectives while preserving useful diversity across subqueries.

### 3. Citation-graph expansion

The system models relationships between legal sources through article references, book-level relationships, and court-document co-citations. This allows retrieval to move beyond the semantic neighborhood of the initial query.

### 4. Multi-stage reranking

The expanded candidate pool is refined using court-law co-citation features, TF-IDF-style weighting, and a cross-encoder. Procedural-law positions can be protected because general-purpose rerankers may undervalue procedural authorities.

### 5. Conservative LLM filtering

The final stage evaluates lower-ranked candidates using the German query, candidate article text, and court-usage context. A second confirmation pass is required before a citation is removed, keeping the LLM as a conservative filter rather than allowing it to replace retrieval.

## Research design

```mermaid
flowchart LR
    A["Multilingual legal query"] --> B["Candidate generation"]
    B --> C["Graph-aware expansion"]
    C --> D["Reranking and filtering"]
    D --> E["Citation-level predictions"]
    E --> F["Macro F1 evaluation"]

    B -.-> B1["Dense embeddings + RRF"]
    C -.-> C1["Article references + court-law links"]
    D -.-> D1["Cross-encoder + LLM confirmation"]
```

The public repository separates stable, testable primitives from the original research scripts. This makes the core retrieval, graph, reranking, metric, and LLM-decision logic easier to inspect without pretending that the full competition environment can be recreated without its data and model services.

## Reproduce

### Install

```bash
python -m venv .venv
.venv\\Scripts\\activate        # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### Run the data-free smoke demo

```bash
python scripts/run_demo.py
```

The demo exercises query-ranking fusion and one-hop citation-graph expansion using a tiny in-memory example. It is intentionally not a benchmark evaluation and does not require competition data.

### Inspect the research entry points

```bash
python scripts/check_environment.py
python scripts/run_law_retrieval.py --help
python scripts/run_llm_filter.py --help
```

The full research pipeline requires locally available competition files, precomputed embeddings, model/API configuration, and the directory layout documented in [`docs/reproduction.md`](docs/reproduction.md). The data boundary is described in [`docs/data.md`](docs/data.md).

## Repository map

```text
.
├── code/
│   ├── prepare/          # Corpus preparation, translation, and embeddings
│   ├── prompts/          # LLM prompt builders
│   ├── dev/              # Retrieval, graph, reranking, and service utilities
│   └── archives/         # Historical preprocessing utilities
├── src/legal_rag/        # Stable, testable public package primitives
│   ├── retrieval/        # RRF and score normalization
│   ├── graph/            # Citation-neighbor expansion
│   ├── reranking/        # Multi-signal score combination
│   ├── evaluation/       # Citation-level metrics
│   └── llm/              # Decision parsing and confirmation policy
├── scripts/              # Public entry points and smoke demo
├── configs/              # Documented research defaults
├── examples/             # Data-free example input
├── tests/                # Unit tests for core primitives
└── docs/                 # Architecture, evaluation, reproduction, and data notes
```

### Documentation guide

| Document | Purpose |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | Component responsibilities and system design |
| [`docs/pipeline.md`](docs/pipeline.md) | Detailed retrieval pipeline |
| [`docs/evaluation.md`](docs/evaluation.md) | Metrics and result interpretation |
| [`docs/experiment-log.md`](docs/experiment-log.md) | Experiments and ablations |
| [`docs/reproduction.md`](docs/reproduction.md) | Execution levels and setup |
| [`docs/data.md`](docs/data.md) | Data boundary and local schemas |
| [`docs/migration-map.md`](docs/migration-map.md) | Relationship between public package and research scripts |
| [`docs/architecture.mmd`](docs/architecture.mmd) | Editable Mermaid architecture source |

## Quality checks

The repository runs lightweight CI checks on every push and pull request:

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q src scripts code
python scripts/run_demo.py
```

## Data and responsible use

Competition data, hidden labels, test queries, and derived labels are intentionally not committed to this repository. This keeps the public code aligned with the competition's data-sharing restrictions. See [`docs/data.md`](docs/data.md) for the local file contract.

This is a legal information retrieval system, not a legal-advice system. It ranks potentially relevant sources and does not determine the correct legal interpretation of a matter. Results may be incomplete, noisy, or sensitive to the corpus, language, model, and evaluation protocol.

## Competition and citation

- [LLM Agentic Legal Information Retrieval on Kaggle](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)
- [`CITATION.cff`](CITATION.cff)
- [`LICENSE`](LICENSE)

## License

The source code is released under the [MIT License](LICENSE). Competition data and derived artifacts are not included and remain subject to their original terms and the competition rules.
