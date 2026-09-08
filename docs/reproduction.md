# Reproduction guide

## Reproduction levels

The repository is organized around three levels of reproducibility:

1. **Code inspection:** understand the complete retrieval and reranking method without downloading competition data.
2. **Research reproduction:** obtain permitted competition inputs locally, configure models and credentials, then run the documented stages.
3. **Full competition reproduction:** run the complete pipeline with the original corpus, embeddings, model configuration, and evaluation protocol.

## Environment

The current scripts use Python and OpenAI-compatible model APIs. Install the dependencies from `requirements.txt`, then create a local `.env` file for any required credentials.

The dependency list includes the scientific Python packages imported by the current source tree. Model-serving binaries and heavyweight model runtimes remain optional and depend on the selected execution backend.

```bash
python -m pip install -r requirements.txt
```

The LLM filtering entry point imports the OpenAI-compatible client at startup, so the dependency installation is required even when requesting command help.

Run the dependency-light core tests with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Check the local environment without requiring competition data:

```bash
python scripts/check_environment.py
```

Use `--strict` in CI or before a full run to fail when declared dependencies
or the required local corpus files are missing.

Run the public data-free smoke demo:

```bash
python scripts/run_demo.py
```

The demo validates the public package boundary and illustrates RRF plus
one-hop graph expansion. It is intentionally separate from competition
evaluation and uses no competition data.

The repository also runs these checks automatically through GitHub Actions on
pushes and pull requests.

## Pipeline documentation

Read [`pipeline.md`](pipeline.md) before running the full system. It records the inputs, outputs, parameters, and commands for:

1. German subquery embedding retrieval
2. Legal-book and domain boosting
3. Law-reference and court-law graph expansion
4. Conservative court-law TF-IDF reranking
5. Weighted cross-encoder reranking
6. Two-stage LLM citation elimination

## Configuration requirements

The full pipeline requires locally available corpus and embedding files, plus any credentials needed by the selected embedding, cross-encoder, or LLM backend. Do not place those files or credentials in Git.

## Reproduction status

The current repository is a curated public code release. A small, data-free smoke test and a fully pinned environment are follow-up hardening tasks; the original competition run depended on private/local artifacts that are intentionally not redistributed.
