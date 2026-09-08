# Data and repository policy

## Competition data boundary

This repository publishes source code, configuration examples, documentation, and methods. It does not publish the Kaggle competition data, hidden labels, test queries, or derived labels.

The competition rules allow code and methods to be shared publicly but restrict redistributing competition data. Users must obtain the data through the permitted Kaggle workflow and comply with the competition rules and any source-data licenses.

## Corpus description

The competition combines Swiss legal sources, including federal law articles and a large court-decision corpus. Public descriptions of the competition corpus commonly report approximately 2.5 million court-decision considerations in addition to the law-article collection. The exact number depends on the competition release and preprocessing; it should not be interpreted as 2.5 million independent statutes.

## Expected local inputs

The research pipeline refers to local files such as:

- `data/test.parquet`
- `data/train.parquet`
- `data/laws_db.parquet`
- `data/court_db.parquet`
- `data/zembed_law_domain_embedding.parquet`
- generated subquery and embedding parquet files

These files are intentionally excluded from version control. Before running the full pipeline, verify the expected columns in the corresponding scripts and use only data obtained through an authorized competition or dataset workflow.

## Secrets

Never commit API keys, tokens, private data, hidden labels, or generated files derived from hidden evaluation data. Use a local `.env` file and keep it ignored by Git.
