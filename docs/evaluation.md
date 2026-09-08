# Evaluation

## Primary metric

The competition evaluates citation-level retrieval using macro F1 on a hidden test set. The final competition submission associated with this project achieved a macro F1 score of **0.23** and a **top-50 Kaggle ranking**.

The score above is the official CV-facing result. The experiment log contains additional local validation and ablation scores; those numbers must always be labeled with their split and evaluation protocol.

## Recommended reporting format

When adding new results, record:

- dataset split
- candidate depth
- retrieval configuration
- embedding model
- reranker model
- LLM configuration, if used
- macro F1 and micro F1 where available
- recall@K where available
- whether the result is local validation or Kaggle leaderboard performance

## Result hierarchy

The public README should lead with the final official submission result. Detailed experiments should remain in `docs/experiment-log.md` so readers can distinguish the headline result from intermediate research findings.
