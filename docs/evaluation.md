# Evaluation

## Primary metric

The competition evaluates citation-level retrieval using macro F1 on a hidden test set. The completed private leaderboard places this project **44th of 584 teams (top 8%)**, with a final private Macro F1 of **0.22252**.

The best documented public-leaderboard Macro F1 is **0.27569**, compared with the project's initial public submission of **0.17058**. This is an absolute increase of **0.10511** and a relative improvement of **61.62%**.

Public and private leaderboard scores are computed on different hidden-test subsets. They must be reported separately and must not be presented as directly interchangeable.

## Result summary

| Result | Value |
| --- | ---: |
| Final private leaderboard rank | 44 / 584 teams |
| Final percentile | Top 8% |
| Final private Macro F1 | 0.22252 |
| Best documented public Macro F1 | 0.27569 |
| Initial public Macro F1 | 0.17058 |
| Public-score relative improvement | 61.62% |

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
- whether the result is local validation, public leaderboard, or final private leaderboard performance

## Result hierarchy

The public README should lead with the final rank and contextualize it with the number of teams. It may show the best public score as a secondary result, provided that the public/private distinction is explicit. Detailed experiments should remain in `docs/experiment-log.md` so readers can distinguish leaderboard results from local validation.
