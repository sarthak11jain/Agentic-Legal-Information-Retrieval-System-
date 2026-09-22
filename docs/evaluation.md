# Evaluation and result provenance

The supported demo reports Macro F1 only against its bundled synthetic labels.
That score validates deterministic behavior; it is not a legal-retrieval
benchmark result.

| Result | Value | Provenance |
| --- | ---: | --- |
| Final private leaderboard rank | 44 / 584 teams | Kaggle private leaderboard |
| Final private Macro F1 | 0.22252 | Kaggle private leaderboard |
| Best documented public Macro F1 | 0.27569 | Project experiment record |
| Initial public Macro F1 | 0.17058 | Project experiment record |

Public and private leaderboard scores are calculated on different hidden-test
splits and must never be compared as equivalent measurements. Treat these as
attributed competition results. A release must record its source URL, code
revision, configuration, backend/model, split, and score before publishing a
new claim.
