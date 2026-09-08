# Law-Side Retrieval Pipeline

This document describes the final law-side retrieval pipeline that produced the current best Kaggle state.

The non-LLM retrieval stack is run from `code/dev/law-only/lawside-retrieval/law_document_side_retrieval.py`.

The final LLM citation-elimination layer is run from `code/dev/law-only/lawside-retrieval/trials/llm_law_submission_reranker.py`.

# Step 1: German Subquery Embedding Retrieval

**Input:** German subqueries with zembed embeddings

**Output:** Initial embedding-based ranked law candidates per original query

## Process

For each original query, we take its generated German subqueries.

Each German subquery has its own `zembed_embedding`.

For every German subquery, we compare its embedding against the law article embeddings in `data/laws_db.parquet`.

For each German subquery, we keep only the top `200` most similar law articles.

Then, for each original query, we merge all of its German-subquery top-200 law lists using RRF.

The RRF score is:

```text
score += 1 / (60 + rank)
```

So if the same law article appears strongly across multiple German subqueries, it receives more RRF score and moves higher in the original query's candidate ranking.

If two law articles have similar RRF scores, the best cosine similarity seen across the German subqueries is used as a tie-breaker.

The result is one initial embedding-based ranked law candidate list per original query.

At this stage, the ranking only uses German subquery-to-law embedding retrieval plus RRF merging. It does not yet include book boosting, exact citation hints, court-law expansion, TF-IDF reranking, cross-encoder reranking, or LLM filtering.

## Data Required

`data/test.parquet`

Provides the original `query_id` values, so German subquery results can be grouped back to the correct original query.

`data/test_sub_query_zembed.parquet`

Contains the generated German subqueries and their `zembed_embedding`. This is the main retrieval input for this step.

`data/laws_db.parquet`

Contains the law article citations and law embeddings searched against by the German subquery embeddings.

# Step 2: Query Hint Parsing and Book/Domain Boosting

**Input:** Raw RRF-ranked law candidates, German query/subqueries, law-domain embeddings, and book-network data

**Output:** Boosted law candidate list with exact citation and likely-book signals applied

## Process

After the initial German subquery embedding retrieval, the pipeline prepares extra signals that help identify which law books are likely relevant for the original query.

First, it parses law hints from the original query text, German translation text, decomposed query text, pre-parsed citation columns when present, and German subquery text.

This includes exact law citation hints when they are explicitly mentioned, for example:

```text
Art. 30 BV
Art. 97 Abs. 1 OR
```

From those parsed citations, the pipeline also extracts book-code hints, for example:

```text
BV
OR
```

These are not fixed book codes. They depend entirely on what is found in the query/subquery text.

Then, the pipeline uses `data/zembed_law_domain_embedding.parquet` to score likely law domains. These domain embeddings are averaged embeddings for groups of law articles inside a law book or legal domain.

For each original query, the German subquery embeddings are compared against these domain embeddings. For each German subquery, the current implementation keeps the top `10` matching law domains before merging the domain results with RRF.

Those domain matches are converted into book-level scores. If several highly ranked domains belong to the same book, that book gets stronger evidence.

The book score combines:

```text
domain match count
domain RRF score
best domain cosine similarity
```

The current weighting is:

```text
0.45 * normalized_domain_count
+ 0.45 * normalized_domain_rrf
+ 0.10 * normalized_best_domain_cosine
```

If the query parser extracted a book code from the query/subquery text, that book receives an additional hint weight.

Then the book scores are expanded through book-reference networks. These networks capture which law books tend to co-occur or cite each other.

The book networks are built from:

`data/laws_db.parquet`

Law-document references between law articles/books.

`data/train.parquet`

Book co-occurrence patterns from gold training citations.

`data/court_db.parquet`

Court-document law references used to build another book co-occurrence network.

After expansion, the top likely books are selected. In the current best setup, the pipeline keeps the top `4` book-network scores.

Finally, the raw RRF candidate list from Step 1 is rescored. A candidate law article can receive extra score if:

- it was directly parsed as an exact citation from the query/subquery text
- its book was directly parsed from the query/subquery text
- its book is one of the likely books from the domain/book-network step

Current boost values in the best setup are:

```text
exact citation boost = 0.10
direct book-code boost = 0.005
book-network boost = 0.01
book-network top books = 4
book-network hint weight = 0.25
book-network expansion weight = 0.40
```

The result is a boosted law candidate list that still starts from the German subquery RRF retrieval, but now has stronger ranking support for exact citations and likely legal books.

## Data Required

`data/test.parquet`

Provides the original query rows and query text used for parsing hints. When available, `german_translation`, `decomposed_queries_raw`, `pre_near_matches`, and `pre_citations` are also used by the hint parser.

`data/test_sub_query_zembed.parquet`

Provides the German subqueries, their text, and their `zembed_embedding`. The text is used for hint parsing, and the embeddings are used for law-domain matching.

`data/laws_db.parquet`

Provides law citations, source books, law references, and law-book reference information used for candidate scoring and law-book networks.

`data/zembed_law_domain_embedding.parquet`

Provides precomputed domain-level law embeddings. These are used to identify likely relevant law domains and books.

`data/train.parquet`

Provides gold citation co-occurrence patterns used to build one of the book-reference networks.

`data/court_db.parquet`

Provides court-document law references used to build another book co-occurrence network.

# Step 3: Law-Reference Graph Expansion and Court-Law Exact-Anchor Expansion

**Input:** Boosted law candidates, parsed exact citation hints, law-reference graph data, and court-law co-citation data

**Output:** Wider boosted candidate set with law-reference neighbors and court-law exact-anchor neighbors added

## Process

This step adds new candidate laws, not just boosts existing ones.

There are two expansion signals.

First, the pipeline uses the law-reference graph from `data/laws_db.parquet`.

The law corpus has article-to-article references. The pipeline builds incoming and outgoing law-reference neighbors from those reference columns. Then, from the top raw retrieved candidates, it can add nearby referenced laws.

Current defaults:

```text
graph_mode = both
graph_seed_k = 200
graph_boost = 0.1
```

So it looks at the top `200` candidates from the German subquery RRF list, then adds and boosts both incoming and outgoing law-reference neighbors.

Second, the pipeline uses court-law exact-anchor expansion from `data/court_db.parquet`.

Here the idea is: if the query already strongly points to some exact law article, or the first retrieved laws are strong anchors, then we look in court documents for laws that commonly appear with those anchor laws.

Current defaults:

```text
court-law exact-anchor ranked_k = 5
court-law exact-anchor top_targets = 50
court-law exact-anchor min_count = 2
court-law exact-anchor boost = 0.08
```

So it uses:

- exact citations parsed from the query/subqueries
- plus the top `5` current ranked laws

as anchors, then expands to up to `50` court co-cited target laws per anchor, only where the co-citation count is at least `2`.

The output of this step is a wider boosted candidate set. It still keeps the original embedding/RRF candidates, but now also includes law-reference neighbors and court-law exact-anchor neighbors.

## Data Required

`data/laws_db.parquet`

Provides article-to-article law references used to build incoming and outgoing law-reference graph neighbors.

`data/court_db.parquet`

Provides court-document law references used to build court-law co-citation relationships for exact-anchor expansion.

# Step 4: Conservative Court-Law TF-IDF Reranking

**Input:** Expanded candidate set from German subquery retrieval, book/domain boosts, law-reference graph expansion, and court-law exact-anchor expansion

**Output:** Conservatively reranked candidate pool using court-document law co-citation strength

## Process

After Step 3, the pipeline has a wider candidate set from:

- German subquery RRF retrieval
- query/book/domain boosts
- law-reference graph expansion
- court-law exact-anchor expansion

Now the pipeline applies a conservative TF-IDF rerank using court-document law co-citations.

The idea is:

If the current candidate set contains anchor laws, we look at court documents that cite those laws. Then we count what other laws are frequently cited in those same court documents. Common laws that appear everywhere should not dominate, so the score uses an IDF-style penalty.

Current defaults:

```text
court-law tfidf candidate_k = 100
court-law tfidf top_targets_per_anchor = 50
court-law tfidf min_count = 2
court-law tfidf boost = 0.0025
```

So the process is:

1. Take the top `100` candidates after the previous boosts and expansions.

2. Use those candidates as anchors into the court-law TF-IDF index built from `data/court_db.parquet`.

3. For each anchor, find up to `50` related law targets from court co-citation.

4. Apply a small TF-IDF-style boost of `0.0025`.

5. Rerank conservatively, so TF-IDF acts as a small ordering signal rather than a separate retrieval system.

The output is a reranked top candidate pool that keeps the previous retrieval structure, but slightly promotes laws that are strongly supported by court co-citation patterns.

## Data Required

`data/court_db.parquet`

Provides court-document law references used to build the court-law TF-IDF co-citation index.

# Step 5: Weighted Cross-Encoder Reranking on Top 50

**Input:** Top candidate pool after conservative court-law TF-IDF reranking

**Output:** Cross-encoder-weighted top-20 law candidates

## Process

After the conservative court-law TF-IDF rerank, the pipeline applies the cross-encoder reranking layer.

This is not a full replacement of the ranking. It is a weighted rerank over the top `50` candidate laws.

Current defaults from the best setup:

```text
cross_encoder_rerank_top_k = 50
cross_encoder_rerank_mode = weighted
cross_encoder_rerank_model = zerank-2

tfidf_weight = 0.40
cosine_weight = 0.20
cross_encoder_weight = 0.40
```

For each query, the pipeline takes the top `50` law candidates from the previous stage.

For each candidate, it combines three signals:

```text
court-law TF-IDF score
embedding cosine score
cross-encoder score
```

The weighted score is:

```text
0.40 * normalized_tfidf_score
+ 0.20 * normalized_cosine_score
+ 0.40 * normalized_cross_encoder_score
```

The cross-encoder uses the German query text and the law article text to judge whether the article is relevant to the query.

There is one important constraint:

Procedural-law slots are frozen during this cross-encoder step.

That means procedural candidates are not sent to the cross-encoder for semantic reranking. Their positions are preserved, because procedural relevance can be hard for general rerankers to judge correctly.

## Data Required

`data/test.parquet`

Provides the German query text used by the cross-encoder.

`data/laws_db.parquet`

Provides law article text for the candidate citations being reranked.

`context/proc_vs_subs.md`

Provides the procedural law book-code list used to freeze procedural-law slots during cross-encoder reranking.

`context/test_scores_cross_encoder_top50_substantive_freeze_proc.csv`

Stores or reuses cached cross-encoder scores so the same query/citation pairs do not need to be rescored repeatedly.

`.env`

Provides the API key used by the cross-encoder reranker.

# Step 6: Two-Stage LLM Citation Elimination With Court Context

**Input:** Top-20 law candidates from the weighted cross-encoder reranking stage

**Output:** Final law citation submission after LLM citation elimination

## Process

This step is handled by `code/dev/law-only/lawside-retrieval/trials/llm_law_submission_reranker.py`.

It takes the non-LLM top-20 submission from Step 5 and applies an LLM filter to remove citations that look relevant but are probably not citation-worthy.

The key idea is:

We do not ask the LLM to rerank everything. We ask it to eliminate weak citations.

Current best setup:

```text
rerank_mode = conservative-court-context
model = qwen/qwen3.6-35b-a3b
reasoning = enabled
freeze_top_k = 5
first_pass_batch_size = 2
confirmation_batch_size = 2
court_context_top_k = 1
court_context_max_chars = 0
temperature = 0
parallel_requests = 10
```

The process is:

1. Start from the top-20 candidates generated by Step 5.

2. Freeze ranks `1-5`. These are kept without LLM filtering.

3. Send ranks `6-20` to the LLM.

This LLM stage does not skip procedural laws. The only automatic freeze rule is rank-based: ranks `1-5` are kept. Ranks `6-20` are eligible for LLM filtering when their law article text exists.

4. For each candidate law, include:

- German query
- candidate citation
- law article text from `data/laws_db.parquet`
- one court usage context from `data/court_db.parquet`

5. The first LLM pass decides `KEEP` or proposes `REMOVE`.

6. Any proposed removal goes into a second confirmation pass.

7. A citation is removed only if the confirmation pass also says `REMOVE`.

This was the best Kaggle setup:

```text
Base retrieval before LLM: 0.23573
Two-stage LLM with court context: 0.25869
```

So this final layer improves precision by removing noisy lower-ranked citations while keeping the high-confidence top ranks safe.

## Data Required

`context/test_submission_tfidf_cosine_ce_weighted_top50.csv`

Provides the Step 5 top-20 law candidates used as input to the LLM elimination layer.

`data/test.parquet`

Provides the German query text.

`data/laws_db.parquet`

Provides the law article text for each candidate citation.

`data/court_db.parquet`

Provides court-document usage context for candidate laws.

`.env`

Provides the `OPENROUTER_API_KEY` used for the LLM calls.

# Reproduce Current Best Outputs

## Base Non-LLM Retrieval

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/dev/law-only/lawside-retrieval/law_document_side_retrieval.py \
  --test-path data/test.parquet \
  --test-subquery-path data/test_sub_query_zembed.parquet \
  --output-path context/test_submission_tfidf_cosine_ce_weighted_top50.csv \
  --detail-path context/test_details_tfidf_cosine_ce_weighted_top50.csv \
  --hint-path context/test_hints_tfidf_cosine_ce_weighted_top50.csv \
  --cross-encoder-rerank-score-path context/test_scores_cross_encoder_top50_substantive_freeze_proc.csv
```

This produces the KAGGLE-STATE-3 base submission:

```text
context/test_submission_tfidf_cosine_ce_weighted_top50.csv
```

Kaggle score:

```text
0.23573
```

## Final LLM-Filtered Submission

```bash
set -a; source .env; set +a

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/dev/law-only/lawside-retrieval/trials/llm_law_submission_reranker.py \
  --dataset test \
  --rerank-mode conservative-court-context \
  --submission-path context/test_submission_tfidf_cosine_ce_weighted_top50.csv \
  --llm-backend openrouter \
  --model qwen/qwen3.6-35b-a3b \
  --reasoning-enabled \
  --row-start 1 \
  --row-end 40 \
  --parallel-requests 10 \
  --max-tokens 5000 \
  --temperature 0 \
  --top-p 1 \
  --top-k 1 \
  --court-context-freeze-top-k 5 \
  --court-context-batch-size 2 \
  --court-context-confirm-removals \
  --court-context-confirm-batch-size 2 \
  --court-context-top-k 1 \
  --court-context-max-chars 0 \
  --output-path context/test_submission_openrouter_qwen36_reasoning_confirm.csv \
  --prediction-output-path context/test_llm_predictions_openrouter_qwen36_reasoning_confirm.csv \
  --batch-output-path context/test_llm_batches_openrouter_qwen36_reasoning_confirm.parquet \
  --summary-output-path context/test_llm_summary_openrouter_qwen36_reasoning_confirm.csv \
  --overwrite
```

This produces the KAGGLE-STATE-4 best submission:

```text
context/test_submission_openrouter_qwen36_reasoning_confirm.csv
```

Kaggle score:

```text
0.25869
```
