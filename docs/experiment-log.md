STARTING-STATE

$ python3 code/dev/law-only/lawside-retrieval/generate_zembed_combined_law_submission.py --test-path data/test.parquet --test-subquery-path data/test_sub_query_zembed.parquet --output-path context/test_submission_zembed_tuned_top4.csv --detail-path context/test_details_zembed_tuned_top4.csv --hint-path context/test_hints_zembed_tuned_top4.csv --book-network-top-books 4 --book-network-boost 0.01 --book-boost 0.005 --graph-mode both --graph-seed-k 200 --graph-boost 0.1


this is giving us a kaggle score of 0.17058

we have done no cross encoder reranking and no llm reranking 

LOCAL-STATE-1

Val baseline for court-law-reference trials:
- tuned law_document_side_retrieval settings: book_network_top_books=4, book_network_boost=0.01, book_boost=0.005, graph_mode=both, graph_seed_k=200, graph_boost=0.1
- top20: macro F1 0.1873, micro F1 0.1834, hits 32/149
- recall@100: macro 0.4442, micro 0.3893, hits 58/149

Court-law-reference trial results:

1. Naive court law co-citation graph
- idea: top retrieved laws expand to other laws co-cited in court_db.law_references
- flags: court_law_graph_boost=0.08, court_law_graph_seed_k=100, court_law_graph_min_count=2
- top20: macro F1 0.1899, micro F1 0.1891, hits 33/149
- recall@100: macro 0.4492, micro 0.3960, hits 59/149
- read: weak signal, mostly small reshuffling.

2. Exact-anchor court expansion
- idea: parsed exact query law citations expand through court_db.law_references co-citations
- flags: court_law_exact_anchor_boost=0.08, court_law_exact_anchor_top_targets=50, court_law_exact_anchor_min_count=2
- top20: macro F1 0.2103, micro F1 0.2063, hits 36/149
- recall@100: macro 0.4509, micro 0.3960, hits 59/149
- read: useful precision/top20 ranking signal.

3. Exact-anchor + top5 retrieved anchors
- idea: parsed exact query law citations plus ranked[:5] expand through court_db.law_references co-citations
- flags: same as exact-anchor, plus court_law_exact_anchor_ranked_k=5
- top20: macro F1 0.2192, micro F1 0.2178, hits 38/149
- recall@100: macro 0.4881, micro 0.4497, hits 67/149
- read: best of these three; more aggressive and can drift in broad books, but improves both top20 and recall@100.

KAGGLE-STATE-1

$ PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/dev/law-only/lawside-retrieval/law_document_side_retrieval.py --test-path data/test.parquet --test-subquery-path data/test_sub_query_zembed.parquet --output-path context/test_submission_law_doc_exact_anchor_top5.csv --detail-path context/test_details_law_doc_exact_anchor_top5.csv --hint-path context/test_hints_law_doc_exact_anchor_top5.csv --book-network-top-books 4 --book-network-boost 0.01 --book-boost 0.005 --graph-mode both --graph-seed-k 200 --graph-boost 0.1 --court-law-exact-anchor-boost 0.08 --court-law-exact-anchor-top-targets 50 --court-law-exact-anchor-min-count 2 --court-law-exact-anchor-ranked-k 5

this uses law_document_side_retrieval with the previous tuned law-side settings plus court-law exact-anchor expansion:
- parsed exact query law citations are used as anchors
- top 5 retrieved law citations are also used as anchors
- anchors expand through court_db.law_references co-citations
- expansion boost is 0.08, top targets per anchor is 50, min co-citation count is 2
- no court-law naive graph boost, no cross encoder reranking, and no llm reranking

Kaggle score: 0.19293

Improvement over STARTING-STATE:
- old Kaggle score: 0.17058
- new Kaggle score: 0.19293
- absolute gain: +0.02235
- relative gain: +13.10%


LOCAL-STATE-2

Court-law TF-IDF reranking trials:

1. Aggressive TF-IDF first, then exact-anchor + top5
- idea: use court_db.law_references as a TF-IDF co-citation reranker over top200 candidates, then apply the exact-anchor + top5 court expansion layer.
- command shape:
  - `law_document_side_retrieval.py`
  - `--court-law-tfidf-rerank-boost 0.04`
  - `--court-law-tfidf-rerank-candidate-k 200`
  - `--court-law-tfidf-rerank-top-targets 200`
  - `--court-law-exact-anchor-boost 0.08`
  - `--court-law-exact-anchor-ranked-k 5`
- val looked very promising:
  - top20 macro F1 0.2777
  - micro F1 0.2751
  - hits 48/149
  - recall@100 hits 71/149
- Kaggle dropped:
  - previous best Kaggle score: 0.19293
  - aggressive TF-IDF Kaggle score: 0.18961
  - absolute change: -0.00332
- read: this was too aggressive. It changed 390/800 test predictions compared with exact-anchor + top5, so TF-IDF became a second retrieval system rather than a light reranker. The val set is small, so the large val jump did not generalize.

2. Conservative exact-anchor first, then small TF-IDF rerank
- changed approach:
  - first run exact-anchor + top5 and keep top100 candidates
  - then apply TF-IDF only as a small final rerank inside that top100
  - TF-IDF boost 0.0025
  - top targets per anchor 50
- val was slightly lower than aggressive, but much less jumpy:
  - top20 macro F1 0.2670
  - micro F1 0.2636
  - hits 46/149
  - churn vs exact-anchor + top5 on val: 50/200 predictions
- generated file:
  - `context/test_submission_exact_anchor_top5_tfidf_conservative.csv`
- test churn vs exact-anchor + top5:
  - 183/800 predictions changed
- read: this kept most of the TF-IDF validation gain while being much less aggressive.

KAGGLE-STATE-2

Conservative TF-IDF rerank after exact-anchor + top5:

Step 1: generate exact-anchor + top5 top100 candidates:

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/dev/law-only/lawside-retrieval/law_document_side_retrieval.py --test-path data/test.parquet --test-subquery-path data/test_sub_query_zembed.parquet --output-path context/test_submission_law_doc_exact_anchor_top5_top100_for_conservative_tfidf.csv --detail-path context/test_details_law_doc_exact_anchor_top5_top100_for_conservative_tfidf.csv --hint-path context/test_hints_law_doc_exact_anchor_top5_top100_for_conservative_tfidf.csv --final-k 100 --book-network-top-books 4 --book-network-boost 0.01 --book-boost 0.005 --graph-mode both --graph-seed-k 200 --graph-boost 0.1 --court-law-exact-anchor-boost 0.08 --court-law-exact-anchor-top-targets 50 --court-law-exact-anchor-min-count 2 --court-law-exact-anchor-ranked-k 5`

Step 2: apply `trials.court_law_tfidf_rerank.rerank_court_law_tfidf_candidates` with:
- `boost=0.0025`
- `candidate_k=100`
- `top_targets_per_anchor=50`
- output top20 to `context/test_submission_exact_anchor_top5_tfidf_conservative.csv`

Kaggle score: 0.20865

Improvement over KAGGLE-STATE-1:
- old Kaggle score: 0.19293
- new Kaggle score: 0.20865
- absolute gain: +0.01572
- relative gain: +8.15%

Improvement over STARTING-STATE:
- old Kaggle score: 0.17058
- new Kaggle score: 0.20865
- absolute gain: +0.03807
- relative gain: +22.32%


LOCAL-STATE-3

Cross-encoder weighted reranking after conservative TF-IDF:

Best local setup:
- keep KAGGLE-STATE-2 as the candidate pipeline:
  - exact-anchor + top5 court-law expansion
  - standalone conservative court-law TF-IDF rerank inside top100
- then apply a final top50 rerank layer:
  - freeze procedural-law positions using `context/proc_vs_subs.md`
  - send only non-procedural laws to `zerank-2`
  - query text is `german_translation`
  - final combined score is `0.40 * tfidf_score + 0.20 * cosine_score + 0.40 * cross_encoder_score`

Val result on the art-only metric:
- conservative TF-IDF baseline:
  - macro F1 0.2670
  - micro F1 0.2636
  - hits 46/149
- weighted TF-IDF + cosine + CE final rerank:
  - macro F1 0.2934
  - micro F1 0.2865
  - hits 50/149
  - churn vs conservative baseline: 35/200 top20 predictions

All-gold val result:
- macro F1 0.2404
- micro F1 0.2217
- hits 50/251

Failed / rejected variants:
- aggressive TF-IDF-first experiment from LOCAL-STATE-2 looked strong on val:
  - macro F1 0.2777
  - hits 48/149
  - Kaggle dropped to 0.18961 from 0.19293
- root cause:
  - it changed 390/800 test predictions and turned TF-IDF into a second retrieval system instead of a light reranker
  - the small val set rewarded that over-aggression, but it did not generalize
- fix:
  - make exact-anchor + top5 the stable anchor expansion first
  - use conservative TF-IDF as a small top100 rerank
  - use cross-encoder as a weighted feature on top50, not as a hard full reorder
  - freeze procedural-law slots so the CE layer only moves substantive laws

No-standalone-TF-IDF ablation:
- exact-anchor + top5, then only final TF-IDF + cosine + CE rerank
- art-only val:
  - macro F1 0.2877
  - micro F1 0.2808
  - hits 49/149
- read:
  - worse than the layered setup, and more aggressive versus the conservative baseline
  - keep the standalone conservative TF-IDF layer before final CE weighting


KAGGLE-STATE-3

Best Kaggle setup:

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/dev/law-only/lawside-retrieval/law_document_side_retrieval.py --test-path data/test.parquet --test-subquery-path data/test_sub_query_zembed.parquet --output-path context/test_submission_tfidf_cosine_ce_weighted_top50.csv --detail-path context/test_details_tfidf_cosine_ce_weighted_top50.csv --hint-path context/test_hints_tfidf_cosine_ce_weighted_top50.csv --cross-encoder-rerank-score-path context/test_scores_cross_encoder_top50_substantive_freeze_proc.csv`

This now uses the default best settings in `law_document_side_retrieval.py`:
- exact-anchor + top5 court-law expansion
- standalone conservative TF-IDF rerank:
  - candidate_k=100
  - top_targets_per_anchor=50
  - boost=0.0025
- final cross-encoder weighted rerank:
  - top_k=50
  - mode=weighted
  - TF-IDF weight=0.40
  - cosine weight=0.20
  - cross-encoder weight=0.40
  - model=zerank-2
  - procedural-law slots frozen

Generated files:
- `context/test_submission_tfidf_cosine_ce_weighted_top50.csv`
- `context/test_details_tfidf_cosine_ce_weighted_top50.csv`
- `context/test_scores_cross_encoder_top50_substantive_freeze_proc.csv`

Kaggle score: 0.23573

Improvement over KAGGLE-STATE-2:
- old Kaggle score: 0.20865
- new Kaggle score: 0.23573
- absolute gain: +0.02708
- relative gain: +12.98%

Improvement over STARTING-STATE:
- old Kaggle score: 0.17058
- new Kaggle score: 0.23573
- absolute gain: +0.06515
- relative gain: +38.19%


KAGGLE-STATE-4

LLM citation-elimination layer on top of KAGGLE-STATE-3:

Base input:
- `context/test_submission_tfidf_cosine_ce_weighted_top50.csv`
- OpenRouter model: `qwen/qwen3.6-35b-a3b`
- reasoning enabled
- freeze ranks 1-5
- first-pass / confirmation output is a filtered top20 submission

Results:
- court docs in both first pass and confirmation: Kaggle score 0.25869
- court docs only in confirmation: Kaggle score 0.25570
- no court docs in either stage: Kaggle score 0.24481

Read:
- the two-stage LLM removal pipeline is useful even without court docs
- adding court docs only in confirmation improves over no-court, so court usage helps restore false removals
- adding court docs in both stages is still best, so the court passage is also useful during the first removal decision

Best submission, court docs in both stages:

`set -a; source .env; set +a`

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/dev/law-only/lawside-retrieval/trials/llm_law_submission_reranker.py --dataset test --rerank-mode conservative-court-context --submission-path context/test_submission_tfidf_cosine_ce_weighted_top50.csv --llm-backend openrouter --model qwen/qwen3.6-35b-a3b --reasoning-enabled --row-start 1 --row-end 40 --parallel-requests 10 --max-tokens 5000 --temperature 0 --top-p 1 --top-k 1 --court-context-freeze-top-k 5 --court-context-batch-size 2 --court-context-confirm-removals --court-context-confirm-batch-size 2 --court-context-top-k 1 --court-context-max-chars 0 --output-path context/test_submission_openrouter_qwen36_reasoning_confirm.csv --prediction-output-path context/test_llm_predictions_openrouter_qwen36_reasoning_confirm.csv --batch-output-path context/test_llm_batches_openrouter_qwen36_reasoning_confirm.parquet --summary-output-path context/test_llm_summary_openrouter_qwen36_reasoning_confirm.csv --overwrite`

Confirmation-only court docs ablation:

`set -a; source .env; set +a`

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/dev/law-only/lawside-retrieval/trials/llm_law_submission_reranker.py --dataset test --rerank-mode conservative-no-court-first-court-confirm --submission-path context/test_submission_tfidf_cosine_ce_weighted_top50.csv --llm-backend openrouter --model qwen/qwen3.6-35b-a3b --reasoning-enabled --row-start 1 --row-end 40 --parallel-requests 10 --max-tokens 5000 --temperature 0 --top-p 1 --top-k 1 --court-context-freeze-top-k 5 --no-court-batch-size 4 --court-context-confirm-removals --court-context-confirm-batch-size 2 --court-context-top-k 1 --court-context-max-chars 0 --output-path context/test_submission_openrouter_qwen36_hybrid_no_court_first_court_confirm.csv --prediction-output-path context/test_llm_predictions_openrouter_qwen36_hybrid_no_court_first_court_confirm.csv --batch-output-path context/test_llm_batches_openrouter_qwen36_hybrid_no_court_first_court_confirm.parquet --summary-output-path context/test_llm_summary_openrouter_qwen36_hybrid_no_court_first_court_confirm.csv --overwrite`

No-court ablation:

`set -a; source .env; set +a`

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/dev/law-only/lawside-retrieval/trials/llm_law_submission_reranker.py --dataset test --rerank-mode conservative-no-court-context --submission-path context/test_submission_tfidf_cosine_ce_weighted_top50.csv --llm-backend openrouter --model qwen/qwen3.6-35b-a3b --reasoning-enabled --row-start 1 --row-end 40 --parallel-requests 10 --max-tokens 5000 --temperature 0 --top-p 1 --top-k 1 --court-context-freeze-top-k 5 --no-court-batch-size 4 --court-context-confirm-removals --no-court-confirm-batch-size 4 --output-path context/test_submission_openrouter_qwen36_reasoning_confirm_no_court.csv --prediction-output-path context/test_llm_predictions_openrouter_qwen36_reasoning_confirm_no_court.csv --batch-output-path context/test_llm_batches_openrouter_qwen36_reasoning_confirm_no_court.parquet --summary-output-path context/test_llm_summary_openrouter_qwen36_reasoning_confirm_no_court.csv --overwrite`

Improvement over KAGGLE-STATE-3:
- old Kaggle score: 0.23573
- new Kaggle score: 0.25869
- absolute gain: +0.02296
- relative gain: +9.74%

Improvement over STARTING-STATE:
- old Kaggle score: 0.17058
- new Kaggle score: 0.25869
- absolute gain: +0.08811
- relative gain: +51.65%


LOCAL-STATE-5

Court-grounded law retrieval default pipeline:

- script: `code/dev/law-only/court-graph-retrieval/court_document_side_retrieval.py`
- idea: still produce law citation predictions, but use query-similar court documents as a bridge to find law articles cited in similar cases.
- default active path:
  - court retrieval mode: `per-subquery`
  - per-subquery courts: `20`
  - law similarity mode: `max-subquery`
  - score blend: `0.4 * law cosine + 0.6 * court graph score`
  - MMR: `lambda=0.8`, pool `50`, final top `20`

Validation command:

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/dev/law-only/court-graph-retrieval/court_document_side_retrieval.py --mode val --output-csv context/val_court_document_side_retrieval_default.csv`

Validation result:

- Val Art-only Macro F1@20: `0.3290`
- output: `context/val_court_document_side_retrieval_default.csv`


KAGGLE-STATE-5

Court-grounded law retrieval default submission:

Test submission command:

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/dev/law-only/court-graph-retrieval/court_document_side_retrieval.py --mode test --output-csv context/test_court_document_side_retrieval_default.csv`

Kaggle result:

- Kaggle score: `0.25226`
- output: `context/test_court_document_side_retrieval_default.csv`

Read:

- this is a separate law retrieval route from the law-document-side pipeline and does not use the final LLM citation filter.
- the strong score confirms that court-grounded law retrieval is independently useful: similar court documents provide high-quality law candidates through their cited law references.


LOCAL-STATE-6

Court-grounded law retrieval with conservative cross-encoder reranking:

- script: `code/dev/law-only/court-graph-retrieval/court_document_side_retrieval.py`
- idea: keep the court-grounded retrieval score as the dominant signal, then use `zerank-2` only as a light final rerank over the top `50` candidates.
- cross-encoder rerank setup:
  - top candidates sent to rerank layer: `50`
  - model: `zerank-2`
  - query text: `german_translation`
  - procedural-law slots frozen
  - final rerank blend: `0.80 * base_court_grounded_score + 0.20 * cross_encoder_score`
  - cosine rerank weight: `0.00`

Validation command:

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/dev/law-only/court-graph-retrieval/court_document_side_retrieval.py --mode val --cross-encoder-rerank-top-k 50 --cross-encoder-rerank-model zerank-2 --cross-encoder-rerank-query-column german_translation --cross-encoder-rerank-score-path context/val_court_graph_ce_cache_top50_base080_ce020.csv --cross-encoder-rerank-base-weight 0.80 --cross-encoder-rerank-cosine-weight 0.00 --cross-encoder-rerank-ce-weight 0.20 --output-csv context/val_court_graph_ce_top50_base080_ce020.csv`

Validation result:

- Val Art-only Macro F1@20: `0.3476`
- baseline court-grounded Val Art-only Macro F1@20: `0.3290`
- output: `context/val_court_graph_ce_top50_base080_ce020.csv`
- scores: `context/val_court_graph_ce_cache_top50_base080_ce020.csv`

Read:

- the failed first CE attempt used too much CE/cosine influence and dropped to `0.3003` on val.
- the useful pattern is not full CE reranking; it is a conservative rerank where the original court-grounded score remains dominant.


KAGGLE-STATE-6

Court-grounded law retrieval with conservative cross-encoder reranking:

Test submission command:

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/dev/law-only/court-graph-retrieval/court_document_side_retrieval.py --mode test --cross-encoder-rerank-top-k 50 --cross-encoder-rerank-model zerank-2 --cross-encoder-rerank-query-column german_translation --cross-encoder-rerank-score-path context/test_court_graph_ce_top50_base080_ce020_scores.csv --cross-encoder-rerank-base-weight 0.80 --cross-encoder-rerank-cosine-weight 0.00 --cross-encoder-rerank-ce-weight 0.20 --output-csv context/test_court_graph_ce_top50_base080_ce020.csv`

Kaggle result:

- Kaggle score: `0.26348`
- output: `context/test_court_graph_ce_top50_base080_ce020.csv`
- scores: `context/test_court_graph_ce_top50_base080_ce020_scores.csv`

Improvement over the court-grounded baseline, KAGGLE-STATE-5:

- old Kaggle score: `0.25226`
- new Kaggle score: `0.26348`
- absolute gain: `+0.01122`
- relative gain: `+4.45%`


LOCAL-STATE-7

Court-grounded law retrieval with tail-only LLM citation elimination:

- script: `code/dev/law-only/court-graph-retrieval/trials/llm_law_submission_reranker.py`
- input: `context/val_court_graph_ce_top50_base080_ce020.csv`
- idea: keep the strong court-graph + CE ranking, then use the LLM only as a conservative tail cleaner.
- LLM setup:
  - base ranking from LOCAL-STATE-6
  - freeze ranks `1-10`
  - send only ranks `11-20`
  - include top `1` court usage passage per candidate
  - use the shared conservative law-citation prompt
  - run confirmation pass for first-pass REMOVE decisions
  - score using Art-only gold citations, matching the court-grounded retrieval metric

Validation command:

`set -a; source .env; set +a`

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/dev/law-only/court-graph-retrieval/trials/llm_law_submission_reranker.py --dataset val --submission-path context/val_court_graph_ce_top50_base080_ce020.csv --rerank-mode conservative-court-context --llm-backend openrouter --model qwen/qwen3.6-35b-a3b --reasoning-enabled --row-start 1 --row-end 10 --llm-rank-start 11 --llm-rank-end 20 --court-context-freeze-top-k 10 --court-context-batch-size 2 --court-context-confirm-removals --court-context-confirm-batch-size 2 --court-context-top-k 1 --court-context-max-chars 0 --parallel-requests 10 --max-tokens 8000 --temperature 0 --top-p 1 --top-k 1 --output-path context/val_court_graph_ce_top50_base080_ce020_llm_tail.csv --prediction-output-path context/val_court_graph_llm_tail_predictions.csv --batch-output-path context/val_court_graph_llm_tail_batches.parquet --summary-output-path context/val_court_graph_llm_tail_summary.csv --overwrite`

Validation result:

- before Val Art-only Macro F1@20: `0.3476`
- after Val Art-only Macro F1@20: `0.3668`
- absolute local gain: `+0.0192`
- output: `context/val_court_graph_ce_top50_base080_ce020_llm_tail.csv`
- predictions: `context/val_court_graph_llm_tail_predictions.csv`
- batches: `context/val_court_graph_llm_tail_batches.parquet`
- summary: `context/val_court_graph_llm_tail_summary.csv`

Read:

- ranks `11-20` are much noisier than ranks `1-10`, so tail-only elimination is safer than full-list LLM filtering.
- the LLM removed `31/81` tail noise candidates and `4/19` tail gold candidates on val.
- despite imperfect tail precision, the final Art-only macro F1 improved because precision increased enough while preserving the stronger first 10 ranks.


KAGGLE-STATE-7

Court-grounded law retrieval with tail-only LLM citation elimination:

Test submission command:

`set -a; source .env; set +a`

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python code/dev/law-only/court-graph-retrieval/trials/llm_law_submission_reranker.py --dataset test --submission-path context/test_court_graph_ce_top50_base080_ce020.csv --rerank-mode conservative-court-context --llm-backend openrouter --model qwen/qwen3.6-35b-a3b --reasoning-enabled --row-start 1 --row-end 40 --llm-rank-start 11 --llm-rank-end 20 --court-context-freeze-top-k 10 --court-context-batch-size 2 --court-context-confirm-removals --court-context-confirm-batch-size 2 --court-context-top-k 1 --court-context-max-chars 0 --parallel-requests 10 --max-tokens 8000 --temperature 0 --top-p 1 --top-k 1 --output-path context/test_court_graph_ce_top50_base080_ce020_llm_tail.csv --prediction-output-path context/test_court_graph_llm_tail_predictions.csv --batch-output-path context/test_court_graph_llm_tail_batches.parquet --summary-output-path context/test_court_graph_llm_tail_summary.csv --overwrite`

Kaggle result:

- Kaggle score: `0.27569`
- output: `context/test_court_graph_ce_top50_base080_ce020_llm_tail.csv`
- predictions: `context/test_court_graph_llm_tail_predictions.csv`
- batches: `context/test_court_graph_llm_tail_batches.parquet`
- summary: `context/test_court_graph_llm_tail_summary.csv`

Improvement over KAGGLE-STATE-6:

- old Kaggle score: `0.26348`
- new Kaggle score: `0.27569`
- absolute gain: `+0.01221`
- relative gain: `+4.63%`

Improvement over the court-grounded baseline, KAGGLE-STATE-5:

- old Kaggle score: `0.25226`
- new Kaggle score: `0.27569`
- absolute gain: `+0.02343`
- relative gain: `+9.29%`
