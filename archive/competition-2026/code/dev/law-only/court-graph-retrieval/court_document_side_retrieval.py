"""
Dual-path graph retrieval + MMR diversification using zembed embeddings.

Pipeline:
  Path 1 (court-first): Query → top courts by cosine → cited laws → graph score
  Path 2 (law-first + co-citation): Query → top laws by cosine → courts citing
    those laws → co-cited laws → co-citation frequency score
  Union candidates from both paths, blend three signals, apply MMR.

Modes:
- test: writes Kaggle submission CSV
- val : computes Art-only Macro F1 and (optionally) writes predictions CSV
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from trials.substantive_cross_encoder_rerank import (
    load_env_file,
    load_procedural_books,
    load_score_cache,
    rerank_substantive_slots_cosine_weighted,
)


def find_base_dir(path: Path) -> Path:
    for parent in [path, *path.parents]:
        if (parent / "data").exists() and (parent / "code").exists():
            return parent
    raise RuntimeError(f"Could not find repository root from {path}")


BASE_DIR = find_base_dir(Path(__file__).resolve())

DEFAULT_TEST_SUBQUERY_PATH = BASE_DIR / "data/test_sub_query_zembed.parquet"
DEFAULT_VAL_SUBQUERY_PATH = BASE_DIR / "data/val_sub_query_zembed.parquet"
DEFAULT_LAW_DB_PATH = BASE_DIR / "data/laws_db.parquet"
DEFAULT_COURT_DB_PATH = BASE_DIR / "data/court_db.parquet"
DEFAULT_TOP_COURTS = 200
DEFAULT_ALPHA = 0.4
DEFAULT_FINAL_K = 20
DEFAULT_MMR_LAMBDA = 0.8
DEFAULT_MMR_POOL = 50
DEFAULT_COURT_RETRIEVAL = "per-subquery"
DEFAULT_PER_SQ_TOP_COURTS = 20
DEFAULT_CROSS_ENCODER_SCORE_PATH = BASE_DIR / "context/court_graph_cross_encoder_scores.csv"
DEFAULT_PROCEDURAL_BOOK_PATH = BASE_DIR / "context/proc_vs_subs.md"
ARTICLE_CITATION_RE = re.compile(
    r"^Art\.\s+(?P<article>\S+)(?:\s+Abs\.\s+(?P<paragraph>\S+))?\s+(?P<book>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Court graph + MMR submission pipeline")
    parser.add_argument("--mode", choices=["test", "val"], default="test")
    parser.add_argument("--test-subquery-path", type=Path, default=DEFAULT_TEST_SUBQUERY_PATH)
    parser.add_argument("--val-subquery-path", type=Path, default=DEFAULT_VAL_SUBQUERY_PATH)
    parser.add_argument("--law-db-path", type=Path, default=DEFAULT_LAW_DB_PATH)
    parser.add_argument("--court-db-path", type=Path, default=DEFAULT_COURT_DB_PATH)
    parser.add_argument(
        "--court-parquet-batch-size",
        type=int,
        default=2048,
        help="Rows per parquet batch when streaming the large court DB.",
    )
    parser.add_argument("--top-courts", type=int, default=DEFAULT_TOP_COURTS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--final-k", type=int, default=DEFAULT_FINAL_K)
    parser.add_argument(
        "--dynamic-score-threshold",
        type=float,
        default=None,
        help="Optional per-row output filter: keep citations whose selection score is at least this value.",
    )
    parser.add_argument(
        "--dynamic-score-min-k",
        type=int,
        default=0,
        help="Minimum citations to keep when --dynamic-score-threshold is active.",
    )
    parser.add_argument(
        "--dynamic-score-max-k",
        type=int,
        default=0,
        help="Maximum citations to keep when --dynamic-score-threshold is active; 0 uses --final-k.",
    )
    parser.add_argument(
        "--dynamic-score-keep-top",
        type=int,
        default=0,
        help="Always keep the first N citations before applying --dynamic-score-threshold.",
    )
    parser.add_argument(
        "--dynamic-score-keep-injected",
        action="store_true",
        default=False,
        help="Always keep hard-injected pre_citations/pre_near citations before applying the score threshold.",
    )
    parser.add_argument("--mmr-lambda", type=float, default=DEFAULT_MMR_LAMBDA)
    parser.add_argument("--mmr-pool", type=int, default=DEFAULT_MMR_POOL)
    parser.add_argument(
        "--court-seed-overlap-boost",
        type=float,
        default=0.0,
        help="Adds boost * overlap_count to court score when court cites seed hints",
    )
    parser.add_argument(
        "--seed-query-blend",
        type=float,
        default=0.0,
        help="Blend avg query embedding with seed-centroid embedding for court retrieval",
    )
    parser.add_argument(
        "--seed-sim-weight",
        type=float,
        default=0.0,
        help="Add weight * normalized seed-similarity to candidate law scores",
    )
    parser.add_argument(
        "--seed-max-sim-weight",
        type=float,
        default=0.0,
        help="Add weight * normalized max-similarity to any seed law embedding",
    )
    parser.add_argument(
        "--boost-pre-citations",
        type=float,
        default=0.0,
        help="Additive score boost for each pre_citations entry",
    )
    parser.add_argument(
        "--boost-pre-near",
        type=float,
        default=0.0,
        help="Additive score boost for each pre_near_matches entry",
    )
    parser.add_argument(
        "--inject-pre-citations",
        type=int,
        default=0,
        help="Hard-inject first N pre_citations into final top-k",
    )
    parser.add_argument(
        "--inject-pre-near",
        type=int,
        default=0,
        help="Hard-inject first N pre_near_matches into final top-k",
    )
    parser.add_argument(
        "--cocit-seed-k",
        type=int,
        default=0,
        help="Top-K laws by cosine to use as seeds for co-citation path (0 = disabled)",
    )
    parser.add_argument(
        "--cocit-weight",
        type=float,
        default=0.0,
        help="Weight for co-citation frequency score in the blend (0.0 = path 2 disabled)",
    )
    parser.add_argument(
        "--cocit-idf",
        action="store_true",
        default=False,
        help="Apply IDF damping to co-citation frequencies",
    )
    parser.add_argument(
        "--seed-cocit-weight",
        type=float,
        default=0.0,
        help="Additive weight for seed-anchored co-citation (pre_citations/pre_near → courts → co-cited laws)",
    )
    parser.add_argument(
        "--seed-court-expand",
        action="store_true",
        default=False,
        help="Add courts citing seed laws to the top-court set (graph-scored alongside cosine courts)",
    )
    parser.add_argument(
        "--direct-cosine-inject",
        type=int,
        default=0,
        help="(Legacy post-scoring) Inject top-K cited-only laws by direct cosine",
    )
    parser.add_argument(
        "--cosine-pool-inject",
        type=int,
        default=0,
        help="Pre-scoring: add top-K laws by direct cosine to candidate pool before scoring",
    )
    parser.add_argument(
        "--law-rrf-pool-inject",
        type=int,
        default=0,
        help="Pre-scoring: add top-K laws by per-subquery RRF over law embeddings",
    )
    parser.add_argument(
        "--law-rrf-top-per-subquery",
        type=int,
        default=200,
        help="Number of law hits per subquery used to build --law-rrf-pool-inject candidates",
    )
    parser.add_argument(
        "--law-rrf-k",
        type=float,
        default=60.0,
        help="RRF denominator for --law-rrf-pool-inject",
    )
    parser.add_argument(
        "--law-rrf-weight",
        type=float,
        default=0.0,
        help="Optional additive normalized RRF score for law-RRF injected candidates",
    )
    parser.add_argument(
        "--law-tfidf-pool-inject",
        type=int,
        default=0,
        help="Pre-scoring: add top-K law documents by lexical TF-IDF similarity",
    )
    parser.add_argument(
        "--law-tfidf-weight",
        type=float,
        default=0.0,
        help="Optional additive normalized TF-IDF score for injected law-text candidates",
    )
    parser.add_argument(
        "--law-tfidf-query-source",
        choices=["german-translation", "subqueries-joined", "german-plus-subqueries"],
        default="german-translation",
        help="Query text used for law-document TF-IDF retrieval",
    )
    parser.add_argument(
        "--law-tfidf-max-features",
        type=int,
        default=200000,
        help="Maximum TF-IDF vocabulary size; 0 means unlimited",
    )
    parser.add_argument(
        "--law-tfidf-ngram-max",
        type=int,
        default=2,
        help="Maximum word n-gram length for law-document TF-IDF retrieval",
    )
    parser.add_argument(
        "--court-law-blend",
        type=float,
        default=0.0,
        help="Blend court text embedding with avg of its cited law embeddings (0=text only, 1=laws only)",
    )
    parser.add_argument(
        "--procedural-inject",
        type=int,
        default=0,
        help="Pre-scoring: add N most frequently cited laws to candidate pool",
    )
    parser.add_argument(
        "--book-prior-top-candidates",
        type=int,
        default=0,
        help="Use the top N scored law candidates to infer source-book priors (0 = disabled)",
    )
    parser.add_argument(
        "--book-prior-top-books",
        type=int,
        default=0,
        help="Number of inferred source books to softly boost when book priors are enabled",
    )
    parser.add_argument(
        "--book-prior-weight",
        type=float,
        default=0.0,
        help="Additive weight for normalized candidate-derived source-book priors",
    )
    parser.add_argument(
        "--book-pool-inject-top-candidates",
        type=int,
        default=0,
        help="Infer source books from top scored candidates and inject law candidates from those books (0 = disabled)",
    )
    parser.add_argument(
        "--book-pool-inject-top-books",
        type=int,
        default=0,
        help="Number of inferred source books to use for book-pool candidate injection",
    )
    parser.add_argument(
        "--book-pool-inject-per-book",
        type=int,
        default=0,
        help="Maximum embedding-nearest laws to inject per inferred source book",
    )
    parser.add_argument(
        "--book-pool-inject-score-scale",
        type=float,
        default=0.65,
        help="Injected law score as best existing score for that book times this scale and within-book cosine factor",
    )
    parser.add_argument(
        "--article-family-expand-top-candidates",
        type=int,
        default=0,
        help="Add same-article/base/paragraph sibling laws from top N scored candidates before final MMR/CE",
    )
    parser.add_argument(
        "--article-family-expand-max-siblings",
        type=int,
        default=0,
        help="Maximum sibling laws to add per article family; 0 means no per-family cap",
    )
    parser.add_argument(
        "--article-family-expand-score-scale",
        type=float,
        default=0.98,
        help="Expanded sibling base score as parent_score * scale",
    )
    parser.add_argument(
        "--article-family-expand-skip-procedural",
        action="store_true",
        default=False,
        help="Do not expand article families whose source book is listed as procedural",
    )
    parser.add_argument(
        "--law-reference-expand-top-candidates",
        type=int,
        default=0,
        help="Add statute-reference neighbors from top N scored candidates before final MMR/CE",
    )
    parser.add_argument(
        "--law-reference-expand-max-neighbors",
        type=int,
        default=0,
        help="Maximum law-reference neighbors to add per candidate; 0 means no per-candidate cap",
    )
    parser.add_argument(
        "--law-reference-expand-score-scale",
        type=float,
        default=0.70,
        help="Expanded law-reference neighbor score as parent_score * scale",
    )
    parser.add_argument(
        "--law-reference-expand-mode",
        choices=["outgoing", "incoming", "both"],
        default="outgoing",
        help="Use outgoing statute references, incoming references, or both",
    )
    parser.add_argument(
        "--law-reference-expand-skip-procedural",
        action="store_true",
        default=False,
        help="Do not add law-reference neighbors whose source book is procedural",
    )
    parser.add_argument(
        "--court-retrieval",
        choices=["avg-subquery", "english-query", "german-query", "per-subquery",
                 "union-avg-full", "union-avg-german", "union-all-three"],
        default=DEFAULT_COURT_RETRIEVAL,
        help="How to build the query vector for court retrieval",
    )
    parser.add_argument(
        "--per-sq-top-courts",
        type=int,
        default=DEFAULT_PER_SQ_TOP_COURTS,
        help="Per-subquery mode: top courts per sub-query before union",
    )
    parser.add_argument(
        "--law-sim-mode",
        choices=["max-subquery", "avg-subquery"],
        default="max-subquery",
        help="How to compute law cosine similarity: max or avg across sub-queries",
    )
    parser.add_argument("--cross-encoder-rerank-top-k", type=int, default=0)
    parser.add_argument("--cross-encoder-rerank-model", default="zerank-2")
    parser.add_argument("--cross-encoder-rerank-query-column", default="german_translation")
    parser.add_argument(
        "--cross-encoder-rerank-query-source",
        choices=["meta-column", "subqueries-joined", "german-plus-subqueries"],
        default="meta-column",
        help="Text sent as reranker query: a query_meta column, joined sub_query_de rows, or German query plus subqueries.",
    )
    parser.add_argument(
        "--cross-encoder-rerank-subquery-limit",
        type=int,
        default=0,
        help="Limit joined subqueries used as CE query text; 0 uses all subqueries.",
    )
    parser.add_argument("--cross-encoder-rerank-score-path", type=Path, default=DEFAULT_CROSS_ENCODER_SCORE_PATH)
    parser.add_argument("--cross-encoder-rerank-procedural-path", type=Path, default=DEFAULT_PROCEDURAL_BOOK_PATH)
    parser.add_argument("--cross-encoder-rerank-api-key", default=None)
    parser.add_argument("--cross-encoder-rerank-env-path", type=Path, default=BASE_DIR / ".env")
    parser.add_argument("--cross-encoder-rerank-latency", choices=["fast", "slow"], default=None)
    parser.add_argument("--cross-encoder-rerank-timeout-s", type=int, default=120)
    parser.add_argument("--cross-encoder-rerank-document-max-chars", type=int, default=4000)
    parser.add_argument(
        "--cross-encoder-rerank-score-mode",
        choices=["score", "rank"],
        default="score",
        help="Normalize base/cosine/CE scores by min-max score scale or by within-pool rank.",
    )
    parser.add_argument(
        "--cross-encoder-rerank-aggregate-mode",
        choices=["single", "max", "mean"],
        default="single",
        help="Score against one CE query, or aggregate CE over the query plus individual subqueries.",
    )
    parser.add_argument(
        "--cross-encoder-rerank-include-source-book",
        action="store_true",
        default=False,
        help="Include source_book metadata in the law text sent to the cross-encoder.",
    )
    parser.add_argument("--cross-encoder-rerank-base-weight", type=float, default=0.0)
    parser.add_argument("--cross-encoder-rerank-cosine-weight", type=float, default=0.50)
    parser.add_argument("--cross-encoder-rerank-ce-weight", type=float, default=0.50)
    parser.add_argument("--output-csv", default="", help="Optional output path override")
    return parser.parse_args()


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def minmax_norm(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return score_map
    vals = list(score_map.values())
    mn = min(vals)
    mx = max(vals)
    rg = (mx - mn) if mx > mn else 1.0
    return {k: (v - mn) / rg for k, v in score_map.items()}


def mmr_select(
    candidate_scores: dict[str, float],
    law_to_idx: dict[str, int],
    law_emb_mat: np.ndarray,
    lambda_mmr: float,
    pool_k: int,
    final_k: int,
) -> list[str]:
    ranked = sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)
    pool = ranked[: min(pool_k, len(ranked))]
    docs = [d for d, _ in pool]
    rel = np.array([s for _, s in pool], dtype=np.float32)

    if len(docs) <= final_k or lambda_mmr >= 0.999:
        return docs[:final_k]

    idxs = [law_to_idx[d] for d in docs]
    emb = law_emb_mat[idxs]  # already normalized
    sim = emb @ emb.T

    selected = [0]  # always keep top relevance first
    selected_set = {0}
    while len(selected) < final_k and len(selected) < len(docs):
        best_i = None
        best_score = -1e9
        for i in range(len(docs)):
            if i in selected_set:
                continue
            max_sim = float(np.max(sim[i, selected]))
            score = lambda_mmr * float(rel[i]) - (1.0 - lambda_mmr) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)
        selected_set.add(best_i)

    return [docs[i] for i in selected]


def parse_semicolon_list(raw: object) -> list[str]:
    return [x.strip() for x in str(raw).split(";") if x.strip()]


def article_family_key(citation: object) -> tuple[str, str] | None:
    match = ARTICLE_CITATION_RE.match(str(citation).strip())
    if not match:
        return None
    return match.group("book").strip(), match.group("article").strip()


def article_family_sort_key(citation: str) -> tuple[int, str]:
    match = ARTICLE_CITATION_RE.match(citation.strip())
    if not match:
        return (99, citation)
    paragraph = match.group("paragraph")
    if paragraph is None:
        return (0, citation)
    return (1, paragraph)


def load_court_graph_embeddings(
    path: Path,
    *,
    batch_size: int,
) -> tuple[dict[str, list[str]], list[str], np.ndarray]:
    if batch_size <= 0:
        raise ValueError("--court-parquet-batch-size must be positive")

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for streaming the court DB parquet") from error

    court_to_laws: dict[str, list[str]] = {}
    court_citations: list[str] = []
    embedding_chunks: list[np.ndarray] = []

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["citation", "embedding", "law_references"],
    ):
        columns = batch.to_pydict()
        batch_embeddings: list[np.ndarray] = []
        for citation_raw, embedding_raw, refs_raw in zip(
            columns["citation"],
            columns["embedding"],
            columns["law_references"],
        ):
            refs = [x.strip() for x in str(refs_raw).split(";") if x.strip()]
            if not refs:
                continue
            citation = str(citation_raw).strip()
            if not citation:
                continue
            court_to_laws[citation] = refs
            court_citations.append(citation)
            batch_embeddings.append(np.asarray(embedding_raw, dtype=np.float32))

        if batch_embeddings:
            embedding_chunks.append(np.stack(batch_embeddings).astype(np.float32, copy=False))

    if embedding_chunks:
        court_emb_mat = np.vstack(embedding_chunks).astype(np.float32, copy=False)
    else:
        court_emb_mat = np.empty((0, 0), dtype=np.float32)
    return court_to_laws, court_citations, l2_normalize_rows(court_emb_mat)


def main() -> None:
    args = parse_args()
    cross_encoder_enabled = args.cross_encoder_rerank_top_k > 0
    dynamic_score_enabled = args.dynamic_score_threshold is not None
    dynamic_score_max_k = args.dynamic_score_max_k or args.final_k
    book_prior_enabled = (
        args.book_prior_weight > 0.0
        and args.book_prior_top_candidates > 0
        and args.book_prior_top_books > 0
    )
    book_pool_enabled = (
        args.book_pool_inject_top_candidates > 0
        and args.book_pool_inject_top_books > 0
        and args.book_pool_inject_per_book > 0
        and args.book_pool_inject_score_scale > 0.0
    )
    if args.dynamic_score_min_k < 0:
        raise ValueError("--dynamic-score-min-k must be >= 0")
    if args.dynamic_score_max_k < 0:
        raise ValueError("--dynamic-score-max-k must be >= 0")
    if args.dynamic_score_keep_top < 0:
        raise ValueError("--dynamic-score-keep-top must be >= 0")
    if dynamic_score_enabled and args.dynamic_score_min_k > dynamic_score_max_k:
        raise ValueError("--dynamic-score-min-k must be <= resolved --dynamic-score-max-k")
    if args.book_prior_top_candidates < 0:
        raise ValueError("--book-prior-top-candidates must be >= 0")
    if args.book_prior_top_books < 0:
        raise ValueError("--book-prior-top-books must be >= 0")
    if args.book_pool_inject_top_candidates < 0:
        raise ValueError("--book-pool-inject-top-candidates must be >= 0")
    if args.book_pool_inject_top_books < 0:
        raise ValueError("--book-pool-inject-top-books must be >= 0")
    if args.book_pool_inject_per_book < 0:
        raise ValueError("--book-pool-inject-per-book must be >= 0")
    if args.book_pool_inject_score_scale < 0:
        raise ValueError("--book-pool-inject-score-scale must be >= 0")
    if args.cross_encoder_rerank_subquery_limit < 0:
        raise ValueError("--cross-encoder-rerank-subquery-limit must be >= 0")
    if args.law_rrf_pool_inject < 0:
        raise ValueError("--law-rrf-pool-inject must be >= 0")
    if args.law_rrf_top_per_subquery <= 0:
        raise ValueError("--law-rrf-top-per-subquery must be positive")
    if args.law_rrf_k <= 0:
        raise ValueError("--law-rrf-k must be positive")
    if args.law_tfidf_pool_inject < 0:
        raise ValueError("--law-tfidf-pool-inject must be >= 0")
    if args.law_tfidf_max_features < 0:
        raise ValueError("--law-tfidf-max-features must be >= 0")
    if args.law_tfidf_ngram_max <= 0:
        raise ValueError("--law-tfidf-ngram-max must be positive")
    if args.article_family_expand_top_candidates < 0:
        raise ValueError("--article-family-expand-top-candidates must be >= 0")
    if args.article_family_expand_max_siblings < 0:
        raise ValueError("--article-family-expand-max-siblings must be >= 0")
    if args.article_family_expand_score_scale < 0:
        raise ValueError("--article-family-expand-score-scale must be >= 0")
    if args.law_reference_expand_top_candidates < 0:
        raise ValueError("--law-reference-expand-top-candidates must be >= 0")
    if args.law_reference_expand_max_neighbors < 0:
        raise ValueError("--law-reference-expand-max-neighbors must be >= 0")
    if args.law_reference_expand_score_scale < 0:
        raise ValueError("--law-reference-expand-score-scale must be >= 0")

    seed_suffix = (
        f"_bp{str(args.boost_pre_citations).replace('.', '')}"
        f"_bn{str(args.boost_pre_near).replace('.', '')}"
        f"_ip{args.inject_pre_citations}_in{args.inject_pre_near}"
        f"_csb{str(args.court_seed_overlap_boost).replace('.', '')}"
        f"_sqb{str(args.seed_query_blend).replace('.', '')}"
        f"_ssw{str(args.seed_sim_weight).replace('.', '')}"
        f"_smsw{str(args.seed_max_sim_weight).replace('.', '')}"
    )
    cocit_suffix = ""
    if args.cocit_seed_k > 0 and args.cocit_weight > 0.0:
        cocit_suffix = (
            f"_csk{args.cocit_seed_k}"
            f"_cw{str(args.cocit_weight).replace('.', '')}"
            f"{'_cidf' if args.cocit_idf else ''}"
        )
    cross_encoder_suffix = ""
    if cross_encoder_enabled:
        cross_encoder_suffix = (
            f"_ce{args.cross_encoder_rerank_top_k}"
            f"_cw{str(args.cross_encoder_rerank_cosine_weight).replace('.', '')}"
            f"_zew{str(args.cross_encoder_rerank_ce_weight).replace('.', '')}"
        )
    retrieval_suffix = args.court_retrieval.replace("-", "_")
    if args.court_retrieval == "per-subquery":
        retrieval_suffix += f"_k{args.per_sq_top_courts}"
    run_suffix = (
        f"{retrieval_suffix}_a{str(args.alpha).replace('.', '')}"
        f"_mmr_l{str(args.mmr_lambda).replace('.', '')}"
        f"_p{args.mmr_pool}{seed_suffix}{cocit_suffix}{cross_encoder_suffix}_top{args.final_k}"
    )

    if args.mode == "test":
        sub_queries_path = args.test_subquery_path
        query_meta_path = BASE_DIR / "data/test.parquet"
        gold_path = None
        default_out = (
            BASE_DIR
            / f"context/submission_court_graph_zembed_{run_suffix}.csv"
        )
    else:
        sub_queries_path = args.val_subquery_path
        query_meta_path = BASE_DIR / "data/val.parquet"
        gold_path = BASE_DIR / "data/val.parquet"
        default_out = (
            BASE_DIR
            / f"context/val_court_graph_zembed_{run_suffix}.csv"
        )

    out_path = Path(args.output_csv) if args.output_csv else default_out

    print(f"Mode         : {args.mode}")
    print(f"Top courts   : {args.top_courts}")
    print(f"Alpha        : {args.alpha}")
    print(f"Final K      : {args.final_k}")
    if dynamic_score_enabled:
        print(
            "Dynamic K    : "
            f"threshold={args.dynamic_score_threshold}, "
            f"min={args.dynamic_score_min_k}, "
            f"max={dynamic_score_max_k}, "
            f"keep_top={args.dynamic_score_keep_top}, "
            f"keep_injected={args.dynamic_score_keep_injected}"
        )
    print(f"MMR lambda   : {args.mmr_lambda}")
    print(f"MMR pool     : {args.mmr_pool}")
    print(f"Court boost  : {args.court_seed_overlap_boost}")
    print(f"Court retr   : {args.court_retrieval}")
    print(f"Per-SQ courts: {args.per_sq_top_courts}")
    print(f"Law sim mode : {args.law_sim_mode}")
    print(f"Seed q blend : {args.seed_query_blend}")
    print(f"Seed sim wt  : {args.seed_sim_weight}")
    print(f"Seed max sim : {args.seed_max_sim_weight}")
    print(f"Boost pre    : citations={args.boost_pre_citations}, near={args.boost_pre_near}")
    print(f"Inject pre   : citations={args.inject_pre_citations}, near={args.inject_pre_near}")
    print(f"Cocit seed-K : {args.cocit_seed_k}")
    print(f"Cocit weight : {args.cocit_weight}")
    print(f"Cocit IDF    : {args.cocit_idf}")
    print(f"Seed cocit wt: {args.seed_cocit_weight}")
    print(f"Seed court   : {args.seed_court_expand}")
    print(f"Direct cos K : {args.direct_cosine_inject}")
    print(f"Cos pool inj : {args.cosine_pool_inject}")
    print(
        "Law RRF inj  : "
        f"{args.law_rrf_pool_inject} "
        f"(top_per_subquery={args.law_rrf_top_per_subquery}, "
        f"k={args.law_rrf_k}, weight={args.law_rrf_weight})"
    )
    print(
        "Law TFIDF inj: "
        f"{args.law_tfidf_pool_inject} "
        f"(weight={args.law_tfidf_weight}, "
        f"query_source={args.law_tfidf_query_source}, "
        f"max_features={args.law_tfidf_max_features}, "
        f"ngram_max={args.law_tfidf_ngram_max})"
    )
    print(f"Court-law bld: {args.court_law_blend}")
    print(f"Proced. inj  : {args.procedural_inject}")
    print(
        "Book prior   : "
        f"top_candidates={args.book_prior_top_candidates}, "
        f"top_books={args.book_prior_top_books}, "
        f"weight={args.book_prior_weight}"
    )
    print(
        "Book pool inj: "
        f"top_candidates={args.book_pool_inject_top_candidates}, "
        f"top_books={args.book_pool_inject_top_books}, "
        f"per_book={args.book_pool_inject_per_book}, "
        f"score_scale={args.book_pool_inject_score_scale}"
    )
    print(
        "Art family   : "
        f"top_candidates={args.article_family_expand_top_candidates}, "
        f"max_siblings={args.article_family_expand_max_siblings}, "
        f"score_scale={args.article_family_expand_score_scale}, "
        f"skip_procedural={args.article_family_expand_skip_procedural}"
    )
    print(
        "Law ref exp  : "
        f"top_candidates={args.law_reference_expand_top_candidates}, "
        f"max_neighbors={args.law_reference_expand_max_neighbors}, "
        f"score_scale={args.law_reference_expand_score_scale}, "
        f"mode={args.law_reference_expand_mode}, "
        f"skip_procedural={args.law_reference_expand_skip_procedural}"
    )
    print(f"Cross enc top: {args.cross_encoder_rerank_top_k}")
    if cross_encoder_enabled:
        print(f"Cross enc mdl: {args.cross_encoder_rerank_model}")
        print(
            "Cross enc qry: "
            f"source={args.cross_encoder_rerank_query_source}, "
            f"column={args.cross_encoder_rerank_query_column}, "
            f"subquery_limit={args.cross_encoder_rerank_subquery_limit}"
        )
        print(f"Cross enc doc: include_source_book={args.cross_encoder_rerank_include_source_book}")
        print(f"Cross enc agg: {args.cross_encoder_rerank_aggregate_mode}")
        print(
            "Cross enc wts: "
            f"base={args.cross_encoder_rerank_base_weight}, "
            f"cosine={args.cross_encoder_rerank_cosine_weight}, "
            f"ce={args.cross_encoder_rerank_ce_weight}"
        )
    print(f"Sub-queries  : {sub_queries_path}")
    print(f"Query meta   : {query_meta_path}")
    print(f"Law DB       : {args.law_db_path}")
    print(f"Court DB     : {args.court_db_path}")
    print(f"Court batch  : {args.court_parquet_batch_size}")
    print(f"Output       : {out_path}")
    print()

    sub_queries = pd.read_parquet(sub_queries_path)
    meta_cols = ["query_id", "pre_citations", "pre_near_matches"]
    if args.court_retrieval in ("english-query", "union-avg-full", "union-all-three"):
        meta_cols.append("zembed-full-german-query")
    if args.court_retrieval in ("german-query", "union-avg-german", "union-all-three"):
        if "zembed-full-german-query" not in meta_cols:
            meta_cols.append("zembed-full-german-query")
    if cross_encoder_enabled and args.cross_encoder_rerank_query_source in (
        "meta-column",
        "german-plus-subqueries",
    ):
        meta_cols.append(args.cross_encoder_rerank_query_column)
    if args.law_tfidf_pool_inject > 0 and args.law_tfidf_query_source in (
        "german-translation",
        "german-plus-subqueries",
    ):
        meta_cols.append("german_translation")
    meta_cols = list(dict.fromkeys(meta_cols))
    query_meta = pd.read_parquet(query_meta_path, columns=meta_cols)
    law_columns = ["citation", "embedding"]
    if cross_encoder_enabled or book_prior_enabled or book_pool_enabled or args.law_tfidf_pool_inject > 0:
        law_columns.extend(["document", "source_book"])
    if args.law_reference_expand_top_candidates > 0:
        law_columns.extend(["references", "source_book"])
    law_columns = list(dict.fromkeys(law_columns))
    law_docs = pd.read_parquet(
        args.law_db_path,
        columns=law_columns,
    )
    court_to_laws, court_citations, court_emb_mat = load_court_graph_embeddings(
        args.court_db_path,
        batch_size=args.court_parquet_batch_size,
    )
    court_law_sets = (
        [set(court_to_laws[c]) for c in court_citations]
        if args.court_seed_overlap_boost > 0.0
        else []
    )

    law_citations = law_docs["citation"].tolist()
    law_emb_mat = np.array(law_docs["embedding"].tolist(), dtype=np.float32)
    law_emb_mat = l2_normalize_rows(law_emb_mat)
    law_to_idx = {c: i for i, c in enumerate(law_citations)}
    article_family_to_citations: dict[tuple[str, str], list[str]] = defaultdict(list)
    if args.article_family_expand_top_candidates > 0:
        for citation in law_citations:
            key = article_family_key(citation)
            if key is not None:
                article_family_to_citations[key].append(citation)
        for key in list(article_family_to_citations):
            article_family_to_citations[key] = sorted(
                article_family_to_citations[key],
                key=article_family_sort_key,
            )

    procedural_books: set[str] = set()
    citation_to_document: dict[str, str] = {}
    citation_to_source_book: dict[str, str] = {}
    cross_encoder_score_cache: dict[tuple[str, str], float] = {}
    cross_encoder_score_rows: list[dict[str, object]] = []
    cross_encoder_api_calls = 0
    query_text_by_q: dict[str, str] = {}
    query_text_variants_by_q: dict[str, list[tuple[str, str]]] = {}
    if cross_encoder_enabled:
        load_env_file(args.cross_encoder_rerank_env_path)
        procedural_books = load_procedural_books(args.cross_encoder_rerank_procedural_path)
        citation_to_document = dict(zip(law_docs["citation"], law_docs["document"].astype(str)))
        cross_encoder_score_cache = load_score_cache(
            args.cross_encoder_rerank_score_path,
            model=args.cross_encoder_rerank_model,
        )
        if args.cross_encoder_rerank_query_source == "meta-column":
            query_text_by_q = dict(
                zip(
                    query_meta["query_id"].astype(str),
                    query_meta[args.cross_encoder_rerank_query_column].astype(str),
                )
            )
        else:
            subquery_limit = args.cross_encoder_rerank_subquery_limit
            joined_subqueries: dict[str, str] = {}
            for qid, group in sub_queries.groupby("query_id", sort=False):
                ordered = group.sort_values("sub_id") if "sub_id" in group.columns else group
                parts = [str(x).strip() for x in ordered["sub_query_de"].tolist() if str(x).strip()]
                if subquery_limit > 0:
                    parts = parts[:subquery_limit]
                joined_subqueries[str(qid)] = "\n".join(f"- {part}" for part in parts)
            if args.cross_encoder_rerank_query_source == "subqueries-joined":
                query_text_by_q = joined_subqueries
            else:
                meta_query_by_q = dict(
                    zip(
                        query_meta["query_id"].astype(str),
                        query_meta[args.cross_encoder_rerank_query_column].astype(str),
                    )
                )
                query_text_by_q = {
                    qid: (
                        f"{meta_query_by_q.get(qid, '').strip()}\n\n"
                        f"Relevante Suchaspekte:\n{joined_subqueries.get(qid, '')}"
                    ).strip()
                    for qid in joined_subqueries
                }
        query_text_variants_by_q = {qid: [("main", text)] for qid, text in query_text_by_q.items()}
        if args.cross_encoder_rerank_aggregate_mode != "single":
            subquery_limit = args.cross_encoder_rerank_subquery_limit
            for qid, group in sub_queries.groupby("query_id", sort=False):
                ordered = group.sort_values("sub_id") if "sub_id" in group.columns else group
                subquery_texts = [str(x).strip() for x in ordered["sub_query_de"].tolist() if str(x).strip()]
                if subquery_limit > 0:
                    subquery_texts = subquery_texts[:subquery_limit]
                variants = list(query_text_variants_by_q.get(str(qid), []))
                variants.extend((f"subquery_{idx}", text) for idx, text in enumerate(subquery_texts, start=1))
                query_text_variants_by_q[str(qid)] = variants
        print(f"Cross enc proc: {len(procedural_books):,} books")
        print(f"Cross enc cache: {len(cross_encoder_score_cache):,} scores")
    if "source_book" in law_docs.columns:
        citation_to_source_book = dict(zip(law_docs["citation"], law_docs["source_book"].astype(str)))

    book_to_law_indices: dict[str, np.ndarray] = {}
    if book_pool_enabled:
        if not citation_to_source_book:
            raise ValueError("--book-pool-inject requires law source_book metadata")
        tmp_book_to_indices: defaultdict[str, list[int]] = defaultdict(list)
        for idx, citation in enumerate(law_citations):
            book = citation_to_source_book.get(citation, "")
            if book:
                tmp_book_to_indices[book].append(idx)
        book_to_law_indices = {
            book: np.asarray(indices, dtype=np.intp)
            for book, indices in tmp_book_to_indices.items()
            if indices
        }
        print(f"Book pools    : {len(book_to_law_indices):,} source books")

    law_out_refs: dict[str, list[str]] = {}
    law_in_refs: dict[str, list[str]] = defaultdict(list)
    if args.law_reference_expand_top_candidates > 0:
        if "references" not in law_docs.columns:
            raise ValueError("--law-reference-expand-top-candidates requires law references")
        for row in law_docs[["citation", "references"]].itertuples(index=False):
            citation = str(row.citation)
            refs = [ref for ref in parse_semicolon_list(row.references) if ref in law_to_idx and ref != citation]
            if refs:
                law_out_refs[citation] = refs
                for ref in refs:
                    law_in_refs[ref].append(citation)
        print(
            f"Law reference graph: outgoing={len(law_out_refs):,}, "
            f"incoming={len(law_in_refs):,}"
        )

    law_tfidf_scores_by_q: dict[str, dict[str, float]] = {}
    if args.law_tfidf_pool_inject > 0:
        if "document" not in law_docs.columns:
            raise ValueError("--law-tfidf-pool-inject requires law document text")
        tfidf_query_text_by_q: dict[str, str] = {}
        joined_subqueries: dict[str, str] = {}
        if args.law_tfidf_query_source in ("subqueries-joined", "german-plus-subqueries"):
            for qid, group in sub_queries.groupby("query_id", sort=False):
                ordered = group.sort_values("sub_id") if "sub_id" in group.columns else group
                parts = [str(x).strip() for x in ordered["sub_query_de"].tolist() if str(x).strip()]
                joined_subqueries[str(qid)] = " ".join(parts)
        if args.law_tfidf_query_source == "german-translation":
            tfidf_query_text_by_q = dict(
                zip(query_meta["query_id"].astype(str), query_meta["german_translation"].astype(str))
            )
        elif args.law_tfidf_query_source == "subqueries-joined":
            tfidf_query_text_by_q = joined_subqueries
        else:
            german_by_q = dict(
                zip(query_meta["query_id"].astype(str), query_meta["german_translation"].astype(str))
            )
            tfidf_query_text_by_q = {
                qid: f"{german_by_q.get(qid, '')} {joined_subqueries.get(qid, '')}".strip()
                for qid in set(german_by_q) | set(joined_subqueries)
            }

        max_features = args.law_tfidf_max_features or None
        print("Building law TF-IDF index...")
        tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            max_df=0.90,
            min_df=1,
            max_features=max_features,
            ngram_range=(1, args.law_tfidf_ngram_max),
            sublinear_tf=True,
            norm="l2",
        )
        law_tfidf_matrix = tfidf_vectorizer.fit_transform(law_docs["document"].astype(str).tolist())
        tfidf_query_ids = sorted(tfidf_query_text_by_q)
        query_tfidf_matrix = tfidf_vectorizer.transform([tfidf_query_text_by_q[qid] for qid in tfidf_query_ids])
        local_top = min(args.law_tfidf_pool_inject, len(law_citations))
        for qi, qid in enumerate(tfidf_query_ids):
            sim = (query_tfidf_matrix[qi] @ law_tfidf_matrix.T).toarray().ravel()
            if local_top >= len(sim):
                top_idx = np.argsort(sim)[::-1]
            else:
                top_idx = np.argpartition(sim, -local_top)[-local_top:]
                top_idx = top_idx[np.argsort(sim[top_idx])[::-1]]
            law_tfidf_scores_by_q[qid] = {
                law_citations[int(i)]: float(sim[int(i)])
                for i in top_idx
                if sim[int(i)] > 0.0
            }
        print(
            f"Law TF-IDF index: docs={law_tfidf_matrix.shape[0]:,}, "
            f"features={law_tfidf_matrix.shape[1]:,}, queries={len(law_tfidf_scores_by_q):,}"
        )

    # Enrich court embeddings by blending with cited law embeddings
    if args.court_law_blend > 0.0:
        beta = args.court_law_blend
        enriched = 0
        for ci, court_cit in enumerate(court_citations):
            cited_laws = court_to_laws.get(court_cit, [])
            law_idxs = [law_to_idx[l] for l in cited_laws if l in law_to_idx]
            if law_idxs:
                law_avg = law_emb_mat[law_idxs].mean(axis=0)
                law_avg = law_avg / max(np.linalg.norm(law_avg), 1e-9)
                court_emb_mat[ci] = (1.0 - beta) * court_emb_mat[ci] + beta * law_avg
                enriched += 1
        court_emb_mat = l2_normalize_rows(court_emb_mat)
        print(f"Court-law blend: enriched {enriched}/{len(court_citations)} courts (beta={beta})")

    pre_citations_by_q: dict[str, list[str]] = {}
    pre_near_by_q: dict[str, list[str]] = {}
    for _, r in tqdm(query_meta.iterrows(), total=len(query_meta), desc="Processing query metadata", unit="query"):
        qid = r["query_id"]
        pre_citations_by_q[qid] = [
            c for c in parse_semicolon_list(r.get("pre_citations", "")) if c in law_to_idx
        ]
        pre_near_by_q[qid] = [
            c for c in parse_semicolon_list(r.get("pre_near_matches", "")) if c in law_to_idx
        ]

    # Global law degree for IDF damping + reverse index for co-citation path
    law_degree: Counter[str] = Counter()
    law_to_courts: dict[str, set[str]] = defaultdict(set)
    for court_cit, refs in court_to_laws.items():
        for law in refs:
            law_degree[law] += 1
            law_to_courts[law].add(court_cit)

    court_cit_to_idx = (
        {court_citation: idx for idx, court_citation in enumerate(court_citations)}
        if args.seed_court_expand
        else {}
    )

    cited_law_idxs: list[int] = []
    if args.direct_cosine_inject > 0:
        cited_law_idxs = [law_to_idx[law] for law in law_degree if law in law_to_idx]
        print(f"Cited-only laws: {len(cited_law_idxs)} (for direct cosine injection)")

    # Pre-compute procedural law list (most frequently cited across all courts)
    procedural_laws: list[str] = []
    if args.procedural_inject > 0:
        sorted_by_freq = sorted(
            ((law, cnt) for law, cnt in law_degree.items() if law in law_to_idx),
            key=lambda x: x[1],
            reverse=True,
        )
        procedural_laws = [law for law, _ in sorted_by_freq[: args.procedural_inject]]
        print(f"Procedural laws: {len(procedural_laws)} (top by court citation frequency)")
        for pl, cnt in sorted_by_freq[: min(5, args.procedural_inject)]:
            print(f"  {pl}: cited in {cnt} courts")

    total_court_docs = len(court_to_laws)
    cocit_idf_map: dict[str, float] = {}
    if args.cocit_idf:
        max_idf = 0.0
        for law, courts in law_to_courts.items():
            idf_val = math.log(total_court_docs / max(len(courts), 1))
            cocit_idf_map[law] = idf_val
            if idf_val > max_idf:
                max_idf = idf_val
        if max_idf > 0:
            for law in cocit_idf_map:
                cocit_idf_map[law] /= max_idf

    query_ids = sorted(sub_queries["query_id"].unique())
    sub_embs_by_q: dict[str, np.ndarray] = {}
    avg_emb_by_q: dict[str, np.ndarray] = {}
    for qid in query_ids:
        emb = np.array(
            sub_queries[sub_queries["query_id"] == qid]["zembed_embedding"].tolist(),
            dtype=np.float32,
        )
        emb = l2_normalize_rows(emb)
        sub_embs_by_q[qid] = emb
        avg = emb.mean(axis=0)
        avg_emb_by_q[qid] = avg / np.linalg.norm(avg)

    if args.court_retrieval in ("english-query", "german-query"):
        emb_col = "zembed-full-german-query"
        override_emb_by_q: dict[str, np.ndarray] = {}
        for _, r in query_meta.iterrows():
            qid = r["query_id"]
            vec = np.array(r[emb_col], dtype=np.float32)
            vec = vec / np.linalg.norm(vec)
            override_emb_by_q[qid] = vec
        print(f"Court retrieval: using {args.court_retrieval} full-query embedding ({len(override_emb_by_q)} queries)")

    full_query_emb_by_q: dict[str, np.ndarray] = {}
    german_query_emb_by_q: dict[str, np.ndarray] = {}
    if args.court_retrieval in ("union-avg-full", "union-avg-german", "union-all-three"):
        for _, r in query_meta.iterrows():
            qid = r["query_id"]
            if "zembed-full-german-query" in r.index:
                vec = np.array(r["zembed-full-german-query"], dtype=np.float32)
                full_query_emb_by_q[qid] = vec / max(np.linalg.norm(vec), 1e-9)
                vec = np.array(r["zembed-full-german-query"], dtype=np.float32)
                german_query_emb_by_q[qid] = vec / max(np.linalg.norm(vec), 1e-9)
        print(f"Court retrieval: {args.court_retrieval} (en={len(full_query_emb_by_q)}, de={len(german_query_emb_by_q)})")

    rows = []
    pred_map: dict[str, list[str]] = {}
    for qid in query_ids:
        pre_citations = pre_citations_by_q.get(qid, [])
        pre_near = pre_near_by_q.get(qid, [])
        seed_context = list(pre_citations)
        for law in pre_near:
            if law not in seed_context:
                seed_context.append(law)

        seed_centroid: np.ndarray | None = None
        if seed_context:
            seed_idxs = [law_to_idx[x] for x in seed_context]
            seed_avg = law_emb_mat[seed_idxs].mean(axis=0)
            seed_norm = np.linalg.norm(seed_avg)
            if seed_norm > 0:
                seed_centroid = seed_avg / seed_norm

        if args.court_retrieval in ("english-query", "german-query"):
            q_vec = override_emb_by_q[qid]
        else:
            q_vec = avg_emb_by_q[qid]
        if seed_centroid is not None and args.seed_query_blend > 0.0:
            q_vec = (1.0 - args.seed_query_blend) * q_vec + args.seed_query_blend * seed_centroid
            q_vec = q_vec / np.linalg.norm(q_vec)

        scoring_court_idx: set[int] = set()
        if args.court_retrieval == "per-subquery":
            sq_embs = sub_embs_by_q[qid]  # (n_sq, 1024)
            sq_court_scores = sq_embs @ court_emb_mat.T  # (n_sq, n_courts)
            court_scores = sq_court_scores.max(axis=0)  # max similarity across sub-queries
            per_sq_k = args.per_sq_top_courts
            union_idx = set()
            for sq_i in range(sq_embs.shape[0]):
                top_k = np.argsort(sq_court_scores[sq_i])[::-1][:per_sq_k]
                union_idx.update(int(x) for x in top_k)
            top_court_idx = np.array(sorted(union_idx), dtype=np.intp)
        elif args.court_retrieval in ("union-avg-full", "union-avg-german", "union-all-three"):
            avg_scores = (avg_emb_by_q[qid].reshape(1, -1) @ court_emb_mat.T).flatten()
            court_scores = avg_scores  # graph scoring always uses avg_subquery
            top_avg = set(int(i) for i in np.argsort(avg_scores)[::-1][:args.top_courts])
            scoring_court_idx = top_avg  # only avg_subquery courts for graph scoring
            union_set = set(top_avg)
            if args.court_retrieval in ("union-avg-full", "union-all-three") and qid in full_query_emb_by_q:
                full_scores = (full_query_emb_by_q[qid].reshape(1, -1) @ court_emb_mat.T).flatten()
                top_full = set(int(i) for i in np.argsort(full_scores)[::-1][:args.top_courts])
                union_set |= top_full
            if args.court_retrieval in ("union-avg-german", "union-all-three") and qid in german_query_emb_by_q:
                german_scores = (german_query_emb_by_q[qid].reshape(1, -1) @ court_emb_mat.T).flatten()
                top_german = set(int(i) for i in np.argsort(german_scores)[::-1][:args.top_courts])
                union_set |= top_german
            top_court_idx = np.array(sorted(union_set), dtype=np.intp)
        else:
            q_emb = q_vec.reshape(1, -1)
            court_scores = (q_emb @ court_emb_mat.T).flatten()
            if seed_context and args.court_seed_overlap_boost > 0.0:
                seed_context_set = set(seed_context)
                boosted = np.empty_like(court_scores)
                for i, law_set in enumerate(court_law_sets):
                    overlap = len(seed_context_set & law_set)
                    boosted[i] = court_scores[i] + args.court_seed_overlap_boost * float(overlap)
                court_scores = boosted
            top_court_idx = np.argsort(court_scores)[::-1][: args.top_courts]

        # Expand court set with courts that cite seed laws
        if args.seed_court_expand and seed_context:
            top_court_set = set(int(i) for i in top_court_idx)
            for sc_law in seed_context:
                for sc_court in law_to_courts.get(sc_law, set()):
                    ci = court_cit_to_idx.get(sc_court)
                    if ci is not None and ci not in top_court_set:
                        top_court_set.add(ci)
            top_court_idx = np.array(sorted(top_court_set), dtype=np.intp)

        # --- Path 1: Court-first graph candidates ---
        graph_raw: dict[str, float] = {}
        candidates: set[str] = set()
        law_rrf_scores: dict[str, float] = {}
        law_tfidf_scores = law_tfidf_scores_by_q.get(str(qid), {})
        is_union_mode = args.court_retrieval in ("union-avg-full", "union-avg-german", "union-all-three")
        for ci in top_court_idx:
            court_cit = court_citations[ci]
            c_score = float(court_scores[ci])
            for law in court_to_laws.get(court_cit, []):
                if law in law_to_idx:
                    candidates.add(law)
                    if not is_union_mode or ci in scoring_court_idx:
                        graph_raw[law] = graph_raw.get(law, 0.0) + c_score

        # --- Pre-scoring: inject top cosine laws into candidate pool ---
        if args.cosine_pool_inject > 0:
            all_cos = (sub_embs_by_q[qid] @ law_emb_mat.T).max(axis=0)
            top_cos_idx = np.argsort(all_cos)[::-1]
            injected_cos = 0
            for li in top_cos_idx:
                if injected_cos >= args.cosine_pool_inject:
                    break
                law_cit = law_citations[li]
                if law_cit not in candidates:
                    candidates.add(law_cit)
                    injected_cos += 1

        # --- Pre-scoring: inject a law-side per-subquery RRF pool ---
        if args.law_rrf_pool_inject > 0:
            sub_embs_all = sub_embs_by_q[qid]
            law_rrf_raw: defaultdict[str, float] = defaultdict(float)
            law_best_cos: dict[str, float] = {}
            local_top = min(args.law_rrf_top_per_subquery, len(law_citations))
            law_sims = sub_embs_all @ law_emb_mat.T
            for scores in law_sims:
                if local_top >= len(law_citations):
                    top_law_idx = np.argsort(scores)[::-1]
                else:
                    top_law_idx = np.argpartition(scores, -local_top)[-local_top:]
                    top_law_idx = top_law_idx[np.argsort(scores[top_law_idx])[::-1]]
                for rank, li in enumerate(top_law_idx, start=1):
                    law_cit = law_citations[int(li)]
                    law_rrf_raw[law_cit] += 1.0 / (args.law_rrf_k + rank)
                    law_best_cos[law_cit] = max(law_best_cos.get(law_cit, -1.0), float(scores[int(li)]))

            ranked_rrf_laws = sorted(
                law_rrf_raw,
                key=lambda law: (law_rrf_raw[law], law_best_cos.get(law, -1.0), law),
                reverse=True,
            )[: args.law_rrf_pool_inject]
            for law_cit in ranked_rrf_laws:
                candidates.add(law_cit)
            law_rrf_scores = {law: law_rrf_raw[law] for law in ranked_rrf_laws}

        # --- Pre-scoring: inject lexical law-document candidates ---
        if law_tfidf_scores:
            for law_cit in law_tfidf_scores:
                candidates.add(law_cit)

        # --- Pre-scoring: inject procedural laws into candidate pool ---
        if args.procedural_inject > 0:
            for pl in procedural_laws:
                candidates.add(pl)

        # --- Path 2: Law-first co-citation expansion ---
        cocit_raw: dict[str, float] = {}
        if args.cocit_seed_k > 0 and args.cocit_weight > 0.0:
            sub_embs_all = sub_embs_by_q[qid]
            best_per_law = (sub_embs_all @ law_emb_mat.T).max(axis=0)
            seed_idx = np.argsort(best_per_law)[::-1][: args.cocit_seed_k]
            seed_set = {law_citations[i] for i in seed_idx}

            related_courts: set[str] = set()
            for cit in seed_set:
                related_courts.update(law_to_courts.get(cit, set()))

            cocit_freq: Counter[str] = Counter()
            for rc in related_courts:
                for law_ref in court_to_laws.get(rc, []):
                    cocit_freq[law_ref] += 1

            for law_ref, freq in cocit_freq.items():
                if law_ref in law_to_idx:
                    candidates.add(law_ref)
                    idf_weight = cocit_idf_map.get(law_ref, 1.0) if args.cocit_idf else 1.0
                    cocit_raw[law_ref] = float(freq) * idf_weight

        # --- Compute per-candidate cosine similarity ---
        sub_embs = sub_embs_by_q[qid]
        avg_vec = avg_emb_by_q[qid]
        cos_scores: dict[str, float] = {}
        for law in candidates:
            li = law_to_idx[law]
            if args.law_sim_mode == "max-subquery":
                score = float((sub_embs @ law_emb_mat[li].reshape(1, -1).T).max())
            else:
                score = float(avg_vec @ law_emb_mat[li])
            cos_scores[law] = score

        graph_idf = {
            law: graph_raw[law] * (1.0 / math.log(1 + law_degree.get(law, 1)))
            for law in graph_raw
        }
        cos_norm = minmax_norm(cos_scores)
        graph_norm = minmax_norm(graph_idf)
        cocit_norm = minmax_norm(cocit_raw)

        gamma = args.cocit_weight
        remaining = 1.0 - gamma
        w_cos = args.alpha * remaining
        w_graph = (1.0 - args.alpha) * remaining

        final_scores = {
            law: (
                w_cos * cos_norm.get(law, 0.0)
                + w_graph * graph_norm.get(law, 0.0)
                + gamma * cocit_norm.get(law, 0.0)
            )
            for law in candidates
        }

        if law_rrf_scores and args.law_rrf_weight > 0.0:
            law_rrf_norm = minmax_norm(law_rrf_scores)
            for law in candidates:
                final_scores[law] = final_scores.get(law, 0.0) + args.law_rrf_weight * law_rrf_norm.get(
                    law, 0.0
                )

        if law_tfidf_scores and args.law_tfidf_weight > 0.0:
            law_tfidf_norm = minmax_norm(law_tfidf_scores)
            for law in candidates:
                final_scores[law] = final_scores.get(law, 0.0) + args.law_tfidf_weight * law_tfidf_norm.get(
                    law, 0.0
                )

        if seed_centroid is not None and args.seed_sim_weight > 0.0:
            seed_sim_scores = {
                law: float(law_emb_mat[law_to_idx[law]] @ seed_centroid) for law in candidates
            }
            seed_sim_norm = minmax_norm(seed_sim_scores)
            for law in candidates:
                final_scores[law] = final_scores.get(law, 0.0) + args.seed_sim_weight * seed_sim_norm.get(
                    law, 0.0
                )

        if seed_context and args.seed_max_sim_weight > 0.0:
            seed_mat = law_emb_mat[[law_to_idx[x] for x in seed_context]]
            seed_max_scores = {
                law: float((seed_mat @ law_emb_mat[law_to_idx[law]].reshape(-1, 1)).max())
                for law in candidates
            }
            seed_max_norm = minmax_norm(seed_max_scores)
            for law in candidates:
                final_scores[law] = final_scores.get(law, 0.0) + args.seed_max_sim_weight * seed_max_norm.get(
                    law, 0.0
                )

        # Seed-anchored co-citation: use exact pre_citations/pre_near as seeds
        if seed_context and args.seed_cocit_weight > 0.0:
            seed_courts: set[str] = set()
            for sc in seed_context:
                seed_courts.update(law_to_courts.get(sc, set()))
            if seed_courts:
                sc_freq: Counter[str] = Counter()
                for sc_court in seed_courts:
                    for law_ref in court_to_laws.get(sc_court, []):
                        if law_ref not in set(seed_context):
                            sc_freq[law_ref] += 1
                sc_raw: dict[str, float] = {}
                for law_ref, freq in sc_freq.items():
                    if law_ref in law_to_idx:
                        candidates.add(law_ref)
                        idf_w = 1.0 / math.log(1 + law_degree.get(law_ref, 1))
                        sc_raw[law_ref] = float(freq) * idf_w
                sc_norm = minmax_norm(sc_raw)
                for law_ref in sc_norm:
                    if law_ref not in cos_scores:
                        li = law_to_idx[law_ref]
                        cos_scores[law_ref] = float((sub_embs @ law_emb_mat[li].reshape(1, -1).T).max())
                    final_scores[law_ref] = final_scores.get(law_ref, 0.0) + args.seed_cocit_weight * sc_norm.get(
                        law_ref, 0.0
                    )

        if args.boost_pre_citations > 0.0:
            for law in pre_citations:
                final_scores[law] = final_scores.get(law, 0.0) + args.boost_pre_citations
        if args.boost_pre_near > 0.0:
            for law in pre_near:
                final_scores[law] = final_scores.get(law, 0.0) + args.boost_pre_near

        if book_prior_enabled and citation_to_source_book:
            ranked_for_books = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)[
                : args.book_prior_top_candidates
            ]
            book_scores: dict[str, float] = {}
            for law, score in ranked_for_books:
                book = citation_to_source_book.get(law)
                if book:
                    book_scores[book] = book_scores.get(book, 0.0) + float(score)
            if book_scores:
                top_book_scores = dict(
                    sorted(book_scores.items(), key=lambda kv: kv[1], reverse=True)[
                        : args.book_prior_top_books
                    ]
                )
                book_prior_norm = minmax_norm(top_book_scores)
                for law in list(final_scores):
                    book = citation_to_source_book.get(law)
                    if book in book_prior_norm:
                        final_scores[law] += args.book_prior_weight * book_prior_norm[book]

        if book_pool_enabled and citation_to_source_book:
            ranked_for_book_pool = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)[
                : args.book_pool_inject_top_candidates
            ]
            book_sum_scores: dict[str, float] = {}
            book_best_scores: dict[str, float] = {}
            for law, score in ranked_for_book_pool:
                book = citation_to_source_book.get(law)
                if not book:
                    continue
                score_f = float(score)
                book_sum_scores[book] = book_sum_scores.get(book, 0.0) + score_f
                book_best_scores[book] = max(book_best_scores.get(book, 0.0), score_f)

            top_books = sorted(
                book_sum_scores,
                key=lambda book: (book_sum_scores[book], book_best_scores.get(book, 0.0), book),
                reverse=True,
            )[: args.book_pool_inject_top_books]

            for book in top_books:
                book_indices = book_to_law_indices.get(book)
                if book_indices is None or book_indices.size == 0:
                    continue
                book_embs = law_emb_mat[book_indices]
                if args.law_sim_mode == "max-subquery":
                    book_cos = (sub_embs @ book_embs.T).max(axis=0)
                else:
                    book_cos = avg_vec @ book_embs.T

                top_local_count = min(
                    max(args.book_pool_inject_per_book * 4, args.book_pool_inject_per_book),
                    len(book_indices),
                )
                if top_local_count >= len(book_indices):
                    top_local = np.argsort(book_cos)[::-1]
                else:
                    top_local = np.argpartition(book_cos, -top_local_count)[-top_local_count:]
                    top_local = top_local[np.argsort(book_cos[top_local])[::-1]]

                top_values = [float(book_cos[local_idx]) for local_idx in top_local]
                local_min = min(top_values) if top_values else 0.0
                local_max = max(top_values) if top_values else 1.0
                local_range = max(local_max - local_min, 1e-9)
                parent_score = book_best_scores.get(book, 0.0)
                added_for_book = 0
                for local_idx in top_local:
                    if added_for_book >= args.book_pool_inject_per_book:
                        break
                    law_idx = int(book_indices[int(local_idx)])
                    law_cit = law_citations[law_idx]
                    if law_cit in final_scores:
                        continue
                    cosine = float(book_cos[int(local_idx)])
                    cosine_factor = 0.5 + 0.5 * ((cosine - local_min) / local_range)
                    final_scores[law_cit] = (
                        parent_score
                        * args.book_pool_inject_score_scale
                        * cosine_factor
                    )
                    cos_scores[law_cit] = cosine
                    candidates.add(law_cit)
                    added_for_book += 1

        if args.law_reference_expand_top_candidates > 0:
            ranked_for_refs = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)[
                : args.law_reference_expand_top_candidates
            ]
            for parent_law, parent_score in ranked_for_refs:
                ref_neighbors: list[str] = []
                if args.law_reference_expand_mode in ("outgoing", "both"):
                    ref_neighbors.extend(law_out_refs.get(parent_law, []))
                if args.law_reference_expand_mode in ("incoming", "both"):
                    ref_neighbors.extend(law_in_refs.get(parent_law, []))

                added_for_parent = 0
                seen_ref_neighbors: set[str] = set()
                for neighbor in ref_neighbors:
                    if neighbor in seen_ref_neighbors or neighbor == parent_law:
                        continue
                    seen_ref_neighbors.add(neighbor)
                    if (
                        args.law_reference_expand_skip_procedural
                        and citation_to_source_book.get(neighbor, "") in procedural_books
                    ):
                        continue
                    if neighbor not in final_scores:
                        final_scores[neighbor] = (
                            float(parent_score) * args.law_reference_expand_score_scale
                        )
                        candidates.add(neighbor)
                        added_for_parent += 1
                    if (
                        args.law_reference_expand_max_neighbors > 0
                        and added_for_parent >= args.law_reference_expand_max_neighbors
                    ):
                        break

        if args.article_family_expand_top_candidates > 0 and article_family_to_citations:
            ranked_for_families = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)[
                : args.article_family_expand_top_candidates
            ]
            for parent_law, parent_score in ranked_for_families:
                family_key = article_family_key(parent_law)
                if family_key is None:
                    continue
                if (
                    args.article_family_expand_skip_procedural
                    and citation_to_source_book.get(parent_law, "") in procedural_books
                ):
                    continue
                siblings = article_family_to_citations.get(family_key, [])
                added_for_family = 0
                for sibling in siblings:
                    if sibling == parent_law:
                        continue
                    if (
                        args.article_family_expand_skip_procedural
                        and citation_to_source_book.get(sibling, "") in procedural_books
                    ):
                        continue
                    if sibling not in final_scores:
                        final_scores[sibling] = float(parent_score) * args.article_family_expand_score_scale
                        candidates.add(sibling)
                        added_for_family += 1
                    if (
                        args.article_family_expand_max_siblings > 0
                        and added_for_family >= args.article_family_expand_max_siblings
                    ):
                        break

        # Direct cosine injection: top-K cited-only laws by cosine, post-scoring
        if args.direct_cosine_inject > 0 and cited_law_idxs:
            cited_embs = law_emb_mat[cited_law_idxs]
            cited_cos = (sub_embs @ cited_embs.T).max(axis=0)
            top_cited = np.argsort(cited_cos)[::-1]
            existing_vals = sorted(final_scores.values())
            inject_pct = int(len(existing_vals) * 0.75)
            inject_score = existing_vals[inject_pct] if existing_vals else 0.0
            added_cos = 0
            for ci in top_cited:
                if added_cos >= args.direct_cosine_inject:
                    break
                law_cit = law_citations[cited_law_idxs[ci]]
                if law_cit not in final_scores:
                    final_scores[law_cit] = inject_score
                    added_cos += 1

        if cross_encoder_enabled:
            top_mmr = mmr_select(
                candidate_scores=final_scores,
                law_to_idx=law_to_idx,
                law_emb_mat=law_emb_mat,
                lambda_mmr=args.mmr_lambda,
                pool_k=max(args.mmr_pool, args.cross_encoder_rerank_top_k),
                final_k=args.cross_encoder_rerank_top_k,
            )
            top_mmr, score_rows, api_calls = rerank_substantive_slots_cosine_weighted(
                query_id=str(qid),
                query_text=query_text_by_q.get(str(qid), ""),
                query_texts=query_text_variants_by_q.get(str(qid)),
                aggregate_mode=args.cross_encoder_rerank_aggregate_mode,
                candidates=top_mmr,
                citation_to_document=citation_to_document,
                citation_to_source_book=citation_to_source_book,
                procedural_books=procedural_books,
                score_cache=cross_encoder_score_cache,
                top_k=args.cross_encoder_rerank_top_k,
                model=args.cross_encoder_rerank_model,
                base_scores=final_scores,
                cosine_scores=cos_scores,
                base_weight=args.cross_encoder_rerank_base_weight,
                cosine_weight=args.cross_encoder_rerank_cosine_weight,
                cross_encoder_weight=args.cross_encoder_rerank_ce_weight,
                api_key=args.cross_encoder_rerank_api_key,
                latency=args.cross_encoder_rerank_latency,
                timeout_s=args.cross_encoder_rerank_timeout_s,
                max_document_chars=args.cross_encoder_rerank_document_max_chars,
                score_mode=args.cross_encoder_rerank_score_mode,
                include_source_book=args.cross_encoder_rerank_include_source_book,
            )
            cross_encoder_score_rows.extend(score_rows)
            cross_encoder_api_calls += api_calls
            selection_scores = dict(final_scores)
            for row in score_rows:
                citation = str(row.get("citation", "")).strip()
                if not citation:
                    continue
                combined_score = row.get("combined_score")
                if combined_score is not None and pd.notna(combined_score):
                    selection_scores[citation] = float(combined_score)
                else:
                    base_score = row.get("base_score")
                    if base_score is not None and pd.notna(base_score):
                        selection_scores[citation] = float(base_score)
        else:
            top_mmr = mmr_select(
                candidate_scores=final_scores,
                law_to_idx=law_to_idx,
                law_emb_mat=law_emb_mat,
                lambda_mmr=args.mmr_lambda,
                pool_k=args.mmr_pool,
                final_k=args.final_k,
            )
            selection_scores = dict(final_scores)

        forced: list[str] = []
        if args.inject_pre_citations > 0 or args.inject_pre_near > 0:
            forced.extend(pre_citations[: args.inject_pre_citations])
            for law in pre_near[: args.inject_pre_near]:
                if law not in forced:
                    forced.append(law)

            seen_forced = set(forced)
            merged = forced + [x for x in top_mmr if x not in seen_forced]
            top_mmr = merged

        # Fill from plain ranking if needed
        fill_target_k = dynamic_score_max_k if dynamic_score_enabled else args.final_k
        if len(top_mmr) < fill_target_k:
            plain_rank = [x for x, _ in sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)]
            seen = set(top_mmr)
            for c in plain_rank:
                if c not in seen:
                    top_mmr.append(c)
                    seen.add(c)
                if len(top_mmr) >= fill_target_k:
                    break

        if dynamic_score_enabled:
            threshold = float(args.dynamic_score_threshold)
            forced_set = set(forced)
            filtered: list[str] = []
            seen_filtered: set[str] = set()

            def add_filtered(citation: str) -> None:
                if citation not in seen_filtered and len(filtered) < dynamic_score_max_k:
                    filtered.append(citation)
                    seen_filtered.add(citation)

            for rank, citation in enumerate(top_mmr, start=1):
                keep = False
                if args.dynamic_score_keep_top > 0 and rank <= args.dynamic_score_keep_top:
                    keep = True
                if args.dynamic_score_keep_injected and citation in forced_set:
                    keep = True
                if selection_scores.get(citation, float("-inf")) >= threshold:
                    keep = True
                if keep:
                    add_filtered(citation)

            if len(filtered) < args.dynamic_score_min_k:
                for citation in top_mmr:
                    if len(filtered) >= args.dynamic_score_min_k:
                        break
                    add_filtered(citation)

            top_mmr = filtered[:dynamic_score_max_k]
        else:
            top_mmr = top_mmr[: args.final_k]
        pred_map[qid] = top_mmr
        rows.append({"query_id": qid, "predicted_citations": ";".join(top_mmr)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path}")
    if cross_encoder_enabled:
        args.cross_encoder_rerank_score_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cross_encoder_score_rows).to_csv(
            args.cross_encoder_rerank_score_path,
            index=False,
        )
        print(f"Cross enc API calls: {cross_encoder_api_calls:,}")
        print(f"Cross enc score rows: {len(cross_encoder_score_rows):,}")
        print(f"Cross enc scores: {args.cross_encoder_rerank_score_path}")

    if gold_path is not None:
        val_gold_df = pd.read_parquet(gold_path, columns=["query_id", "gold_citations"])
        gold_map = {}
        for _, r in val_gold_df.iterrows():
            arts = [
                x.strip()
                for x in str(r["gold_citations"]).split(";")
                if x.strip().startswith("Art.")
            ]
            gold_map[r["query_id"]] = set(arts)

        f1_scores = []
        for qid in query_ids:
            pred = set(pred_map[qid])
            gold = gold_map.get(qid, set())
            if not pred and not gold:
                f1 = 1.0
            elif not pred or not gold:
                f1 = 0.0
            else:
                tp = len(pred & gold)
                precision = tp / len(pred)
                recall = tp / len(gold)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            f1_scores.append(f1)
        macro_f1 = float(np.mean(f1_scores))
        print(f"Val Art-only Macro F1@{args.final_k}: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
