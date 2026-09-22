#!/usr/bin/env python3
"""Generate a law-only Kaggle submission with the best combined recipe.

This applies the validation-best law ranking stack to test queries:
  - embedding subquery RRF over the selected law corpus;
  - exact law citation hard include / boost from query hints;
  - book-code boost from query hints;
  - validated law-reference graph neighbor boost.
  - court-law exact-anchor expansion, followed by conservative TF-IDF reranking.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = REPOSITORY_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_rag.graph import neighbors_for as core_neighbors_for
from legal_rag.retrieval import minmax_normalize

from trials.court_law_exact_anchor import (
    build_court_law_exact_anchor_index,
    score_exact_anchor_neighbors,
)
from trials.court_law_tfidf_rerank import (
    build_court_law_tfidf_index,
    rerank_court_law_tfidf_candidates,
    score_court_law_tfidf_candidates,
)
from trials.substantive_cross_encoder_rerank import (
    load_env_file,
    load_procedural_books,
    load_score_cache,
    rerank_substantive_slots,
    rerank_substantive_slots_weighted,
)


BASE_DIR = REPOSITORY_ROOT
DEFAULT_TEST_PATH = BASE_DIR / "data/test.parquet"
DEFAULT_TEST_SUBQUERY_PATH = BASE_DIR / "data/test_sub_query_zembed.parquet"
DEFAULT_LAW_PATH = BASE_DIR / "data/laws_db.parquet"
DEFAULT_DOMAIN_PATH = BASE_DIR / "data/zembed_law_domain_embedding.parquet"
DEFAULT_TRAIN_PATH = BASE_DIR / "data/train.parquet"
DEFAULT_COURT_CASE_PATH = BASE_DIR / "data/court_db.parquet"
DEFAULT_OUTPUT_PATH = BASE_DIR / "context/submission_zembed_combined_law_top20.csv"
DEFAULT_DETAIL_PATH = BASE_DIR / "context/test_zembed_combined_law_top20_details.csv"
DEFAULT_HINT_PATH = BASE_DIR / "context/test_query_law_hint_parse_law.csv"
DEFAULT_CROSS_ENCODER_SCORE_PATH = BASE_DIR / "context/law_cross_encoder_rerank_scores.csv"
DEFAULT_PROCEDURAL_BOOK_PATH = BASE_DIR / "context/proc_vs_subs.md"

CANON_RE = re.compile(r"^Art\.\s*(?P<art>\S+?)(?:\s+Abs\.\s*(?P<abs>\S+?))?\s+(?P<book>.+)$")
MENTION_RE = re.compile(
    r"\b(?:Art\.?|Artikel)\s*"
    r"(?P<art>\d+(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|[a-z](?:bis)?)?)"
    r"(?:\s*(?:Abs\.?|Absatz)\s*(?P<abs>\d+(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|[a-z](?:bis)?)?))?"
    r"(?:\s*(?:lit\.?|Bst\.?|Ziff\.?)\s*[a-z](?:bis)?)?"
    r"\s+(?P<book>[A-ZÄÖÜ][A-Za-zÄÖÜäöü0-9+.\- ]{1,24})"
)

NON_BOOK_TOKENS = {
    "ABS",
    "ABS.",
    "ABSATZ",
    "ABSÄTZE",
    "BUCHSTABE",
    "BUCHSTABEN",
    "BST",
    "BST.",
    "ZIFF",
    "ZIFF.",
    "ZIFFER",
    "ZIFFERN",
    "UND",
    "ODER",
    "SATZ",
    "SÄTZE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate zembed combined law-only test submission.")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--test-subquery-path", type=Path, default=DEFAULT_TEST_SUBQUERY_PATH)
    parser.add_argument("--law-path", type=Path, default=DEFAULT_LAW_PATH)
    parser.add_argument("--subquery-embedding-column", default="zembed_embedding")
    parser.add_argument("--law-embedding-column", default="embedding")
    parser.add_argument("--domain-path", type=Path, default=DEFAULT_DOMAIN_PATH)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--court-case-path", type=Path, default=DEFAULT_COURT_CASE_PATH)
    parser.add_argument(
        "--court-parquet-batch-size",
        type=int,
        default=2048,
        help="Rows per parquet batch when streaming the large court DB law_references column.",
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--detail-path", type=Path, default=DEFAULT_DETAIL_PATH)
    parser.add_argument("--hint-path", type=Path, default=DEFAULT_HINT_PATH)
    parser.add_argument("--top-per-subquery", type=int, default=200)
    parser.add_argument("--final-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--exact-boost", type=float, default=0.10)
    parser.add_argument("--book-boost", type=float, default=0.005)
    parser.add_argument("--book-network-top-books", type=int, default=4)
    parser.add_argument("--book-network-boost", type=float, default=0.01)
    parser.add_argument("--book-network-hint-weight", type=float, default=0.25)
    parser.add_argument("--book-network-weight", type=float, default=0.40)
    parser.add_argument("--min-network-count", type=int, default=2)
    parser.add_argument("--graph-mode", choices=["incoming", "outgoing", "both", "none"], default="both")
    parser.add_argument("--graph-seed-k", type=int, default=200)
    parser.add_argument("--graph-boost", type=float, default=0.1)
    parser.add_argument("--court-law-exact-anchor-min-count", type=int, default=2)
    parser.add_argument("--court-law-exact-anchor-top-targets", type=int, default=50)
    parser.add_argument("--court-law-exact-anchor-ranked-k", type=int, default=5)
    parser.add_argument("--court-law-exact-anchor-boost", type=float, default=0.08)
    parser.add_argument("--court-law-tfidf-rerank-min-count", type=int, default=2)
    parser.add_argument("--court-law-tfidf-rerank-candidate-k", type=int, default=100)
    parser.add_argument("--court-law-tfidf-rerank-top-targets", type=int, default=50)
    parser.add_argument("--court-law-tfidf-rerank-boost", type=float, default=0.0025)
    parser.add_argument("--skip-standalone-tfidf-rerank", action="store_true")
    parser.add_argument("--cross-encoder-rerank-top-k", type=int, default=50)
    parser.add_argument("--cross-encoder-rerank-mode", choices=["weighted", "ce_only"], default="weighted")
    parser.add_argument("--cross-encoder-rerank-model", default="zerank-2")
    parser.add_argument("--cross-encoder-rerank-query-column", default="german_translation")
    parser.add_argument("--cross-encoder-rerank-score-path", type=Path, default=DEFAULT_CROSS_ENCODER_SCORE_PATH)
    parser.add_argument("--cross-encoder-rerank-procedural-path", type=Path, default=DEFAULT_PROCEDURAL_BOOK_PATH)
    parser.add_argument("--cross-encoder-rerank-api-key", default=None)
    parser.add_argument("--cross-encoder-rerank-env-path", type=Path, default=BASE_DIR / ".env")
    parser.add_argument("--cross-encoder-rerank-latency", choices=["fast", "slow"], default=None)
    parser.add_argument("--cross-encoder-rerank-timeout-s", type=int, default=120)
    parser.add_argument("--cross-encoder-rerank-document-max-chars", type=int, default=4000)
    parser.add_argument("--cross-encoder-rerank-tfidf-weight", type=float, default=0.40)
    parser.add_argument("--cross-encoder-rerank-cosine-weight", type=float, default=0.20)
    parser.add_argument("--cross-encoder-rerank-ce-weight", type=float, default=0.40)
    return parser.parse_args()


def split_refs(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def norm_space(value: object) -> str:
    return re.sub(r"\s+", " ", "" if pd.isna(value) else str(value)).strip()


def norm_book(value: object) -> str:
    return norm_space(value).strip(".,;:()[]").upper()


def normalize_candidate_book(value: object, book_norm_to_canonical: dict[str, str]) -> str:
    candidate = norm_book(value)
    if candidate in book_norm_to_canonical:
        return book_norm_to_canonical[candidate]
    without_footnote = re.sub(r"(?<=[A-ZÄÖÜa-zäöü])\d{1,3}$", "", candidate)
    if without_footnote in book_norm_to_canonical:
        return book_norm_to_canonical[without_footnote]
    first_token = candidate.split()[0] if candidate else ""
    if first_token in book_norm_to_canonical:
        return book_norm_to_canonical[first_token]
    first_without_footnote = re.sub(r"(?<=[A-ZÄÖÜa-zäöü])\d{1,3}$", "", first_token)
    if first_without_footnote in book_norm_to_canonical:
        return book_norm_to_canonical[first_without_footnote]
    return ""


def citation_book(citation: object) -> str:
    match = CANON_RE.match(str(citation).strip())
    return match.group("book") if match else ""


def split_document(document: object) -> tuple[str, str]:
    text = norm_space(document)
    title_match = re.search(r"\bGesetz:\s*(.*?)(?=\s*(?:Regelungsbereich:|Normtext:)|$)", text)
    normtext_match = re.search(r"\bNormtext:\s*(.*)$", text)
    title = title_match.group(1).strip() if title_match else ""
    body = normtext_match.group(1).strip() if normtext_match else text
    return title, body


def as_matrix(values: pd.Series) -> np.ndarray:
    matrix = np.asarray(values.tolist(), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def minmax(values: dict[str, float]) -> dict[str, float]:
    """Compatibility wrapper around the public score-normalization helper."""

    return minmax_normalize(values)


def directed_confidence(
    pair_counts: Counter[tuple[str, str]],
    source_counts: Counter[str],
    *,
    min_count: int,
) -> dict[str, dict[str, float]]:
    adjacency: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for (src, dst), count in pair_counts.items():
        if src != dst and count >= min_count and source_counts[src]:
            adjacency[src][dst] = count / source_counts[src]
    return dict(adjacency)


def add_undirected_pairs(books: set[str], pair_counts: Counter[tuple[str, str]]) -> None:
    for left, right in itertools.combinations(sorted(books), 2):
        pair_counts[(left, right)] += 1
        pair_counts[(right, left)] += 1


def build_law_doc_book_network(laws: pd.DataFrame, *, min_count: int) -> dict[str, dict[str, float]]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    source_counts: Counter[str] = Counter()
    for row in laws.itertuples(index=False):
        src = norm_space(row.source_book)
        if not src:
            continue
        for dst in split_refs(getattr(row, "target_books", "")):
            dst = norm_space(dst)
            if src and dst and src != dst:
                pair_counts[(src, dst)] += 1
                source_counts[src] += 1
    return directed_confidence(pair_counts, source_counts, min_count=min_count)


def build_train_book_network(path: Path, *, min_count: int) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    train = pd.read_parquet(path, columns=["gold_citations"])
    pair_counts: Counter[tuple[str, str]] = Counter()
    source_counts: Counter[str] = Counter()
    for row in train.itertuples(index=False):
        books = {citation_book(citation) for citation in split_refs(row.gold_citations)}
        books.discard("")
        if books:
            source_counts.update(books)
            add_undirected_pairs(books, pair_counts)
    return directed_confidence(pair_counts, source_counts, min_count=min_count)


def iter_court_law_references(path: Path, *, batch_size: int) -> object:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for streaming court law references") from error

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["law_references"]):
        for value in batch.column("law_references").to_pylist():
            yield value


def build_court_book_network(
    path: Path,
    *,
    min_count: int,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    if batch_size <= 0:
        raise ValueError("--court-parquet-batch-size must be positive")
    pair_counts: Counter[tuple[str, str]] = Counter()
    source_counts: Counter[str] = Counter()
    for law_references in iter_court_law_references(path, batch_size=batch_size):
        books = {citation_book(citation) for citation in split_refs(law_references)}
        books.discard("")
        if books:
            source_counts.update(books)
            add_undirected_pairs(books, pair_counts)
    return directed_confidence(pair_counts, source_counts, min_count=min_count)


def build_graph_neighbors_from_law_columns(laws: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    outgoing: defaultdict[str, set[str]] = defaultdict(set)
    incoming: defaultdict[str, set[str]] = defaultdict(set)
    citation_set = set(laws["citation"].astype(str))
    for row in laws.itertuples(index=False):
        source = norm_space(row.citation)
        for target in split_refs(getattr(row, "references", "")):
            target = norm_space(target)
            if source and target and source != target and target in citation_set:
                outgoing[source].add(target)
                incoming[target].add(source)
    return {"outgoing": dict(outgoing), "incoming": dict(incoming)}


def required_law_columns(embedding_column: str, *, include_document: bool) -> list[str]:
    columns = [
        "citation",
        embedding_column,
        "references",
        "source_book",
        "target_books",
    ]
    if include_document:
        columns.append("document")
    return columns


def neighbors_for(
    citation: str,
    mode: str,
    graph_neighbors: dict[str, dict[str, set[str]]],
) -> set[str]:
    """Compatibility wrapper preserving the legacy invalid-mode behavior."""

    if mode not in {"incoming", "outgoing", "both"}:
        return set()
    return core_neighbors_for(citation, mode, graph_neighbors)


def build_citation_lookup(citations: set[str]) -> dict[tuple[str, str, str], str]:
    lookup: dict[tuple[str, str, str], str] = {}
    for citation in citations:
        match = CANON_RE.match(citation)
        if not match:
            continue
        book = norm_book(match.group("book"))
        art = match.group("art").lower()
        abs_no = (match.group("abs") or "").lower()
        lookup[(book, art, abs_no)] = citation
    return lookup


def split_title(title: object) -> tuple[str, str]:
    text = norm_space(title)
    if " - " not in text:
        return text, ""
    base, domain = text.split(" - ", 1)
    return base.strip(), domain.strip()


def build_domain_embeddings(args: argparse.Namespace) -> pd.DataFrame:
    merged = pd.read_parquet(
        args.law_path,
        columns=["citation", args.law_embedding_column, "document"],
    )
    merged["citation"] = merged["citation"].astype(str).map(norm_space)
    merged["title"] = merged["document"].map(lambda value: split_document(value)[0])
    merged["domain_id"] = merged["title"]
    missing_title = merged["domain_id"].eq("")
    merged.loc[missing_title, "domain_id"] = (
        "UNKNOWN_TITLE::" + merged.loc[missing_title, "citation"].astype(str)
    )
    merged["book_code"] = merged["citation"].map(citation_book)

    rows: list[dict[str, object]] = []
    for domain_id, group in merged.groupby("domain_id", sort=True):
        vectors = np.asarray(group[args.law_embedding_column].tolist(), dtype=np.float32)
        avg = vectors.mean(axis=0, dtype=np.float32)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm

        title = "" if str(domain_id).startswith("UNKNOWN_TITLE::") else str(domain_id)
        book_description, domain = split_title(title)
        book_counts = group["book_code"].value_counts()
        book_code = str(book_counts.index[0]) if len(book_counts) else ""

        rows.append(
            {
                "domain_id": str(domain_id),
                "book_code": book_code,
                "book_description": book_description,
                "domain": domain,
                "title": title,
                "num_laws": int(len(group)),
                args.law_embedding_column: avg.astype(np.float32).tolist(),
            }
        )

    domain_df = pd.DataFrame(rows)
    args.domain_path.parent.mkdir(parents=True, exist_ok=True)
    domain_df.to_parquet(args.domain_path, index=False)
    return domain_df


def expand_book_scores(
    seed_scores: dict[str, float],
    networks: dict[str, dict[str, dict[str, float]]],
    *,
    network_weight: float,
) -> dict[str, float]:
    scores: defaultdict[str, float] = defaultdict(float)
    for book, value in seed_scores.items():
        scores[book] += value
    for adjacency in networks.values():
        for book, seed_score in seed_scores.items():
            for neighbor, confidence in adjacency.get(book, {}).items():
                scores[neighbor] += network_weight * seed_score * confidence
    return dict(scores)


def build_book_network_scores(
    *,
    args: argparse.Namespace,
    test: pd.DataFrame,
    subqueries: pd.DataFrame,
    hints_by_query: dict[str, dict[str, object]],
    laws: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    if args.book_network_top_books <= 0 or args.book_network_boost <= 0:
        return {}

    domains = pd.read_parquet(args.domain_path) if args.domain_path.exists() else build_domain_embeddings(args)
    for candidate_column in [args.law_embedding_column, "embedding", "zembed_embedding"]:
        if candidate_column in domains.columns:
            domain_embedding_column = candidate_column
            break
    else:
        raise SystemExit(f"No supported embedding column found in {args.domain_path}")
    domain_ids = domains["domain_id"].astype(str).to_numpy()
    domain_books = domains["book_code"].astype(str).to_numpy()
    domain_matrix = as_matrix(domains[domain_embedding_column])
    networks = {
        "law": build_law_doc_book_network(laws, min_count=args.min_network_count),
        "train": build_train_book_network(args.train_path, min_count=args.min_network_count),
        "court": build_court_book_network(
            args.court_case_path,
            min_count=args.min_network_count,
            batch_size=args.court_parquet_batch_size,
        ),
    }

    scores_by_query: dict[str, dict[str, float]] = {}
    local_top = min(10, len(domain_ids))
    sort_column = "sub_id" if "sub_id" in subqueries.columns else "subquery_id"
    for row in test.itertuples(index=False):
        query_id = str(row.query_id)
        query_subs = subqueries[subqueries["query_id"].astype(str).eq(query_id)].sort_values(sort_column)
        if query_subs.empty:
            scores_by_query[query_id] = {}
            continue
        similarities = as_matrix(query_subs[args.subquery_embedding_column]) @ domain_matrix.T

        domain_rrf: defaultdict[int, float] = defaultdict(float)
        domain_best: dict[int, float] = {}
        for scores in similarities:
            top_idx = np.argpartition(scores, -local_top)[-local_top:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            for rank, domain_idx in enumerate(top_idx, start=1):
                idx = int(domain_idx)
                cosine = float(scores[domain_idx])
                domain_rrf[idx] += 1.0 / (args.rrf_k + rank)
                domain_best[idx] = max(domain_best.get(idx, -1.0), cosine)

        ranked_domains = sorted(
            domain_rrf,
            key=lambda idx: (domain_rrf[idx], domain_best.get(idx, -1.0), str(domain_ids[idx])),
            reverse=True,
        )[:100]

        book_count: defaultdict[str, float] = defaultdict(float)
        book_rrf: defaultdict[str, float] = defaultdict(float)
        book_best: dict[str, float] = {}
        for domain_idx in ranked_domains:
            book = str(domain_books[domain_idx])
            book_count[book] += 1.0
            book_rrf[book] += domain_rrf[domain_idx]
            book_best[book] = max(book_best.get(book, -1.0), domain_best.get(domain_idx, -1.0))

        count_n = minmax(book_count)
        rrf_n = minmax(book_rrf)
        best_n = minmax(book_best)
        seed_scores = {
            book: 0.45 * count_n.get(book, 0.0)
            + 0.45 * rrf_n.get(book, 0.0)
            + 0.10 * best_n.get(book, 0.0)
            for book in set(book_count)
        }
        for book in hints_by_query.get(query_id, {}).get("books", set()):
            if book:
                seed_scores[book] = seed_scores.get(book, 0.0) + args.book_network_hint_weight

        expanded = expand_book_scores(seed_scores, networks, network_weight=args.book_network_weight)
        top_scores = {
            book: expanded[book]
            for book in sorted(
                expanded,
                key=lambda book: (expanded[book], seed_scores.get(book, 0.0), book),
                reverse=True,
            )[: args.book_network_top_books]
        }
        max_score = max(top_scores.values(), default=1.0)
        if max_score > 0:
            top_scores = {book: float(value) / max_score for book, value in top_scores.items()}
        scores_by_query[query_id] = top_scores

    return scores_by_query


def parse_query_hints(
    *,
    test: pd.DataFrame,
    subqueries: pd.DataFrame,
    citation_set: set[str],
    citation_lookup: dict[tuple[str, str, str], str],
    book_norm_to_canonical: dict[str, str],
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    sub_text_by_query = (
        subqueries.groupby("query_id")["sub_query_de"]
        .apply(lambda values: "\n".join(str(value) for value in values if str(value).strip()))
        .to_dict()
    )
    hints: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    for row in test.itertuples(index=False):
        query_id = str(row.query_id)
        text_parts = [
            getattr(row, "query", ""),
            getattr(row, "german_translation", ""),
            getattr(row, "decomposed_queries_raw", ""),
            sub_text_by_query.get(query_id, ""),
        ]
        text = "\n".join(str(part) for part in text_parts if str(part).strip())

        exact: list[str] = []
        seen_exact: set[str] = set()
        books: set[str] = set()

        for raw_ref in split_refs(getattr(row, "pre_near_matches", "")):
            canonical = norm_space(raw_ref)
            if canonical in citation_set and canonical not in seen_exact:
                exact.append(canonical)
                seen_exact.add(canonical)
                books.add(citation_book(canonical))

        for raw_ref in split_refs(getattr(row, "pre_citations", "")):
            canonical = norm_space(raw_ref)
            if canonical in citation_set and canonical not in seen_exact:
                exact.append(canonical)
                seen_exact.add(canonical)
                books.add(citation_book(canonical))

        for match in MENTION_RE.finditer(text):
            art = (match.group("art") or "").lower()
            abs_no = (match.group("abs") or "").lower()
            book = normalize_candidate_book(match.group("book"), book_norm_to_canonical)
            if not book or norm_book(book) in NON_BOOK_TOKENS:
                continue
            books.add(book)
            canonical = citation_lookup.get((norm_book(book), art, abs_no))
            if canonical and canonical not in seen_exact:
                exact.append(canonical)
                seen_exact.add(canonical)

        hints[query_id] = {"exact": exact, "books": books}
        rows.append(
            {
                "query_id": query_id,
                "parsed_exact_citations": ";".join(exact),
                "parsed_books": ";".join(sorted(book for book in books if book)),
            }
        )

    return hints, pd.DataFrame(rows)


def retrieve_zembed_rrf(
    *,
    test: pd.DataFrame,
    subqueries: pd.DataFrame,
    law_citations: np.ndarray,
    law_matrix: np.ndarray,
    subquery_embedding_column: str,
    top_per_subquery: int,
    rrf_k: float,
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    ranked_by_query: dict[str, list[str]] = {}
    rrf_scores_by_query: dict[str, dict[str, float]] = {}
    cosine_scores_by_query: dict[str, dict[str, float]] = {}
    local_top = min(top_per_subquery, len(law_citations))
    sort_column = "sub_id" if "sub_id" in subqueries.columns else "subquery_id"
    for row in test.itertuples(index=False):
        query_id = str(row.query_id)
        query_subs = subqueries[subqueries["query_id"].astype(str).eq(query_id)].sort_values(sort_column)
        if query_subs.empty:
            ranked_by_query[query_id] = []
            rrf_scores_by_query[query_id] = {}
            cosine_scores_by_query[query_id] = {}
            continue
        sub_matrix = as_matrix(query_subs[subquery_embedding_column])
        similarities = sub_matrix @ law_matrix.T
        rrf_scores: defaultdict[str, float] = defaultdict(float)
        best_cosine: dict[str, float] = {}

        for scores in similarities:
            top_idx = np.argpartition(scores, -local_top)[-local_top:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            for rank, law_idx in enumerate(top_idx, start=1):
                citation = str(law_citations[law_idx])
                rrf_scores[citation] += 1.0 / (rrf_k + rank)
                best_cosine[citation] = max(best_cosine.get(citation, -1.0), float(scores[law_idx]))

        ranked_by_query[query_id] = sorted(
            rrf_scores,
            key=lambda citation: (rrf_scores[citation], best_cosine.get(citation, -1.0), citation),
            reverse=True,
        )
        rrf_scores_by_query[query_id] = dict(rrf_scores)
        cosine_scores_by_query[query_id] = dict(best_cosine)
    return ranked_by_query, rrf_scores_by_query, cosine_scores_by_query


def apply_combined_boost(
    *,
    ranked: list[str],
    hints: dict[str, object],
    book_network_scores: dict[str, float],
    graph_neighbors: dict[str, dict[str, set[str]]],
    court_law_exact_anchor_index: dict[str, dict[str, float]],
    final_k: int,
    rrf_k: float,
    exact_boost: float,
    book_boost: float,
    book_network_boost: float,
    graph_mode: str,
    graph_seed_k: int,
    graph_boost: float,
    court_law_exact_anchor_top_targets: int,
    court_law_exact_anchor_ranked_k: int,
    court_law_exact_anchor_boost: float,
    return_scores: bool = False,
) -> list[str] | tuple[list[str], dict[str, float]]:
    exact = list(hints.get("exact", []))
    books = set(hints.get("books", set()))
    candidates: list[str] = []
    seen: set[str] = set()

    for citation in exact + ranked:
        if citation and citation not in seen:
            seen.add(citation)
            candidates.append(citation)

    scores = {
        citation: 1.0 / (rrf_k + rank)
        for rank, citation in enumerate(ranked, start=1)
    }

    if graph_mode != "none" and graph_boost > 0:
        for rank, seed_citation in enumerate(ranked[:graph_seed_k], start=1):
            seed_score = 1.0 / (rrf_k + rank)
            for neighbor in neighbors_for(seed_citation, graph_mode, graph_neighbors):
                if neighbor not in seen:
                    seen.add(neighbor)
                    candidates.append(neighbor)
                scores[neighbor] = scores.get(neighbor, 0.0) + graph_boost * seed_score

    exact_anchor_seeds = list(dict.fromkeys(exact + ranked[:court_law_exact_anchor_ranked_k]))
    exact_anchor_scores = score_exact_anchor_neighbors(
        exact_anchors=exact_anchor_seeds,
        index=court_law_exact_anchor_index,
        boost=court_law_exact_anchor_boost,
        top_targets_per_anchor=court_law_exact_anchor_top_targets,
    )
    for citation, boost_value in exact_anchor_scores.items():
        if citation not in seen:
            seen.add(citation)
            candidates.append(citation)
        scores[citation] = scores.get(citation, 0.0) + boost_value

    for citation in candidates:
        value = scores.get(citation, 0.0)
        if citation in exact:
            value += exact_boost
        book = citation_book(citation)
        if book in books:
            value += book_boost
        value += book_network_boost * book_network_scores.get(book, 0.0)
        scores[citation] = value

    ranked_candidates = sorted(
        candidates,
        key=lambda citation: (scores[citation], citation),
        reverse=True,
    )[:final_k]
    if return_scores:
        return ranked_candidates, {citation: scores[citation] for citation in candidates}
    return ranked_candidates


def main() -> None:
    args = parse_args()
    cross_encoder_enabled = args.cross_encoder_rerank_top_k > 0
    test = pd.read_parquet(args.test_path)
    subqueries = pd.read_parquet(args.test_subquery_path)
    laws = pd.read_parquet(
        args.law_path,
        columns=required_law_columns(
            args.law_embedding_column,
            include_document=cross_encoder_enabled,
        ),
    )
    laws["citation"] = laws["citation"].astype(str).map(norm_space)
    laws["source_book"] = laws["source_book"].astype(str).map(norm_space)
    citation_set = set(laws["citation"])
    graph_neighbors = build_graph_neighbors_from_law_columns(laws)
    graph_edge_rows = sum(len(targets) for targets in graph_neighbors["outgoing"].values())
    court_law_exact_anchor_index = (
        build_court_law_exact_anchor_index(
            args.court_case_path,
            valid_citations=citation_set,
            min_count=args.court_law_exact_anchor_min_count,
            batch_size=args.court_parquet_batch_size,
        )
        if args.court_law_exact_anchor_boost > 0 and args.court_law_exact_anchor_top_targets > 0
        else {}
    )
    court_law_exact_anchor_edges = sum(
        len(targets) for targets in court_law_exact_anchor_index.values()
    )
    court_law_tfidf_index = (
        build_court_law_tfidf_index(
            args.court_case_path,
            valid_citations=citation_set,
            min_count=args.court_law_tfidf_rerank_min_count,
            batch_size=args.court_parquet_batch_size,
        )
        if args.court_law_tfidf_rerank_boost > 0 and args.court_law_tfidf_rerank_candidate_k > 0
        else {}
    )
    court_law_tfidf_edges = sum(len(targets) for targets in court_law_tfidf_index.values())

    law_books = sorted({citation_book(citation) for citation in laws["citation"]})
    book_norm_to_canonical = {norm_book(book): book for book in law_books if book}
    citation_lookup = build_citation_lookup(citation_set)

    hints_by_query, hint_df = parse_query_hints(
        test=test,
        subqueries=subqueries,
        citation_set=citation_set,
        citation_lookup=citation_lookup,
        book_norm_to_canonical=book_norm_to_canonical,
    )
    book_network_scores_by_query = build_book_network_scores(
        args=args,
        test=test,
        subqueries=subqueries,
        hints_by_query=hints_by_query,
        laws=laws,
    )
    law_citations = np.asarray(laws["citation"].tolist(), dtype=object)
    law_matrix = as_matrix(laws[args.law_embedding_column])

    ranked_by_query, rrf_scores_by_query, cosine_scores_by_query = retrieve_zembed_rrf(
        test=test,
        subqueries=subqueries,
        law_citations=law_citations,
        law_matrix=law_matrix,
        subquery_embedding_column=args.subquery_embedding_column,
        top_per_subquery=args.top_per_subquery,
        rrf_k=args.rrf_k,
    )

    procedural_books: set[str] = set()
    citation_to_document: dict[str, str] = {}
    citation_to_source_book: dict[str, str] = {}
    cross_encoder_score_cache: dict[tuple[str, str], float] = {}
    cross_encoder_score_rows: list[dict[str, object]] = []
    cross_encoder_api_calls = 0
    if cross_encoder_enabled:
        load_env_file(args.cross_encoder_rerank_env_path)
        procedural_books = load_procedural_books(args.cross_encoder_rerank_procedural_path)
        citation_to_source_book = dict(zip(laws["citation"], laws["source_book"]))
        citation_to_document = dict(zip(laws["citation"], laws["document"].astype(str)))
        cross_encoder_score_cache = load_score_cache(
            args.cross_encoder_rerank_score_path,
            model=args.cross_encoder_rerank_model,
        )

    output_rows: list[dict[str, str]] = []
    detail_rows: list[dict[str, object]] = []
    for row in test.itertuples(index=False):
        query_id = str(row.query_id)
        ranked = ranked_by_query.get(query_id, [])
        hints = hints_by_query.get(query_id, {"exact": [], "books": set()})
        if court_law_tfidf_index:
            pre_tfidf_candidates = apply_combined_boost(
                ranked=ranked,
                hints=hints,
                book_network_scores=book_network_scores_by_query.get(query_id, {}),
                graph_neighbors=graph_neighbors,
                court_law_exact_anchor_index=court_law_exact_anchor_index,
                final_k=args.court_law_tfidf_rerank_candidate_k,
                rrf_k=args.rrf_k,
                exact_boost=args.exact_boost,
                book_boost=args.book_boost,
                book_network_boost=args.book_network_boost,
                graph_mode=args.graph_mode,
                graph_seed_k=args.graph_seed_k,
                graph_boost=args.graph_boost,
                court_law_exact_anchor_top_targets=args.court_law_exact_anchor_top_targets,
                court_law_exact_anchor_ranked_k=args.court_law_exact_anchor_ranked_k,
                court_law_exact_anchor_boost=args.court_law_exact_anchor_boost,
            )
            tfidf_scores = score_court_law_tfidf_candidates(
                pre_tfidf_candidates,
                index=court_law_tfidf_index,
                boost=args.court_law_tfidf_rerank_boost,
                candidate_k=args.court_law_tfidf_rerank_candidate_k,
                top_targets_per_anchor=args.court_law_tfidf_rerank_top_targets,
                rrf_k=args.rrf_k,
            )
            if args.skip_standalone_tfidf_rerank:
                reranked_candidates = pre_tfidf_candidates
            else:
                reranked_candidates = rerank_court_law_tfidf_candidates(
                    pre_tfidf_candidates,
                    index=court_law_tfidf_index,
                    boost=args.court_law_tfidf_rerank_boost,
                    candidate_k=args.court_law_tfidf_rerank_candidate_k,
                    top_targets_per_anchor=args.court_law_tfidf_rerank_top_targets,
                    rrf_k=args.rrf_k,
                )
            candidate_pool = reranked_candidates
            base_top_citations = pre_tfidf_candidates[: args.final_k]
        else:
            candidate_pool = apply_combined_boost(
                ranked=ranked,
                hints=hints,
                book_network_scores=book_network_scores_by_query.get(query_id, {}),
                graph_neighbors=graph_neighbors,
                court_law_exact_anchor_index=court_law_exact_anchor_index,
                final_k=max(args.final_k, args.cross_encoder_rerank_top_k),
                rrf_k=args.rrf_k,
                exact_boost=args.exact_boost,
                book_boost=args.book_boost,
                book_network_boost=args.book_network_boost,
                graph_mode=args.graph_mode,
                graph_seed_k=args.graph_seed_k,
                graph_boost=args.graph_boost,
                court_law_exact_anchor_top_targets=args.court_law_exact_anchor_top_targets,
                court_law_exact_anchor_ranked_k=args.court_law_exact_anchor_ranked_k,
                court_law_exact_anchor_boost=args.court_law_exact_anchor_boost,
            )
            tfidf_scores = {}
            base_top_citations = ranked[: args.final_k]
        if cross_encoder_enabled:
            query_text = ""
            if args.cross_encoder_rerank_query_column in test.columns:
                query_text = norm_space(getattr(row, args.cross_encoder_rerank_query_column, ""))
            if not query_text:
                query_text = norm_space(getattr(row, "query", ""))
            if args.cross_encoder_rerank_mode == "ce_only":
                candidate_pool, score_rows, api_calls = rerank_substantive_slots(
                    query_id=query_id,
                    query_text=query_text,
                    candidates=candidate_pool,
                    citation_to_document=citation_to_document,
                    citation_to_source_book=citation_to_source_book,
                    procedural_books=procedural_books,
                    score_cache=cross_encoder_score_cache,
                    top_k=args.cross_encoder_rerank_top_k,
                    model=args.cross_encoder_rerank_model,
                    api_key=args.cross_encoder_rerank_api_key,
                    latency=args.cross_encoder_rerank_latency,
                    timeout_s=args.cross_encoder_rerank_timeout_s,
                    max_document_chars=args.cross_encoder_rerank_document_max_chars,
                )
            else:
                candidate_pool, score_rows, api_calls = rerank_substantive_slots_weighted(
                    query_id=query_id,
                    query_text=query_text,
                    candidates=candidate_pool,
                    citation_to_document=citation_to_document,
                    citation_to_source_book=citation_to_source_book,
                    procedural_books=procedural_books,
                    score_cache=cross_encoder_score_cache,
                    top_k=args.cross_encoder_rerank_top_k,
                    model=args.cross_encoder_rerank_model,
                    tfidf_scores=tfidf_scores,
                    cosine_scores=cosine_scores_by_query.get(query_id, {}),
                    tfidf_weight=args.cross_encoder_rerank_tfidf_weight,
                    cosine_weight=args.cross_encoder_rerank_cosine_weight,
                    cross_encoder_weight=args.cross_encoder_rerank_ce_weight,
                    api_key=args.cross_encoder_rerank_api_key,
                    latency=args.cross_encoder_rerank_latency,
                    timeout_s=args.cross_encoder_rerank_timeout_s,
                    max_document_chars=args.cross_encoder_rerank_document_max_chars,
                )
            cross_encoder_score_rows.extend(score_rows)
            cross_encoder_api_calls += api_calls
        prediction = candidate_pool[: args.final_k]
        output_rows.append({"query_id": query_id, "predicted_citations": ";".join(prediction)})
        detail_rows.append(
            {
                "query_id": query_id,
                "pred_count": len(prediction),
                "parsed_exact_citations": ";".join(hints["exact"]),
                "parsed_books": ";".join(sorted(hints["books"])),
                "book_network_books": ";".join(book_network_scores_by_query.get(query_id, {})),
                "base_top_citations": ";".join(base_top_citations),
                "predicted_citations": ";".join(prediction),
            }
        )

    submission = pd.DataFrame(output_rows)
    details = pd.DataFrame(detail_rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.detail_path.parent.mkdir(parents=True, exist_ok=True)
    args.hint_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_path, index=False)
    details.to_csv(args.detail_path, index=False)
    hint_df.to_csv(args.hint_path, index=False)
    if cross_encoder_enabled:
        args.cross_encoder_rerank_score_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cross_encoder_score_rows).to_csv(
            args.cross_encoder_rerank_score_path,
            index=False,
        )

    print(f"Test rows         : {len(test):,}")
    print(f"Subquery rows     : {len(subqueries):,}")
    print(f"Law corpus rows   : {len(laws):,}")
    print(f"Law embedding col : {args.law_embedding_column}")
    print(f"Query embed col   : {args.subquery_embedding_column}")
    print("Graph source      : law parquet columns")
    print(f"Graph edge rows   : {graph_edge_rows:,}")
    print(f"Exact anchor edges: {court_law_exact_anchor_edges:,}")
    print(f"Exact anchor top-k: {args.court_law_exact_anchor_ranked_k}")
    print(f"TF-IDF edges      : {court_law_tfidf_edges:,}")
    print(f"TF-IDF candidate k: {args.court_law_tfidf_rerank_candidate_k}")
    print(f"TF-IDF standalone : {not args.skip_standalone_tfidf_rerank}")
    print(f"Cross encoder top : {args.cross_encoder_rerank_top_k}")
    if cross_encoder_enabled:
        print(f"Cross encoder mode: {args.cross_encoder_rerank_mode}")
        print(
            "Cross encoder wts : "
            f"tfidf={args.cross_encoder_rerank_tfidf_weight}, "
            f"cosine={args.cross_encoder_rerank_cosine_weight}, "
            f"ce={args.cross_encoder_rerank_ce_weight}"
        )
        print(f"Cross encoder mdl : {args.cross_encoder_rerank_model}")
        print(f"Cross encoder proc: {len(procedural_books):,} books")
        print(f"Cross encoder API : {cross_encoder_api_calls:,} calls")
        print(f"Cross encoder rows: {len(cross_encoder_score_rows):,}")
        print(f"Cross encoder CSV : {args.cross_encoder_rerank_score_path}")
    print(f"Final citations/q : {args.final_k}")
    print(f"Book net top books: {args.book_network_top_books}")
    print(f"Book net boost    : {args.book_network_boost}")
    print(f"Submission        : {args.output_path}")
    print(f"Details           : {args.detail_path}")
    print(f"Hints             : {args.hint_path}")
    print(submission.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
