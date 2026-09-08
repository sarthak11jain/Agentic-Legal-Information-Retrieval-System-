#!/usr/bin/env python3
"""Generate pre_citations and pre_near_matches for query CSV/parquet files.

This is an archive utility for reconstructing the pre-parsed citation hints
already present in data/val.parquet and data/test.parquet. It does not modify
the input file; it writes a separate parquet/CSV and can compare generated
columns against existing columns.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LAWS_PATH = BASE_DIR / "data/laws_db.parquet"

CANON_RE = re.compile(
    r"^Art\.\s*(?P<art>\S+?)(?:\s+Abs\.\s*(?P<abs>\S+?))?\s+(?P<book>.+)$"
)
MENTION_RE = re.compile(
    r"\b(?:Art\.?|Artikel)\s*"
    r"(?P<art>\d+(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|[a-z](?:bis)?)?)"
    r"(?:\s*(?:Abs\.?|Absatz)\s*(?P<abs>\d+(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|[a-z](?:bis)?)?))?"
    r"(?:\s*(?:lit\.?|Bst\.?)\s*(?P<lit>[a-z](?:bis)?))?"
    r"\s+(?P<book>[A-ZÄÖÜ][A-Za-zÄÖÜäöü0-9+.\-]{1,24})"
)

# The existing val/test columns include one foreign procedural reference
# normalized to its nearest Swiss procedural book. Keep this tiny fallback
# explicit instead of hiding it inside fuzzy matching.
BOOK_FALLBACKS = {
    "CO": "OR",
    "CPLR": "ZPO",
}

# These paragraphs exist in the current laws_db snapshot, but the already
# stored val/test pre_near_matches columns were generated without them. Keep
# this compatibility layer explicit so the old columns can be reproduced while
# the default current-laws mode remains transparent.
LEGACY_VAL_TEST_NEAR_EXCLUSIONS = {
    "Art. 364 Abs. 3 OR",
    "Art. 248 Abs. 2 OR",
    "Art. 934 Abs. 3 ZGB",
    "Art. 83 Abs. 2 SVG",
    "Art. 5 Abs. 1bis PrHG",
    "Art. 5 Abs. 2 PrHG",
    "Art. 400 Abs. 2 OR",
    "Art. 839 Abs. 4 ZGB",
    "Art. 839 Abs. 5 ZGB",
    "Art. 839 Abs. 6 ZGB",
    "Art. 255 Abs. 1bis StPO",
    "Art. 255 Abs. 3 StPO",
    "Art. 176 Abs. 2 ZGB",
}

NON_BOOK_TOKENS = {
    "ABS",
    "ABS.",
    "ABSATZ",
    "UND",
    "ODER",
    "AND",
    "OF",
    "THE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pre_citations/pre_near_matches from query text."
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--laws-path", type=Path, default=DEFAULT_LAWS_PATH)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--comparison-path", type=Path, default=None)
    parser.add_argument(
        "--text-columns",
        default="query",
        help="Comma-separated text columns to parse. Existing val/test used query only.",
    )
    parser.add_argument("--query-id-column", default="query_id")
    parser.add_argument(
        "--near-compat-mode",
        choices=["current-laws", "legacy-val-test"],
        default="current-laws",
        help=(
            "current-laws returns every matching article/paragraph in laws_db. "
            "legacy-val-test reproduces the already stored val/test columns."
        ),
    )
    parser.add_argument(
        "--generated-only",
        action="store_true",
        help=(
            "Write only query_id/generated_pre_citations/generated_pre_near_matches. "
            "Default preserves all input columns and appends pre_citations/pre_near_matches."
        ),
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported input extension for {path}; use .csv or .parquet")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    raise SystemExit(f"Unsupported output extension for {path}; use .csv or .parquet")


def norm_space(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def norm_book(value: object) -> str:
    return norm_space(value).strip(".,;:()[]").upper()


def split_refs(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def dedupe_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def citation_sort_key(citation: str) -> tuple[int, str]:
    match = CANON_RE.match(citation)
    if not match:
        return (10_000_000, citation)
    art = match.group("art")
    abs_no = match.group("abs") or ""
    art_match = re.match(r"(\d+)(.*)", art)
    abs_match = re.match(r"(\d+)(.*)", abs_no)
    art_num = int(art_match.group(1)) if art_match else 10_000_000
    abs_num = int(abs_match.group(1)) if abs_match else -1
    return (art_num * 10_000 + abs_num, citation)


def build_law_indexes(
    laws_path: Path,
    *,
    near_compat_mode: str,
) -> tuple[set[str], dict[str, str], dict[tuple[str, str], list[str]]]:
    laws = pd.read_parquet(laws_path, columns=["citation", "source_book"])
    citation_set = {norm_space(value) for value in laws["citation"].astype(str)}
    book_norm_to_canonical: dict[str, str] = {}
    for value in laws["source_book"].astype(str):
        canonical = norm_space(value)
        book_norm_to_canonical.setdefault(norm_book(canonical), canonical)
    article_to_citations: defaultdict[tuple[str, str], list[str]] = defaultdict(list)

    for citation in laws["citation"].astype(str):
        canonical = norm_space(citation)
        match = CANON_RE.match(canonical)
        if not match:
            continue
        book = norm_book(match.group("book"))
        art = match.group("art").lower()
        if (
            near_compat_mode == "legacy-val-test"
            and canonical in LEGACY_VAL_TEST_NEAR_EXCLUSIONS
        ):
            continue
        article_to_citations[(book, art)].append(canonical)

    for key in list(article_to_citations):
        article_to_citations[key] = sorted(
            dedupe_ordered(article_to_citations[key]),
            key=citation_sort_key,
        )

    return citation_set, book_norm_to_canonical, dict(article_to_citations)


_LAW_INDEX_CACHE: dict[
    tuple[str, str],
    tuple[set[str], dict[str, str], dict[tuple[str, str], list[str]]],
] = {}


def get_law_indexes(
    laws_path: Path = DEFAULT_LAWS_PATH,
    *,
    near_compat_mode: str = "current-laws",
) -> tuple[set[str], dict[str, str], dict[tuple[str, str], list[str]]]:
    """Load and cache law citation indexes used for pre-citation parsing."""
    key = (str(laws_path.resolve()), near_compat_mode)
    if key not in _LAW_INDEX_CACHE:
        _LAW_INDEX_CACHE[key] = build_law_indexes(
            laws_path,
            near_compat_mode=near_compat_mode,
        )
    return _LAW_INDEX_CACHE[key]


def generate_pre_citations_for_query(
    query: str,
    *,
    laws_path: Path = DEFAULT_LAWS_PATH,
    near_compat_mode: str = "current-laws",
) -> tuple[str, str]:
    """Return semicolon-separated pre_citations and pre_near_matches for one query."""
    _, book_norm_to_canonical, article_to_citations = get_law_indexes(
        laws_path,
        near_compat_mode=near_compat_mode,
    )
    pre_citations, pre_near = generate_for_text(
        norm_space(query),
        book_norm_to_canonical=book_norm_to_canonical,
        article_to_citations=article_to_citations,
    )
    return ";".join(pre_citations), ";".join(pre_near)


def normalize_book(raw_book: str, book_norm_to_canonical: dict[str, str]) -> str:
    book = norm_book(raw_book)
    if book in NON_BOOK_TOKENS:
        return ""
    if book in book_norm_to_canonical:
        return book_norm_to_canonical[book]
    if book in BOOK_FALLBACKS:
        return BOOK_FALLBACKS[book]
    return ""


def format_pre_citation(art: str, abs_no: str, lit: str, book: str) -> str:
    parts = [f"Art. {art}"]
    if abs_no:
        parts.append(f"Abs. {abs_no}")
    if lit:
        parts.append(f"lit. {lit}")
    parts.append(book)
    return " ".join(parts)


def generate_for_text(
    text: str,
    *,
    book_norm_to_canonical: dict[str, str],
    article_to_citations: dict[tuple[str, str], list[str]],
) -> tuple[list[str], list[str]]:
    pre_citations: list[str] = []
    pre_near: list[str] = []

    for match in MENTION_RE.finditer(text):
        art = (match.group("art") or "").lower()
        abs_no = (match.group("abs") or "").lower()
        lit = (match.group("lit") or "").lower()
        book = normalize_book(match.group("book"), book_norm_to_canonical)
        if not art or not book:
            continue

        pre_citations.append(format_pre_citation(art, abs_no, lit, book))
        pre_near.extend(article_to_citations.get((norm_book(book), art), []))

    return dedupe_ordered(pre_citations), dedupe_ordered(pre_near)


def generate_columns(args: argparse.Namespace) -> pd.DataFrame:
    data = read_table(args.input_path).reset_index(drop=True)
    text_columns = [column.strip() for column in args.text_columns.split(",") if column.strip()]
    missing_text_columns = [column for column in text_columns if column not in data.columns]
    if missing_text_columns:
        raise SystemExit(f"Missing text columns in {args.input_path}: {missing_text_columns}")
    if args.query_id_column not in data.columns:
        raise SystemExit(f"Missing query id column in {args.input_path}: {args.query_id_column}")

    _, book_norm_to_canonical, article_to_citations = build_law_indexes(
        args.laws_path,
        near_compat_mode=args.near_compat_mode,
    )

    rows: list[dict[str, object]] = []
    for row in data.itertuples(index=False):
        query_id = str(getattr(row, args.query_id_column))
        texts = [norm_space(getattr(row, column)) for column in text_columns]
        text = "\n".join(part for part in texts if part)
        pre_citations, pre_near = generate_for_text(
            text,
            book_norm_to_canonical=book_norm_to_canonical,
            article_to_citations=article_to_citations,
        )
        rows.append(
            {
                args.query_id_column: query_id,
                "generated_pre_citations": ";".join(pre_citations),
                "generated_pre_near_matches": ";".join(pre_near),
            }
        )

    generated = pd.DataFrame(rows)
    if args.generated_only and not args.overwrite_existing:
        return generated

    output = data.copy()
    output["pre_citations"] = generated["generated_pre_citations"]
    output["pre_near_matches"] = generated["generated_pre_near_matches"]
    return output


def compare_existing(
    input_path: Path,
    generated: pd.DataFrame,
    *,
    query_id_column: str,
) -> pd.DataFrame:
    existing = read_table(input_path).reset_index(drop=True)
    if not {"pre_citations", "pre_near_matches"}.issubset(existing.columns):
        return pd.DataFrame()

    merged = existing[[query_id_column, "pre_citations", "pre_near_matches"]].copy()
    generated_cols = generated[[query_id_column]].copy()
    if {"generated_pre_citations", "generated_pre_near_matches"}.issubset(
        generated.columns
    ):
        generated_cols["generated_pre_citations"] = generated["generated_pre_citations"]
        generated_cols["generated_pre_near_matches"] = generated[
            "generated_pre_near_matches"
        ]
    else:
        generated_cols["generated_pre_citations"] = generated["pre_citations"]
        generated_cols["generated_pre_near_matches"] = generated["pre_near_matches"]
    merged = merged.merge(generated_cols, on=query_id_column, how="left")
    merged["expected_pre_citations"] = merged["pre_citations"].map(norm_space)
    merged["expected_pre_near_matches"] = merged["pre_near_matches"].map(norm_space)
    merged["pre_citations_match"] = (
        merged["expected_pre_citations"] == merged["generated_pre_citations"].map(norm_space)
    )
    merged["pre_near_matches_match"] = (
        merged["expected_pre_near_matches"] == merged["generated_pre_near_matches"].map(norm_space)
    )
    return merged[
        [
            query_id_column,
            "expected_pre_citations",
            "generated_pre_citations",
            "pre_citations_match",
            "expected_pre_near_matches",
            "generated_pre_near_matches",
            "pre_near_matches_match",
        ]
    ]


def main() -> None:
    args = parse_args()
    generated = generate_columns(args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(generated, args.output_path)

    print(f"Input      : {args.input_path}")
    print(f"Output     : {args.output_path}")
    print(f"Rows       : {len(generated):,}")
    print(f"Text cols  : {args.text_columns}")
    print(f"Near mode  : {args.near_compat_mode}")

    comparison = compare_existing(
        args.input_path,
        generated,
        query_id_column=args.query_id_column,
    )
    if not comparison.empty:
        comparison_path = args.comparison_path or args.output_path.with_suffix(".comparison.csv")
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(comparison_path, index=False)
        citation_matches = int(comparison["pre_citations_match"].sum())
        near_matches = int(comparison["pre_near_matches_match"].sum())
        both_matches = int(
            (
                comparison["pre_citations_match"]
                & comparison["pre_near_matches_match"]
            ).sum()
        )
        print(f"Comparison : {comparison_path}")
        print(f"pre_citations matches  : {citation_matches}/{len(comparison)}")
        print(f"pre_near_matches match : {near_matches}/{len(comparison)}")
        print(f"both columns match     : {both_matches}/{len(comparison)}")


if __name__ == "__main__":
    main()
