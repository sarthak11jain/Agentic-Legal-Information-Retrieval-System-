#!/usr/bin/env python3
"""Build a lightweight court -> law references file using the full law corpus.

This rebuilds court-law references with the full law document set as the
whitelist, not train-derived book coverage. The output intentionally excludes
text, court references, and embeddings to keep the parquet small.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_COURT_PATH = BASE_DIR / "data/court_db.parquet"
DEFAULT_LAWS_PATH = BASE_DIR / "data/laws_db.parquet"
DEFAULT_OUTPUT_PATH = BASE_DIR / "context/court_law_references_fixed.parquet"

_ORDINAL = r"(?:bis|ter|quater|quinquies|sexies|septies|octies|novies)"
ART_TOKEN = rf"\d+(?:{_ORDINAL}|[a-z](?:{_ORDINAL})?)?"

LAW_MENTION_RE = re.compile(
    rf"\b(?:Art\.?|Artikel)\s*"
    rf"(?P<art>{ART_TOKEN})"
    rf"(?:\s*(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*(?P<abs>{ART_TOKEN}))?"
    rf"(?:\s*(?:lit\.?|Bst\.?|Ziff\.?)\s*[a-z](?:bis)?)?"
    rf"\s+(?P<book>[A-Za-zÄÖÜäöü]{{2,20}}"
    rf"(?:\s+[A-Za-zÄÖÜäöü]{{2,20}})?)",
    flags=re.IGNORECASE,
)
CANON_RE = re.compile(
    r"^Art\.\s*(?P<art>\S+?)(?:\s+Abs\.\s*(?P<abs>\S+?))?\s+(?P<book>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact court-law references using full law corpus whitelist."
    )
    parser.add_argument("--court-path", type=Path, default=DEFAULT_COURT_PATH)
    parser.add_argument("--laws-path", type=Path, default=DEFAULT_LAWS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--text-column", default="document")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--write-batch-size", type=int, default=50_000)
    return parser.parse_args()


def norm_book(book: str) -> str:
    return re.sub(r"\s+", " ", book.strip()).upper()


def build_law_lookup(laws_path: Path) -> dict[tuple[str, str, str], str]:
    """Build (BOOK, article, paragraph) -> normalized law reference lookup."""
    if laws_path.suffix.lower() == ".csv":
        laws = pd.read_csv(laws_path, usecols=["citation"])
    else:
        laws = pd.read_parquet(laws_path, columns=["citation"])
    lookup: dict[tuple[str, str, str], str] = {}
    article_keys: set[tuple[str, str]] = set()

    for citation in laws["citation"].astype(str):
        canonical = citation.strip()
        match = CANON_RE.match(canonical)
        if not match:
            continue

        book = norm_book(match.group("book"))
        art = match.group("art").lower()
        abs_no = (match.group("abs") or "").lower()

        # Keep article-only mentions distinct from paragraph-level citations.
        # Otherwise "Art. 30 BV" can invent "Art. 30 Abs. 1 BV".
        article_keys.add((book, art))
        lookup[(book, art, abs_no)] = canonical

    for book, art in article_keys:
        lookup.setdefault((book, art, ""), f"Art. {art} {book}")

    return lookup


def extract_law_refs(text: str, lookup: dict[tuple[str, str, str], str]) -> list[str]:
    """Extract canonical law citations from text when present in the whitelist."""
    if not text:
        return []

    seen: set[str] = set()
    out: list[str] = []

    for match in LAW_MENTION_RE.finditer(text):
        art = (match.group("art") or "").strip().lower()
        abs_no = (match.group("abs") or "").strip().lower()
        book_raw = norm_book(match.group("book") or "")
        if not art or not book_raw:
            continue

        found = lookup.get((book_raw, art, abs_no))
        if not found and abs_no:
            found = lookup.get((book_raw, art, ""))

        if not found and " " in book_raw:
            first_word = book_raw.split()[0]
            found = lookup.get((first_word, art, abs_no))
            if not found and abs_no:
                found = lookup.get((first_word, art, ""))

        if found and found not in seen:
            seen.add(found)
            out.append(found)

    return out


def flush_rows(
    writer: pq.ParquetWriter | None,
    output_path: Path,
    batch_citations: list[str],
    batch_law_references: list[str],
) -> tuple[pq.ParquetWriter, int]:
    table = pa.table(
        {
            "citation": pa.array(batch_citations, type=pa.string()),
            "law_references": pa.array(batch_law_references, type=pa.string()),
        }
    )
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
    writer.write_table(table)
    return writer, len(batch_citations)


def main() -> None:
    args = parse_args()
    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Court input : {args.court_path}")
    print(f"Law whitelist: {args.laws_path}")
    print(f"Output      : {output_path}")
    print()

    law_lookup = build_law_lookup(args.laws_path)
    print(f"Law lookup keys : {len(law_lookup):,}")

    parquet_file = pq.ParquetFile(args.court_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"Court rows      : {total_rows:,}")
    print(f"Text column     : {args.text_column}")

    court_to_laws: dict[str, set[str]] = {}
    processed = 0
    matched_mentions = 0
    t0 = time.time()

    for record_batch in parquet_file.iter_batches(
        batch_size=args.batch_size, columns=["citation", args.text_column]
    ):
        citations = record_batch.column(0).to_pylist()
        texts = record_batch.column(1).to_pylist()

        for citation, text in zip(citations, texts):
            refs = extract_law_refs(text or "", law_lookup)
            if not refs:
                continue
            citation_str = str(citation).strip()
            court_to_laws.setdefault(citation_str, set()).update(refs)
            matched_mentions += len(refs)

        processed += len(citations)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0.0
        print(
            f"  {processed:>10,} / {total_rows:,} "
            f"({100 * processed / total_rows:5.1f}%) "
            f"{rate:,.0f} rows/s "
            f"courts={len(court_to_laws):,}"
        )

    print("\nWriting compact parquet")
    writer: pq.ParquetWriter | None = None
    batch_citations: list[str] = []
    batch_law_references: list[str] = []
    written = 0

    for citation, refs in sorted(court_to_laws.items()):
        if not refs:
            continue
        batch_citations.append(citation)
        batch_law_references.append(";".join(sorted(refs)))

        if len(batch_citations) >= args.write_batch_size:
            writer, n_written = flush_rows(
                writer, output_path, batch_citations, batch_law_references
            )
            written += n_written
            print(f"  written={written:,}")
            batch_citations, batch_law_references = [], []

    if batch_citations:
        writer, n_written = flush_rows(
            writer, output_path, batch_citations, batch_law_references
        )
        written += n_written

    if writer is not None:
        writer.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output rows          : {written:,}")
    print(f"Matched law mentions : {matched_mentions:,}")
    print(f"Output saved to      : {output_path}")


if __name__ == "__main__":
    main()
