#!/usr/bin/env python3
"""Update court-document -> court-citation references in court_db.

This parses only court citations from `data/court_db.parquet`; it does not parse
law citations. Only the `court_references` column is replaced; all other columns
and row order are preserved.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parents[4]
DEFAULT_COURT_PATH = BASE_DIR / "data/court_db.parquet"

BGE_MENTION_RE = re.compile(
    r"BGE\s+(?P<vol>\d{1,3})\s+"
    r"(?P<section>[IVX]+[a-z]?)\s+"
    r"(?P<page>\d+)"
    r"(?:\s+(?:E\.|cons\.?|Erw\.?)\s*"
    r"(?P<consideration>"
    r"(?:[IVX]+\.)?"
    r"(?:\d+[a-z]?)"
    r"(?:\.\d+[a-z]?)*"
    r"(?:(?:/[a-z]+)|(?:-\d+[a-z]?))?"
    r"))?",
)
DOCKET_MENTION_RE = re.compile(
    r"(?<!\w)"
    r"(?P<docket>\d[A-Z]+[_.]\d+/\d{2,4})"
    r"(?:\s+(?:E\.?|cons\.?|Erw\.?)\s*"
    r"(?P<consideration>\d+[a-z]?(?:\.\d+[a-z]?)*))?",
)
_BGE_BASE_RE = re.compile(r"^(BGE\s+\d+\s+[IVX]+[a-z]?\s+\d+)")
_DOCKET_BASE_RE = re.compile(r"^(\d+[A-Z]+[_.]\d+/\d{2,4})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update court_references in court_db.")
    parser.add_argument("--court-path", type=Path, default=DEFAULT_COURT_PATH)
    parser.add_argument("--text-column", default="document")
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def norm_docket(docket: str) -> str:
    return docket.replace(".", "_", 1) if "." in docket and "_" not in docket else docket


def base_case(citation: str) -> str | None:
    match = _BGE_BASE_RE.match(citation)
    if match:
        return match.group(1)
    match = _DOCKET_BASE_RE.match(citation)
    if match:
        return norm_docket(match.group(1))
    return None


def is_self_cite(doc_citation: str, ref_citation: str) -> bool:
    doc = doc_citation.strip()
    ref = ref_citation.strip()
    if ref == doc:
        return True
    doc_base = base_case(doc)
    return bool(doc_base and ref == doc_base)


def extract_court_refs(text: str, doc_citation: str, valid_courts: set[str]) -> list[str]:
    if not text:
        return []

    seen: set[str] = set()
    refs: list[str] = []

    def add_ref(citation: str) -> None:
        if citation in seen:
            return
        if is_self_cite(doc_citation, citation):
            return
        if citation in valid_courts:
            seen.add(citation)
            refs.append(citation)

    for match in BGE_MENTION_RE.finditer(text):
        vol = match.group("vol")
        section = match.group("section")
        page = match.group("page")
        consideration = match.group("consideration")
        if consideration:
            add_ref(f"BGE {vol} {section} {page} E. {consideration}")
        else:
            add_ref(f"BGE {vol} {section} {page}")

    for match in DOCKET_MENTION_RE.finditer(text):
        docket = norm_docket(match.group("docket"))
        consideration = match.group("consideration")
        if consideration:
            add_ref(f"{docket} E. {consideration}")
        add_ref(docket)

    return refs


def main() -> None:
    args = parse_args()
    tmp_path = args.court_path.with_suffix(args.court_path.suffix + ".tmp")

    parquet_file = pq.ParquetFile(args.court_path)
    columns = set(parquet_file.schema_arrow.names)
    if "citation" not in columns:
        raise SystemExit(f"`citation` column missing from {args.court_path}")
    if args.text_column not in columns:
        raise SystemExit(f"`{args.text_column}` column missing from {args.court_path}")
    if "court_references" not in columns:
        raise SystemExit(f"`court_references` column missing from {args.court_path}")

    print(f"Court input : {args.court_path}")
    print(f"Text column : {args.text_column}")
    print("Updating    : court_references")
    print()

    valid_courts: set[str] = set()
    for record_batch in parquet_file.iter_batches(
        batch_size=args.batch_size,
        columns=["citation"],
    ):
        valid_courts.update(str(value).strip() for value in record_batch.column(0).to_pylist())
    valid_courts.discard("")
    print(f"Valid court citations: {len(valid_courts):,}")

    total_rows = parquet_file.metadata.num_rows
    processed = 0
    matched_mentions = 0
    writer: pq.ParquetWriter | None = None
    t0 = time.time()

    for record_batch in parquet_file.iter_batches(
        batch_size=args.batch_size,
    ):
        table = record_batch.to_table()
        names = table.schema.names
        citation_idx = names.index("citation")
        text_idx = names.index(args.text_column)
        court_refs_idx = names.index("court_references")
        citations = table.column(citation_idx).to_pylist()
        texts = table.column(text_idx).to_pylist()

        updated_court_references: list[str] = []
        for citation, text in zip(citations, texts):
            citation_str = str(citation).strip()
            refs = extract_court_refs(str(text or ""), citation_str, valid_courts)
            updated_court_references.append(";".join(refs))
            matched_mentions += len(refs)

        updated_table = table.set_column(
            court_refs_idx,
            "court_references",
            updated_court_references,
        )
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, updated_table.schema, compression="snappy")
        writer.write_table(updated_table)

        processed += len(citations)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0.0
        print(
            f"  {processed:>10,} / {total_rows:,} "
            f"({100 * processed / total_rows:5.1f}%) "
            f"{rate:,.0f} rows/s "
            f"updated={processed:,}"
        )

    if writer is not None:
        writer.close()
        tmp_path.replace(args.court_path)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Rows updated             : {processed:,}")
    print(f"Matched court references : {matched_mentions:,}")
    print(f"Updated file             : {args.court_path}")


if __name__ == "__main__":
    main()
