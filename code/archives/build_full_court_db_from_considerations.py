#!/usr/bin/env python3
"""Rebuild a full court_db-style parquet from court_considerations.

Input:
  - data/court_considerations.csv
  - data/laws_db.parquet

Output:
  - citation
  - document
  - law_references
  - court_references
  - embedding

The embedding column is intentionally empty for now.
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_COURT_CONSIDERATIONS_PATH = BASE_DIR / "data/court_considerations.csv"
DEFAULT_LAWS_PATH = BASE_DIR / "data/laws_db.parquet"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data/court_db_new.parquet"


ORDINAL = r"(?:bis|ter|quater|quinquies|sexies|septies|octies|novies)"
ART_TOKEN = rf"\d+(?:{ORDINAL}|[a-z](?:{ORDINAL})?)?"
ARTICLE_PREFIX = r"(?:aArt\.?|Art\.?|Artikel|art\.?)"
BOOK_TOKEN_PATTERN = (
    rf"(?:[A-ZÄÖÜ][A-Za-zÄÖÜäöü0-9+.\-]{{1,20}}"
    rf"(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü0-9+.\-]{{1,20}})?)"
    rf"|(?:\d{{2,4}}(?:\.\d{{1,4}})*)"
)
QUALIFIER_PATTERN = (
    rf"(?:\s*(?:lit\.?|Bst\.?|Ziff\.?|Ziffer)\s*[a-z0-9](?:bis)?"
    rf"(?:\s*(?:,|und|oder|et|-|–)\s*[a-z0-9](?:bis)?)*)*"
)

CANON_RE = re.compile(
    r"^Art\.\s*(?P<art>\S+?)(?:\s+Abs\.\s*(?P<abs>\S+?))?\s+(?P<book>.+)$"
)

EXPLICIT_CODE_RE = re.compile(
    rf"\b{ARTICLE_PREFIX}\s*"
    rf"(?P<art>{ART_TOKEN})"
    rf"(?:\s*(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*"
    rf"(?P<abs>{ART_TOKEN}))?"
    rf"{QUALIFIER_PATTERN}"
    rf"\s+(?P<book>{BOOK_TOKEN_PATTERN})",
)

EXPLICIT_ABS_CODE_RE = re.compile(
    rf"\b{ARTICLE_PREFIX}\s*"
    rf"(?P<art>{ART_TOKEN})"
    rf"\s*(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*"
    rf"(?P<abs_part>"
    rf"{ART_TOKEN}"
    rf"(?:\s*(?:,|und|oder|et|-|–|\bbis\b)\s*{ART_TOKEN})*"
    rf")"
    rf"{QUALIFIER_PATTERN}"
    rf"\s+(?P<book>{BOOK_TOKEN_PATTERN})",
)

EXPLICIT_REPEATED_ABS_CODE_RE = re.compile(
    rf"\b{ARTICLE_PREFIX}\s*"
    rf"(?P<art>{ART_TOKEN})"
    rf"(?P<abs_segments>"
    rf"(?:\s*(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*{ART_TOKEN}"
    rf"{QUALIFIER_PATTERN}"
    rf"\s*(?:,|und|oder|et)\s*)"
    rf"(?:\s*(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*{ART_TOKEN}"
    rf"{QUALIFIER_PATTERN}"
    rf"(?:\s*(?:,|und|oder|et)\s*)?)+"
    rf")"
    rf"\s+(?P<book>{BOOK_TOKEN_PATTERN})",
)

SHARED_TRAILING_BOOK_RE = re.compile(
    rf"\b{ARTICLE_PREFIX}\s*"
    rf"(?P<art>{ART_TOKEN})"
    rf"(?:\s*(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*(?P<abs>{ART_TOKEN}))?"
    rf"{QUALIFIER_PATTERN}"
    rf"\s*(?:in Verbindung mit|i\.V\.m\.|iVm)\s+"
    rf"{ARTICLE_PREFIX}\s*"
    rf"{ART_TOKEN}"
    rf"(?:\s*(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*{ART_TOKEN})?"
    rf"{QUALIFIER_PATTERN}"
    rf"\s+(?P<book>{BOOK_TOKEN_PATTERN})",
)

NAMED_LAW_RE = re.compile(
    rf"\b{ARTICLE_PREFIX}\s*"
    rf"(?P<art>{ART_TOKEN})"
    rf"(?:\s*(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*(?P<abs>{ART_TOKEN}))?"
    rf"(?:\s*(?:lit\.?|Bst\.?|Ziff\.?)\s*[a-z](?:bis)?)?"
    rf"\s+(?P<prefix>des|der|dem)\s+"
    rf"(?P<law_name>[A-ZÄÖÜa-zäöü][A-ZÄÖÜa-zäöü0-9 \-]+?)"
    rf"(?=\s+(?:und|oder|sowie)\s+(?:Art\.?|Artikel)\b|[,.;:()\n]|$)",
)

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
_DOCKET_BASE_RE = re.compile(r"^(\d[A-Z]+[_.]\d+/\d{2,4})")

NOISE_BOOK_TOKENS = {
    "ABS",
    "ABS.",
    "ABSATZ",
    "BUCHSTABE",
    "BUCHSTABEN",
    "BST",
    "BST.",
    "ZIFF",
    "ZIFF.",
    "ZIFFER",
    "UND",
    "ODER",
    "SATZ",
}

BOOK_ALIASES = {
    "CST": "BV",
    "CST.": "BV",
    "CC": "ZGB",
    "CO": "OR",
    "CP": "StGB",
    "CPP": "StPO",
    "CPC": "ZPO",
    "LTF": "BGG",
    "LIA": "VStG",
    "LEI": "AIG",
    "LETR": "AIG",
}

NAME_ALIASES = {
    "OBLIGATIONENRECHTS": "OR",
    "ZIVILGESETZBUCHES": "ZGB",
    "STRAFGESETZBUCHES": "StGB",
    "BUNDESGERICHTSGESETZES": "BGG",
    "AUSLÄNDER- UND INTEGRATIONSGESETZES": "AIG",
    "INVALIDENVERSICHERUNGSGESETZES": "IVG",
    "ARBEITSLOSENVERSICHERUNGSGESETZES": "AVIG",
    "MEHRWERTSTEUERGESETZES": "MWSTG",
    "BETREIBUNGS- UND KONKURSGESETZES": "SchKG",
    "KOLLEKTIVANLAGENGESETZES": "KAG",
    "FINANZMARKTINFRASTRUKTURGESETZES": "FinfraG",
    "ASYLGESETZES": "AsylG",
}


def norm_space(value: object) -> str:
    return re.sub(r"\s+", " ", "" if pd.isna(value) else str(value)).strip()


def norm_book(value: object) -> str:
    book = norm_space(value).strip(".,;:()[]").upper()
    return BOOK_ALIASES.get(book, book)


def normalize_candidate_book(raw_book: object, book_norms: set[str]) -> str:
    candidate = norm_book(raw_book)
    if candidate in book_norms:
        return candidate

    first_token = candidate.split()[0] if candidate else ""
    if first_token in BOOK_ALIASES:
        first_token = norm_book(first_token)
    if first_token in book_norms:
        return first_token

    without_footnote = re.sub(r"(?<=[A-ZÄÖÜa-zäöü])\d{1,3}$", "", candidate)
    if without_footnote in book_norms:
        return without_footnote
    return candidate


def parse_canonical_law(citation: object) -> tuple[str, str, str] | None:
    match = CANON_RE.match(norm_space(citation))
    if not match:
        return None
    book = norm_book(match.group("book"))
    art = match.group("art").lower()
    abs_no = (match.group("abs") or "").lower()
    return book, art, abs_no


def split_law_title(document: object) -> str:
    text = norm_space(document)
    match = re.search(r"\bGesetz:\s*(.*?)(?=\s+Regelungsbereich:|\s+Normtext:|$)", text)
    return match.group(1).strip() if match else ""


def normalize_law_name(raw_name: str) -> str:
    cleaned = norm_space(raw_name)
    cleaned = re.sub(r"(?<=[A-ZÄÖÜa-zäöü])\d{1,3}\b", "", cleaned)
    cleaned = re.sub(r"\bvom\s+\d{1,2}\..*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(vom|von|über)\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(".,;:()[]").upper()


def genitive_variants(name: str) -> set[str]:
    variants = {name}
    replacements = {
        "GESETZ": "GESETZES",
        "BUCH": "BUCHES",
        "RECHT": "RECHTS",
        "ÜBEREINKOMMEN": "ÜBEREINKOMMENS",
        "ABKOMMEN": "ABKOMMENS",
        "REGLEMENT": "REGLEMENTS",
    }
    for ending, genitive in replacements.items():
        if name.endswith(ending):
            variants.add(name[: -len(ending)] + genitive)
    return variants


def add_alias(alias_map: dict[str, str], raw_name: str, book: str) -> None:
    normalized = normalize_law_name(raw_name)
    if not normalized or normalized in {"GESETZ", "VERORDNUNG", "BUNDESGESETZ"}:
        return
    if len(normalized) < 5:
        return
    for variant in genitive_variants(normalized):
        alias_map.setdefault(variant, norm_book(book))


def expand_abs_part(abs_part: str) -> list[str]:
    """Expand explicit paragraph lists like `1 und 2` or `1-3`.

    This only expands paragraphs that are written in the text. It does not infer
    a default paragraph from an article-only citation.
    """
    text = norm_space(abs_part).lower()
    expanded: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        if value and value not in seen:
            expanded.append(value)
            seen.add(value)

    for match in re.finditer(r"(?<![a-z])(?P<start>\d+)\s*(?:-|–|\bbis\b)\s*(?P<end>\d+)(?![a-z])", text):
        start = int(match.group("start"))
        end = int(match.group("end"))
        step = 1 if end >= start else -1
        if abs(end - start) <= 20:
            for value in range(start, end + step, step):
                add(str(value))

    for token in re.findall(ART_TOKEN, text):
        add(token)

    return expanded


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


class LawReferenceExtractor:
    def __init__(self, laws_path: Path) -> None:
        laws = pd.read_parquet(laws_path, columns=["citation", "document", "source_book"])
        self.book_norms: set[str] = set()
        self.exact_lookup: dict[tuple[str, str, str], str] = {}
        self.article_lookup: dict[tuple[str, str], str] = {}
        self.name_aliases: dict[str, str] = {key: norm_book(value) for key, value in NAME_ALIASES.items()}

        seen_books: set[str] = set()
        for row in laws.itertuples(index=False):
            canonical = norm_space(row.citation)
            parsed = parse_canonical_law(canonical)
            if not parsed:
                continue
            book, art, abs_no = parsed
            self.book_norms.add(book)
            self.exact_lookup[(book, art, abs_no)] = canonical
            if not abs_no:
                self.article_lookup[(book, art)] = canonical
            else:
                self.article_lookup.setdefault((book, art), f"Art. {art} {book}")

            source_book = norm_space(row.source_book)
            source_book_norm = norm_book(source_book)
            if source_book_norm and source_book_norm not in seen_books:
                seen_books.add(source_book_norm)
                title = split_law_title(row.document)
                main_title = title.split(" - ", 1)[0]
                for paren in re.findall(r"\(([^()]*)\)", main_title):
                    for piece in [part.strip() for part in paren.split(",")]:
                        if norm_book(piece) != source_book_norm:
                            add_alias(self.name_aliases, piece, source_book)
                title_without_date = re.sub(
                    r"\bvom\s+\d{1,2}\..*$",
                    "",
                    main_title,
                    flags=re.IGNORECASE,
                )
                short_match = re.search(
                    r"\b(?:über|betreffend)\s+(?:das|die|den|der)?\s*(.+)$",
                    title_without_date,
                    flags=re.IGNORECASE,
                )
                if short_match:
                    add_alias(self.name_aliases, short_match.group(1), source_book)

    def extract(self, text: object, counters: Counter[str]) -> list[str]:
        text_str = norm_space(text)
        if not text_str:
            counters["empty_text"] += 1
            return []

        seen: set[str] = set()
        refs: list[str] = []

        def add_ref(ref: str | None, counter: str) -> None:
            if not ref or ref in seen:
                return
            seen.add(ref)
            refs.append(ref)
            counters[counter] += 1

        for match in SHARED_TRAILING_BOOK_RE.finditer(text_str):
            art = (match.group("art") or "").lower()
            abs_no = (match.group("abs") or "").lower()
            book = normalize_candidate_book(match.group("book"), self.book_norms)
            first_token = book.split()[0] if book else ""
            if book in NOISE_BOOK_TOKENS or first_token in NOISE_BOOK_TOKENS:
                counters["shared_book_noise_book"] += 1
                continue
            ref = self.resolve(book, art, abs_no)
            if not ref and " " in book:
                ref = self.resolve(book.split()[0], art, abs_no)
            add_ref(ref, "shared_book_hit")
            if not ref:
                counters["shared_book_miss"] += 1

        for match in EXPLICIT_REPEATED_ABS_CODE_RE.finditer(text_str):
            art = (match.group("art") or "").lower()
            abs_values = re.findall(
                rf"(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*({ART_TOKEN})",
                match.group("abs_segments") or "",
            )
            book = normalize_candidate_book(match.group("book"), self.book_norms)
            first_token = book.split()[0] if book else ""
            if book in NOISE_BOOK_TOKENS or first_token in NOISE_BOOK_TOKENS:
                counters["explicit_repeated_abs_noise_book"] += 1
                continue
            for abs_no in abs_values:
                ref = self.resolve(book, art, abs_no.lower())
                if not ref and " " in book:
                    ref = self.resolve(book.split()[0], art, abs_no.lower())
                add_ref(ref, "explicit_repeated_abs_hit")
                if not ref:
                    counters["explicit_repeated_abs_miss"] += 1

        for match in EXPLICIT_ABS_CODE_RE.finditer(text_str):
            art = (match.group("art") or "").lower()
            abs_values = expand_abs_part(match.group("abs_part") or "")
            book = normalize_candidate_book(match.group("book"), self.book_norms)
            first_token = book.split()[0] if book else ""
            if book in NOISE_BOOK_TOKENS or first_token in NOISE_BOOK_TOKENS:
                counters["explicit_abs_noise_book"] += 1
                continue

            for abs_no in abs_values:
                ref = self.resolve(book, art, abs_no)
                if not ref and " " in book:
                    ref = self.resolve(book.split()[0], art, abs_no)
                add_ref(ref, "explicit_abs_hit")
                if not ref:
                    counters["explicit_abs_miss"] += 1

        for match in EXPLICIT_CODE_RE.finditer(text_str):
            art = (match.group("art") or "").lower()
            abs_no = (match.group("abs") or "").lower()
            book = normalize_candidate_book(match.group("book"), self.book_norms)
            first_token = book.split()[0] if book else ""
            if book in NOISE_BOOK_TOKENS or first_token in NOISE_BOOK_TOKENS:
                counters["explicit_noise_book"] += 1
                continue

            ref = self.resolve(book, art, abs_no)
            if not ref and " " in book:
                ref = self.resolve(book.split()[0], art, abs_no)
            add_ref(ref, "explicit_hit")
            if not ref:
                counters["explicit_miss"] += 1

        for match in NAMED_LAW_RE.finditer(text_str):
            book = self.resolve_law_name(match.group("law_name") or "")
            if not book:
                counters["named_miss_book"] += 1
                continue
            art = (match.group("art") or "").lower()
            abs_no = (match.group("abs") or "").lower()
            ref = self.resolve(book, art, abs_no)
            add_ref(ref, "named_hit")
            if not ref:
                counters["named_miss_ref"] += 1

        if not refs:
            counters["no_law_reference"] += 1
        return refs

    def resolve(self, book: str, art: str, abs_no: str) -> str | None:
        book = norm_book(book)
        if not book or not art:
            return None
        if abs_no:
            ref = self.exact_lookup.get((book, art, abs_no))
            if ref:
                return ref
        return self.article_lookup.get((book, art))

    def resolve_law_name(self, raw_name: str) -> str | None:
        cleaned = normalize_law_name(raw_name)
        if cleaned in self.name_aliases:
            return self.name_aliases[cleaned]
        for variant in genitive_variants(cleaned):
            if variant in self.name_aliases:
                return self.name_aliases[variant]
        for alias in sorted(self.name_aliases, key=len, reverse=True):
            if len(alias) >= 6 and cleaned.startswith(alias + " "):
                return self.name_aliases[alias]
        return None

    @staticmethod
    def _drop_redundant_article_refs(refs: list[str]) -> list[str]:
        specific_article_keys: set[tuple[str, str]] = set()
        parsed_by_ref: dict[str, tuple[str, str, str] | None] = {}
        for ref in refs:
            parsed = parse_canonical_law(ref)
            parsed_by_ref[ref] = parsed
            if parsed is None:
                continue
            book, art, abs_no = parsed
            if abs_no:
                specific_article_keys.add((book, art))

        if not specific_article_keys:
            return refs

        filtered: list[str] = []
        for ref in refs:
            parsed = parsed_by_ref.get(ref)
            if parsed is not None:
                book, art, abs_no = parsed
                if not abs_no and (book, art) in specific_article_keys:
                    continue
            filtered.append(ref)
        return filtered


def make_output_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("citation", pa.large_string()),
            pa.field("document", pa.large_string()),
            pa.field("law_references", pa.large_string()),
            pa.field("court_references", pa.large_string()),
            pa.field("embedding", pa.list_(pa.float32())),
        ]
    )


def write_rows(writer: pq.ParquetWriter | None, output_path: Path, rows: list[dict[str, object]]) -> pq.ParquetWriter:
    schema = make_output_schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    if writer is None:
        writer = pq.ParquetWriter(output_path, schema, compression="snappy")
    writer.write_table(table)
    return writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full court_db-style parquet from court_considerations and laws_db."
    )
    parser.add_argument("--court-considerations-path", type=Path, default=DEFAULT_COURT_CONSIDERATIONS_PATH)
    parser.add_argument("--laws-path", type=Path, default=DEFAULT_LAWS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--write-batch-size", type=int, default=50_000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def collect_detected_citations(
    *,
    court_considerations_path: Path,
    extractor: LawReferenceExtractor,
    chunk_size: int,
    max_rows: int | None,
) -> tuple[set[str], Counter[str], int]:
    print("Pass 1: detecting court rows with valid laws")
    counters: Counter[str] = Counter()
    detected: set[str] = set()
    processed = 0
    t0 = time.time()

    for chunk in pd.read_csv(court_considerations_path, chunksize=chunk_size):
        if max_rows is not None:
            remaining = max_rows - processed
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        for row in chunk.itertuples(index=False):
            citation = norm_space(row.citation)
            if citation and extractor.extract(row.text, counters):
                detected.add(citation)

        processed += len(chunk)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0.0
        print(f"  {processed:>10,} rows | detected={len(detected):>8,} | {rate:,.0f} rows/s")

    print(f"Detected court rows: {len(detected):,}")
    print(f"Detection counters : {dict(counters.most_common())}")
    print()
    return detected, counters, processed


def write_rebuilt_court_db(
    *,
    args: argparse.Namespace,
    extractor: LawReferenceExtractor,
    detected_citations: set[str],
    valid_courts: set[str],
) -> tuple[int, Counter[str], Counter[str]]:
    print("Pass 2: writing rebuilt court_db parquet")
    if args.output_path.exists():
        args.output_path.unlink()

    writer: pq.ParquetWriter | None = None
    rows_buffer: list[dict[str, object]] = []
    law_counters: Counter[str] = Counter()
    court_counters: Counter[str] = Counter()
    processed = 0
    written = 0
    written_citations: set[str] = set()
    t0 = time.time()

    for chunk in pd.read_csv(args.court_considerations_path, chunksize=args.chunk_size):
        if args.max_rows is not None:
            remaining = args.max_rows - processed
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

        for row in chunk.itertuples(index=False):
            citation = norm_space(row.citation)
            if citation not in detected_citations:
                continue
            if citation in written_citations:
                court_counters["duplicate_source_citation_skipped"] += 1
                continue

            document = norm_space(row.text)
            law_refs = extractor.extract(document, law_counters)
            if not law_refs:
                continue

            court_refs = extract_court_refs(document, citation, valid_courts)
            court_counters["rows_with_court_refs" if court_refs else "rows_without_court_refs"] += 1
            court_counters["court_ref_mentions"] += len(court_refs)
            rows_buffer.append(
                {
                    "citation": citation,
                    "document": document,
                    "law_references": ";".join(law_refs),
                    "court_references": ";".join(court_refs),
                    "embedding": [],
                }
            )
            written_citations.add(citation)

            if len(rows_buffer) >= args.write_batch_size:
                writer = write_rows(writer, args.output_path, rows_buffer)
                written += len(rows_buffer)
                rows_buffer = []

        processed += len(chunk)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0.0
        print(f"  {processed:>10,} rows | written={written + len(rows_buffer):>8,} | {rate:,.0f} rows/s")

    if rows_buffer:
        writer = write_rows(writer, args.output_path, rows_buffer)
        written += len(rows_buffer)
    if writer is not None:
        writer.close()
    elif not args.output_path.exists():
        empty = pa.Table.from_pylist([], schema=make_output_schema())
        pq.write_table(empty, args.output_path, compression="snappy")

    print(f"Output rows written: {written:,}")
    print(f"Law parse counters : {dict(law_counters.most_common())}")
    print(f"Court ref counters : {dict(court_counters.most_common())}")
    print()
    return written, law_counters, court_counters


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Court considerations : {args.court_considerations_path}")
    print(f"Laws DB              : {args.laws_path}")
    print(f"Output               : {args.output_path}")
    print(f"Max rows             : {args.max_rows if args.max_rows is not None else 'all'}")
    print()

    t0 = time.time()
    extractor = LawReferenceExtractor(args.laws_path)
    print(f"Law books         : {len(extractor.book_norms):,}")
    print(f"Law exact refs    : {len(extractor.exact_lookup):,}")
    print(f"Law article refs  : {len(extractor.article_lookup):,}")
    print(f"Named law aliases : {len(extractor.name_aliases):,}")
    print()

    detected_citations, _, _ = collect_detected_citations(
        court_considerations_path=args.court_considerations_path,
        extractor=extractor,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
    )
    valid_courts = set(detected_citations)
    print(f"Valid court-reference targets: {len(valid_courts):,}")
    print()

    write_rebuilt_court_db(
        args=args,
        extractor=extractor,
        detected_citations=detected_citations,
        valid_courts=valid_courts,
    )

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
