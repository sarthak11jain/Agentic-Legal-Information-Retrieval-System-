#!/usr/bin/env python3
"""Audit law-body citation parsing before building a law-to-law graph.

The goal is diagnostics, not final graph construction. The parser separates:
  1. explicit book-code references: "Art. 8 ATSG";
  2. internal same-book references: "Artikel 11 Absatz 4";
  3. named-law references: "Artikel 270 des Obligationenrechts";
  4. unresolved or likely foreign/non-corpus references.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LAWS_PATH = BASE_DIR / "data/laws_db.parquet"
DEFAULT_OUTPUT_DIR = BASE_DIR / "context/law_reference_parser_audit"

ORDINAL = r"(?:bis|ter|quater|quinquies|sexies|septies|octies|novies)"
ART_TOKEN = rf"\d+(?:{ORDINAL}|[a-z](?:{ORDINAL})?)?"
CANON_RE = re.compile(
    r"^Art\.\s*(?P<art>\S+?)(?:\s+Abs\.\s*(?P<abs>\S+?))?\s+(?P<book>.+)$"
)

# Explicit code references: Art. 8 ATSG, Artikel 221 Absatz 1 StPO.
EXPLICIT_CODE_RE = re.compile(
    rf"\b(?:Art\.?|Artikel)\s*"
    rf"(?P<art>{ART_TOKEN})"
    rf"(?:\s*(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*(?P<abs>{ART_TOKEN}))?"
    rf"(?:\s*(?:lit\.?|Bst\.?|Ziff\.?)\s*[a-z](?:bis)?)?"
    rf"\s+(?P<book>[A-ZÄÖÜ][A-Za-zÄÖÜäöü0-9+.\-]{{1,20}}"
    rf"(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü0-9+.\-]{{1,20}})?)",
)

# Any article mention, including internal references without code.
ARTICLE_RE = re.compile(
    rf"\b(?:Art\.?|Artikel)\s*"
    rf"(?P<art>{ART_TOKEN})"
    rf"(?:\s*(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*(?P<abs>{ART_TOKEN}))?"
    rf"(?:\s*(?:lit\.?|Bst\.?|Ziff\.?)\s*[a-z](?:bis)?)?",
    flags=re.IGNORECASE,
)

NOISE_AFTER_ARTICLE_RE = re.compile(
    r"^\s*(?:"
    r"Buchstabe|Buchstaben|Absatz|Absätze|Ziffer|Ziff\.?|lit\.?|Bst\.?|"
    r"und|oder|bis|erfüllen|sinngemäss|bleibt|für|des|der|dieser|dieses"
    r")\b",
    flags=re.IGNORECASE,
)

NAMED_LAW_RE = re.compile(
    rf"\b(?:Art\.?|Artikel)\s*"
    rf"(?P<art>{ART_TOKEN})"
    rf"(?:\s*(?:Abs\.?|Absatz|al\.?|cpv\.?)\s*(?P<abs>{ART_TOKEN}))?"
    rf"(?:\s*(?:lit\.?|Bst\.?|Ziff\.?)\s*[a-z](?:bis)?)?"
    rf"\s+(?P<prefix>des|der|dem)\s+"
    rf"(?P<law_name>[A-ZÄÖÜa-zäöü][A-ZÄÖÜa-zäöü0-9 \-]+?)"
    rf"(?=\s+(?:und|oder|sowie)\s+(?:Art\.?|Artikel)\b|[,.;:()\n]|$)",
)


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
    parser = argparse.ArgumentParser(description="Audit law-body reference parsing.")
    parser.add_argument("--laws-path", type=Path, default=DEFAULT_LAWS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-limit", type=int, default=80)
    return parser.parse_args()


def norm_space(value: object) -> str:
    return re.sub(r"\s+", " ", "" if pd.isna(value) else str(value)).strip()


def norm_book(book: object) -> str:
    text = norm_space(book).strip(".,;:()[]")
    return text.upper()


def normalize_candidate_book(book: object, book_norms: set[str]) -> str:
    candidate = norm_book(book)
    if candidate in book_norms:
        return candidate
    without_footnote = re.sub(r"(?<=[A-ZÄÖÜa-zäöü])\d{1,3}$", "", candidate)
    if without_footnote in book_norms:
        return without_footnote
    first_token = candidate.split()[0] if candidate else ""
    if first_token in book_norms:
        return first_token
    first_without_footnote = re.sub(r"(?<=[A-ZÄÖÜa-zäöü])\d{1,3}$", "", first_token)
    if first_without_footnote in book_norms:
        return first_without_footnote
    return candidate


def parse_citation(citation: object) -> tuple[str, str, str, str] | None:
    match = CANON_RE.match(norm_space(citation))
    if not match:
        return None
    art = match.group("art").lower()
    abs_no = (match.group("abs") or "").lower()
    book = norm_space(match.group("book"))
    return art, abs_no, book, norm_book(book)


def build_lookup(
    laws: pd.DataFrame,
) -> tuple[dict[tuple[str, str, str], str], dict[str, tuple[str, str, str, str]], set[str]]:
    lookup: dict[tuple[str, str, str], str] = {}
    parsed_by_citation: dict[str, tuple[str, str, str, str]] = {}
    book_norms: set[str] = set()
    for citation in laws["citation"].astype(str):
        parsed = parse_citation(citation)
        if not parsed:
            continue
        art, abs_no, book, book_norm = parsed
        book_norms.add(book_norm)
        canonical = norm_space(citation)
        parsed_by_citation[canonical] = parsed
        lookup[(book_norm, art, abs_no)] = canonical
    return lookup, parsed_by_citation, book_norms


def resolve(lookup: dict[tuple[str, str, str], str], book_norm: str, art: str, abs_no: str) -> str | None:
    found = lookup.get((book_norm, art, abs_no))
    if not found and abs_no:
        found = lookup.get((book_norm, art, ""))
    return found


def has_trailing_foreign_book_list(tail: str, book_norms: set[str], source_book_norm: str) -> bool:
    """Catch mentions like "Artikel 19, 19a und 20 BVG" before internal parsing."""
    normalized_tail = norm_space(tail).upper()
    if not re.match(r"^(?:[,;/]|\s|UND\b|ODER\b|\d)", normalized_tail):
        return False
    for book in sorted(book_norms, key=len, reverse=True):
        if book == source_book_norm or not re.search(r"[A-ZÄÖÜ]", book):
            continue
        pattern = rf"(?<![\w.+-]){re.escape(book)}\d{{0,3}}(?![\w.+-])"
        if re.search(pattern, normalized_tail):
            return True
    return False


def split_document(document: object) -> tuple[str, str]:
    text = norm_space(document)
    title_match = re.search(r"\bGesetz:\s*(.*?)(?=\s+Regelungsbereich:|\s+Normtext:|$)", text)
    normtext_match = re.search(r"\bNormtext:\s*(.*)$", text)
    title = title_match.group(1).strip() if title_match else ""
    body = normtext_match.group(1).strip() if normtext_match else text
    return title, body


def load_laws(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        laws = pd.read_parquet(path, columns=["citation", "document"])
        title_text = laws["document"].map(split_document)
        laws["title"] = title_text.map(lambda value: value[0])
        laws["text"] = title_text.map(lambda value: value[1])
        return laws[["citation", "text", "title"]]
    return pd.read_csv(path, usecols=["citation", "text", "title"])


def normalize_law_name(raw_name: str) -> str:
    cleaned = norm_space(raw_name)
    cleaned = re.sub(r"(?<=[A-ZÄÖÜa-zäöü])\d{1,3}\b", "", cleaned)
    cleaned = re.sub(r"\bvom\s+\d{1,2}\..*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(vom|vom|von|über)\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = norm_space(cleaned).strip(".,;:()[]")
    return cleaned.upper()


def genitive_variants(name: str) -> set[str]:
    variants = {name}
    replacements = {
        "GESETZ": "GESETZES",
        "BUCH": "BUCHES",
        "RECHT": "RECHTS",
        "ÜBEREINKOMMEN": "ÜBEREINKOMMENS",
        "ABKOMMEN": "ABKOMMENS",
        "BESCHLUSS": "BESCHLUSSES",
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
        alias_map.setdefault(variant, book)


def build_law_name_aliases(laws: pd.DataFrame, parsed_by_citation: dict[str, tuple[str, str, str, str]]) -> dict[str, str]:
    aliases = {key: norm_book(value) for key, value in NAME_ALIASES.items()}
    seen_books: set[str] = set()
    for row in laws.itertuples(index=False):
        citation = norm_space(row.citation)
        parsed = parsed_by_citation.get(citation)
        if not parsed:
            continue
        _, _, book, book_norm = parsed
        if book_norm in seen_books:
            continue
        seen_books.add(book_norm)
        title = norm_space(getattr(row, "title", ""))
        main_title = title.split(" - ", 1)[0]

        for paren in re.findall(r"\(([^()]*)\)", main_title):
            pieces = [piece.strip() for piece in paren.split(",")]
            for piece in pieces:
                if norm_book(piece) != book_norm:
                    add_alias(aliases, piece, book)

        title_without_date = re.sub(r"\bvom\s+\d{1,2}\..*$", "", main_title, flags=re.IGNORECASE)
        short_match = re.search(
            r"\b(?:über|betreffend)\s+(?:das|die|den|der)?\s*(.+)$",
            title_without_date,
            flags=re.IGNORECASE,
        )
        if short_match:
            add_alias(aliases, short_match.group(1), book)
    return aliases


def resolve_book_name(raw_name: str, aliases: dict[str, str]) -> str | None:
    cleaned = normalize_law_name(raw_name)
    if cleaned in aliases:
        return aliases[cleaned]
    for variant in genitive_variants(cleaned):
        if variant in aliases:
            return aliases[variant]
    for alias in sorted(aliases, key=len, reverse=True):
        if len(alias) < 6:
            continue
        if cleaned.startswith(alias + " "):
            return aliases[alias]
    return None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    laws = load_laws(args.laws_path)
    lookup, parsed_by_citation, book_norms = build_lookup(laws)
    law_name_aliases = build_law_name_aliases(laws, parsed_by_citation)
    citation_set = set(laws["citation"].astype(str).map(norm_space))

    rows: list[dict[str, object]] = []
    hard_cases: list[dict[str, object]] = []
    counters: Counter[str] = Counter()
    unmatched_phrases: Counter[str] = Counter()

    for row in laws.itertuples(index=False):
        source = norm_space(row.citation)
        source_parsed = parsed_by_citation.get(source)
        if not source_parsed:
            continue
        source_book = source_parsed[2]
        source_book_norm = source_parsed[3]
        text = str(row.text or "")

        matched_spans: list[tuple[int, int]] = []

        for match in EXPLICIT_CODE_RE.finditer(text):
            art = (match.group("art") or "").lower()
            abs_no = (match.group("abs") or "").lower()
            book_raw = normalize_candidate_book(match.group("book"), book_norms)
            first_book_token = book_raw.split()[0] if book_raw else ""
            if (
                book_raw in NON_BOOK_TOKENS
                or first_book_token in NON_BOOK_TOKENS
                or (book_raw not in book_norms and first_book_token not in book_norms)
            ):
                counters["explicit_book_code_noise_skipped"] += 1
                continue
            target = resolve(lookup, book_raw, art, abs_no)
            if not target and " " in book_raw:
                target = resolve(lookup, book_raw.split()[0], art, abs_no)
            kind = "explicit_book_code_valid" if target else "explicit_book_code_unresolved"
            counters[kind] += 1
            matched_spans.append(match.span())
            if target and target != source:
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "kind": "explicit_book_code",
                        "mention": norm_space(match.group(0)),
                        "source_book": source_book,
                        "target_book": parse_citation(target)[2] if parse_citation(target) else "",
                    }
                )
            elif not target:
                unmatched_phrases[book_raw] += 1
                if len(hard_cases) < args.sample_limit:
                    hard_cases.append(
                        {
                            "case_type": kind,
                            "source": source,
                            "mention": norm_space(match.group(0)),
                            "context": norm_space(text[max(0, match.start() - 80) : match.end() + 80]),
                        }
                    )

        for match in NAMED_LAW_RE.finditer(text):
            # Skip when already covered by explicit code parser.
            if any(not (match.end() <= start or match.start() >= end) for start, end in matched_spans):
                continue
            book_code = resolve_book_name(match.group("law_name") or "", law_name_aliases)
            art = (match.group("art") or "").lower()
            abs_no = (match.group("abs") or "").lower()
            target = resolve(lookup, norm_book(book_code), art, abs_no) if book_code else None
            kind = "named_law_valid" if target else "named_law_unresolved"
            counters[kind] += 1
            if target and target != source:
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "kind": "named_law",
                        "mention": norm_space(match.group(0)),
                        "source_book": source_book,
                        "target_book": parse_citation(target)[2] if parse_citation(target) else "",
                    }
                )
            elif not target:
                unmatched_phrases[norm_space(match.group("law_name"))] += 1
                if len(hard_cases) < args.sample_limit:
                    hard_cases.append(
                        {
                            "case_type": kind,
                            "source": source,
                            "mention": norm_space(match.group(0)),
                            "context": norm_space(text[max(0, match.start() - 80) : match.end() + 80]),
                        }
                    )

        for match in ARTICLE_RE.finditer(text):
            if any(not (match.end() <= start or match.start() >= end) for start, end in matched_spans):
                continue
            tail = text[match.end() : match.end() + 40]
            if NOISE_AFTER_ARTICLE_RE.match(tail):
                counters["internal_noise_skipped"] += 1
                continue
            art = (match.group("art") or "").lower()
            abs_no = (match.group("abs") or "").lower()
            if not abs_no and has_trailing_foreign_book_list(tail, book_norms, source_book_norm):
                counters["internal_trailing_book_list_skipped"] += 1
                continue
            target = resolve(lookup, source_book_norm, art, abs_no)
            kind = "same_book_internal_valid" if target else "same_book_internal_unresolved"
            counters[kind] += 1
            if target and target != source:
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "kind": "same_book_internal",
                        "mention": norm_space(match.group(0)),
                        "source_book": source_book,
                        "target_book": parse_citation(target)[2] if parse_citation(target) else "",
                    }
                )
            elif not target and len(hard_cases) < args.sample_limit:
                hard_cases.append(
                    {
                        "case_type": kind,
                        "source": source,
                        "mention": norm_space(match.group(0)),
                        "context": norm_space(text[max(0, match.start() - 80) : match.end() + 80]),
                    }
                )

    edges = pd.DataFrame(rows).drop_duplicates()
    if len(edges):
        edges = edges[
            edges["source"].astype(str).map(norm_space).isin(citation_set)
            & edges["target"].astype(str).map(norm_space).isin(citation_set)
        ].copy()
    hard = pd.DataFrame(hard_cases)
    unmatched = pd.DataFrame(
        [
            {"phrase": phrase, "count": count}
            for phrase, count in unmatched_phrases.most_common(100)
        ]
    )
    summary = pd.DataFrame(
        [{"metric": key, "count": value} for key, value in counters.most_common()]
    )

    edges_path = args.output_dir / "law_reference_parser_edges.csv"
    hard_path = args.output_dir / "law_reference_parser_hard_cases.csv"
    unmatched_path = args.output_dir / "law_reference_parser_unmatched_phrases.csv"
    summary_path = args.output_dir / "law_reference_parser_summary.csv"
    edges.to_csv(edges_path, index=False)
    hard.to_csv(hard_path, index=False)
    unmatched.to_csv(unmatched_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Laws rows       : {len(laws):,}")
    print(f"Edges           : {len(edges):,}")
    print(f"Unique sources  : {edges['source'].nunique() if len(edges) else 0:,}")
    print(f"Unique targets  : {edges['target'].nunique() if len(edges) else 0:,}")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Edges      : {edges_path}")
    print(f"Hard cases : {hard_path}")
    print(f"Unmatched  : {unmatched_path}")
    print(f"Summary    : {summary_path}")


if __name__ == "__main__":
    main()
