#!/usr/bin/env python3
"""Apply the LLM law-citation filter to a court-grounded law submission."""

from __future__ import annotations

import os
import argparse
import heapq
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm

import numpy as np
import pandas as pd


def find_base_dir(path: Path) -> Path:
    for parent in [path, *path.parents]:
        if (parent / "data").exists() and (parent / "code").exists():
            return parent
    raise RuntimeError(f"Could not find repository root from {path}")


BASE_DIR = find_base_dir(Path(__file__).resolve())
CODE_DIR = BASE_DIR / "code"
UTILS_DIR = CODE_DIR / "dev/utils"
for path in [CODE_DIR, UTILS_DIR]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from call_llm import (  # noqa: E402
    DEFAULT_FIREWORKS_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    call_llm,
    get_vllm_client,
)
from prompts.llm_law_reranking_prompt import (  # noqa: E402
    LABELS,
    build_conservative_llm_law_reranking_prompt,
    build_conservative_no_court_law_reranking_prompt,
    build_conservative_no_court_removal_confirmation_prompt,
    build_conservative_removal_confirmation_prompt,
)


DEFAULT_SUBMISSION_PATH = BASE_DIR / "context/test_court_graph_ce_top50_base080_ce020.csv"
DEFAULT_VAL_SUBMISSION_PATH = BASE_DIR / "context/val_court_graph_ce_top50_base080_ce020.csv"
DEFAULT_TEST_QUERY_PATH = BASE_DIR / "data/test.parquet"
DEFAULT_VAL_QUERY_PATH = BASE_DIR / "data/val.parquet"
DEFAULT_LAW_PATH = BASE_DIR / "data/laws_db.parquet"
DEFAULT_COURT_PATH = BASE_DIR / "data/court_db.parquet"
DEFAULT_ENV_PATH = BASE_DIR / ".env"
DEFAULT_OUTPUT_PATH = BASE_DIR / "context/test_court_graph_ce_top50_base080_ce020_llm_tail.csv"
DEFAULT_VAL_OUTPUT_PATH = BASE_DIR / "context/val_court_graph_ce_top50_base080_ce020_llm_tail.csv"
DEFAULT_PREDICTION_OUTPUT_PATH = (
    BASE_DIR / "context/test_court_graph_llm_tail_predictions.csv"
)
DEFAULT_VAL_PREDICTION_OUTPUT_PATH = (
    BASE_DIR / "context/val_court_graph_llm_tail_predictions.csv"
)
DEFAULT_BATCH_OUTPUT_PATH = BASE_DIR / "context/test_court_graph_llm_tail_batches.parquet"
DEFAULT_VAL_BATCH_OUTPUT_PATH = BASE_DIR / "context/val_court_graph_llm_tail_batches.parquet"
DEFAULT_SUMMARY_OUTPUT_PATH = BASE_DIR / "context/test_court_graph_llm_tail_summary.csv"
DEFAULT_VAL_SUMMARY_OUTPUT_PATH = BASE_DIR / "context/val_court_graph_llm_tail_summary.csv"
CANON_RE = re.compile(r"^Art\.\s*\S+(?:\s+Abs\.\s*\S+)?\s+(.+)$")
DEFAULT_VLLM_CONTEXT_WINDOWS = {
    "qwen36": 14384,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-filter a court-grounded law submission.")
    parser.add_argument(
        "--dataset",
        choices=["test", "val"],
        default="test",
        help="Dataset whose query/gold file should be used.",
    )
    parser.add_argument("--submission-path", type=Path, default=DEFAULT_SUBMISSION_PATH)
    parser.add_argument("--query-path", type=Path, default=None)
    parser.add_argument(
        "--query-column",
        default="",
        help="Query text column; defaults to german_translation when available.",
    )
    parser.add_argument("--gold-column", default="gold_citations")
    parser.set_defaults(score_art_only_gold=True)
    parser.add_argument(
        "--score-art-only-gold",
        dest="score_art_only_gold",
        action="store_true",
        help="Score only Art.-style law gold citations. This matches the court-grounded law retrieval metric.",
    )
    parser.add_argument(
        "--score-all-gold",
        dest="score_art_only_gold",
        action="store_false",
        help="Score against all entries in the gold_citations column.",
    )
    parser.add_argument(
        "--input-top-k",
        type=int,
        default=20,
        help="When submission-path is a long score file, use this many citations per query.",
    )
    parser.add_argument(
        "--rerank-mode",
        choices=[
            "conservative-court-context",
            "conservative-no-court-first-court-confirm",
            "conservative-no-court-context",
        ],
        default="conservative-court-context",
        help="LLM reranking experiment mode.",
    )
    parser.add_argument("--law-path", type=Path, default=DEFAULT_LAW_PATH)
    parser.add_argument("--court-path", type=Path, default=DEFAULT_COURT_PATH)
    parser.add_argument("--env-path", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--prediction-output-path", type=Path, default=DEFAULT_PREDICTION_OUTPUT_PATH)
    parser.add_argument("--batch-output-path", type=Path, default=DEFAULT_BATCH_OUTPUT_PATH)
    parser.add_argument("--summary-output-path", type=Path, default=DEFAULT_SUMMARY_OUTPUT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--row-start", type=int, default=1)
    parser.add_argument("--row-end", type=int, default=0)
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--court-context-batch-size", type=int, default=2)
    parser.add_argument("--court-context-confirm-batch-size", type=int, default=2)
    parser.add_argument(
        "--no-court-batch-size",
        type=int,
        default=4,
        help="Candidate articles per prompt for conservative-no-court-context mode.",
    )
    parser.add_argument(
        "--no-court-confirm-batch-size",
        type=int,
        default=4,
        help="Candidate articles per confirmation prompt for conservative-no-court-context mode.",
    )
    parser.add_argument(
        "--court-context-confirm-removals",
        action="store_true",
        help="Run a second safety-check pass over first-pass REMOVE decisions.",
    )
    parser.add_argument("--court-context-freeze-top-k", type=int, default=10)
    parser.add_argument(
        "--llm-rank-start",
        type=int,
        default=11,
        help="First 1-based citation rank that may be sent to the LLM.",
    )
    parser.add_argument(
        "--llm-rank-end",
        type=int,
        default=20,
        help="Last 1-based citation rank that may be sent to the LLM.",
    )
    parser.add_argument("--court-context-top-k", type=int, default=1)
    parser.add_argument(
        "--court-context-max-chars",
        type=int,
        default=0,
        help="Maximum chars per court usage passage; 0 disables truncation.",
    )
    parser.add_argument("--court-context-parquet-batch-size", type=int, default=2048)
    parser.add_argument(
        "--query-embedding-column",
        default="zembed-full-german-query",
        help="Embedding column used to pick query-similar court usage passages.",
    )
    parser.add_argument(
        "--parallel-requests",
        type=int,
        default=3,
        help="Maximum number of LLM batch requests in flight at once.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["openrouter", "openai", "fireworks", "vllm"],
        default="fireworks",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument(
        "--llm-context-window",
        type=int,
        default=0,
        help=(
            "Maximum model context tokens for dynamic vLLM output budgeting. "
            "0 auto-detects known model aliases such as qwen36; negative disables."
        ),
    )
    parser.add_argument(
        "--llm-token-buffer",
        type=int,
        default=512,
        help="Safety buffer reserved from the model context window before setting max_tokens.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="",
        help=(
            "Optional local path or Hugging Face model id for prompt token counting. "
            "When set, this overrides the tokenizer root reported by vLLM."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.set_defaults(reasoning_enabled=False)
    parser.add_argument("--reasoning-enabled", dest="reasoning_enabled", action="store_true")
    parser.add_argument("--no-reasoning", dest="reasoning_enabled", action="store_false")
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default="none",
    )
    parser.add_argument("--provider-only", default="")
    parser.add_argument("--disable-provider-fallbacks", action="store_true")
    return parser.parse_args()


def resolve_dataset_args(args: argparse.Namespace) -> None:
    if args.query_path is None:
        args.query_path = DEFAULT_VAL_QUERY_PATH if args.dataset == "val" else DEFAULT_TEST_QUERY_PATH
    if args.dataset == "val" and args.submission_path == DEFAULT_SUBMISSION_PATH:
        args.submission_path = DEFAULT_VAL_SUBMISSION_PATH
    if not args.query_column:
        args.query_column = "german_translation"
    if args.output_path is None:
        args.output_path = DEFAULT_VAL_OUTPUT_PATH if args.dataset == "val" else DEFAULT_OUTPUT_PATH
    if args.dataset == "val" and args.prediction_output_path == DEFAULT_PREDICTION_OUTPUT_PATH:
        args.prediction_output_path = DEFAULT_VAL_PREDICTION_OUTPUT_PATH
    if args.dataset == "val" and args.batch_output_path == DEFAULT_BATCH_OUTPUT_PATH:
        args.batch_output_path = DEFAULT_VAL_BATCH_OUTPUT_PATH
    if args.dataset == "val" and args.summary_output_path == DEFAULT_SUMMARY_OUTPUT_PATH:
        args.summary_output_path = DEFAULT_VAL_SUMMARY_OUTPUT_PATH


def split_refs(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def load_submission_input(path: Path, *, input_top_k: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if {"query_id", "predicted_citations"}.issubset(frame.columns):
        return frame[["query_id", "predicted_citations"]].copy()

    if {"query_id", "citation"}.issubset(frame.columns):
        if input_top_k < 1:
            raise ValueError("--input-top-k must be >= 1 when using a long score file")
        if "original_rank" in frame.columns:
            frame = frame.sort_values(["query_id", "original_rank"], kind="stable")
        else:
            frame = frame.sort_values(["query_id"], kind="stable")
        rows = []
        for query_id, group in frame.groupby("query_id", sort=False):
            citations = reconstruct_score_file_order(group).head(input_top_k).tolist()
            rows.append(
                {
                    "query_id": str(query_id),
                    "predicted_citations": ";".join(citations),
                }
            )
        return pd.DataFrame(rows)

    raise ValueError(
        f"{path} must contain either query_id+predicted_citations or query_id+citation"
    )


def reconstruct_score_file_order(group: pd.DataFrame) -> pd.Series:
    """Rebuild final frozen-slot order from a cross-encoder score/debug file."""
    if not {"original_rank", "sent_to_reranker"}.issubset(group.columns):
        return group["citation"].astype(str)

    score_column = ""
    for candidate in ["combined_score", "score"]:
        if candidate in group.columns:
            score_column = candidate
            break
    if not score_column:
        return group["citation"].astype(str)

    ordered = group.sort_values("original_rank", kind="stable").reset_index(drop=True)
    citations = ordered["citation"].astype(str).tolist()
    movable_positions: list[int] = []
    movable: list[tuple[float, int, str]] = []
    for index, row in enumerate(ordered.itertuples(index=False)):
        sent_to_reranker = str(getattr(row, "sent_to_reranker")).lower() == "true"
        score = getattr(row, score_column)
        if sent_to_reranker and pd.notna(score):
            movable_positions.append(index)
            movable.append((float(score), -index, str(getattr(row, "citation"))))

    movable.sort(reverse=True)
    rebuilt = citations[:]
    for position, (_, __, citation) in zip(movable_positions, movable):
        rebuilt[position] = citation
    return pd.Series(rebuilt)


def citation_book(citation: object) -> str:
    match = CANON_RE.match(str(citation).strip())
    return match.group(1).strip() if match else ""


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in __import__("os").environ:
            __import__("os").environ[key] = value


def parse_decisions(raw_response: str) -> tuple[dict[str, str], str]:
    text = raw_response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\[.*\]", text, flags=re.S)
    payload = match.group(0) if match else text
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        fallback = parse_decisions_fallback(text)
        if fallback:
            return fallback, f"json_decode_error_regex_fallback: {error}"
        return {}, f"json_decode_error: {error}"
    if not isinstance(parsed, list):
        fallback = parse_decisions_fallback(text)
        if fallback:
            return fallback, "json_root_not_list_regex_fallback"
        return {}, "json_root_not_list"

    decisions: dict[str, str] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().upper()
        decision = str(item.get("decision", "")).strip().upper()
        if label and decision in {"KEEP", "REMOVE"}:
            decisions[label] = decision
    return decisions, ""


def parse_decisions_fallback(raw_response: str) -> dict[str, str]:
    decisions: dict[str, str] = {}
    pattern = re.compile(
        r'"label"\s*:\s*"(?P<label>[A-D])".*?"decision"\s*:\s*"(?P<decision>KEEP|REMOVE)"',
        flags=re.S,
    )
    for match in pattern.finditer(raw_response):
        decisions[match.group("label")] = match.group("decision")
    return decisions


def resolve_model(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    if args.llm_backend == "openai":
        return DEFAULT_OPENAI_MODEL
    if args.llm_backend == "fireworks":
        return DEFAULT_FIREWORKS_MODEL
    return DEFAULT_MODEL


def build_request_extra_body(args: argparse.Namespace) -> dict[str, Any]:
    if args.llm_backend != "openrouter":
        return {}
    provider: dict[str, Any] = {}
    if args.provider_only:
        provider["only"] = [part.strip() for part in args.provider_only.split(",") if part.strip()]
    if args.disable_provider_fallbacks:
        provider["allow_fallbacks"] = False
    return {"provider": provider} if provider else {}


def resolve_llm_context_window(args: argparse.Namespace) -> int:
    if args.llm_context_window < 0:
        return 0
    if args.llm_context_window > 0:
        return args.llm_context_window
    if args.llm_backend != "vllm":
        return 0

    _, server_context_window = resolve_vllm_model_info(str(args.model))
    if server_context_window > 0:
        return server_context_window

    model = str(args.model).lower()
    for alias, context_window in DEFAULT_VLLM_CONTEXT_WINDOWS.items():
        if alias in model:
            return context_window
    return 0


@lru_cache(maxsize=16)
def resolve_vllm_model_info(model: str) -> tuple[str, int]:
    try:
        models = get_vllm_client().models.list()
    except Exception:
        return model, 0

    model_data = getattr(models, "data", []) or []
    for item in model_data:
        model_id = str(getattr(item, "id", "") or "")
        if model_id != model:
            continue
        root = str(getattr(item, "root", "") or "").strip()
        max_model_len = getattr(item, "max_model_len", 0) or 0
        try:
            max_model_len = int(max_model_len)
        except (TypeError, ValueError):
            max_model_len = 0
        return root or model, max_model_len
    return model, 0


def resolve_vllm_model_root(model: str) -> str:
    model_root, _ = resolve_vllm_model_info(model)
    return model_root


def resolve_tokenizer_source(args: argparse.Namespace) -> str:
    if str(args.tokenizer_path).strip():
        return str(args.tokenizer_path).strip()
    if args.llm_backend == "vllm":
        return resolve_vllm_model_root(str(args.model))
    return str(args.model)


@lru_cache(maxsize=16)
def load_chat_tokenizer(model_or_root: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_or_root,
        trust_remote_code=True,
        local_files_only=False,
    )


def estimate_chat_prompt_tokens(
    messages: list[dict[str, str]],
    *,
    args: argparse.Namespace,
) -> int:
    text = "\n".join(
        f"{message.get('role', '')}: {message.get('content', '')}"
        for message in messages
    )
    # Conservative fallback for Qwen/vLLM aliases where the exact tokenizer may
    # not be locally available.
    char_estimate = math.ceil(len(text) / 3.0) + 16 + 12 * len(messages)
    explicit_tokenizer_path = bool(str(args.tokenizer_path).strip())
    if args.llm_backend == "vllm" or explicit_tokenizer_path:
        try:
            model_root = resolve_tokenizer_source(args)
            tokenizer = load_chat_tokenizer(model_root)
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            return len(token_ids)
        except Exception as error:
            if explicit_tokenizer_path:
                raise RuntimeError(
                    f"failed_to_load_tokenizer_path: {args.tokenizer_path}"
                ) from error
            pass

    try:
        import tiktoken  # type: ignore

        encoder = tiktoken.get_encoding("cl100k_base")
        tokenizer_estimate = len(encoder.encode(text)) + 16 + 12 * len(messages)
        return max(char_estimate, tokenizer_estimate)
    except Exception:
        return char_estimate


def resolve_effective_max_tokens(
    args: argparse.Namespace,
    messages: list[dict[str, str]],
) -> dict[str, int | bool]:
    requested_max_tokens = max(1, int(args.max_tokens))
    context_window = resolve_llm_context_window(args)
    estimated_prompt_tokens = estimate_chat_prompt_tokens(messages, args=args)
    if context_window <= 0:
        return {
            "requested_max_tokens": requested_max_tokens,
            "effective_max_tokens": requested_max_tokens,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "llm_context_window": context_window,
            "max_tokens_reduced": False,
        }

    token_buffer = max(0, int(args.llm_token_buffer))
    available_output_tokens = context_window - estimated_prompt_tokens - token_buffer
    effective_max_tokens = min(requested_max_tokens, max(1, available_output_tokens))
    return {
        "requested_max_tokens": requested_max_tokens,
        "effective_max_tokens": effective_max_tokens,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "llm_context_window": context_window,
        "max_tokens_reduced": effective_max_tokens < requested_max_tokens,
    }


def select_rows(frame: pd.DataFrame, row_start: int, row_end: int) -> tuple[pd.DataFrame, int]:
    if row_start < 1:
        raise ValueError("--row-start is 1-based and must be >= 1")
    if row_end and row_end < row_start:
        raise ValueError("--row-end must be >= --row-start")
    resolved_end = min(row_end if row_end else len(frame), len(frame))
    return frame.iloc[row_start - 1 : resolved_end].copy(), resolved_end


def chunked(values: list[dict[str, object]], size: int) -> list[list[dict[str, object]]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def build_prompt_for_style(
    prompt_style: str,
    german_query: str,
    citations: list[str],
    documents: list[str],
    court_contexts: list[list[dict[str, object]]] | None = None,
) -> str:
    if prompt_style == "conservative":
        return build_conservative_llm_law_reranking_prompt(
            german_query,
            citations,
            documents,
            court_contexts=court_contexts,
        )
    if prompt_style == "conservative_confirm":
        return build_conservative_removal_confirmation_prompt(
            german_query,
            citations,
            documents,
            court_contexts=court_contexts,
        )
    if prompt_style == "conservative_no_court":
        return build_conservative_no_court_law_reranking_prompt(
            german_query,
            citations,
            documents,
        )
    if prompt_style == "conservative_no_court_confirm":
        return build_conservative_no_court_removal_confirmation_prompt(
            german_query,
            citations,
            documents,
        )
    raise ValueError(f"Unsupported prompt style: {prompt_style}")


def normalize_embedding(value: object) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if vector.ndim != 1 or vector.size == 0:
        return None
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        return None
    return vector / norm


def truncate_text(value: object, max_chars: int) -> str:
    text = str(value).strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def format_reference_list(refs: list[str], *, max_refs: int = 12) -> str:
    if len(refs) <= max_refs:
        return ";".join(refs)
    return ";".join(refs[:max_refs]) + f";...(+{len(refs) - max_refs})"


def build_court_usage_contexts(
    *,
    args: argparse.Namespace,
    selected: pd.DataFrame,
    queries: pd.DataFrame,
) -> dict[tuple[str, str], list[dict[str, object]]]:
    query_to_citations: dict[str, set[str]] = defaultdict(set)
    for row in tqdm(selected.itertuples(index=False), total=len(selected), desc="Building court usage contexts", unit="row"):
        query_id = str(row.query_id)
        for rank_position, citation in enumerate(split_refs(row.predicted_citations), start=1):
            if (
                rank_position < args.llm_rank_start
                or rank_position > args.llm_rank_end
                or rank_position <= args.court_context_freeze_top_k
            ):
                continue
            query_to_citations[query_id].add(citation)

    return build_court_usage_contexts_for_targets(
        args=args,
        queries=queries,
        query_to_citations=query_to_citations,
    )


def build_court_usage_contexts_for_targets(
    *,
    args: argparse.Namespace,
    queries: pd.DataFrame,
    query_to_citations: dict[str, set[str]],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    if args.court_context_top_k < 1:
        return {}
    if args.query_embedding_column not in queries.columns:
        raise ValueError(
            f"query embedding column {args.query_embedding_column!r} is required for "
            "court context mode"
        )

    query_embeddings: dict[str, np.ndarray] = {}
    for _, row in queries.iterrows():
        query_id = str(row["query_id"])
        embedding = normalize_embedding(row[args.query_embedding_column])
        if embedding is not None:
            query_embeddings[query_id] = embedding

    citation_to_query_ids: dict[str, set[str]] = defaultdict(set)
    for query_id, citations in query_to_citations.items():
        if query_id not in query_embeddings:
            continue
        for citation in citations:
            citation_to_query_ids[citation].add(query_id)

    target_citations = set(citation_to_query_ids)
    if not target_citations:
        return {}

    heaps: dict[tuple[str, str], list[tuple[float, int, dict[str, object]]]] = defaultdict(list)
    row_counter = 0

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for court context streaming") from error

    parquet = pq.ParquetFile(args.court_path)
    columns = ["citation", "document", "law_references", "embedding"]
    for batch in parquet.iter_batches(
        batch_size=args.court_context_parquet_batch_size,
        columns=columns,
    ):
        batch_columns = batch.to_pydict()
        for court_citation, document, law_references, embedding in zip(
            batch_columns["citation"],
            batch_columns["document"],
            batch_columns["law_references"],
            batch_columns["embedding"],
        ):
            refs = list(dict.fromkeys(split_refs(law_references)))
            matched_refs = [ref for ref in refs if ref in target_citations]
            if not matched_refs:
                continue
            court_embedding = normalize_embedding(embedding)
            if court_embedding is None:
                continue

            law_reference_count = len(refs)
            for citation in matched_refs:
                other_refs = [ref for ref in refs if ref != citation]
                for query_id in citation_to_query_ids[citation]:
                    query_embedding = query_embeddings.get(query_id)
                    if query_embedding is None or query_embedding.shape != court_embedding.shape:
                        continue
                    similarity = float(np.dot(query_embedding, court_embedding))
                    context = {
                        "court_citation": str(court_citation),
                        "law_references": format_reference_list(refs),
                        "other_law_references": format_reference_list(other_refs),
                        "law_reference_count": law_reference_count,
                        "similarity": similarity,
                        "document": truncate_text(document, args.court_context_max_chars),
                    }
                    key = (query_id, citation)
                    item = (similarity, row_counter, context)
                    heap = heaps[key]
                    if len(heap) < args.court_context_top_k:
                        heapq.heappush(heap, item)
                    elif similarity > heap[0][0]:
                        heapq.heapreplace(heap, item)
                    row_counter += 1

    return {
        key: [context for _, __, context in sorted(heap, reverse=True)]
        for key, heap in heaps.items()
    }


def run_conservative_court_context_mode(
    *,
    args: argparse.Namespace,
    submission: pd.DataFrame,
    queries: pd.DataFrame,
    selected: pd.DataFrame,
    row_end: int,
    citation_to_document: dict[str, str],
    citation_to_source_book: dict[str, str],
    query_to_text: dict[str, str],
    request_extra_body: dict[str, Any],
) -> None:
    use_court_first_pass = args.rerank_mode == "conservative-court-context"
    use_court_confirm = args.rerank_mode in {
        "conservative-court-context",
        "conservative-no-court-first-court-confirm",
    }
    first_pass_prompt_style = (
        "conservative" if use_court_first_pass else "conservative_no_court"
    )
    confirm_prompt_style = (
        "conservative_confirm"
        if use_court_confirm
        else "conservative_no_court_confirm"
    )
    prompt_style_label = (
        "conservative_court_context"
        if use_court_first_pass
        else "conservative_no_court_context"
    )
    confirm_prompt_style_label = (
        "conservative_court_confirm"
        if use_court_confirm
        else "conservative_no_court_confirm"
    )
    if args.rerank_mode == "conservative-no-court-first-court-confirm":
        vote_group_prefix = "conservative_no_court_first_court_confirm"
    elif use_court_first_pass:
        vote_group_prefix = "conservative_court"
    else:
        vote_group_prefix = "conservative_no_court"
    first_pass_batch_size = (
        args.court_context_batch_size
        if use_court_first_pass
        else args.no_court_batch_size
    )
    confirm_batch_size = (
        args.court_context_confirm_batch_size
        if use_court_confirm
        else args.no_court_confirm_batch_size
    )

    if args.court_context_batch_size < 1 or args.court_context_batch_size > len(LABELS):
        raise ValueError(f"--court-context-batch-size must be between 1 and {len(LABELS)}")
    if args.court_context_confirm_batch_size < 1 or args.court_context_confirm_batch_size > len(LABELS):
        raise ValueError(
            f"--court-context-confirm-batch-size must be between 1 and {len(LABELS)}"
        )
    if args.no_court_batch_size < 1 or args.no_court_batch_size > len(LABELS):
        raise ValueError(f"--no-court-batch-size must be between 1 and {len(LABELS)}")
    if args.no_court_confirm_batch_size < 1 or args.no_court_confirm_batch_size > len(LABELS):
        raise ValueError(
            f"--no-court-confirm-batch-size must be between 1 and {len(LABELS)}"
        )
    if args.court_context_freeze_top_k < 0:
        raise ValueError("--court-context-freeze-top-k must be >= 0")
    if args.llm_rank_start < 1:
        raise ValueError("--llm-rank-start must be >= 1")
    if args.llm_rank_end < args.llm_rank_start:
        raise ValueError("--llm-rank-end must be >= --llm-rank-start")

    print(f"Dataset         : {args.dataset}")
    print(f"Submission rows : {len(submission):,}")
    print(f"Submission input: {args.submission_path}")
    print(f"Query path      : {args.query_path}")
    print(f"Query column    : {args.query_column}")
    print(f"LLM rows        : {args.row_start}-{row_end} ({len(selected):,} rows)")
    print(f"LLM backend     : {args.llm_backend}")
    print(f"LLM model       : {args.model}")
    context_window = resolve_llm_context_window(args)
    if context_window > 0:
        print(f"LLM context     : {context_window:,} tokens, buffer {max(0, args.llm_token_buffer):,}")
    else:
        print("LLM context     : dynamic max_tokens disabled")
    if str(args.tokenizer_path).strip():
        print(f"Tokenizer path  : {args.tokenizer_path}")
    print(f"Rerank mode     : {args.rerank_mode}")
    print(f"Freeze ranks    : 1-{args.court_context_freeze_top_k}")
    print(f"LLM rank window : {args.llm_rank_start}-{args.llm_rank_end}")
    print(f"Batch size      : {first_pass_batch_size}")
    print(f"Confirm batch   : {confirm_batch_size}")
    if use_court_first_pass or use_court_confirm:
        print(f"Court top-k     : {args.court_context_top_k}")
    print(f"Confirm removals: {bool(args.court_context_confirm_removals)}")
    print(f"Parallel reqs   : {max(1, args.parallel_requests)}")

    if use_court_first_pass:
        print("Building court usage contexts...")
        court_contexts = build_court_usage_contexts(
            args=args,
            selected=selected,
            queries=queries,
        )
        print(f"Court contexts  : {len(court_contexts):,} query/citation pairs")
    elif use_court_confirm:
        court_contexts = {}
        print("Court contexts  : confirmation pass only")
    else:
        court_contexts = {}
        print("Court contexts  : skipped")

    selected_query_ids = set(selected["query_id"].astype(str))
    candidate_states: dict[str, dict[str, str]] = {}
    prediction_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, str]] = []
    batch_jobs: list[dict[str, Any]] = []
    confirm_candidates: list[dict[str, object]] = []
    prediction_index_by_key: dict[tuple[str, str], int] = {}
    global_batch_id = 0
    hit_batch_limit = False

    for row in submission.itertuples(index=False):
        query_id = str(row.query_id)
        original = split_refs(row.predicted_citations)
        if query_id not in selected_query_ids:
            continue

        german_query = query_to_text.get(query_id, "")
        candidate_state: dict[str, str] = {citation: "KEEP" for citation in original}
        candidate_states[query_id] = candidate_state
        sendable: list[dict[str, object]] = []

        for rank_position, citation in enumerate(original, start=1):
            source_book = citation_to_source_book.get(citation, citation_book(citation))
            has_document = citation in citation_to_document and bool(citation_to_document[citation].strip())
            freeze_reason = ""
            if rank_position < args.llm_rank_start:
                freeze_reason = "before_llm_window"
            elif rank_position > args.llm_rank_end:
                freeze_reason = "after_llm_window"
            elif rank_position <= args.court_context_freeze_top_k:
                freeze_reason = "top_rank"
            elif not has_document:
                freeze_reason = "missing_document"

            context_items = court_contexts.get((query_id, citation), [])
            if freeze_reason:
                prediction_rows.append(
                    {
                        "query_id": query_id,
                        "batch_id": None,
                        "label": "",
                        "rank_position": rank_position,
                        "citation": citation,
                        "source_book": source_book,
                        "sent_to_llm": False,
                        "prompt_style": "freeze",
                        "freeze_reason": freeze_reason,
                        "first_pass_decision": "KEEP",
                        "model_decision": "KEEP",
                        "removed": False,
                        "confirmation_decision": "",
                        "confirmation_batch_id": None,
                        "confirmation_parse_error": "",
                        "confirmation_finish_reason": "",
                        "parse_error": "",
                        "finish_reason": "",
                        "court_context_count": len(context_items),
                        "court_context_citations": ";".join(
                            str(item.get("court_citation", "")) for item in context_items
                        ),
                        "court_context_similarities": ";".join(
                            f"{float(item.get('similarity', 0.0)):.4f}" for item in context_items
                        ),
                        "german_query": german_query,
                        "document": citation_to_document.get(citation, ""),
                    }
                )
            else:
                sendable.append(
                    {
                        "citation": citation,
                        "source_book": source_book,
                        "rank_position": rank_position,
                        "court_contexts": context_items,
                    }
                )

        local_batch_id = 0
        for batch in chunked(sendable, first_pass_batch_size):
            if args.limit_batches > 0 and global_batch_id >= args.limit_batches:
                hit_batch_limit = True
                break
            batch_citations = [str(item["citation"]) for item in batch]
            batch_documents = [
                citation_to_document.get(citation, "") for citation in batch_citations
            ]
            batch_contexts = [
                list(item.get("court_contexts", [])) for item in batch
            ]
            prompt = build_prompt_for_style(
                first_pass_prompt_style,
                german_query,
                batch_citations,
                batch_documents,
                court_contexts=batch_contexts,
            )
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            batch_jobs.append(
                {
                    "query_id": query_id,
                    "batch_id": global_batch_id,
                    "local_batch_id": local_batch_id,
                    "vote_group_id": f"{query_id}:{vote_group_prefix}:{local_batch_id}",
                    "vote_index": 0,
                    "vote_temperature": args.temperature,
                    "temperature": args.temperature,
                    "prompt_style": prompt_style_label,
                    "parse_mode": "decision",
                    "prompt_hash": prompt_hash,
                    "prompt": prompt,
                    "batch": batch,
                }
            )
            global_batch_id += 1
            local_batch_id += 1

        if hit_batch_limit:
            break

    print(f"LLM batch jobs  : {len(batch_jobs):,}")
    batch_results = execute_batch_jobs(
        batch_jobs,
        args=args,
        request_extra_body=request_extra_body,
    )

    for result in batch_results:
        query_id = str(result["query_id"])
        batch = result["batch"]
        decisions = result["decisions"]
        parse_error = result["parse_error"]
        finish_reason = result["finish_reason"]
        batch_rows.append(
            {
                "query_id": query_id,
                "batch_id": result["batch_id"],
                "local_batch_id": result["local_batch_id"],
                "llm_backend": args.llm_backend,
                "model": args.model,
                "prompt_style": result["prompt_style"],
                "parse_mode": result["parse_mode"],
                "vote_group_id": result["vote_group_id"],
                "vote_index": result["vote_index"],
                "prompt_hash": result["prompt_hash"],
                "prompt": result["prompt"],
                "raw_response": result["raw_response"],
                "finish_reason": finish_reason,
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "max_tokens": result["effective_max_tokens"],
                "requested_max_tokens": result["requested_max_tokens"],
                "estimated_prompt_tokens": result["estimated_prompt_tokens"],
                "llm_context_window": result["llm_context_window"],
                "max_tokens_reduced": result["max_tokens_reduced"],
                "temperature": result["temperature"],
                "top_p": args.top_p,
                "top_k": args.top_k,
                "parse_error": parse_error,
            }
        )

        candidate_state = candidate_states.get(query_id, {})
        for label, item in zip(LABELS, batch):
            citation = str(item["citation"])
            decision = decisions.get(label, "KEEP")
            removed = decision == "REMOVE"
            candidate_state[citation] = decision
            context_items = list(item.get("court_contexts", []))
            prediction_row = {
                "query_id": query_id,
                "batch_id": result["batch_id"],
                "label": label,
                "rank_position": item["rank_position"],
                "citation": citation,
                "source_book": item["source_book"],
                "sent_to_llm": True,
                "prompt_style": prompt_style_label,
                "freeze_reason": "",
                "first_pass_decision": decision,
                "model_decision": decision,
                "removed": removed,
                "confirmation_decision": "",
                "confirmation_batch_id": None,
                "confirmation_parse_error": "",
                "confirmation_finish_reason": "",
                "parse_error": parse_error,
                "finish_reason": finish_reason,
                "court_context_count": len(context_items),
                "court_context_citations": ";".join(
                    str(context.get("court_citation", "")) for context in context_items
                ),
                "court_context_similarities": ";".join(
                    f"{float(context.get('similarity', 0.0)):.4f}"
                    for context in context_items
                ),
                "german_query": query_to_text.get(query_id, ""),
                "document": citation_to_document.get(citation, ""),
            }
            prediction_index_by_key[(query_id, citation)] = len(prediction_rows)
            prediction_rows.append(prediction_row)
            if removed:
                confirm_candidates.append(
                    {
                        "query_id": query_id,
                        "citation": citation,
                        "source_book": item["source_book"],
                        "rank_position": item["rank_position"],
                        "court_contexts": context_items,
                    }
                )

    if args.court_context_confirm_removals and confirm_candidates:
        if use_court_confirm and not use_court_first_pass:
            confirm_targets: dict[str, set[str]] = defaultdict(set)
            for item in confirm_candidates:
                confirm_targets[str(item["query_id"])].add(str(item["citation"]))

            print("Building court usage contexts for confirmation candidates...")
            confirm_court_contexts = build_court_usage_contexts_for_targets(
                args=args,
                queries=queries,
                query_to_citations=confirm_targets,
            )
            print(
                "Confirm contexts : "
                f"{len(confirm_court_contexts):,} query/citation pairs"
            )
            for item in confirm_candidates:
                key = (str(item["query_id"]), str(item["citation"]))
                item["court_contexts"] = confirm_court_contexts.get(key, [])

        confirm_by_query: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in confirm_candidates:
            confirm_by_query[str(item["query_id"])].append(item)

        confirm_jobs: list[dict[str, Any]] = []
        for query_id, items in confirm_by_query.items():
            german_query = query_to_text.get(query_id, "")
            for local_batch_id, batch in enumerate(
                chunked(items, confirm_batch_size)
            ):
                batch_citations = [str(item["citation"]) for item in batch]
                batch_documents = [
                    citation_to_document.get(citation, "") for citation in batch_citations
                ]
                batch_contexts = [
                    list(item.get("court_contexts", [])) for item in batch
                ]
                prompt = build_prompt_for_style(
                    confirm_prompt_style,
                    german_query,
                    batch_citations,
                    batch_documents,
                    court_contexts=batch_contexts,
                )
                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                confirm_jobs.append(
                    {
                        "query_id": query_id,
                        "batch_id": global_batch_id,
                        "local_batch_id": local_batch_id,
                        "vote_group_id": f"{query_id}:{vote_group_prefix}_confirm:{local_batch_id}",
                        "vote_index": 0,
                        "vote_temperature": args.temperature,
                        "temperature": args.temperature,
                        "prompt_style": confirm_prompt_style_label,
                        "parse_mode": "decision",
                        "prompt_hash": prompt_hash,
                        "prompt": prompt,
                        "batch": batch,
                    }
                )
                global_batch_id += 1

        print(f"Confirm jobs    : {len(confirm_jobs):,}")
        confirm_results = execute_batch_jobs(
            confirm_jobs,
            args=args,
            request_extra_body=request_extra_body,
        )

        for result in confirm_results:
            query_id = str(result["query_id"])
            batch = result["batch"]
            decisions = result["decisions"]
            parse_error = result["parse_error"]
            finish_reason = result["finish_reason"]
            batch_rows.append(
                {
                    "query_id": query_id,
                    "batch_id": result["batch_id"],
                    "local_batch_id": result["local_batch_id"],
                    "llm_backend": args.llm_backend,
                    "model": args.model,
                    "prompt_style": result["prompt_style"],
                    "parse_mode": result["parse_mode"],
                    "vote_group_id": result["vote_group_id"],
                    "vote_index": result["vote_index"],
                    "prompt_hash": result["prompt_hash"],
                    "prompt": result["prompt"],
                    "raw_response": result["raw_response"],
                    "finish_reason": finish_reason,
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["total_tokens"],
                    "max_tokens": result["effective_max_tokens"],
                    "requested_max_tokens": result["requested_max_tokens"],
                    "estimated_prompt_tokens": result["estimated_prompt_tokens"],
                    "llm_context_window": result["llm_context_window"],
                    "max_tokens_reduced": result["max_tokens_reduced"],
                    "temperature": result["temperature"],
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "parse_error": parse_error,
                }
            )

            candidate_state = candidate_states.get(query_id, {})
            for label, item in zip(LABELS, batch):
                citation = str(item["citation"])
                confirmation_decision = decisions.get(label, "KEEP")
                final_decision = "REMOVE" if confirmation_decision == "REMOVE" else "KEEP"
                candidate_state[citation] = final_decision
                row_index = prediction_index_by_key.get((query_id, citation))
                if row_index is not None:
                    prediction_rows[row_index].update(
                        {
                            "model_decision": final_decision,
                            "removed": final_decision == "REMOVE",
                            "confirmation_decision": confirmation_decision,
                            "confirmation_batch_id": result["batch_id"],
                            "confirmation_parse_error": parse_error,
                            "confirmation_finish_reason": finish_reason,
                        }
                    )

    for row in submission.itertuples(index=False):
        query_id = str(row.query_id)
        original = split_refs(row.predicted_citations)
        candidate_state = candidate_states.get(query_id)
        if candidate_state is None:
            final = original
        else:
            final = [
                citation
                for citation in original
                if candidate_state.get(citation, "KEEP") != "REMOVE"
            ]
        output_rows.append({"query_id": query_id, "predicted_citations": ";".join(final)})

    output = pd.DataFrame(output_rows)
    predictions = pd.DataFrame(prediction_rows)
    batches = pd.DataFrame(batch_rows)
    summary = summarize(
        submission,
        output,
        predictions,
        batches,
        queries=queries,
        gold_column=args.gold_column,
        score_art_only_gold=args.score_art_only_gold,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.prediction_output_path.parent.mkdir(parents=True, exist_ok=True)
    args.batch_output_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)
    predictions.to_csv(args.prediction_output_path, index=False)
    batches.to_parquet(args.batch_output_path, index=False)
    summary.to_csv(args.summary_output_path, index=False)

    print(f"Output          : {args.output_path}")
    print(f"Predictions     : {args.prediction_output_path}")
    print(f"Batches         : {args.batch_output_path}")
    print(f"Summary         : {args.summary_output_path}")
    print(summary.to_string(index=False))


def execute_batch_job(
    job: dict[str, Any],
    *,
    args: argparse.Namespace,
    request_extra_body: dict[str, Any],
) -> dict[str, Any]:
    messages = [{"role": "user", "content": job["prompt"]}]
    token_budget = resolve_effective_max_tokens(args, messages)
    try:
        response = call_llm(
            messages,
            model=args.model,
            backend=args.llm_backend,
            reasoning_enabled=args.reasoning_enabled,
            openai_reasoning_effort=args.openai_reasoning_effort,
            temperature=float(job.get("temperature", args.temperature)),
            max_tokens=int(token_budget["effective_max_tokens"]),
            top_p=args.top_p,
            top_k=args.top_k,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            extra_body=request_extra_body,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("llm_response_missing_choices")
        choice = choices[0]
        raw_response = str(choice.message.content or "")
        finish_reason = str(getattr(choice, "finish_reason", "") or "")
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
        decisions, parse_error = parse_decisions(raw_response)
        if finish_reason == "length" and not parse_error:
            parse_error = "finish_reason_length"
    except Exception as error:
        raw_response = ""
        finish_reason = "error"
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        decisions = {}
        parse_error = f"llm_call_error: {type(error).__name__}: {str(error)[:500]}"

    return {
        **job,
        "raw_response": raw_response,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        **token_budget,
        "temperature": float(job.get("temperature", args.temperature)),
        "decisions": decisions,
        "parse_error": parse_error,
    }


def execute_batch_jobs(
    jobs: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    request_extra_body: dict[str, Any],
) -> list[dict[str, Any]]:
    if not jobs:
        return []

    max_workers = max(1, args.parallel_requests)
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    next_index = 0
    futures: dict[Future[dict[str, Any]], int] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while next_index < len(jobs) and len(futures) < max_workers:
            futures[
                executor.submit(
                    execute_batch_job,
                    jobs[next_index],
                    args=args,
                    request_extra_body=request_extra_body,
                )
            ] = next_index
            next_index += 1

        completed_count = 0
        progress = tqdm(total=len(jobs), desc="LLM batches", unit="batch")

        try:
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    job_index = futures.pop(future)
                    results[job_index] = future.result()
                    completed_count += 1
                    if progress is not None:
                        progress.update(1)
                    elif completed_count % max_workers == 0 or completed_count == len(jobs):
                        print(f"Completed LLM batches: {completed_count:,}/{len(jobs):,}")
                    if next_index < len(jobs):
                        futures[
                            executor.submit(
                                execute_batch_job,
                                jobs[next_index],
                                args=args,
                                request_extra_body=request_extra_body,
                            )
                        ] = next_index
                        next_index += 1
        finally:
            if progress is not None:
                progress.close()

    return [result for result in results if result is not None]


def main() -> None:
    args = parse_args()
    resolve_dataset_args(args)
    args.model = resolve_model(args)
    load_env_file(args.env_path)

    for path in [args.output_path, args.prediction_output_path, args.batch_output_path, args.summary_output_path]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    submission = load_submission_input(args.submission_path, input_top_k=args.input_top_k)
    query_columns = ["query_id", args.query_column]
    if args.rerank_mode in {
        "conservative-court-context",
        "conservative-no-court-first-court-confirm",
    }:
        query_columns.append(args.query_embedding_column)
    if args.gold_column:
        available_columns = pd.read_parquet(args.query_path).columns
        if (
            args.rerank_mode
            in {
                "conservative-court-context",
                "conservative-no-court-first-court-confirm",
            }
            and args.query_embedding_column not in available_columns
        ):
            raise ValueError(
                f"{args.query_path} does not contain query embedding column "
                f"{args.query_embedding_column!r}"
            )
        if args.gold_column in available_columns:
            query_columns.append(args.gold_column)
    queries = pd.read_parquet(args.query_path, columns=list(dict.fromkeys(query_columns)))
    laws = pd.read_parquet(args.law_path, columns=["citation", "source_book", "document"])
    laws["citation"] = laws["citation"].astype(str)
    citation_to_document = dict(zip(laws["citation"], laws["document"].astype(str)))
    citation_to_source_book = dict(zip(laws["citation"], laws["source_book"].astype(str)))
    query_to_text = dict(zip(queries["query_id"].astype(str), queries[args.query_column].astype(str)))
    selected, row_end = select_rows(submission, args.row_start, args.row_end)
    request_extra_body = build_request_extra_body(args)

    if args.rerank_mode in {
        "conservative-court-context",
        "conservative-no-court-first-court-confirm",
        "conservative-no-court-context",
    }:
        run_conservative_court_context_mode(
            args=args,
            submission=submission,
            queries=queries,
            selected=selected,
            row_end=row_end,
            citation_to_document=citation_to_document,
            citation_to_source_book=citation_to_source_book,
            query_to_text=query_to_text,
            request_extra_body=request_extra_body,
        )
        return

    raise ValueError(f"Unsupported rerank mode: {args.rerank_mode}")


def summarize(
    submission: pd.DataFrame,
    output: pd.DataFrame,
    predictions: pd.DataFrame,
    batches: pd.DataFrame,
    *,
    queries: pd.DataFrame,
    gold_column: str,
    score_art_only_gold: bool,
) -> pd.DataFrame:
    original_counts = submission["predicted_citations"].map(lambda value: len(split_refs(value)))
    final_counts = output["predicted_citations"].map(lambda value: len(split_refs(value)))
    parse_errors = predictions["parse_error"].fillna("").astype(str) if not predictions.empty else pd.Series([], dtype=str)
    length_batches = (
        int((batches["finish_reason"] == "length").sum())
        if not batches.empty and "finish_reason" in batches
        else 0
    )
    row = {
        "query_count": int(len(output)),
        "candidate_count": int(len(predictions)),
        "sent_to_llm_count": int(predictions["sent_to_llm"].sum()) if not predictions.empty else 0,
        "not_sent_to_llm_count": int((~predictions["sent_to_llm"]).sum()) if not predictions.empty else 0,
        "removed_count": int(predictions["removed"].sum()) if not predictions.empty else 0,
        "remove_rate_sent": (
            float(predictions.loc[predictions["sent_to_llm"], "removed"].mean())
            if not predictions.empty and predictions["sent_to_llm"].any()
            else 0.0
        ),
        "mean_original_count": float(original_counts.mean()),
        "mean_final_count": float(final_counts.mean()),
        "min_final_count": int(final_counts.min()) if len(final_counts) else 0,
        "max_final_count": int(final_counts.max()) if len(final_counts) else 0,
        "parse_error_rows": int(parse_errors.astype(bool).sum()),
        "regex_fallback_rows": int(parse_errors.str.contains("regex_fallback", regex=False).sum()),
        "hard_parse_error_rows": int(
            parse_errors.astype(bool).sum()
            - parse_errors.str.contains("regex_fallback", regex=False).sum()
        ),
        "length_finish_batches": length_batches,
    }
    if gold_column and gold_column in queries.columns:
        original_scores = score_against_gold(
            submission,
            queries,
            gold_column=gold_column,
            art_only=score_art_only_gold,
        )
        final_scores = score_against_gold(
            output,
            queries,
            gold_column=gold_column,
            art_only=score_art_only_gold,
        )
        removal_quality = score_removal_quality(
            predictions,
            queries,
            gold_column=gold_column,
            art_only=score_art_only_gold,
        )
        row.update(
            {
                "score_art_only_gold": bool(score_art_only_gold),
                "before_macro_f1": original_scores["macro_f1"],
                "after_macro_f1": final_scores["macro_f1"],
                "before_micro_f1": original_scores["micro_f1"],
                "after_micro_f1": final_scores["micro_f1"],
                "original_macro_f1": original_scores["macro_f1"],
                "macro_f1": final_scores["macro_f1"],
                "macro_f1_delta": final_scores["macro_f1"] - original_scores["macro_f1"],
                "original_micro_f1": original_scores["micro_f1"],
                "micro_f1": final_scores["micro_f1"],
                "micro_f1_delta": final_scores["micro_f1"] - original_scores["micro_f1"],
                "precision": final_scores["precision"],
                "recall": final_scores["recall"],
                "hits": final_scores["hits"],
                "gold_count": final_scores["gold_count"],
                "predicted_count": final_scores["predicted_count"],
                **removal_quality,
            }
        )
    return pd.DataFrame([row])


def score_removal_quality(
    predictions: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    gold_column: str,
    art_only: bool,
) -> dict[str, float]:
    if predictions.empty:
        return {
            "original_predicted_gold_count": 0,
            "original_predicted_noise_count": 0,
            "gold_removed_count": 0,
            "noise_removed_count": 0,
            "gold_removal_rate": 0.0,
            "noise_removal_rate": 0.0,
            "reranking_quality": 0.0,
            "llm_candidate_gold_count": 0,
            "llm_candidate_noise_count": 0,
            "llm_gold_removed_count": 0,
            "llm_noise_removed_count": 0,
            "llm_gold_removal_rate": 0.0,
            "llm_noise_removal_rate": 0.0,
            "llm_reranking_quality": 0.0,
        }

    gold_by_query = {
        str(row.query_id): set(split_gold_refs(getattr(row, gold_column), art_only=art_only))
        for row in queries.itertuples(index=False)
    }
    candidate_gold_count = 0
    candidate_noise_count = 0
    gold_removed_count = 0
    noise_removed_count = 0
    llm_candidate_gold_count = 0
    llm_candidate_noise_count = 0
    llm_gold_removed_count = 0
    llm_noise_removed_count = 0

    for row in predictions.itertuples(index=False):
        query_id = str(row.query_id)
        if query_id not in gold_by_query:
            continue
        citation = str(row.citation)
        removed = bool(row.removed)
        sent_to_llm = bool(row.sent_to_llm)
        if citation in gold_by_query[query_id]:
            candidate_gold_count += 1
            if sent_to_llm:
                llm_candidate_gold_count += 1
            if removed:
                gold_removed_count += 1
                if sent_to_llm:
                    llm_gold_removed_count += 1
        else:
            candidate_noise_count += 1
            if sent_to_llm:
                llm_candidate_noise_count += 1
            if removed:
                noise_removed_count += 1
                if sent_to_llm:
                    llm_noise_removed_count += 1

    gold_removal_rate = (
        gold_removed_count / candidate_gold_count if candidate_gold_count else 0.0
    )
    noise_removal_rate = (
        noise_removed_count / candidate_noise_count if candidate_noise_count else 0.0
    )
    llm_gold_removal_rate = (
        llm_gold_removed_count / llm_candidate_gold_count
        if llm_candidate_gold_count
        else 0.0
    )
    llm_noise_removal_rate = (
        llm_noise_removed_count / llm_candidate_noise_count
        if llm_candidate_noise_count
        else 0.0
    )
    return {
        "original_predicted_gold_count": candidate_gold_count,
        "original_predicted_noise_count": candidate_noise_count,
        "gold_removed_count": gold_removed_count,
        "noise_removed_count": noise_removed_count,
        "gold_removal_rate": gold_removal_rate,
        "noise_removal_rate": noise_removal_rate,
        "reranking_quality": noise_removal_rate - 3 * gold_removal_rate,
        "llm_candidate_gold_count": llm_candidate_gold_count,
        "llm_candidate_noise_count": llm_candidate_noise_count,
        "llm_gold_removed_count": llm_gold_removed_count,
        "llm_noise_removed_count": llm_noise_removed_count,
        "llm_gold_removal_rate": llm_gold_removal_rate,
        "llm_noise_removal_rate": llm_noise_removal_rate,
        "llm_reranking_quality": llm_noise_removal_rate - 3 * llm_gold_removal_rate,
    }


def score_against_gold(
    predictions: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    gold_column: str,
    art_only: bool,
) -> dict[str, float]:
    gold_by_query = {
        str(row.query_id): set(split_gold_refs(getattr(row, gold_column), art_only=art_only))
        for row in queries.itertuples(index=False)
    }
    total_hits = 0
    total_predicted = 0
    total_gold = 0
    per_query_f1: list[float] = []
    for row in predictions.itertuples(index=False):
        query_id = str(row.query_id)
        if query_id not in gold_by_query:
            continue
        predicted = set(split_refs(row.predicted_citations))
        gold = gold_by_query[query_id]
        hits = len(predicted & gold)
        total_hits += hits
        total_predicted += len(predicted)
        total_gold += len(gold)
        precision = hits / len(predicted) if predicted else 0.0
        recall = hits / len(gold) if gold else 0.0
        per_query_f1.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    precision = total_hits / total_predicted if total_predicted else 0.0
    recall = total_hits / total_gold if total_gold else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    macro_f1 = sum(per_query_f1) / len(per_query_f1) if per_query_f1 else 0.0
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "precision": precision,
        "recall": recall,
        "hits": float(total_hits),
        "gold_count": float(total_gold),
        "predicted_count": float(total_predicted),
    }


def split_gold_refs(value: object, *, art_only: bool) -> list[str]:
    refs = split_refs(value)
    if not art_only:
        return refs
    return [ref for ref in refs if ref.startswith("Art.")]


if __name__ == "__main__":
    main()
