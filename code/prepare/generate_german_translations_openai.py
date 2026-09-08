#!/usr/bin/env python3
"""Translate legal queries to Swiss legal German with an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import openai
import pandas as pd
from dotenv import load_dotenv

try:
    from generate_decomposed_queries_openai import (
        HarmonyRenderer,
        ManagedVLLMServer,
        _one_harmony_request,
        model_supports_temperature,
        parse_json_dict_arg,
        resolve_model_family,
    )
except ModuleNotFoundError:
    from src.generate_decomposed_queries_openai import (
        HarmonyRenderer,
        ManagedVLLMServer,
        _one_harmony_request,
        model_supports_temperature,
        parse_json_dict_arg,
        resolve_model_family,
    )

load_dotenv()

DEFAULT_SYSTEM_PROMPT = (
    "You translate English legal queries into formal Swiss legal German "
    "(Schriftdeutsch). Return only the German translation."
)

DEFAULT_USER_PROMPT_TEMPLATE = """
Translate the following English legal query into formal Swiss legal German.
Preserve the full meaning, names, dates, legal claims, and question structure.
Return only the German translation as plain text.
Do not return JSON, lists, headings, markdown, or commentary.

English legal query:
{{query}}
""".strip()


def load_translation_prompt(path: str | Path, query: str) -> str:
    """Build the translation prompt from the embedded template or an override file."""
    if not path:
        return DEFAULT_USER_PROMPT_TEMPLATE.replace("{{query}}", query.strip())

    text = Path(path).read_text(encoding="utf-8")
    pattern = re.compile(
        r"(\*\*English Legal Query:\*\*\s*)(.*?)(\n\s*---)",
        flags=re.DOTALL,
    )
    if pattern.search(text):
        text = pattern.sub(
            lambda match: f"{match.group(1)}{query.strip()}{match.group(3)}",
            text,
            count=1,
        )
    else:
        text = f"{text.rstrip()}\n\n**English Legal Query:**\n{query.strip()}\n"

    if "GERMAN LEGAL QUERY:" in text:
        text = text.split("GERMAN LEGAL QUERY:", 1)[0].rstrip()

    return (
        text.rstrip()
        + "\n\nIMPORTANT OUTPUT RULE:\n"
        + "Return only the final German translation text. Do not include the "
        + "terminology mapping table, headings, markdown, commentary, or the "
        + "'GERMAN LEGAL QUERY:' label.\n"
    )


def clean_translation(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise RuntimeError("Empty model response.")

    fence = re.search(r"```(?:[a-zA-Z]+)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    for marker in (
        "GERMAN LEGAL QUERY:",
        "Final German Translation:",
        "Final German translation:",
        "Deutsche Übersetzung:",
        "Deutsche Uebersetzung:",
    ):
        if marker in text:
            text = text.split(marker)[-1].strip()

    text = re.sub(r"^\s*[-#*]+\s*", "", text).strip()
    if not text:
        raise RuntimeError("Empty cleaned translation.")
    if looks_like_decomposition_output(text):
        raise RuntimeError("Model returned decomposition JSON instead of a translation.")
    return text


def validate_translation_for_query(query: str, translation: str) -> None:
    query_len = len((query or "").strip())
    translation_len = len((translation or "").strip())
    if not translation_len:
        raise RuntimeError("Empty translation.")
    if looks_like_decomposition_output(translation):
        raise RuntimeError("Translation is decomposition JSON.")
    if query_len > 0 and translation_len > 2 * query_len:
        raise RuntimeError(
            "Translation is suspiciously long "
            f"({translation_len} chars > 2x query length {query_len}); "
            "likely contains unfinished thinking."
        )


def is_valid_cached_translation(query: str, translation: str) -> bool:
    try:
        validate_translation_for_query(query, translation)
    except RuntimeError:
        return False
    return True


def looks_like_decomposition_output(text: str) -> bool:
    lowered = text.lower()
    if (
        "sub_issues" in lowered
        or "search_query_de" in lowered
        or '"dimension"' in lowered
    ):
        return True

    try:
        parsed = json.loads(text)
    except Exception:
        return False

    if isinstance(parsed, str):
        return looks_like_decomposition_output(parsed)
    if not isinstance(parsed, dict):
        return False
    if "sub_issues" in parsed:
        return True
    values = json.dumps(parsed, ensure_ascii=False).lower()
    return "search_query_de" in values or '"dimension"' in values


def _pop_float_config(
    config: dict[str, object], key: str, current: float | None
) -> float | None:
    if key not in config:
        return current
    value = config.pop(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as err:
        raise SystemExit(f"--generation-config {key} must be numeric.") from err


def _pop_int_config(
    config: dict[str, object], key: str, current: int | None
) -> int | None:
    if key not in config:
        return current
    value = config.pop(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise SystemExit(f"--generation-config {key} must be an integer.") from err


def call_translation(
    *,
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    timeout_s: float,
    max_retries: int,
    retry_base_s: float,
    max_output_tokens: int,
    use_harmony: bool,
    harmony_renderer: HarmonyRenderer | None,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    extra_generation_body: dict[str, object],
    query: str,
    label: str,
) -> tuple[str, str]:
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if max_output_tokens > 0:
        kwargs["max_tokens"] = max_output_tokens
    if model_supports_temperature(model):
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty

    extra_body: dict[str, object] = {}
    if top_k is not None:
        extra_body["top_k"] = top_k
    if min_p is not None:
        extra_body["min_p"] = min_p
    if repetition_penalty is not None:
        extra_body["repetition_penalty"] = repetition_penalty
    if extra_generation_body:
        extra_body.update(extra_generation_body)
    if extra_body:
        kwargs["extra_body"] = extra_body

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if use_harmony:
                if harmony_renderer is None:
                    raise RuntimeError("Harmony renderer was not initialized.")
                raw = _one_harmony_request(
                    client=client,
                    renderer=harmony_renderer,
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            else:
                response = client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                raw = message.content or getattr(message, "reasoning", "") or ""
            translation = clean_translation(raw)
            validate_translation_for_query(query, translation)
            return translation, raw
        except (
            openai.APIConnectionError,
            openai.APITimeoutError,
            RuntimeError,
            ValueError,
        ) as err:
            last_err = err
            if attempt < max_retries:
                print(
                    f"[{label}] retry {attempt}/{max_retries} after "
                    f"{type(err).__name__}: {err}",
                    flush=True,
                )
                time.sleep(retry_base_s * attempt)
                continue
            raise
        except openai.APIStatusError as err:
            last_err = err
            if err.status_code in (408, 409, 429, 500, 502, 503, 504) and attempt < max_retries:
                print(
                    f"[{label}] retry {attempt}/{max_retries} after "
                    f"HTTP {err.status_code}: {err}",
                    flush=True,
                )
                time.sleep(retry_base_s * attempt)
                continue
            raise

    raise RuntimeError(f"Request failed after {max_retries} attempts: {last_err}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", default="llm-agentic-legal-information-retrieval/test.csv")
    p.add_argument("--output-parquet", default="data-processed/test_german_translation.parquet")
    p.add_argument("--query-col", default="query")
    p.add_argument("--query-id-col", default="query_id")
    p.add_argument(
        "--prompt-file",
        default="",
        help="Optional markdown prompt file override. Empty uses the embedded default prompt.",
    )
    p.add_argument("--model", default="nvidia/openai/gpt-oss-20b")
    p.add_argument("--model-family", choices=["auto", "gpt-oss", "qwen", "chat"], default="auto")
    p.add_argument("--base-url", default="http://localhost:20128/v1")
    p.add_argument("--api-key-env", default="VLLM_API_KEY")
    p.add_argument("--max-parallel", type=int, default=40)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--presence-penalty", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--generation-config", default="")
    p.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=None,
        help=(
            "Set extra_body.chat_template_kwargs.enable_thinking=true for "
            "OpenAI-compatible chat backends that support it."
        ),
    )
    p.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help=(
            "Set extra_body.chat_template_kwargs.enable_thinking=false for "
            "OpenAI-compatible chat backends that support it."
        ),
    )
    p.add_argument("--max-output-tokens", type=int, default=8192)
    p.add_argument("--timeout-seconds", type=float, default=180.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--retry-base-seconds", type=float, default=2.0)
    p.add_argument("--start-index", type=int, default=1, help="1-based row index.")
    p.add_argument("--end-index", type=int, default=0, help="0 = all rows.")
    p.add_argument("--overwrite-existing", action="store_true")
    p.add_argument("--start-vllm", action="store_true")
    p.add_argument("--use-harmony", action="store_true")
    p.add_argument("--disable-harmony", action="store_true")
    p.add_argument("--vllm-model-path", default="")
    p.add_argument("--served-model-name", default="")
    p.add_argument("--vllm-host", default="localhost")
    p.add_argument("--vllm-port", type=int, default=20128)
    p.add_argument("--server-timeout", type=int, default=900)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--vllm-max-num-seqs", type=int, default=64)
    p.add_argument("--vllm-async-scheduling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--speculative-config", default="")
    p.add_argument("--vllm-extra-args", default="")
    p.add_argument("--vllm-log-file", default="vllm_translation_server.log")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.start_index <= 0:
        raise SystemExit("--start-index must be >= 1.")
    if args.max_parallel <= 0:
        raise SystemExit("--max-parallel must be > 0.")
    if args.max_retries <= 0:
        raise SystemExit("--max-retries must be > 0.")

    generation_config = parse_json_dict_arg(args.generation_config, "--generation-config")
    args.temperature = _pop_float_config(generation_config, "temperature", args.temperature)
    args.top_p = _pop_float_config(generation_config, "top_p", args.top_p)
    args.top_k = _pop_int_config(generation_config, "top_k", args.top_k)
    args.min_p = _pop_float_config(generation_config, "min_p", args.min_p)
    args.presence_penalty = _pop_float_config(
        generation_config, "presence_penalty", args.presence_penalty
    )
    args.repetition_penalty = _pop_float_config(
        generation_config, "repetition_penalty", args.repetition_penalty
    )
    if args.enable_thinking is not None:
        chat_template_kwargs = generation_config.get("chat_template_kwargs", {})
        if chat_template_kwargs is None:
            chat_template_kwargs = {}
        if not isinstance(chat_template_kwargs, dict):
            raise SystemExit(
                "--generation-config chat_template_kwargs must be a JSON object."
            )
        chat_template_kwargs = dict(chat_template_kwargs)
        chat_template_kwargs["enable_thinking"] = bool(args.enable_thinking)
        generation_config["chat_template_kwargs"] = chat_template_kwargs

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        if "openai.com" in args.base_url:
            raise SystemExit(f"Missing API key. Set environment variable {args.api_key_env}.")
        api_key = "vllm-local"

    server: ManagedVLLMServer | None = None
    if args.start_vllm:
        base_url_was_set = any(
            x == "--base-url" or x.startswith("--base-url=") for x in sys.argv[1:]
        )
        if not base_url_was_set:
            args.base_url = f"http://{args.vllm_host}:{args.vllm_port}/v1"
        args.served_model_name = args.served_model_name or args.model
        server = ManagedVLLMServer(args, api_key=api_key)
        args.base_url = server.start()

    model_family = resolve_model_family(args.model, args.vllm_model_path, args.model_family)
    use_harmony = bool(
        not args.disable_harmony
        and model_family == "gpt-oss"
        and (args.use_harmony or args.start_vllm)
    )
    harmony_renderer = HarmonyRenderer("medium") if use_harmony else None
    request_model = args.served_model_name or args.model

    out_path = Path(args.output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(args.input_csv).fillna("")
    for col in (args.query_id_col, args.query_col):
        if col not in input_df.columns:
            raise SystemExit(f"Missing column in input CSV: {col}")

    n = len(input_df)
    start_i = args.start_index - 1
    end_i = n if args.end_index <= 0 else min(args.end_index, n)

    existing_raw: dict[str, str] = {}
    if out_path.exists() and not args.overwrite_existing:
        old = pd.read_parquet(out_path).fillna("")
        if {"query_id", "query", "german_translation"}.issubset(old.columns):
            existing_raw = {
                qid: value
                for qid, query, value in zip(
                    old["query_id"].astype(str),
                    old["query"].astype(str),
                    old["german_translation"].astype(str),
                )
                if is_valid_cached_translation(query, value)
            }
        elif {"query_id", "german_translation"}.issubset(old.columns):
            existing_raw = {
                qid: value
                for qid, value in zip(
                    old["query_id"].astype(str),
                    old["german_translation"].astype(str),
                )
                if value.strip() and not looks_like_decomposition_output(value)
            }

    rows: list[dict[str, str]] = []
    lock = threading.Lock()
    counters = {"written": 0, "skipped": 0, "failed": 0}

    worklist: list[tuple[int, str, str]] = []
    for idx in range(start_i, end_i):
        row = input_df.iloc[idx]
        qid = str(row[args.query_id_col])
        query = str(row[args.query_col]).strip()
        if not query:
            counters["failed"] += 1
            print(f"[{idx + 1}/{end_i}] FAIL {qid}: empty query", flush=True)
            continue
        if qid in existing_raw and is_valid_cached_translation(query, existing_raw[qid]):
            rows.append(
                {
                    "query_id": qid,
                    "query": query,
                    "german_translation": existing_raw[qid],
                }
            )
            counters["skipped"] += 1
            continue
        worklist.append((idx, qid, query))

    print(f"Input:      {args.input_csv} ({n} rows)")
    print(f"Output:     {out_path}")
    print(f"Model:      {request_model}")
    print(f"Base URL:   {args.base_url}")
    print(f"Mode:       {'harmony completions' if use_harmony else 'chat completions'}")
    if generation_config:
        print(f"Extra body: {generation_config}")
    print(f"Range:      rows {start_i + 1}..{end_i}")
    print(f"Work:       {len(worklist)} rows ({counters['skipped']} skipped)")
    if worklist:
        sample_idx, sample_qid, sample_query = worklist[0]
        sample_prompt = load_translation_prompt(args.prompt_file, sample_query)
        print("\n=== Sample API Input ===")
        print(f"Row:        {sample_idx + 1}")
        print(f"Query ID:   {sample_qid}")
        print("\n[SYSTEM]")
        print(DEFAULT_SYSTEM_PROMPT)
        print("\n[USER]")
        print(sample_prompt)
        print("=== End Sample API Input ===\n")
    else:
        print("Sample API Input: no rows will be sent to the API.")

    def process_row(item: tuple[int, str, str]) -> None:
        idx, qid, query = item
        try:
            prompt = load_translation_prompt(args.prompt_file, query)
            translation, _raw = call_translation(
                prompt=prompt,
                api_key=api_key,
                base_url=args.base_url,
                model=request_model,
                temperature=args.temperature,
                timeout_s=args.timeout_seconds,
                max_retries=args.max_retries,
                retry_base_s=args.retry_base_seconds,
                max_output_tokens=args.max_output_tokens,
                use_harmony=use_harmony,
                harmony_renderer=harmony_renderer,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                repetition_penalty=args.repetition_penalty,
                extra_generation_body=generation_config,
                query=query,
                label=qid,
            )
            with lock:
                rows.append(
                    {
                        "query_id": qid,
                        "query": query,
                        "german_translation": translation,
                    }
                )
                counters["written"] += 1
                pd.DataFrame(rows).sort_values("query_id").to_parquet(out_path, index=False)
                print(f"[{idx + 1}/{end_i}] OK   {qid}", flush=True)
        except Exception as err:
            with lock:
                counters["failed"] += 1
                print(f"[{idx + 1}/{end_i}] FAIL {qid}: {err}", flush=True)

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = [ex.submit(process_row, item) for item in worklist]
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Translating",
                unit="row",
            )
            for fut in iterator:
                fut.result()
        except ImportError:
            for done, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                fut.result()
                print(f"Progress: {done}/{len(futures)}", flush=True)

    final_df = pd.DataFrame(rows, columns=["query_id", "query", "german_translation"])
    if not final_df.empty:
        final_df = final_df.drop_duplicates("query_id", keep="last").sort_values("query_id")
    final_df.to_parquet(out_path, index=False)

    elapsed = time.time() - started
    print(
        f"Done in {elapsed:.1f}s - written: {counters['written']}, "
        f"skipped: {counters['skipped']}, failed: {counters['failed']}"
    )
    print(f"Saved: {out_path}")

    if server is not None:
        server.stop()


if __name__ == "__main__":
    main()
