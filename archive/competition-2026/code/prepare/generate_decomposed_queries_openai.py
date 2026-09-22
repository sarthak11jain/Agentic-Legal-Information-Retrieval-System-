#!/usr/bin/env python3
"""
Generate decomposed German search queries for each legal query using an
OpenAI-compatible Chat Completions endpoint (defaults to a local vLLM server
running ``nvidia/openai/gpt-oss-20b``).

Reads queries from the input CSV or parquet file, decomposes each into
sub-issues using the 6-dimension framework, and stores the result as a new column
``decomposed_queries_german``. Queries are processed **in parallel** with a
ThreadPoolExecutor, mirroring ``src/run_llm_reranking_openai.py``.

The column stores a JSON string with the full sub_issues array. The raw model
text is preserved in ``decomposed_queries_raw``. Output schema is unchanged so
the downstream parquet/embedding pipeline is unaffected.
"""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
import polars as pl

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

QUERY_PLACEHOLDER = "{{query}}"

DEFAULT_SYSTEM_PROMPT = """
You are a senior Swiss legal researcher who has argued cases before the Federal Supreme Court (Bundesgericht). Your task is to decompose a complex legal query into focused sub-issues and generate a precise German search query for each.

You will receive a legal query in English describing a factual situation with one or more legal questions. The retrieval corpus consists of Swiss federal law articles written in German -- short, declarative legal rules (typically 1-4 sentences each).

Your decomposed search queries will be used to generate embeddings for semantic search against this corpus. The more precisely each query captures the legal concept in the vocabulary of Swiss statute text, the better the retrieval will work.

CRITICAL: You must go far beyond the explicit questions in the query. A Swiss court decision cites not just the articles that answer the question asked, but also:
- Articles that DEFINE the legal concepts used (e.g. what "disability" or "official" means)
- Articles that establish WHICH LAW APPLIES to the situation
- Articles governing the PROCEDURAL PATHWAY (appeal rights, deadlines, standing)
- Articles about CONSEQUENCES (sentencing, costs, compensation) even if the query only asks about liability
- GENERAL PRINCIPLES (burden of proof, good faith, abuse of rights) when facts are disputed
- CONSTITUTIONAL provisions when fundamental rights are at stake (detention, criminal accusation, right to be heard)
- The FEDERAL APPEAL DEADLINE (almost every Swiss court decision cites this)

Think like a judge writing a full decision, not like a student answering a homework question.
""".strip()

DEFAULT_USER_PROMPT_TEMPLATE = """
## QUERY
{{query}}

---

## INSTRUCTIONS

Decompose this query by working through SIX mandatory dimensions. Each dimension may produce 0-3 sub-issues depending on the query. The total should be between 6 and 15 sub-issues.

### Dimension A: Explicit Legal Questions
Read the query's explicit questions (usually at the end, phrased as "Can...", "May...", "Is..."). For each distinct question, create a sub-issue.

### Dimension B: Legal Definitions
Identify every key legal concept in the query that has a statutory definition in Swiss law. Courts always cite the definition article even when the concept seems obvious. Examples:
- If the query mentions disability -> there is an article defining "Invalidität"
- If the query involves a public official -> there is an article defining "Beamte"
- If the query mentions work incapacity -> there is an article defining "Arbeitsunfähigkeit"
- If the query involves capacity to act -> there is an article defining "Urteilsfähigkeit"

Create a sub-issue for each key legal term that has a statutory definition.

### Dimension C: Legal Characterization
When the facts could fall under more than one legal framework, courts must determine which one applies. Examples:
- Is the arrangement a work contract (Werkvertrag), mandate (Auftrag), gift (Schenkung), or employment (Arbeitsvertrag)?
- Is the transfer inter vivos or mortis causa?
- Is it theft, robbery, or misappropriation?
- Is the act intentional, negligent, or innocent?

If the facts are ambiguous about which legal framework applies, create a sub-issue for each plausible characterization.

### Dimension D: Consequences & Remedies
Courts always address what follows from their legal conclusion, even when the query only asks "is there liability?" Think about:
- Criminal cases: sentencing (suspended sentence, probation, concurrent penalties), duty to state sentencing reasons
- Civil cases: damages calculation, specific performance, injunctions
- All cases: who bears the costs of proceedings, compensation for wrongful prosecution/detention
- Family cases: maintenance calculation, enforcement mechanisms, child protection measures

Create sub-issues for the consequences that logically follow from the query's scenario.

### Dimension E: Procedural Pathway
Every Swiss court decision cites the articles that establish its own jurisdiction and the procedural rules it follows. Think about:
- Which court has jurisdiction? (cantonal, federal criminal court, insurance court)
- Who has standing to appeal? What is the appeal deadline?
- What are the formal requirements for the appeal brief?
- Is there a federal appeal to the Bundesgericht? Under which provision (BGG, ATSG)?
- Court organization articles (StBOG for federal criminal matters)

For criminal matters, the procedural stack typically includes: standing to appeal, appealable decisions, appeal deadline, exchange of briefs, who can challenge detention, and court organization.

For social insurance matters: the specific appeal path (ATSG -> cantonal insurance court -> BGG).

IMPORTANT: Almost every Swiss case cites the federal appeal deadline provision. Always generate a sub-issue for this.

### Dimension F: General Principles & Constitutional Rights
When the facts involve any of the following, generate a sub-issue:
- Disputed or conflicting evidence -> burden of proof (Beweislast)
- One party exploiting a technicality or acting in bad faith -> abuse of rights (Rechtsmissbrauch), good faith (Treu und Glauben)
- Judicial discretion needed -> equitable assessment (Ermessen, Billigkeit)
- Deprivation of liberty -> right to be heard (rechtliches Gehör), proportionality
- Criminal accusation -> presumption of innocence, right to be informed of charges
- Procedural fairness complaints -> fundamental procedural principles

---

## FOR EACH SUB-ISSUE, GENERATE:

A focused German search query (`search_query_de`) -- a concise query (10-25 words) that:
- Uses precise Swiss legal terminology (the vocabulary found in Bundesgesetz text)
- Captures the core legal concept with enough specificity to distinguish it from other provisions
- Does NOT include article numbers or statute abbreviations
- Is phrased in the style of Swiss statute language, not as a question

---

## OUTPUT FORMAT

Return a valid JSON object with this structure:

{
  "sub_issues": [
    {
      "id": 1,
      "dimension": "A|B|C|D|E|F",
      "description_en": "Brief English description of the sub-issue",
      "search_query_de": "Focused German search query in statute vocabulary (10-25 words)"
    }
  ]
}

Return ONLY the JSON object. No explanations, no markdown fences, no preamble.
""".strip()


def read_input_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported input file extension for --input: {path}")


def default_output_csv(input_path: str, output_csv: str) -> str:
    if output_csv:
        return output_csv

    input_file = Path(input_path)
    if input_file.suffix.lower() == ".csv":
        return str(input_file)

    return str(input_file.with_name(f"{input_file.stem}_decomposed.csv"))


def output_subquery_parquet_path(output_csv: str) -> str:
    return str(Path(output_csv).with_suffix(".parquet"))


def _connect_host(host: str) -> str:
    return "localhost" if host in {"0.0.0.0", "::"} else host


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((_connect_host(host), port)) == 0


def _wait_for_openai_server(
    *,
    base_url: str,
    api_key: str,
    timeout_s: int,
    proc: subprocess.Popen | None = None,
) -> None:
    client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=10.0)
    health_url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    started = time.time()
    last_err: Exception | None = None

    while time.time() - started <= timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server exited before ready (code {proc.returncode})."
            )

        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError) as err:
            last_err = err

        try:
            client.models.list()
            return
        except Exception as err:
            last_err = err

        time.sleep(2)

    raise TimeoutError(
        f"vLLM server was not ready after {timeout_s}s at {base_url}: {last_err}"
    )


class ManagedVLLMServer:
    """Own one optional vLLM subprocess for gpt-oss decomposition."""

    def __init__(self, args: argparse.Namespace, api_key: str) -> None:
        self.args = args
        self.api_key = api_key
        self.proc: subprocess.Popen | None = None
        self.log_fh = None
        self.started = False
        self.base_url = args.base_url or f"http://{args.vllm_host}:{args.vllm_port}/v1"

    def command(self) -> list[str]:
        model_path = self.args.vllm_model_path or self.args.model
        served_model = self.args.served_model_name or self.args.model
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--served-model-name",
            served_model,
            "--tensor-parallel-size",
            str(self.args.tensor_parallel_size),
            "--max-num-seqs",
            str(self.args.vllm_max_num_seqs),
            "--gpu-memory-utilization",
            str(self.args.gpu_memory_utilization),
            "--host",
            self.args.vllm_host,
            "--port",
            str(self.args.vllm_port),
            "--dtype",
            self.args.dtype,
            "--max-model-len",
            str(self.args.max_model_len),
            "--disable-log-stats",
            "--enable-prefix-caching",
            "--trust-remote-code",
            "--api-key",
            self.api_key,
        ]
        if self.args.kv_cache_dtype:
            cmd += ["--kv-cache-dtype", self.args.kv_cache_dtype]
        if self.args.vllm_async_scheduling:
            cmd.append("--async-scheduling")
        if self.args.speculative_config:
            cmd += ["--speculative-config", self.args.speculative_config]
        if self.args.vllm_extra_args:
            cmd += shlex.split(self.args.vllm_extra_args)
        return cmd

    def start(self) -> str:
        if _port_is_open(self.args.vllm_host, self.args.vllm_port):
            print(
                f"vLLM port {self.args.vllm_port} is already open; "
                f"attaching to {self.base_url}",
                flush=True,
            )
            _wait_for_openai_server(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout_s=self.args.server_timeout,
            )
            return self.base_url

        cmd = self.command()
        print(f"Starting vLLM: {' '.join(cmd)}", flush=True)
        if self.args.vllm_log_file:
            self.log_fh = open(self.args.vllm_log_file, "a", encoding="utf-8")
            stdout = self.log_fh
        else:
            stdout = subprocess.DEVNULL

        env_vars = os.environ.copy()
        env_vars["LD_LIBRARY_PATH"]="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/lib64"
        self.proc = subprocess.Popen(
            cmd,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env_vars,
        )
        self.started = True
        atexit.register(self.stop)
        _wait_for_openai_server(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout_s=self.args.server_timeout,
            proc=self.proc,
        )
        print(f"vLLM ready: {self.base_url}", flush=True)
        return self.base_url

    def stop(self) -> None:
        if not self.started or self.proc is None:
            return
        self.started = False
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        if self.log_fh is not None:
            self.log_fh.close()
            self.log_fh = None
        print("vLLM server stopped.", flush=True)


def load_prompts(path: str) -> tuple[str, str]:
    if not path:
        return DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT_TEMPLATE

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.findall(r"```\n(.*?)```", content, re.DOTALL)
    if len(blocks) < 2:
        raise RuntimeError(
            f"Expected at least 2 code-fenced blocks in {path}, found {len(blocks)}. "
            "Need a System Prompt block and a User Prompt Template block."
        )

    system_prompt = blocks[0].strip()
    user_prompt_template = blocks[1].strip()

    if QUERY_PLACEHOLDER not in user_prompt_template:
        raise RuntimeError(
            f"User prompt template must contain {QUERY_PLACEHOLDER} placeholder."
        )

    return system_prompt, user_prompt_template


def parse_json_response(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        raise RuntimeError("Empty model response.")

    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: extract first JSON object substring (gpt-oss is a reasoning
    # model and may wrap the JSON in prose).
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        obj2 = json.loads(text[start : end + 1])
        if isinstance(obj2, dict):
            return obj2
    print(raw)
    raise RuntimeError("Model response is not valid JSON object.")


def parse_json_dict_arg(raw: str, arg_name: str) -> dict[str, object]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as err:
        raise SystemExit(f"{arg_name} must be valid JSON dict: {err}") from err
    if not isinstance(obj, dict):
        raise SystemExit(f"{arg_name} must be a JSON object/dict.")
    return obj


_STATUTE_ABBREVS = [
    "IPRG", "OR", "ZGB", "StPO", "BGG", "ATSG", "StGB", "SchKG", "BV", "VwVG",
    "ZPO", "IVG", "UVG", "SVG", "OG", "StBOG", "DSG", "URG", "KG", "FusG",
    "VVG", "AHVG", "BVG", "MWSTG", "ELG", "EOG", "FamZG", "PartG", "BGFA",
]
_ART_RE = re.compile(r"\bArt\.?\s*\d+\s*[a-zA-Z]?\b", re.IGNORECASE)
_ABBR_RE = re.compile(r"\b(?:%s)\b" % "|".join(_STATUTE_ABBREVS))
_SUBDIV_RE = re.compile(
    r"\b(?:Abs|lit|Ziff|Ziffer|Bst|Buchstabe)\b\.?", re.IGNORECASE
)
_DIGIT_RE = re.compile(r"\d+")
_KEEP_RE = re.compile(r"[^0-9A-Za-zÄÖÜäöüß \-]")
_WS_RE = re.compile(r"\s+")


def normalize_search_query(s: str) -> str:
    """Deterministic Swiss-orthography + citation-leak scrub for one
    ``search_query_de``. Idempotent; safe to run repeatedly.

    Guarantees Swiss spelling ("ss", never "ß"), removes leaked article
    numbers / statute abbreviations / subdivision tokens / stray digits,
    strips soft-hyphens and exotic dashes, and drops residual punctuation.
    Does NOT split CamelCase or invented compounds — that is the prompt's
    job; this is purely an orthography safety net.
    """
    if not s:
        return s
    s = s.replace("ß", "ss").replace("ẞ", "SS")          # Swiss spelling
    s = s.replace("­", "")                          # soft hyphen
    s = (
        s.replace("‑", "-")                         # non-breaking hyphen
        .replace("–", " ")                          # en dash
        .replace("—", " ")                          # em dash
    )
    s = _ART_RE.sub(" ", s)        # "Art. 46", "Art 12a"
    s = _ABBR_RE.sub(" ", s)       # IPRG, OR, ZGB, …
    s = _SUBDIV_RE.sub(" ", s)     # Abs. lit. Ziff.
    s = _DIGIT_RE.sub(" ", s)      # stray numbers
    s = _KEEP_RE.sub(" ", s)       # drop residual punctuation
    s = re.sub(r"(?:^|\s)-+(?:\s|$)", " ", s)  # orphan hyphens
    s = _WS_RE.sub(" ", s).strip()
    return s


def model_supports_temperature(model: str) -> bool:
    m = (model or "").strip().lower()
    # Real OpenAI gpt-5* models reject the temperature parameter.
    # Local vLLM models always accept it.
    return not m.startswith("gpt-5")


def resolve_model_family(model: str, model_path: str, requested: str) -> str:
    if requested != "auto":
        return requested
    text = f"{model} {model_path}".lower()
    if "gpt-oss" in text:
        return "gpt-oss"
    if "qwen" in text:
        return "qwen"
    return "chat"


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


def _one_request(client: openai.OpenAI, kwargs: dict) -> str:
    """Perform a single Chat Completions request and return the text."""
    response = client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
    # print(response)
    message = response.choices[0].message
    content = message.content or ""
    if not content:
        # Reasoning models may put the answer in the reasoning field
        content = getattr(message, "reasoning", "") or ""
    return content


class HarmonyRenderer:
    """Render gpt-oss prompts with openai_harmony for vLLM completions."""

    def __init__(self, reasoning_effort: str) -> None:
        try:
            from openai_harmony import (
                Conversation,
                HarmonyEncodingName,
                Message,
                ReasoningEffort,
                Role,
                SystemContent,
                load_harmony_encoding,
            )
        except ImportError as err:
            raise RuntimeError(
                "openai_harmony is required for --use-harmony / local gpt-oss."
            ) from err

        effort_name = (reasoning_effort or "medium").strip().upper()
        effort = getattr(ReasoningEffort, effort_name, None)
        if effort is None:
            raise SystemExit(
                "--reasoning-effort must be one of low, medium, high when "
                "using Harmony."
            )

        self.Conversation = Conversation
        self.Message = Message
        self.Role = Role
        self.SystemContent = SystemContent
        self.effort = effort
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

    def render(self, system_prompt: str, user_prompt: str) -> list[int]:
        system_content = (
            self.SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=self.effort)
        )
        messages = [
            self.Message.from_role_and_content(self.Role.SYSTEM, system_content),
            self.Message.from_role_and_content(self.Role.USER, user_prompt),
        ]
        conversation = self.Conversation.from_messages(messages)
        return self.encoding.render_conversation_for_completion(
            conversation, self.Role.ASSISTANT
        )


def _one_harmony_request(
    *,
    client: openai.OpenAI,
    renderer: HarmonyRenderer,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    """Perform one vLLM completions request using Harmony token IDs."""
    prompt_ids = renderer.render(system_prompt, user_prompt)
    kwargs: dict[str, object] = {
        "model": model,
        "prompt": prompt_ids,
        "extra_body": {"stop_token_ids": renderer.stop_token_ids},
    }
    if max_output_tokens > 0:
        kwargs["max_tokens"] = max_output_tokens
    if model_supports_temperature(model):
        kwargs["temperature"] = temperature
    response = client.completions.create(**kwargs)  # type: ignore[arg-type]
    return response.choices[0].text or ""


def call_llm_json(
    *,
    query: str,
    system_prompt: str,
    user_prompt_template: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    timeout_s: float,
    max_retries: int,
    retry_base_s: float,
    max_output_tokens: int,
    reasoning_effort: str,
    include_reasoning_effort: bool = True,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    extra_generation_body: dict[str, object] | None = None,
    use_harmony: bool = False,
    harmony_renderer: HarmonyRenderer | None = None,
    label: str = "",
) -> tuple[dict, str]:
    """Request a decomposition and return (parsed_json, raw_text).

    A single retry loop covers both the HTTP request and parsing/validation:
    API/connection errors AND unusable output (invalid JSON, empty response,
    empty sub_issues) are re-requested up to ``max_retries`` times. With
    temperature sampling a re-request yields a fresh generation, so a bad
    sample usually recovers on retry.
    """
    user_prompt = user_prompt_template.replace(QUERY_PLACEHOLDER, query.strip())

    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    kwargs: dict[str, object] = {"model": model, "messages": messages}
    if max_output_tokens > 0:
        kwargs["max_tokens"] = max_output_tokens
    if model_supports_temperature(model):
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    extra_body: dict[str, object] = {}
    if include_reasoning_effort and reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort
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

    def _retry(attempt: int, err: Exception) -> None:
        print(
            f"[{label}] retry {attempt}/{max_retries} after "
            f"{type(err).__name__}: {err}",
            flush=True,
        )
        time.sleep(retry_base_s * attempt)

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if use_harmony:
                if harmony_renderer is None:
                    raise RuntimeError("Harmony renderer was not initialized.")
                content = _one_harmony_request(
                    client=client,
                    renderer=harmony_renderer,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            else:
                content = _one_request(client, kwargs)
            try:
                parsed = parse_json_response(content)
            except:
                print(content)
                parsed = parse_json_response(content)
            sub_issues = parsed.get("sub_issues", [])
            if not sub_issues:
                raise RuntimeError("No sub_issues in response")
            for s in sub_issues:
                if isinstance(s, dict) and "search_query_de" in s:
                    s["search_query_de"] = normalize_search_query(
                        str(s["search_query_de"])
                    )
            return parsed, content

        except openai.APIStatusError as err:
            last_err = err
            if err.status_code in (408, 409, 429, 500, 502, 503, 504) and attempt < max_retries:
                _retry(attempt, err)
                continue
            raise

        except (openai.APIConnectionError, openai.APITimeoutError) as err:
            last_err = err
            if attempt < max_retries:
                _retry(attempt, err)
                continue
            raise

        except (RuntimeError, ValueError) as err:
            # Unusable model output: invalid JSON (RuntimeError or
            # json.JSONDecodeError ⊂ ValueError), empty response, or empty
            # sub_issues. Re-request a fresh sample.
            last_err = err
            if attempt < max_retries:
                _retry(attempt, err)
                continue
            raise

    raise RuntimeError(f"Request failed after {max_retries} attempts: {last_err}")


def run_normalize_only(args: argparse.Namespace) -> None:
    """One-shot deterministic re-normalization of an already-generated CSV.

    No LLM calls: parse each row's ``--output-col`` JSON, run
    ``normalize_search_query`` over every ``search_query_de``, re-dump, save.
    """
    df = read_input_table(args.input_path).fillna("")
    if args.output_col not in df.columns:
        raise SystemExit(f"Missing column: {args.output_col}")

    output_csv = default_output_csv(args.input_path, args.output_csv)
    rows_touched = 0
    subq_normalized = 0

    for idx in range(len(df)):
        cell = str(df.at[idx, args.output_col]).strip()
        if not cell:
            continue
        try:
            parsed = json.loads(cell)
        except Exception as err:
            print(
                f"[{idx + 1}] SKIP unparseable {args.output_col}: {err}",
                flush=True,
            )
            continue

        sub_issues = parsed.get("sub_issues", []) if isinstance(parsed, dict) else []
        changed = False
        for s in sub_issues:
            if isinstance(s, dict) and "search_query_de" in s:
                old = str(s["search_query_de"])
                new = normalize_search_query(old)
                if new != old:
                    changed = True
                    subq_normalized += 1
                s["search_query_de"] = new

        if changed:
            df.at[idx, args.output_col] = json.dumps(parsed, ensure_ascii=False)
            rows_touched += 1

    df.to_csv(output_csv, index=False)
    print(
        f"Normalize-only: {rows_touched} rows changed, "
        f"{subq_normalized} sub-queries normalized."
    )
    print(f"Saved: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate decomposed German search queries using an "
        "OpenAI-compatible Chat Completions endpoint (parallel)."
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default="llm-agentic-legal-information-retrieval/test.csv",
        help="Input query table. Supports .csv and .parquet.",
    )
    parser.add_argument(
        "--input-csv",
        dest="input_path",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="If empty, overwrite CSV input; parquet input writes <stem>_decomposed.csv.",
    )
    parser.add_argument("--query-col", default="query")
    parser.add_argument("--output-col", default="decomposed_queries_german")
    parser.add_argument("--raw-col", default="decomposed_queries_raw")
    parser.add_argument(
        "--prompt-file",
        default="",
        help="Optional markdown prompt file override. Empty uses the embedded default prompt.",
    )
    parser.add_argument("--model", default="nvidia/openai/gpt-oss-20b")
    parser.add_argument(
        "--model-family",
        choices=["auto", "gpt-oss", "qwen", "chat"],
        default="auto",
        help="Model behavior family. Auto detects gpt-oss and Qwen from model names.",
    )
    parser.add_argument("--base-url", default="http://localhost:20128/v1")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--max-parallel", type=int, default=40)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument(
        "--generation-config",
        default="",
        help="JSON dict of generation args. Known keys include temperature, "
        "top_p, top_k, min_p, presence_penalty, repetition_penalty. Unknown "
        "keys are passed through to vLLM extra_body.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--sleep-between-seconds", type=float, default=0.0)
    parser.add_argument("--start-index", type=int, default=1, help="1-based row index.")
    parser.add_argument("--end-index", type=int, default=0, help="0 = all rows.")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument(
        "--start-vllm",
        action="store_true",
        help="Start or attach to a local vLLM OpenAI-compatible server before generation.",
    )
    parser.add_argument(
        "--use-harmony",
        action="store_true",
        help="Render prompts with openai_harmony and call /v1/completions. "
        "Automatically enabled with --start-vllm for gpt-oss models.",
    )
    parser.add_argument(
        "--disable-harmony",
        action="store_true",
        help="Force Chat Completions even when model-family resolves to gpt-oss.",
    )
    parser.add_argument(
        "--vllm-model-path",
        default="",
        help="Model path/ID passed to vLLM. Defaults to --model.",
    )
    parser.add_argument(
        "--served-model-name",
        default="",
        help="vLLM served model name. Defaults to --model.",
    )
    parser.add_argument("--vllm-host", default="localhost")
    parser.add_argument("--vllm-port", type=int, default=20128)
    parser.add_argument("--server-timeout", type=int, default=900)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=256)
    parser.add_argument(
        "--speculative-config",
        default="",
        help="JSON string passed to vLLM --speculative-config, e.g. "
        '\'{"method":"qwen3_next_mtp","num_speculative_tokens":2}\'.',
    )
    parser.add_argument(
        "--vllm-async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --async-scheduling to vLLM when starting the server.",
    )
    parser.add_argument(
        "--vllm-extra-args",
        default="",
        help="Extra vLLM server args appended verbatim, e.g. "
        "'--enable-chunked-prefill'.",
    )
    parser.add_argument(
        "--vllm-log-file",
        default="vllm_decomposer_server.log",
        help="File for managed vLLM stdout/stderr. Empty disables logging.",
    )
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="Skip the LLM entirely: load --input, re-normalize every "
        "search_query_de in --output-col in place, save, and exit.",
    )
    argv = sys.argv[1:]
    base_url_was_set = any(
        x == "--base-url" or x.startswith("--base-url=") for x in sys.argv[1:]
    )
    kv_cache_dtype_was_set = any(
        x == "--kv-cache-dtype" or x.startswith("--kv-cache-dtype=") for x in argv
    )
    args = parser.parse_args()
    if args.start_vllm and not base_url_was_set:
        args.base_url = f"http://{args.vllm_host}:{args.vllm_port}/v1"

    if args.normalize_only:
        run_normalize_only(args)
        return

    if args.start_index <= 0:
        raise SystemExit("--start-index must be >= 1.")
    if args.max_retries <= 0:
        raise SystemExit("--max-retries must be > 0.")
    if args.max_parallel <= 0:
        raise SystemExit("--max-parallel must be > 0.")
    if args.sleep_between_seconds < 0:
        raise SystemExit("--sleep-between-seconds must be >= 0.")
    if args.vllm_port <= 0:
        raise SystemExit("--vllm-port must be > 0.")
    if args.server_timeout <= 0:
        raise SystemExit("--server-timeout must be > 0.")

    generation_config = parse_json_dict_arg(
        args.generation_config, "--generation-config"
    )
    args.temperature = _pop_float_config(
        generation_config, "temperature", args.temperature
    )
    args.top_p = _pop_float_config(generation_config, "top_p", args.top_p)
    args.top_k = _pop_int_config(generation_config, "top_k", args.top_k)
    args.min_p = _pop_float_config(generation_config, "min_p", args.min_p)
    args.presence_penalty = _pop_float_config(
        generation_config, "presence_penalty", args.presence_penalty
    )
    args.repetition_penalty = _pop_float_config(
        generation_config, "repetition_penalty", args.repetition_penalty
    )
    extra_generation_body = generation_config
    if args.top_p is not None and not (0 <= args.top_p <= 1):
        raise SystemExit("--top-p / generation_config.top_p must be in [0, 1].")
    if args.top_k is not None and args.top_k < 0:
        raise SystemExit("--top-k / generation_config.top_k must be >= 0.")
    if args.min_p is not None and not (0 <= args.min_p <= 1):
        raise SystemExit("--min-p / generation_config.min_p must be in [0, 1].")

    model_family = resolve_model_family(
        args.model, args.vllm_model_path, args.model_family
    )
    if model_family == "qwen" and not kv_cache_dtype_was_set:
        args.kv_cache_dtype = ""

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        if "openai.com" in args.base_url:
            raise SystemExit(f"Missing API key. Set environment variable {args.api_key_env}.")
        api_key = "vllm-local"

    server: ManagedVLLMServer | None = None
    if args.start_vllm:
        args.served_model_name = args.served_model_name or args.model
        server = ManagedVLLMServer(args, api_key=api_key)
        args.base_url = server.start()

    use_harmony = bool(
        not args.disable_harmony
        and model_family == "gpt-oss"
        and (
            args.use_harmony
            or args.start_vllm
        )
    )
    harmony_renderer = HarmonyRenderer(args.reasoning_effort) if use_harmony else None
    request_model = args.served_model_name or args.model
    include_reasoning_effort = model_family == "gpt-oss" and not use_harmony
    top_p = args.top_p
    top_k = args.top_k
    min_p = args.min_p
    presence_penalty = args.presence_penalty
    repetition_penalty = args.repetition_penalty
    if model_family == "qwen":
        top_p = 0.95 if top_p is None else top_p
        top_k = 20 if top_k is None else top_k
        min_p = 0.0 if min_p is None else min_p
        presence_penalty = (
            1.5 if presence_penalty is None else presence_penalty
        )
        repetition_penalty = (
            1.0 if repetition_penalty is None else repetition_penalty
        )

    system_prompt, user_prompt_template = load_prompts(args.prompt_file)
    print(f"Prompt: {args.prompt_file or 'embedded default'}")
    print(f"  System prompt: {len(system_prompt)} chars")
    print(f"  User template: {len(user_prompt_template)} chars")

    df = read_input_table(args.input_path).fillna("")
    if args.query_col not in df.columns:
        raise SystemExit(f"Missing column: {args.query_col}")
    if args.output_col not in df.columns:
        df[args.output_col] = ""
    if args.raw_col not in df.columns:
        df[args.raw_col] = ""

    n = len(df)
    start_i = args.start_index - 1
    end_i = n if args.end_index <= 0 else min(args.end_index, n)

    output_csv = default_output_csv(args.input_path, args.output_csv)

    print(f"Input:      {args.input_path} ({n} rows)")
    print(f"Output:     {output_csv}")
    print(f"Base URL:   {args.base_url}")
    print(f"Model:      {request_model}")
    print(f"Family:     {model_family}")
    print(f"Mode:       {'harmony completions' if use_harmony else 'chat completions'}")
    if model_supports_temperature(request_model):
        print(f"Temperature: {args.temperature}")
    else:
        print("Temperature: ignored for this model")
    if top_p is not None or top_k is not None or min_p is not None:
        print(
            f"Sampling:   top_p={top_p}, top_k={top_k}, min_p={min_p}, "
            f"presence_penalty={presence_penalty}, "
            f"repetition_penalty={repetition_penalty}"
        )
    if extra_generation_body:
        print(f"Extra body: {sorted(extra_generation_body)}")
    print(f"Max parallel: {args.max_parallel}")
    print(f"Range:      rows {start_i + 1}..{end_i}")
    print()

    counters = {"written": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()

    # Build the worklist up front, honoring skip/empty rules.
    worklist: list[tuple[int, str, str]] = []
    for idx in range(start_i, end_i):
        row = df.iloc[idx]
        qid = str(row.get("query_id", f"row_{idx}"))
        query_text = str(row[args.query_col]).strip()
        existing = str(row.get(args.output_col, "")).strip()

        if not query_text:
            counters["failed"] += 1
            print(f"[{idx + 1}/{end_i}] FAIL {qid}: empty query", flush=True)
            continue

        if existing and not args.overwrite_existing:
            counters["skipped"] += 1
            print(f"[{idx + 1}/{end_i}] SKIP {qid} (already has {args.output_col})", flush=True)
            continue

        worklist.append((idx, qid, query_text))

    started = time.time()

    def process_row(item: tuple[int, str, str]) -> None:
        idx, qid, query_text = item
        try:
            parsed, raw_response = call_llm_json(
                query=query_text,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                api_key=api_key,
                base_url=args.base_url,
                model=request_model,
                temperature=args.temperature,
                timeout_s=args.timeout_seconds,
                max_retries=args.max_retries,
                retry_base_s=args.retry_base_seconds,
                max_output_tokens=args.max_output_tokens,
                reasoning_effort=args.reasoning_effort,
                include_reasoning_effort=include_reasoning_effort,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                extra_generation_body=extra_generation_body,
                use_harmony=use_harmony,
                harmony_renderer=harmony_renderer,
                label=qid,
            )
            raw_response = raw_response.strip()
            sub_issues = parsed.get("sub_issues", [])

            n_issues = len(sub_issues)
            dims = sorted(set(s.get("dimension", "?") for s in sub_issues))

            with lock:
                df.at[idx, args.raw_col] = raw_response
                df.at[idx, args.output_col] = json.dumps(parsed, ensure_ascii=False)
                counters["written"] += 1
                df.to_csv(output_csv, index=False)
                print(
                    f"[{idx + 1}/{end_i}] OK   {qid}: {n_issues} sub-issues, "
                    f"dimensions: {dims}",
                    flush=True,
                )

        except Exception as err:
            with lock:
                counters["failed"] += 1
                print(f"[{idx + 1}/{end_i}] FAIL {qid}: {err}", flush=True)

        if args.sleep_between_seconds > 0:
            time.sleep(args.sleep_between_seconds)

    total_work = len(worklist)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = [ex.submit(process_row, item) for item in worklist]
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(
                concurrent.futures.as_completed(futures),
                total=total_work,
                desc="Decomposing",
                unit="row",
            )
            for fut in iterator:
                fut.result()
        except ImportError:
            done = 0
            next_report = 1
            report_every = max(1, total_work // 100)
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
                done += 1
                if done >= next_report or done == total_work:
                    print(
                        f"Progress: {done}/{total_work} rows "
                        f"({done / max(1, total_work):.1%})",
                        flush=True,
                    )
                    next_report = done + report_every

    df = pl.read_csv(output_csv)
    list_row = []
    for row in df.iter_rows(named=True):
        x = json.loads(row["decomposed_queries_german"])
        for s in x["sub_issues"]:
            if not isinstance(s, dict):
                continue
            search_query = str(
                s.get("search_query_de")
                or s.get("query_de")
                or s.get("description_de")
                or s.get("description_en")
                or ""
            ).strip()
            if not search_query:
                continue
            list_row.append({
                "query_id": row["query_id"],
                "sub_id": s.get("id", len(list_row) + 1),
                "sub_query_de": normalize_search_query(search_query)
            })
        # break
    dfx = pl.from_dicts(list_row)
    output_parquet = output_subquery_parquet_path(output_csv)
    dfx.write_parquet(output_parquet)

    elapsed = time.time() - started
    print(
        f"\nDone in {elapsed:.1f}s — written: {counters['written']}, "
        f"skipped: {counters['skipped']}, failed: {counters['failed']}"
    )
    print(f"Saved: {output_csv} and {output_parquet}")
    if server is not None:
        server.stop()


if __name__ == "__main__":
    main()
