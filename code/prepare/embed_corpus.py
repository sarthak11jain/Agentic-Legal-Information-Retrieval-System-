#!/usr/bin/env python3
"""
Unified embedding script — supports voyageai/voyage-4-nano and Qwen3-Embedding.

The model is served by a managed `vllm serve` subprocess (started/stopped by this
script) and accessed over its OpenAI-compatible /v1/embeddings endpoint. Pass
--no-serve to instead attach to an already-running server at --base-url.

Modes:
  queries       Embed text from a configurable column in a query parquet.
  sub-queries   Backward-compatible alias for queries.
  laws          Embed laws_de.csv → law embedding parquet.
  courts        Embed court texts → court embedding parquet.

Backend is auto-detected from --model:
  Contains "Qwen"                      → qwen  (Qwen3-Embedding family)
  Contains "zembed" or "zeroentropy"   → zembed (zembed-1-embedding, 2560/1280 dims)
  Otherwise                            → voyage (voyage-4-nano, default)
Use --backend to override.

Examples:
    # Voyage (default)
    python src/embed_corpus.py queries \\
        --input-parquet data-processed/test_sub_query_voyage_large.parquet \\
        --embed-text-col sub_query_de \\
        --output data-processed/test_sub_query_voyage_nano.parquet

    python src/embed_corpus.py laws \\
        --output data-processed/voyage_law_embedding_nano.parquet

    python src/embed_corpus.py courts \\
        --output data-processed/court_voyage_nano.parquet

    # Qwen
    python src/embed_corpus.py laws \\
        --model Qwen/Qwen3-Embedding-0.6B \\
        --output data-processed/law_embedding_qwen.parquet

    python src/embed_corpus.py courts \\
        --model Qwen/Qwen3-Embedding-0.6B \\
        --court-parquet data-processed/court_law_references.parquet \\
        --output data-processed/court_embedding_qwen.parquet

    # Zembed
    python src/embed_corpus.py laws \\
        --model zeroentropy/zembed-1-embedding \\
        --output data-processed/law_embedding_zembed.parquet

    python src/embed_corpus.py courts \\
        --model zeroentropy/zembed-1-embedding \\
        --output data-processed/court_embedding_zembed.parquet
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "llm-agentic-legal-information-retrieval"
DATA_PROC = PROJECT_ROOT / "data-processed"

# Importable prompt constants (used by embed_bm25_candidates.py and tests)
QUERY_INSTRUCTION = "Instruct: Given a legal query, retrieve the relevant Swiss law article\nQuery: "
VOYAGE_QUERY_PROMPT = "Represent the query for retrieving supporting documents: "
VOYAGE_DOC_PROMPT = "Represent the document for retrieval: "
ZEMBED_QUERY_PROMPT = "<|im_start|>system\nquery<|im_end|>\n<|im_start|>user\n"
ZEMBED_DOC_PROMPT   = "<|im_start|>system\ndocument<|im_end|>\n<|im_start|>user\n"
ZEMBED_SUFFIX       = "<|im_end|>\n"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


def batch_embed(model, texts: list[str], batch_size: int) -> np.ndarray:
    """Embed texts in batches, return L2-normalised float32 matrix (N, dim)."""
    results: list[np.ndarray] = []
    total = len(texts)
    t0 = time.time()
    for start in range(0, total, batch_size):
        chunk = texts[start : start + batch_size]
        vecs = model.embed(chunk)
        results.append(l2_normalize(vecs))
        done = min(start + batch_size, total)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  {done}/{total}  ({rate:.0f} texts/s)", end="\r", flush=True)
    print()
    return np.vstack(results)


def truncate_text(text: str, max_chars: int) -> str:
    s = (text or "").strip()
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _resolve_chunk(n: int, start_idx: int, end_idx: int) -> tuple[int, int, bool]:
    """[start, end) row range over n items (end_idx < 0 → n). Returns
    (start, end, chunked); chunked is True iff it is a strict sub-range."""
    start = max(0, start_idx)
    end = n if end_idx is None or end_idx < 0 else min(end_idx, n)
    if end < start:
        end = start
    return start, end, (start != 0 or end != n)


def _chunk_path(p: Path, start: int, end: int) -> Path:
    """Insert _<start>-<end> before the suffix:
    voyage_law_embedding_nano.parquet → voyage_law_embedding_nano_0-5000.parquet"""
    return p.with_name(f"{p.stem}_{start}-{end}{p.suffix}")


def _shard_dir(output: Path) -> Path:
    """Sibling directory holding delta shards for `output`."""
    return output.parent / (output.stem + ".parts")


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write parquet via a .tmp file + os.replace so a kill never leaves a
    half-written / corrupt file (atomic on the same filesystem)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _reset_shards(output: Path) -> None:
    """--overwrite: drop the base file + any leftover shards so a fresh run
    does not fold stale data at consolidate time."""
    sd = _shard_dir(output)
    if sd.exists():
        shutil.rmtree(sd)
    if output.exists():
        output.unlink()


class _ShardWriter:
    """Background checkpoint writer.

    Each submit() flushes only the *new* delta rows since the last flush to
    its own immutable shard parquet, on a single background thread, so the
    embedding loop (GPU) never blocks on a growing full rewrite. Shards are
    never rewritten → constant per-checkpoint cost and crash-safe resume.
    """

    def __init__(self, output: Path) -> None:
        from concurrent.futures import ThreadPoolExecutor
        self._dir = _shard_dir(output)
        self._dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self._dir.glob("part-*.parquet"))
        self._idx = int(existing[-1].stem.split("-")[1]) if existing else 0
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._future = None

    def submit(self, rows: list[dict]) -> None:
        if not rows:
            return
        snap = list(rows)  # shallow copy on the caller thread → race-free
        self._idx += 1
        path = self._dir / f"part-{self._idx:05d}.parquet"
        if self._future is not None:
            self._future.result()  # single slot: drain prior (tiny) write
        self._future = self._pool.submit(
            _write_parquet_atomic, pd.DataFrame(snap), path
        )
        print(f"\n  [ckpt] queued {len(snap)} rows → {path.name}", flush=True)

    def close(self) -> None:
        if self._future is not None:
            self._future.result()
            self._future = None
        self._pool.shutdown(wait=True)


def _load_done(base: Path) -> set[str]:
    """Citation set already embedded = base parquet + any leftover shards."""
    paths: list[Path] = []
    if base.exists():
        paths.append(base)
    sd = _shard_dir(base)
    if sd.exists():
        paths.extend(sorted(sd.glob("part-*.parquet")))
    done: set[str] = set()
    for p in paths:
        try:
            done.update(pd.read_parquet(p, columns=["citation"])["citation"].tolist())
        except Exception:
            continue
    return done


def _consolidate(output: Path) -> None:
    """Stream-concat base (if any) + all shards → single `output`, then drop
    the shard dir. polars sink (streaming, fast); pyarrow fallback."""
    shard_dir = _shard_dir(output)
    shards = sorted(shard_dir.glob("part-*.parquet")) if shard_dir.exists() else []
    if not shards:
        return
    sources = ([output] if output.exists() else []) + shards
    tmp = output.with_name(output.name + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import polars as pl
        pl.scan_parquet([str(s) for s in sources]).sink_parquet(str(tmp))
    except Exception as e:
        print(f"  [consolidate] polars sink failed ({e}); pyarrow fallback",
              flush=True)
        import pyarrow.parquet as pq
        writer = None
        try:
            for s in sources:
                pf = pq.ParquetFile(s)
                for i in range(pf.num_row_groups):
                    tbl = pf.read_row_group(i)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp, tbl.schema)
                    writer.write_table(tbl)
        finally:
            if writer is not None:
                writer.close()
    os.replace(tmp, output)
    shutil.rmtree(shard_dir)
    n = len(pd.read_parquet(output, columns=["citation"]))
    print(f"  [consolidate] {len(shards)} shards + base → {output} ({n} rows)",
          flush=True)


# ---------------------------------------------------------------------------
# vLLM serve subprocess + OpenAI-compatible client
# ---------------------------------------------------------------------------

def detect_backend(model: str) -> str:
    if "Qwen" in model:
        return "qwen"
    if "zembed" in model.lower() or "zeroentropy" in model.lower():
        return "zembed"
    return "voyage"


def _serve_command(
    args, port: int | None = None, gpu: str | None = None
) -> tuple[list[str], dict[str, str]]:
    """Build the `vllm serve` argv + subprocess env from args/backend.

    port/gpu override args.port and pin the process to one CUDA device group
    via CUDA_VISIBLE_DEVICES. With --num-instances, each instance gets
    tensor_parallel_size * data_parallel_size devices.
    """
    backend = getattr(args, "backend", "") or detect_backend(args.model)
    port = args.port if port is None else port
    cmd = [
        "vllm", "serve", args.model,
        "--runner", "pooling",
        "--convert", "embed",
        "--host", args.host,
        "--port", str(port),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--api-key", args.api_key,
    ]
    if int(args.tensor_parallel_size) > 1:
        cmd += ["--tensor-parallel-size", str(args.tensor_parallel_size)]
    if int(args.data_parallel_size) > 1:
        cmd += ["--data-parallel-size", str(args.data_parallel_size)]
    if backend == "voyage":
        cmd += [
            "--hf-overrides",
            json.dumps({"architectures": ["VoyageQwen3BidirectionalEmbedModel"]}),
            "--pooler-config",
            json.dumps({"pooling_type": "MEAN"}),
        ]
    extra = (getattr(args, "vllm_extra_args", "") or "").split()
    if extra:
        cmd += extra
    env = {**os.environ, "VLLM_ATTENTION_BACKEND": "TRITON_ATTN"}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return cmd, env


def _resolve_gpus(args, n: int) -> list[str]:
    """CUDA_VISIBLE_DEVICES group per instance.

    For example, --num-instances 2 --tensor-parallel-size 2 --gpus 0,1,2,3
    returns ["0,1", "2,3"].
    """
    tp = int(getattr(args, "tensor_parallel_size", 1))
    dp = int(getattr(args, "data_parallel_size", 1))
    if tp <= 0 or dp <= 0:
        raise SystemExit("--tensor-parallel-size and --data-parallel-size must be > 0.")
    group_size = tp * dp
    total_needed = n * group_size
    raw = (getattr(args, "gpus", "") or "").strip()
    ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else [
        str(i) for i in range(total_needed)
    ]
    if len(ids) < total_needed:
        raise SystemExit(
            f"--gpus has {len(ids)} id(s), but --num-instances={n}, "
            f"--tensor-parallel-size={tp}, and --data-parallel-size={dp} "
            f"need {total_needed} GPU id(s)."
        )
    return [
        ",".join(ids[i * group_size : (i + 1) * group_size])
        for i in range(n)
    ]


class VLLMServer:
    """One `vllm serve` subprocess, optionally pinned to a CUDA device group."""

    def __init__(self, args, port: int | None = None, gpu: str | None = None) -> None:
        self.port = args.port if port is None else port
        self.gpu = gpu
        self.cmd, self.env = _serve_command(args, port=self.port, gpu=gpu)
        self.host = args.host
        self.base_url = f"http://{args.host}:{self.port}/v1"
        self.health_url = f"http://{args.host}:{self.port}/health"
        self._proc: subprocess.Popen | None = None
        self._stopped = False

    @property
    def tag(self) -> str:
        g = "" if self.gpu is None else f" gpu={self.gpu}"
        return f"[server:{self.port}{g}]"

    def start(self) -> None:
        print(f"{self.tag} launching: {' '.join(self.cmd)}", flush=True)
        # New session → own process group, so we can reap vLLM's worker children.
        self._proc = subprocess.Popen(self.cmd, env=self.env, start_new_session=True)

    def wait_ready(self, timeout: int) -> None:
        print(f"{self.tag} waiting for {self.health_url} ...", flush=True)
        t0 = time.time()
        while True:
            if self._proc is None or self._proc.poll() is not None:
                code = None if self._proc is None else self._proc.returncode
                raise RuntimeError(
                    f"{self.tag} vllm serve exited early (code {code}) "
                    f"before becoming healthy."
                )
            try:
                with urllib.request.urlopen(self.health_url, timeout=2) as resp:
                    if resp.status == 200:
                        print(f"{self.tag} ready in {time.time() - t0:.0f}s", flush=True)
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            if time.time() - t0 > timeout:
                self.stop()
                raise TimeoutError(
                    f"{self.tag} not healthy after {timeout}s ({self.health_url})"
                )
            print(f"{self.tag} waiting ... {time.time() - t0:.0f}s",
                  end="\r", flush=True)
            time.sleep(3)

    def start_and_wait(self, timeout: int) -> None:
        self.start()
        self.wait_ready(timeout)

    def stop(self) -> None:
        if self._stopped or self._proc is None:
            return
        self._stopped = True
        if self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        print(f"{self.tag} terminated", flush=True)


class EmbedClient:
    """Thin wrapper around an OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        from openai import OpenAI
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=600.0, max_retries=3
        )
        self._model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


class MultiEmbedClient:
    """Fan out one embed() call across N independent servers, in parallel.

    Each call's text list is split into N contiguous shards; shard i goes to
    client i concurrently; results are concatenated back in input order. This
    keeps batch_embed/_embed_with_saves unchanged while spreading load across
    single-GPU servers (no tensor/data-parallel cross-GPU traffic).
    """

    def __init__(self, clients: list[EmbedClient]) -> None:
        from concurrent.futures import ThreadPoolExecutor
        self._clients = clients
        self._pool = ThreadPoolExecutor(max_workers=len(clients))

    def embed(self, texts: list[str]) -> np.ndarray:
        n = len(self._clients)
        if n == 1 or len(texts) < n:
            return self._clients[0].embed(texts)
        k, m = divmod(len(texts), n)
        shards: list[list[str]] = []
        i = 0
        for x in range(n):
            size = k + (1 if x < m else 0)
            shards.append(texts[i: i + size])
            i += size
        futures = [
            self._pool.submit(c.embed, s)
            for c, s in zip(self._clients, shards)
        ]
        return np.vstack([f.result() for f in futures])


class ServerManager:
    """Owns 1..N `vllm serve` processes + the (multi-)embed client.

    --num-instances N starts N single-GPU servers on ports
    --port .. --port+N-1 (GPUs from --gpus or 0..N-1). With --no-serve it
    instead attaches to existing servers (comma-separated --base-url, or
    derived host:port+i).
    """

    def __init__(self, args) -> None:
        self.args = args
        self.n = max(1, int(getattr(args, "num_instances", 1)))
        self.no_serve = bool(getattr(args, "no_serve", False))
        self.servers: list[VLLMServer] = []
        self._urls: list[str] = []
        self._stopped = False

    def start(self) -> None:
        if self.no_serve:
            bu = (self.args.base_url or "").strip()
            if bu:
                self._urls = [u.strip() for u in bu.split(",") if u.strip()]
            else:
                self._urls = [
                    f"http://{self.args.host}:{self.args.port + i}/v1"
                    for i in range(self.n)
                ]
            for u in self._urls:
                print(f"[client] attaching to existing server at {u}")
            return

        gpus = _resolve_gpus(self.args, self.n)
        self.servers = [
            VLLMServer(self.args, port=self.args.port + i, gpu=gpus[i])
            for i in range(self.n)
        ]
        atexit.register(self.stop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_handler)
        for s in self.servers:          # launch all (parallel warmup)
            s.start()
        for s in self.servers:          # then wait each → ~max load time
            s.wait_ready(self.args.server_timeout)
        self._urls = [s.base_url for s in self.servers]

    def client(self):
        clients = [
            EmbedClient(u, self.args.api_key, self.args.model) for u in self._urls
        ]
        return clients[0] if len(clients) == 1 else MultiEmbedClient(clients)

    def _signal_handler(self, signum, frame) -> None:
        self.stop()
        raise KeyboardInterrupt

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for s in self.servers:
            s.stop()


# ---------------------------------------------------------------------------
# Embed loop with periodic checkpoint saves
# ---------------------------------------------------------------------------

def _embed_with_saves(
    model,
    texts: list[str],
    todo_df: pd.DataFrame,
    extra_cols: list[str],
    ckpt_rows: list[dict],
    output: Path,
    args,
    emb_col: str,
) -> list[dict]:
    """Shared inner loop for laws and courts. Returns new_rows.

    Periodic checkpoints write only the new delta rows since the last flush to
    an immutable shard, on a background thread (constant cost, GPU never
    waits). The base parquet stays on disk and is folded in at _consolidate();
    `ckpt_rows` is accepted for signature compatibility but no longer rewritten
    here.
    """
    total = len(texts)
    new_rows: list[dict] = []
    last_save = 0
    writer = _ShardWriter(output)
    t0 = time.time()
    try:
        for start in range(0, total, args.batch_size):
            chunk = texts[start : start + args.batch_size]
            vecs = l2_normalize(model.embed(chunk))
            if 0 < args.dimensions < vecs.shape[1]:
                vecs = vecs[:, : args.dimensions]
                vecs = l2_normalize(vecs)
            for j, vec in enumerate(vecs):
                row: dict = {"citation": todo_df["citation"].iloc[start + j], emb_col: vec}
                for col in extra_cols:
                    row[col] = todo_df[col].iloc[start + j]
                new_rows.append(row)
            done = min(start + args.batch_size, total)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {done}/{total}  ({rate:.0f} texts/s)", end="\r", flush=True)
            if args.save_freq > 0 and len(new_rows) - last_save >= args.save_freq:
                writer.submit(new_rows[last_save:])
                last_save = len(new_rows)
    finally:
        writer.submit(new_rows[last_save:])  # flush tail
        writer.close()
    print()
    return new_rows


# ---------------------------------------------------------------------------
# Mode: queries
# ---------------------------------------------------------------------------

def run_queries(args, model) -> None:
    input_parquet = Path(args.input_parquet)
    output = Path(args.output)
    print(f"Loading {input_parquet} ...")
    df = pd.read_parquet(input_parquet)
    if args.embed_text_col not in df.columns:
        raise SystemExit(
            f"{input_parquet}: missing --embed-text-col '{args.embed_text_col}'. "
            f"Found columns: {list(df.columns)}"
        )
    print(f"  {len(df)} rows; embedding text column '{args.embed_text_col}'")

    start, end, chunked = _resolve_chunk(len(df), args.start_idx, args.end_idx)
    df = df.iloc[start:end].reset_index(drop=True)
    if chunked:
        output = _chunk_path(output, start, end)
        print(f"  Chunk [{start}:{end}] → {output.name}")
    if len(df) == 0:
        print("Empty chunk; nothing to do.")
        return

    texts = [
        args.query_prefix + str(t) + args.suffix
        for t in df[args.embed_text_col].fillna("").tolist()
    ]
    print(f"Embedding {len(texts)} query texts from '{args.embed_text_col}' ...")
    embs = batch_embed(model, texts, args.batch_size)
    if 0 < args.dimensions < embs.shape[1]:
        embs = embs[:, : args.dimensions]
        embs = l2_normalize(embs)

    out_df = df.copy().reset_index(drop=True)
    out_df[args.emb_col] = list(embs)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output, index=False)
    print(f"Saved {len(out_df)} rows → {output}")
    print(f"Embedding dim: {embs.shape[1]}  norm[0]: {np.linalg.norm(embs[0]):.4f}")


# ---------------------------------------------------------------------------
# Mode: laws
# ---------------------------------------------------------------------------

def run_laws(args, model) -> None:
    laws_csv = Path(args.laws_csv)
    output = Path(args.output)
    emb_col = args.emb_col
    print(f"Loading {laws_csv} ...")
    # maybe input is parquet file with "zembed_document" column instead of raw CSV; if so, read that
    if laws_csv.suffix == ".parquet":
        df = pd.read_parquet(laws_csv).fillna("")
    else:
        df = pd.read_csv(laws_csv).fillna("")
    print(f"  {len(df)} laws")

    start, end, chunked = _resolve_chunk(len(df), args.start_idx, args.end_idx)
    df = df.iloc[start:end].reset_index(drop=True)
    if chunked:
        output = _chunk_path(output, start, end)
        print(f"  Chunk [{start}:{end}] → {output.name}")

    if args.overwrite:
        _reset_shards(output)
    done_citations = set() if args.overwrite else _load_done(output)
    if done_citations:
        print(f"  Resuming: {len(done_citations)} already embedded")
    ckpt_rows: list[dict] = []

    todo_df = df[~df["citation"].isin(done_citations)].reset_index(drop=True)
    print(f"  {len(todo_df)} remaining to embed")
    if len(todo_df) == 0:
        print("Nothing to do.")
        return

    max_chars = args.max_chars_per_law
    # check if column "zembed_document" exists, if yes use it instead of constructing from title+text
    if "zembed_document" in todo_df.columns:
        print("Using existing 'zembed_document' column for embedding (truncated to max_chars_per_law)")
        texts = [
            args.doc_prefix + truncate_text(str(r.zembed_document), max_chars) + args.suffix
            for r in todo_df.itertuples(index=False)
        ]
    else:
        texts = [
            # args.doc_prefix + truncate_text(f"{r.title}: {r.citation} - {r.text}", max_chars) + args.suffix
            args.doc_prefix + truncate_text(f"Gesetz: {r.title} {r.text}", max_chars) + args.suffix
            for r in todo_df.itertuples(index=False)
        ]
    print(f"Embedding {len(texts)} law texts ...")
    new_rows = _embed_with_saves(model, texts, todo_df, [], ckpt_rows, output, args, emb_col)

    _consolidate(output)
    n = len(pd.read_parquet(output, columns=["citation"])) if output.exists() else 0
    print(f"Saved {n} rows → {output}")
    if new_rows:
        sample = np.array(new_rows[-1][emb_col])
        print(f"Embedding dim: {sample.shape[0]}  norm[-1]: {np.linalg.norm(sample):.4f}")


# ---------------------------------------------------------------------------
# Mode: courts
# ---------------------------------------------------------------------------

def run_courts(args, model) -> None:
    """Embed the `text` column of a court file (.csv or .parquet,
    auto-detected) and write a parquet with every input column + the
    embedding column. Resume/checkpoint/chunking are keyed on `citation`."""
    output = Path(args.output)
    emb_col = args.emb_col
    max_chars = getattr(args, "max_chars_per_court", 0) or args.max_chars_per_law

    src = Path(args.court_parquet)
    print(f"Loading {src} ...")
    if src.suffix == ".csv":
        df = pd.read_csv(src, dtype=str).fillna("")
    else:
        df = pd.read_parquet(src).fillna("")
    missing = [c for c in ("citation", "text") if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{src}: missing required column(s) {missing}; "
            f"found {list(df.columns)}"
        )
    print(f"  {len(df)} court rows; columns: {list(df.columns)}")

    start, end, chunked = _resolve_chunk(len(df), args.start_idx, args.end_idx)
    df = df.iloc[start:end].reset_index(drop=True)
    if chunked:
        output = _chunk_path(output, start, end)
        print(f"  Chunk [{start}:{end}] → {output.name}")

    if args.overwrite:
        _reset_shards(output)
    done_citations = set() if args.overwrite else _load_done(output)
    if done_citations:
        print(f"  Resuming: {len(done_citations)} already embedded")
    ckpt_rows: list[dict] = []

    todo_df = df[~df["citation"].isin(done_citations)].reset_index(drop=True)
    print(f"  {len(todo_df)} remaining to embed")
    if len(todo_df) == 0:
        print("Nothing to do.")
        return

    texts = [
        args.doc_prefix + truncate_text(str(r.text), max_chars) + args.suffix
        for r in todo_df.itertuples(index=False)
    ]
    print(f"Embedding {len(texts)} court texts ...")
    # Carry every input column through (incl. the full untruncated `text`);
    # `citation` is injected by _embed_with_saves, so exclude it here.
    extra_cols = [c for c in todo_df.columns if c != "citation"]
    new_rows = _embed_with_saves(
        model, texts, todo_df, extra_cols, ckpt_rows, output, args, emb_col
    )
    _consolidate(output)
    n = len(pd.read_parquet(output, columns=["citation"])) if output.exists() else 0
    print(f"Saved {n} rows → {output}")
    if new_rows:
        sample = np.array(new_rows[-1][emb_col])
        print(f"Embedding dim: {sample.shape[0]}  norm[-1]: {np.linalg.norm(sample):.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_server_args(p: argparse.ArgumentParser) -> None:
    """Shared `vllm serve` + OpenAI-client CLI args (reused by other scripts)."""
    p.add_argument("--host", default="localhost",
                   help="Host vllm serve binds to / client connects to.")
    p.add_argument("--port", type=int, default=8000,
                   help="Port vllm serve binds to / client connects to.")
    p.add_argument("--base-url", default="",
                   help="OpenAI base URL. Default: http://{host}:{port}/v1")
    p.add_argument("--api-key", default="vllm-local",
                   help="API key shared between vllm serve and the client.")
    p.add_argument("--no-serve", action="store_true",
                   help="Attach to an already-running server instead of launching one.")
    p.add_argument("--num-instances", type=int, default=1,
                   help="Start N independent vllm serve instances on "
                        "ports --port..--port+N-1 and shard embed work across them "
                        "(each instance uses TP*DP GPUs).")
    p.add_argument("--gpus", default="",
                   help="Comma-separated CUDA device ids assigned in contiguous "
                        "groups of TP*DP per instance. Default 0..needed-1. "
                        "E.g. --num-instances 2 --tensor-parallel-size 2 "
                        "--gpus 0,1,2,3")
    p.add_argument("--server-timeout", type=int, default=900,
                   help="Seconds to wait for vllm serve to become healthy.")
    p.add_argument("--vllm-extra-args", default="",
                   help="Extra args appended verbatim to the vllm serve command.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Embed texts with voyage-4-nano or Qwen3-Embedding "
                    "via a managed vllm serve subprocess."
    )
    p.add_argument(
        "mode",
        choices=["queries", "sub-queries", "laws", "courts", "consolidate"],
                   help="What to embed. 'consolidate' folds <output>.parts/ "
                        "shards into --output and removes them (no GPU/server).")

    # Model / backend
    p.add_argument("--model", default="voyageai/voyage-4-nano",
                   help="HuggingFace model ID.")
    p.add_argument("--backend", default="", choices=["", "voyage", "qwen", "zembed"],
                   help="Force backend. Auto-detected from --model if not set.")
    p.add_argument("--emb-col", default="",
                   help="Output embedding column name. Auto: voyage_nano_embedding, qwen_embedding, or zembed_embedding.")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="Tensor parallel size.")
    p.add_argument("--data-parallel-size", type=int, default=1,
                   help="Data parallel size.")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Texts per /v1/embeddings request (server batches internally).")

    # vLLM serve subprocess + OpenAI client
    add_server_args(p)

    # Matryoshka truncation
    p.add_argument("--dimensions", type=int, default=0,
                   help="Post-embed Matryoshka truncation dim (0 = no truncation). "
                        "voyage: 2048/1024/512/256. zembed: 2560/1280.")
    p.add_argument("--suffix", default="",
                   help="Text appended after each input (auto-set per backend; override here).")

    # Shared output
    p.add_argument("--output", default="", help="Output parquet path (auto-derived if not set).")
    p.add_argument("--overwrite", action="store_true",
                   help="Ignore existing checkpoint and re-embed everything.")
    p.add_argument("--save-freq", type=int, default=5000,
                   help="Save checkpoint every N new rows (0 = only at end).")
    p.add_argument("--start-idx", type=int, default=0,
                   help="Embed only input rows [start-idx:end-idx) "
                        "(by position, applied BEFORE resume-skip). Default 0.")
    p.add_argument("--end-idx", type=int, default=-1,
                   help="Exclusive end row; -1 = to the end. When the range "
                        "is a strict subset, '_<start>-<end>' is appended to "
                        "every output filename so chunks don't collide.")

    # Query-like rows
    p.add_argument(
        "--input",
        "--input-parquet",
        dest="input_parquet",
        default=str(DATA_PROC / "test_sub_query_voyage_large.parquet"),
        help="[queries] Input parquet.",
    )
    p.add_argument(
        "--embed-text-col",
        default="",
        help=(
            "[queries] Column containing text to embed. Default: query for "
            "queries mode, sub_query_de for legacy sub-queries mode."
        ),
    )

    # Laws
    p.add_argument("--laws-csv",
                   default=str(DATA_RAW / "laws_de.csv"),
                   help="[laws] Input CSV with citation, text columns.")
    p.add_argument("--max-chars-per-law", type=int, default=5000,
                   help="[laws/courts] Truncate text to this many chars (0 = no limit).")

    # Courts
    p.add_argument("--court-parquet",
                   default=str(DATA_PROC / "court_law_references.parquet"),
                   help="[courts] Input file (.csv or .parquet, auto-detected) "
                        "with at least 'citation' and 'text' columns. Output "
                        "parquet = every input column + the embedding column.")
    p.add_argument("--max-chars-per-court", type=int, default=0,
                   help="[courts] Truncate the embedded text to this many "
                        "chars (0 = use --max-chars-per-law). Does not affect "
                        "the stored 'text' column.")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode in {"queries", "sub-queries"} and not args.embed_text_col:
        args.embed_text_col = "sub_query_de" if args.mode == "sub-queries" else "query"

    # consolidate: fold leftover shards into --output; no model/server needed.
    if args.mode == "consolidate":
        if not args.output:
            raise SystemExit("consolidate requires --output <parquet path>")
        out = Path(args.output)
        _consolidate(out)
        if not _shard_dir(out).exists():
            print(f"Clean: {out}")
        return

    # Auto-detect backend
    if not args.backend:
        if "qwen" in args.model.lower():
            args.backend = "qwen"
        elif "zembed" in args.model.lower() or "zeroentropy" in args.model.lower():
            args.backend = "zembed"
        elif "voyage" in args.model.lower():
            args.backend = "voyage"
        else:
            raise ValueError("Cannot auto-detect backend from --model; please set --backend explicitly.")

    # Auto-set emb_col
    if not args.emb_col:
        if args.backend == "voyage":
            args.emb_col = "voyage_nano_embedding"
        elif args.backend == "zembed":
            args.emb_col = "zembed_embedding"
        else:
            args.emb_col = "qwen_embedding"

    # Auto-set prompts and suffix
    if args.backend == "voyage":
        args.query_prefix = VOYAGE_QUERY_PROMPT
        args.doc_prefix = VOYAGE_DOC_PROMPT
        if not args.suffix:
            args.suffix = ""
    elif args.backend == "zembed":
        args.query_prefix = ZEMBED_QUERY_PROMPT
        args.doc_prefix = ZEMBED_DOC_PROMPT
        if not args.suffix:
            args.suffix = ZEMBED_SUFFIX
    elif args.backend == "qwen":
        args.query_prefix = QUERY_INSTRUCTION
        args.doc_prefix = ""
        if not args.suffix:
            args.suffix = ""
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    # Default output path
    if not args.output:
        if args.mode in {"queries", "sub-queries"}:
            stem = Path(args.input_parquet).stem
            for tag in ("_voyage_large", "_voyage_nano", "_qwen", "_zembed"):
                stem = stem.replace(tag, "")
            col_tag = args.emb_col.replace("_embedding", "")
            args.output = str(DATA_PROC / f"{stem}_{col_tag}.parquet")
        elif args.mode == "laws":
            if args.backend == "voyage":
                dim_suffix = "" if args.dimensions == 0 else f"_{args.dimensions}"
                args.output = str(DATA_PROC / f"voyage_law_embedding_nano{dim_suffix}.parquet")
            elif args.backend == "zembed":
                args.output = str(DATA_PROC / "law_embedding_zembed.parquet")
            else:
                args.output = str(DATA_PROC / "law_embedding_qwen.parquet")
        else:  # courts
            if args.backend == "voyage":
                dim_suffix = "" if args.dimensions == 0 else f"_{args.dimensions}"
                args.output = str(DATA_PROC / f"court_voyage_nano{dim_suffix}.parquet")
            elif args.backend == "zembed":
                args.output = str(DATA_PROC / "court_embedding_zembed.parquet")
            else:
                args.output = str(DATA_PROC / "court_embedding_qwen.parquet")

    mgr = ServerManager(args)
    try:
        mgr.start()
        model = mgr.client()

        if args.mode in {"queries", "sub-queries"}:
            run_queries(args, model)
        elif args.mode == "laws":
            run_laws(args, model)
        else:
            run_courts(args, model)
    finally:
        mgr.stop()


if __name__ == "__main__":
    main()
