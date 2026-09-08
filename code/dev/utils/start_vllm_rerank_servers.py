#!/usr/bin/env python3
"""Start one or more self-hosted vLLM rerank servers."""

from __future__ import annotations

import argparse
import atexit
import os
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request


def resolve_gpu_groups(args: argparse.Namespace) -> list[str]:
    """Return one CUDA_VISIBLE_DEVICES group per vLLM instance."""
    n = max(1, int(args.num_instances))
    tp = int(args.tensor_parallel_size)
    dp = int(args.data_parallel_size)
    if tp <= 0 or dp <= 0:
        raise SystemExit("--tensor-parallel-size and --data-parallel-size must be > 0.")
    group_size = tp * dp
    total_needed = n * group_size

    raw = (args.gpus or "").strip()
    ids = [part.strip() for part in raw.split(",") if part.strip()] if raw else [
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


def serve_command(args: argparse.Namespace, port: int) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        args.model,
        "--runner",
        "pooling",
        "--host",
        args.host,
        "--port",
        str(port),
        "--served-model-name",
        args.served_model_name,
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--api-key",
        args.api_key,
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(args.data_parallel_size),
    ]
    if args.vllm_extra_args:
        cmd += shlex.split(args.vllm_extra_args)
    return cmd


class VLLMRerankServer:
    def __init__(self, args: argparse.Namespace, port: int, gpu_group: str) -> None:
        self.args = args
        self.port = port
        self.gpu_group = gpu_group
        self.base_url = f"http://{args.host}:{port}"
        self.health_url = f"{self.base_url}/health"
        self.cmd = serve_command(args, port)
        self.proc: subprocess.Popen | None = None
        self.stopped = False

    @property
    def tag(self) -> str:
        return f"[rerank:{self.port} gpu={self.gpu_group}]"

    def start(self) -> None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": self.gpu_group}
        print(f"{self.tag} launching: {' '.join(self.cmd)}", flush=True)
        self.proc = subprocess.Popen(self.cmd, env=env, start_new_session=True)

    def wait_ready(self, timeout_s: int) -> None:
        print(f"{self.tag} waiting for {self.health_url} ...", flush=True)
        started = time.time()
        while True:
            if self.proc is None or self.proc.poll() is not None:
                code = None if self.proc is None else self.proc.returncode
                raise RuntimeError(
                    f"{self.tag} vLLM exited early before ready (code {code})."
                )
            try:
                with urllib.request.urlopen(self.health_url, timeout=2) as resp:
                    if resp.status == 200:
                        print(
                            f"{self.tag} ready in {time.time() - started:.0f}s",
                            flush=True,
                        )
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            if time.time() - started > timeout_s:
                self.stop()
                raise TimeoutError(f"{self.tag} not healthy after {timeout_s}s")
            time.sleep(3)

    def stop(self) -> None:
        if self.stopped or self.proc is None:
            return
        self.stopped = True
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
        print(f"{self.tag} stopped", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="zeroentropy/zerank-2-reranker")
    p.add_argument("--served-model-name", default="zerank-2")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default="vllm-local")
    p.add_argument("--num-instances", type=int, default=1)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--data-parallel-size", type=int, default=1)
    p.add_argument(
        "--gpus",
        default="",
        help="Comma-separated CUDA ids. Assigned in TP*DP groups per instance.",
    )
    p.add_argument("--dtype", default="auto")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--server-timeout", type=int, default=900)
    p.add_argument("--vllm-extra-args", default="")
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.set_defaults(trust_remote_code=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.num_instances <= 0:
        raise SystemExit("--num-instances must be > 0.")
    if args.port <= 0:
        raise SystemExit("--port must be > 0.")

    gpu_groups = resolve_gpu_groups(args)
    servers = [
        VLLMRerankServer(args, port=args.port + i, gpu_group=gpu_groups[i])
        for i in range(args.num_instances)
    ]

    def stop_all() -> None:
        for server in servers:
            server.stop()

    atexit.register(stop_all)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt))

    for server in servers:
        server.start()
    for server in servers:
        server.wait_ready(args.server_timeout)

    urls = ",".join(server.base_url for server in servers)
    print("\nServers ready.", flush=True)
    print(f"export RERANK_BACKEND=vllm", flush=True)
    print(f"export VLLM_RERANK_BASE_URLS={urls}", flush=True)
    print(f"export VLLM_API_KEY={args.api_key}", flush=True)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nStopping rerank servers ...", flush=True)
    finally:
        stop_all()


if __name__ == "__main__":
    main()
