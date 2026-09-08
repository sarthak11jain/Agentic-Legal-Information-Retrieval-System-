"""ZeroEntropy reranker helper utilities."""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from typing import Literal, TypedDict


ZEROENTROPY_RERANK_URL = "https://api.zeroentropy.dev/v1/models/rerank"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "zerank-2"
DEFAULT_VLLM_MODEL = "zerank-2"
API_KEY_ENV = "ZEROENTROPY_API_KEY"
VLLM_API_KEY_ENV = "VLLM_API_KEY"
VLLM_BASE_URL_ENV = "VLLM_RERANK_BASE_URL"
VLLM_BASE_URLS_ENV = "VLLM_RERANK_BASE_URLS"

_VLLM_URL_LOCK = threading.Lock()
_VLLM_URL_INDEX = 0

Latency = Literal["fast", "slow"]


class RerankResult(TypedDict):
    """A scored document reference returned by ZeroEntropy."""

    index: int
    relevance_score: float


def get_api_key(api_key: str | None = None) -> str:
    """Resolve a ZeroEntropy API key from an argument or environment variable."""
    if api_key:
        return api_key

    value = os.environ.get(API_KEY_ENV, "").strip()
    if value:
        return value

    raise ValueError(f"ZeroEntropy API key is required. Set {API_KEY_ENV}.")


def _normalize_vllm_base_url(value: str) -> str:
    # vLLM rerank/score examples mount at /rerank, not /v1/rerank.
    return value.strip().rstrip("/").removesuffix("/v1")


def get_vllm_base_urls(base_url: str | None = None) -> list[str]:
    """Resolve one or more self-hosted vLLM rerank base URLs."""
    raw = (
        base_url
        or os.environ.get(VLLM_BASE_URLS_ENV)
        or os.environ.get(VLLM_BASE_URL_ENV)
        or os.environ.get("VLLM_BASE_URL")
        or DEFAULT_VLLM_BASE_URL
    )
    urls = [_normalize_vllm_base_url(part) for part in raw.split(",") if part.strip()]
    return urls or [DEFAULT_VLLM_BASE_URL]


def get_vllm_base_url(base_url: str | None = None) -> str:
    """Resolve one self-hosted vLLM rerank base URL using round-robin."""
    urls = get_vllm_base_urls(base_url)
    if len(urls) == 1:
        return urls[0]
    global _VLLM_URL_INDEX
    with _VLLM_URL_LOCK:
        url = urls[_VLLM_URL_INDEX % len(urls)]
        _VLLM_URL_INDEX += 1
    return url


def call_vllm_reranker(
    query: str,
    documents: list[str],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = DEFAULT_VLLM_MODEL,
    top_n: int | None = None,
    timeout_s: int = 120,
) -> list[RerankResult]:
    """Rerank documents with a self-hosted vLLM /rerank endpoint."""
    payload: dict[str, object] = {
        "model": model,
        "query": query,
        "documents": documents,
    }
    if top_n is not None:
        payload["top_n"] = top_n

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    resolved_api_key = api_key or os.environ.get(VLLM_API_KEY_ENV, "vllm-local").strip()
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"

    url = f"{get_vllm_base_url(base_url)}/rerank"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace") if error.fp else ""
        raise RuntimeError(f"vLLM rerank HTTP {error.code}: {detail[:500]}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"vLLM rerank request failed: {error}") from error

    return [
        {
            "index": int(result["index"]),
            "relevance_score": float(result["relevance_score"]),
        }
        for result in body.get("results", [])
    ]


def call_reranker(
    query: str,
    documents: list[str],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = DEFAULT_MODEL,
    top_n: int | None = None,
    latency: Latency | None = None,
    timeout_s: int = 120,
    backend: Literal["zeroentropy", "vllm"] | None = None,
) -> list[RerankResult]:
    """Rerank documents for a query with ZeroEntropy."""
    backend = backend or os.getenv("RERANK_BACKEND", "zeroentropy")
    if backend == "vllm":
        vllm_model = model if model != DEFAULT_MODEL else DEFAULT_VLLM_MODEL
        return call_vllm_reranker(
            query,
            documents,
            api_key=api_key,
            base_url=base_url,
            model=vllm_model,
            top_n=top_n,
            timeout_s=timeout_s,
        )
    if backend != "zeroentropy":
        raise ValueError(f"Unsupported reranker backend: {backend}")

    payload: dict[str, object] = {
        "model": model,
        "query": query,
        "documents": documents,
    }
    if top_n is not None:
        payload["top_n"] = top_n
    if latency is not None:
        payload["latency"] = latency

    request = urllib.request.Request(
        ZEROENTROPY_RERANK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {get_api_key(api_key)}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace") if error.fp else ""
        raise RuntimeError(f"ZeroEntropy HTTP {error.code}: {detail[:500]}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"ZeroEntropy request failed: {error}") from error

    return [
        {
            "index": int(result["index"]),
            "relevance_score": float(result["relevance_score"]),
        }
        for result in body.get("results", [])
    ]
