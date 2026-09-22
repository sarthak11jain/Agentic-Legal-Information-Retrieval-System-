"""ZeroEntropy embedding helper utilities."""

from __future__ import annotations

import base64
import json
import os
import struct
import urllib.error
import urllib.request
from typing import Literal

from openai import OpenAI


ZEROENTROPY_EMBED_URL = "https://api.zeroentropy.dev/v1/models/embed"
DEFAULT_VLLM_BASE_URL = "http://localhost:20128/v1"
DEFAULT_MODEL = "zembed-1"
DEFAULT_VLLM_MODEL = "zembed-1"
DEFAULT_DIMENSIONS = 2560
API_KEY_ENV = "ZEROENTROPY_API_KEY"
VLLM_API_KEY_ENV = "VLLM_API_KEY"
VLLM_BASE_URL_ENV = "VLLM_BASE_URL"

EncodingFormat = Literal["float", "base64"]
InputType = Literal["query", "document"]


def get_api_key(api_key: str | None = None) -> str:
    """Resolve a ZeroEntropy API key from an argument or environment variable."""
    if api_key:
        return api_key

    value = os.environ.get(API_KEY_ENV, "").strip()
    if value:
        return value

    raise ValueError(f"ZeroEntropy API key is required. Set {API_KEY_ENV}.")


def decode_embedding(embedding: object, encoding_format: EncodingFormat) -> list[float]:
    """Decode a ZeroEntropy embedding into Python floats."""
    if encoding_format == "float":
        return [float(value) for value in embedding]

    raw = base64.b64decode(str(embedding))
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def get_vllm_client(
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_s: int = 120,
) -> OpenAI:
    """Create an OpenAI-compatible client for a self-hosted vLLM server."""
    resolved_base_url = (
        base_url or os.environ.get(VLLM_BASE_URL_ENV) or DEFAULT_VLLM_BASE_URL
    )
    resolved_api_key = api_key or os.environ.get(VLLM_API_KEY_ENV) or "vllm-local"
    return OpenAI(
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        timeout=timeout_s,
    )


def call_vllm_embedder(
    text: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = DEFAULT_VLLM_MODEL,
    dimensions: int = 0,
    timeout_s: int = 120,
    client: OpenAI | None = None,
) -> list[float]:
    """Embed a single text with a self-hosted vLLM /v1/embeddings endpoint."""
    active_client = client or get_vllm_client(
        api_key=api_key,
        base_url=base_url,
        timeout_s=timeout_s,
    )
    request: dict[str, object] = {"model": model, "input": [text]}
    if dimensions > 0:
        request["dimensions"] = dimensions
    response = active_client.embeddings.create(**request)
    data = getattr(response, "data", []) or []
    if len(data) != 1:
        raise RuntimeError(f"Expected 1 embedding, got {len(data)}")
    return [float(value) for value in data[0].embedding]


def call_embedder(
    text: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = DEFAULT_MODEL,
    dimensions: int = DEFAULT_DIMENSIONS,
    input_type: InputType = "query",
    encoding_format: EncodingFormat = "float",
    timeout_s: int = 120,
    backend: Literal["zeroentropy", "vllm"] = "zeroentropy",
    client: OpenAI | None = None,
) -> list[float]:
    """Embed a single text with ZeroEntropy or a self-hosted vLLM backend."""
    if backend == "vllm":
        vllm_model = model if model != DEFAULT_MODEL else DEFAULT_VLLM_MODEL
        vllm_dimensions = 0 if dimensions == DEFAULT_DIMENSIONS else dimensions
        return call_vllm_embedder(
            text,
            api_key=api_key,
            base_url=base_url,
            model=vllm_model,
            dimensions=vllm_dimensions,
            timeout_s=timeout_s,
            client=client,
        )
    if backend != "zeroentropy":
        raise ValueError(f"Unsupported embedding backend: {backend}")

    payload = {
        "model": model,
        "input_type": input_type,
        "input": [text],
        "dimensions": dimensions,
        "encoding_format": encoding_format,
    }
    request = urllib.request.Request(
        ZEROENTROPY_EMBED_URL,
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

    results = body.get("results", [])
    if len(results) != 1:
        raise RuntimeError(f"Expected 1 embedding, got {len(results)}")

    return decode_embedding(results[0]["embedding"], encoding_format)


def call_embedder_batch(
    texts: list[str],
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    dimensions: int = DEFAULT_DIMENSIONS,
    input_type: InputType = "query",
    encoding_format: EncodingFormat = "float",
    latency: Literal["fast", "slow"] | None = None,
    timeout_s: int = 120,
) -> list[list[float]]:
    """Embed a batch of texts with ZeroEntropy."""
    if not texts:
        return []

    payload: dict[str, object] = {
        "model": model,
        "input_type": input_type,
        "input": texts,
        "dimensions": dimensions,
        "encoding_format": encoding_format,
    }
    if latency:
        payload["latency"] = latency

    request = urllib.request.Request(
        ZEROENTROPY_EMBED_URL,
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

    results = body.get("results", [])
    if len(results) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(results)}")

    return [
        decode_embedding(result["embedding"], encoding_format)
        for result in results
    ]
