"""OpenRouter LLM helper utilities."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
FIREWORKS_CHAT_COMPLETIONS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_VLLM_BASE_URL = "http://localhost:20128/v1"
DEFAULT_MODEL = "mistralai/ministral-14b-2512"
DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_FIREWORKS_MODEL = "accounts/rishav01iitd/deployments/wltiy7wk"
DEFAULT_VLLM_MODEL = "qwen36"

Message = Mapping[str, Any]


def get_client(api_key: str | None = None, base_url: str = OPENROUTER_BASE_URL) -> OpenAI:
    """Create an OpenRouter-backed OpenAI client."""
    resolved_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not resolved_api_key:
        raise ValueError("OPENROUTER_API_KEY is required in the environment or api_key.")

    default_headers = {
        key: value
        for key, value in {
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL"),
            "X-OpenRouter-Title": os.environ.get("OPENROUTER_APP_NAME"),
        }.items()
        if value
    }

    return OpenAI(
        base_url=base_url,
        api_key=resolved_api_key,
        default_headers=default_headers or None,
    )


def get_openai_client(api_key: str | None = None) -> OpenAI:
    """Create a direct OpenAI client."""
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OPENAI_API_KEY is required in the environment or api_key.")

    return OpenAI(api_key=resolved_api_key)


def get_vllm_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    """Create an OpenAI-compatible client for a self-hosted vLLM server."""
    resolved_base_url = base_url or os.environ.get("VLLM_BASE_URL") or DEFAULT_VLLM_BASE_URL
    resolved_api_key = api_key or os.environ.get("VLLM_API_KEY") or "vllm-local"
    return OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)


def call_llm(
    messages: Sequence[Message],
    *,
    model: str = DEFAULT_MODEL,
    backend: str = "openrouter",
    reasoning_enabled: bool = True,
    openai_reasoning_effort: str = "none",
    client: OpenAI | None = None,
    **kwargs: Any,
) -> Any:
    """Call an LLM through OpenRouter chat completions or OpenAI Responses."""
    if backend == "openai":
        return call_openai_responses(
            messages,
            model=model,
            reasoning_effort=openai_reasoning_effort if not reasoning_enabled else "medium",
            client=client,
            **kwargs,
        )
    if backend == "fireworks":
        return call_fireworks_chat(
            messages,
            model=model,
            **kwargs,
        )
    if backend == "vllm":
        return call_vllm_chat(
            messages,
            model=model,
            client=client,
            **kwargs,
        )
    if backend != "openrouter":
        raise ValueError(f"Unsupported LLM backend: {backend}")

    active_client = client or get_client()
    extra_body = kwargs.pop("extra_body", {})
    top_k = kwargs.pop("top_k", None)
    if top_k is not None:
        extra_body = {**extra_body, "top_k": top_k}

    if reasoning_enabled:
        extra_body = {**extra_body, "reasoning": {"enabled": True}}

    # OpenRouter supports provider-specific fields that the OpenAI SDK types do not model.
    completion_create = cast(Any, active_client.chat.completions.create)
    return completion_create(
        model=model,
        messages=list(messages),
        extra_body=extra_body,
        **kwargs,
    )


def call_openai_responses(
    messages: Sequence[Message],
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    reasoning_effort: str = "none",
    client: OpenAI | None = None,
    **kwargs: Any,
) -> Any:
    """Call OpenAI Responses API and normalize the result to the trainer's expected shape."""
    active_client = client or get_openai_client()
    max_tokens = kwargs.pop("max_tokens", None)
    extra_body = kwargs.pop("extra_body", None)
    kwargs.pop("provider", None)
    kwargs.pop("top_k", None)
    kwargs.pop("presence_penalty", None)
    kwargs.pop("frequency_penalty", None)

    request: dict[str, Any] = {
        "model": model,
        "input": messages_to_response_input(messages),
        "reasoning": {"effort": reasoning_effort},
    }
    if max_tokens is not None:
        request["max_output_tokens"] = max_tokens
    if extra_body:
        # OpenRouter-only fields such as provider routing are ignored for direct OpenAI calls.
        pass
    request.update(kwargs)

    response = active_client.responses.create(**request)
    return normalize_openai_response(response)


def call_fireworks_chat(
    messages: Sequence[Message],
    *,
    model: str = DEFAULT_FIREWORKS_MODEL,
    api_key: str | None = None,
    url: str = FIREWORKS_CHAT_COMPLETIONS_URL,
    timeout: int = 300,
    **kwargs: Any,
) -> Any:
    """Call Fireworks chat completions and normalize the result to the trainer shape."""
    resolved_api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
    if not resolved_api_key:
        raise ValueError("FIREWORKS_API_KEY is required in the environment or api_key.")

    max_tokens = kwargs.pop("max_tokens", None)
    extra_body = kwargs.pop("extra_body", None)
    kwargs.pop("openai_reasoning_effort", None)

    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if extra_body:
        # OpenRouter-only fields such as provider routing are ignored for direct Fireworks calls.
        pass
    payload.update(kwargs)

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved_api_key}",
            "User-Agent": "python-requests/2.32.3",
        },
        data=json_dumps(payload).encode("utf-8"),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Fireworks API error {error.code}: {body}") from error

    return normalize_chat_completion_dict(json_loads(body))


def call_vllm_chat(
    messages: Sequence[Message],
    *,
    model: str = DEFAULT_VLLM_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    client: OpenAI | None = None,
    **kwargs: Any,
) -> Any:
    """Call a self-hosted vLLM OpenAI-compatible chat completions endpoint."""
    active_client = client or get_vllm_client(api_key=api_key, base_url=base_url)

    extra_body = dict(kwargs.pop("extra_body", {}) or {})
    # Provider routing is OpenRouter-specific and can make vLLM reject requests.
    extra_body.pop("provider", None)

    top_k = kwargs.pop("top_k", None)
    min_p = kwargs.pop("min_p", None)
    repetition_penalty = kwargs.pop("repetition_penalty", None)
    kwargs.pop("openai_reasoning_effort", None)
    kwargs.pop("provider", None)

    if top_k is not None:
        extra_body["top_k"] = top_k
    if min_p is not None:
        extra_body["min_p"] = min_p
    if repetition_penalty is not None:
        extra_body["repetition_penalty"] = repetition_penalty

    completion_create = cast(Any, active_client.chat.completions.create)
    request: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        **kwargs,
    }
    if extra_body:
        request["extra_body"] = extra_body
    return completion_create(**request)


def normalize_chat_completion_dict(payload: Mapping[str, Any]) -> Any:
    choices = []
    for choice in payload.get("choices", []) or []:
        message = choice.get("message", {}) or {}
        choices.append(
            SimpleNamespace(
                message=SimpleNamespace(content=str(message.get("content", "") or "")),
                finish_reason=str(choice.get("finish_reason", "") or ""),
            )
        )

    usage_payload = payload.get("usage", {}) or {}
    return SimpleNamespace(
        choices=choices,
        usage=SimpleNamespace(
            prompt_tokens=usage_payload.get("prompt_tokens"),
            completion_tokens=usage_payload.get("completion_tokens"),
            total_tokens=usage_payload.get("total_tokens"),
        ),
        raw_response=payload,
    )


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value)


def json_loads(value: str) -> Any:
    import json

    return json.loads(value)


def messages_to_response_input(messages: Sequence[Message]) -> Any:
    if len(messages) == 1 and messages[0].get("role") == "user":
        content = messages[0].get("content", "")
        if isinstance(content, str):
            return content

    converted = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            converted.append(
                {
                    "role": message.get("role", "user"),
                    "content": [{"type": "input_text", "text": content}],
                }
            )
        else:
            converted.append(message)
    return converted


def normalize_openai_response(response: Any) -> Any:
    text = str(getattr(response, "output_text", "") or "")
    if not text:
        text = extract_response_text(response)

    status = str(getattr(response, "status", "") or "")
    incomplete_details = getattr(response, "incomplete_details", None)
    incomplete_reason = (
        str(getattr(incomplete_details, "reason", "") or "")
        if incomplete_details is not None
        else ""
    )
    finish_reason = "stop" if status == "completed" else status
    if incomplete_reason in {"max_output_tokens", "max_tokens"}:
        finish_reason = "length"

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
    total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        raw_response=response,
    )


def extract_response_text(response: Any) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts)
