"""OpenAI-compatible provider client."""

import re
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from openai import APIStatusError, APIConnectionError, AsyncOpenAI, RateLimitError

from .base import CompletionResult
from .headers import collect_rl_headers, extract_remaining_requests

_NO_AUTH_SENTINEL = "sentinel-no-op"


class _NoAuth(httpx.Auth):
    """httpx Auth handler that strips the Authorization header.

    Used for providers like opencode_zen that reject invalid auth headers
    but allow unauthenticated requests.
    """

    def auth_flow(self, request: httpx.Request) -> httpx.Request:
        request.headers.pop("authorization", None)
        yield request


def _make_client(base_url: str, api_key: str, no_auth: bool = False) -> AsyncOpenAI:
    """Create an AsyncOpenAI client.

    When *api_key* is empty/the no-auth sentinel, or *no_auth* is ``True``,
    creates a client with no ``Authorization`` header (for providers like
    opencode_zen that reject invalid auth but allow public access).
    """
    if no_auth or not api_key or api_key == "empty-key-placeholder":
        http_client = httpx.AsyncClient(auth=_NoAuth())
        return AsyncOpenAI(
            base_url=base_url,
            api_key=_NO_AUTH_SENTINEL,
            http_client=http_client,
        )
    return AsyncOpenAI(base_url=base_url, api_key=api_key)

_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL)
_MASK_MIN_LEN = 8
_MASK_SHOW = 4


def _mask_key(api_key: str) -> str:
    """Mask API key for safe logging: show last 4 chars only."""
    if len(api_key) <= _MASK_MIN_LEN:
        return "****" + api_key[-_MASK_SHOW:] if len(api_key) > _MASK_SHOW else "****"
    return api_key[:_MASK_SHOW] + "****" + api_key[-_MASK_SHOW:]


def _strip_thinking(text: str) -> str:
    text = _THINK_CLOSED_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


def _make_chunk_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _build_chunk(
    chunk_id: str,
    created: int,
    model: str,
    delta_content: str | None = None,
    delta_role: str | None = None,
    finish_reason: str | None = None,
    index: int = 0,
    **extra: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Build an OpenAI-format streaming chunk dict."""
    chunk: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
    }
    if delta_content is not None or delta_role is not None or finish_reason is not None:
        delta: dict[str, Any] = {}
        if delta_role is not None:
            delta["role"] = delta_role
        if delta_content is not None:
            delta["content"] = delta_content
        chunk["choices"] = [
            {
                "index": index,
                "delta": delta,
                "finish_reason": finish_reason,
            },
        ]
    chunk.update(extra)
    return chunk


def _build_error_chunk(
    chunk_id: str,
    created: int,
    model: str,
    error: str,
    was_429: bool = False,
) -> dict[str, Any]:
    return _build_chunk(
        chunk_id, created, model,
        x_error=error, x_was_429=was_429,
    )


def _make_stream_gen(
    key_data: dict[str, Any],
    messages: list[dict[str, Any]],
    model: str,
    provider: str,
    api_key: str,
    base_url: str,
    strip_thinking: bool,
    no_auth: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> AsyncGenerator[dict[str, Any], None]:
    """Return an async generator that yields OpenAI-format streaming chunks."""
    chunk_id = _make_chunk_id()
    created = int(time.time())

    async def _gen() -> AsyncGenerator[dict[str, Any], None]:
        client = _make_client(base_url=base_url, api_key=api_key, no_auth=no_auth)
        try:
            raw_stream = await client.chat.completions.with_raw_response.create(
                model=model,
                messages=messages,
                stream=True,  # type: ignore[call-overload]
                stream_options={"include_usage": True},
                **kwargs,
            )
            stream = raw_stream.parse()

            async for event in stream:
                choices_list: list[dict[str, Any]] = []
                if event.choices:
                    choice = event.choices[0]
                    delta = choice.delta
                    delta_out: dict[str, Any] = {}
                    if delta is not None:
                        if delta.role is not None:
                            delta_out["role"] = delta.role

                        has_real_content = delta.content is not None
                        if has_real_content:
                            delta_out["content"] = delta.content

                        # reasoning_content — Z.ai/OpenCode Zen/Cloudflare
                        # reasoning         — Ollama
                        rc = getattr(delta, "reasoning_content", None)
                        if rc is not None:
                            delta_out["reasoning_content"] = rc
                        r = getattr(delta, "reasoning", None)
                        if r is not None:
                            delta_out["reasoning"] = r

                        # FreeLLMAPI's normalizeChoices only folds reasoning into content
                        # on the non-streaming path. For streaming, we forward
                        # reasoning_content as its own field and NEVER touch
                        # content — the client accumulates it correctly.
                    finish = str(choice.finish_reason) if choice.finish_reason else None
                    choices_list = [
                        {
                            "index": choice.index,
                            "delta": delta_out,
                            "finish_reason": finish,
                        },
                    ]

                extra: dict[str, Any] = {}
                if event.usage:
                    extra["x_tokens_used"] = event.usage.total_tokens or 0

                yield {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": choices_list,
                    **extra,
                }

        except RateLimitError as e:
            yield _build_error_chunk(
                chunk_id, created, model,
                error=f"Rate limited (429): {str(e)[:150]}",
                was_429=True,
            )
        except APIStatusError as e:
            yield _build_error_chunk(
                chunk_id, created, model,
                error=f"HTTP {e.status_code} from provider {provider}: {str(e)[:160]}",
            )
        except httpx.TimeoutException as e:
            yield _build_error_chunk(
                chunk_id, created, model,
                error=f"Request to {_mask_key(base_url)} timed out: {str(e)[:100]}",
            )
        except httpx.NetworkError as e:
            yield _build_error_chunk(
                chunk_id, created, model,
                error=f"Network error connecting to {_mask_key(base_url)}: {str(e)[:100]}",
            )
        except httpx.HTTPStatusError as e:
            yield _build_error_chunk(
                chunk_id, created, model,
                error=f"HTTP {e.status_code} from provider {provider}: {str(e)[:160]}",
            )
        except httpx.RequestError as e:
            yield _build_error_chunk(
                chunk_id, created, model,
                error=f"Request error: {str(e)[:100]}",
            )
        except APIConnectionError as e:
            yield _build_error_chunk(
                chunk_id, created, model,
                error=f"Connection error to {_mask_key(base_url)}: {str(e)[:150]}",
            )
        except Exception as e:  # noqa: BLE001
            yield _build_error_chunk(
                chunk_id, created, model,
                error=f"Unexpected error: {type(e).__name__}: {str(e)[:150]}",
            )

    return _gen()


async def complete(
    key_data: dict[str, Any],
    messages: list[dict[str, Any]],
    stream: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> CompletionResult | AsyncGenerator[dict[str, Any], None]:
    """Call an OpenAI-compatible provider with the given key and messages.

    When *stream* is ``False`` (default), returns a :class:`CompletionResult`.
    When *stream* is ``True``, returns an async generator that yields dicts in
    OpenAI streaming chunk format.
    """
    strip_thinking = kwargs.pop("strip_thinking", True)
    model = kwargs.pop("model", None) or key_data["model"]

    if stream:
        return _make_stream_gen(
            key_data, messages, model,
            provider=key_data.get("provider", ""),
            api_key=key_data["api_key"] or "empty-key-placeholder",
            base_url=key_data["base_url"],
            strip_thinking=strip_thinking,
            no_auth=key_data.get("no_auth", False),
            **kwargs,
        )

    provider = key_data.get("provider", "")
    api_key = key_data["api_key"] or "empty-key-placeholder"
    client = _make_client(base_url=key_data["base_url"], api_key=api_key, no_auth=key_data.get("no_auth", False))
    try:
        raw = await client.chat.completions.with_raw_response.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        resp = raw.parse()
        rl_headers = collect_rl_headers(raw.headers)
        remaining = extract_remaining_requests(provider, rl_headers)
        msg = resp.choices[0].message
        text = msg.content or ""
        reasoning_content: str | None = None
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            reasoning_content = msg.reasoning_content
        if not text and reasoning_content:
            text = reasoning_content
        if strip_thinking:
            text = _strip_thinking(text)
        return CompletionResult(
            text=text,
            tokens_used=resp.usage.total_tokens if resp.usage else 0,
            was_429=False,
            remaining_requests=remaining,
            rate_limit_headers=rl_headers,
            reasoning_content=reasoning_content,
        )
    except RateLimitError as e:
        rl_headers = {}
        if hasattr(e, "response") and e.response is not None:
            rl_headers = collect_rl_headers(e.response.headers)
        return CompletionResult(
            text="", tokens_used=0, was_429=True,
            error=f"Rate limited (429): {str(e)[:150]}", rate_limit_headers=rl_headers,
        )
    except APIStatusError as e:
        return CompletionResult(
            text="", tokens_used=0, was_429=False,
            error=f"HTTP {e.status_code} from provider {key_data.get('provider','?')}: {str(e)[:160]}",
        )
    except httpx.TimeoutException as e:
        return CompletionResult(
            text="", tokens_used=0, was_429=False,
            error=f"Request to {_mask_key(key_data.get('base_url',''))} timed out: {str(e)[:100]}",
        )
    except httpx.NetworkError as e:
        return CompletionResult(
            text="", tokens_used=0, was_429=False,
            error=f"Network error connecting to {_mask_key(key_data.get('base_url',''))}: {str(e)[:100]}",
        )
    except httpx.HTTPStatusError as e:
        return CompletionResult(
            text="", tokens_used=0, was_429=False,
            error=f"HTTP {e.status_code} from provider {key_data.get('provider','?')}: {str(e)[:160]}",
        )
    except httpx.RequestError as e:
        return CompletionResult(
            text="", tokens_used=0, was_429=False,
            error=f"Request error: {str(e)[:100]}",
        )
    except APIConnectionError as e:
        return CompletionResult(
            text="", tokens_used=0, was_429=False,
            error=f"Connection error to {_mask_key(key_data.get('base_url',''))}: {str(e)[:150]}",
        )
    # Broad catch to always return structured CompletionResult, not crash
    except Exception as e:  # noqa: BLE001
        return CompletionResult(
            text="", tokens_used=0, was_429=False,
            error=f"Unexpected error: {type(e).__name__}: {str(e)[:150]}",
        )
