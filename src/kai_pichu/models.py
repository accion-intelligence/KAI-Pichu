"""Explicit model transport; no endpoint or API key is discovered implicitly.

Providers: chat_completions and responses speak the OpenAI-shaped HTTP APIs
through urllib; anthropic uses the official ``anthropic`` SDK (optional
dependency); replay serves recorded text for offline tests.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import ModelConfig

# A completion is not streamed and not stored, so re-sending one costs tokens
# and nothing else. Retry only faults that carry no decision from the model.
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 16.0


def retry_delay(error: HTTPError | None, attempt: int) -> float:
    """Honour a delta-seconds Retry-After; otherwise back off exponentially."""
    header = error.headers.get("Retry-After") if error is not None and error.headers else None
    if header:
        try:
            return max(0.0, min(float(header), MAX_BACKOFF_SECONDS))
        except ValueError:
            pass  # The HTTP-date form needs a trusted clock; back off instead.
    return min(BACKOFF_SECONDS * 2 ** attempt, MAX_BACKOFF_SECONDS)


class ModelReplyError(ValueError):
    """The model answered, and the answer cannot be used.

    Truncation, a content filter, an empty body or a response the provider marks
    incomplete all mean a call was made and produced something unusable. That is
    the generator's existing repair case, not a transport fault, so it must reach
    the caller instead of ending the run. A ValueError subclass so existing
    handlers keep working.

    A reply cut off by the output limit carries the text produced so far in
    ``partial_text``; the caller may ask the model to continue it.
    """

    def __init__(self, message: str, *, partial_text: str | None = None):
        super().__init__(message)
        self.partial_text = partial_text

    @property
    def truncated(self) -> bool:
        return self.partial_text is not None


class ModelClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def complete(self, messages: list[dict[str, str]], *, index: int, timeout: float) -> dict[str, Any]:
        config = self.config
        if config.provider == "replay":
            if index >= len(config.responses):
                raise ValueError("replay responses exhausted")
            return {"text": config.responses[index], "usage": {}, "provider": "replay"}
        if config.provider == "anthropic":
            return self._complete_anthropic(messages, timeout=timeout)
        if config.provider == "responses":
            endpoint = "/responses"
            body = {"model": config.model, "input": messages, "max_output_tokens": config.max_tokens,
                    "stream": False, "store": False, **config.extra_body}
        else:
            endpoint = "/chat/completions"
            body = {"model": config.model, "messages": messages, "max_tokens": config.max_tokens,
                    "stream": False, **config.extra_body}
        if config.temperature is not None:
            body["temperature"] = config.temperature
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(config.api_key_env, "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = Request(config.base_url.rstrip("/") + endpoint,
                          data=json.dumps(body).encode(), headers=headers, method="POST")
        # The caller's timeout is the remaining wall budget for this call, so
        # retries share it rather than extending it.
        deadline = time.monotonic() + timeout
        attempt = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"model transport deadline exhausted after {attempt} attempt(s)")
            try:
                with urlopen(request, timeout=min(remaining, config.timeout_seconds)) as response:
                    value = json.loads(response.read(8 * 1024 * 1024))
                break
            except HTTPError as error:
                detail = error.read(8000).decode("utf-8", errors="replace")
                if key:
                    detail = detail.replace(key, "[redacted]")
                message = f"model HTTP {error.code}: {detail}"
                retryable = error.code in RETRY_STATUSES
                delay = retry_delay(error, attempt)
            except (URLError, TimeoutError, OSError) as error:
                # No response reached us, so no model decision was lost.
                message = f"model transport error: {type(error).__name__}: {error}"
                retryable = True
                delay = retry_delay(None, attempt)
            attempt += 1
            if not retryable or attempt >= MAX_ATTEMPTS:
                raise RuntimeError(f"{message} (attempt {attempt}/{MAX_ATTEMPTS})") from None
            if deadline - time.monotonic() <= delay:
                raise RuntimeError(f"{message} (attempt {attempt}/{MAX_ATTEMPTS}; no budget to retry)") from None
            time.sleep(delay)
        if config.provider == "responses":
            output_messages = [item for item in value.get("output", [])
                               if item.get("type") == "message" and item.get("role") == "assistant"]
            if value.get("status") != "completed":
                details = value.get("incomplete_details") or {}
                partial = "".join(part.get("text", "") for item in output_messages
                                  for part in item.get("content", []) if part.get("type") == "output_text")
                cut_off = details.get("reason") == "max_output_tokens" and bool(partial.strip())
                raise ModelReplyError(f"model response incomplete: {value.get('status')}; {details}",
                                      partial_text=partial if cut_off else None)
            final_messages = [item for item in output_messages if item.get("phase") == "final_answer"]
            # Intermediate updates are separate messages, even in JSON mode.
            # Concatenating them can turn two valid objects into invalid JSON or
            # accidentally treat an intermediate query as the final decision.
            answers = final_messages or [item for item in output_messages if item.get("phase") is None]
            texts = {"".join(part.get("text", "") for part in answer.get("content", [])
                             if part.get("type") == "output_text") for answer in answers}
            # Some completed responses repeat an identical final message with a
            # different message ID. Consume it once, retaining both raw messages.
            if len(texts) != 1:
                raise ModelReplyError("model response must contain one final answer message")
            content = texts.pop()
            if not content.strip():
                raise ModelReplyError("model returned no textual candidate/strategy")
            return {"text": content, "usage": value.get("usage", {}),
                    "provider": config.provider, "model": config.model,
                    "response_id": value.get("id"), "finish_reason": value["status"],
                    "transport_attempts": attempt + 1, "output_messages": output_messages}
        choice = value["choices"][0]
        content = choice["message"].get("content")
        if choice.get("finish_reason") in ("length", "content_filter"):
            cut_off = choice["finish_reason"] == "length" and isinstance(content, str) and bool(content.strip())
            raise ModelReplyError(f"model response incomplete: {choice['finish_reason']}",
                                  partial_text=content if cut_off else None)
        if not isinstance(content, str) or not content.strip():
            raise ModelReplyError("model returned no textual candidate/strategy")
        return {"text": content, "usage": value.get("usage", {}),
                "provider": config.provider, "model": config.model,
                "transport_attempts": attempt + 1, "finish_reason": choice.get("finish_reason")}

    def _complete_anthropic(self, messages: list[dict[str, str]], *, timeout: float) -> dict[str, Any]:
        """One Messages API call through the official SDK.

        The system prompt travels in the dedicated ``system`` field. Thinking is
        adaptive unless ``extra_body`` overrides it, and sampling parameters are
        never sent because current Claude models reject them. A refusal or a
        truncated answer is an error: the loop must not treat either as a
        candidate. Fallback models are not enabled, so every recorded response
        comes from the configured model.
        """
        try:
            import anthropic
        except ImportError:
            raise ValueError("provider anthropic needs the official SDK: pip install 'kai-pichu[anthropic]'") from None
        config = self.config
        key = os.environ.get(config.api_key_env, "")
        if not key:
            raise ValueError(f"API key environment variable not set: {config.api_key_env}")
        client = anthropic.Anthropic(api_key=key, base_url=config.base_url or None,
                                     timeout=min(timeout, config.timeout_seconds), max_retries=2)
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        request: dict[str, Any] = {"model": config.model, "max_tokens": config.max_tokens, "messages": turns,
                                   "thinking": {"type": "adaptive"}}
        if system:
            request["system"] = system
        try:
            # Streaming keeps long generations clear of HTTP timeouts.
            with client.messages.stream(**request, extra_body=config.extra_body or None) as stream:
                message = stream.get_final_message()
        except anthropic.APIStatusError as error:
            raise RuntimeError(f"model HTTP {error.status_code}: {error.message}") from None
        except anthropic.APIConnectionError as error:
            raise RuntimeError(f"model connection error: {error}") from None
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise ModelReplyError(f"model refused the request (category: {category})")
        content = "".join(block.text for block in message.content if block.type == "text")
        if message.stop_reason == "max_tokens":
            raise ModelReplyError("model response incomplete: max_tokens",
                                  partial_text=content if content.strip() else None)
        if not content.strip():
            raise ModelReplyError("model returned no textual candidate/strategy")
        return {"text": content, "usage": message.usage.model_dump(), "provider": config.provider,
                "model": message.model, "response_id": message.id, "finish_reason": message.stop_reason,
                "content_blocks": [block.model_dump() for block in message.content]}
