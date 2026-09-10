"""Explicit model transport; no endpoint or API key is discovered implicitly."""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .config import ModelConfig


class ModelClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def complete(self, messages: list[dict[str, str]], *, index: int, timeout: float) -> dict[str, Any]:
        config = self.config
        if config.provider == "replay":
            if index >= len(config.responses):
                raise ValueError("replay responses exhausted")
            return {"text": config.responses[index], "usage": {}, "provider": "replay"}
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
        try:
            with urlopen(request, timeout=min(timeout, config.timeout_seconds)) as response:
                value = json.loads(response.read(8 * 1024 * 1024))
        except HTTPError as error:
            detail = error.read(8000).decode("utf-8", errors="replace")
            if key:
                detail = detail.replace(key, "[redacted]")
            raise RuntimeError(f"model HTTP {error.code}: {detail}") from None
        if config.provider == "responses":
            if value.get("status") != "completed":
                raise ValueError(f"model response incomplete: {value.get('status')}; {value.get('incomplete_details')}")
            output_messages = [item for item in value.get("output", [])
                               if item.get("type") == "message" and item.get("role") == "assistant"]
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
                raise ValueError("model response must contain one final answer message")
            content = texts.pop()
            if not content.strip():
                raise ValueError("model returned no textual candidate/strategy")
            return {"text": content, "usage": value.get("usage", {}),
                    "provider": config.provider, "model": config.model,
                    "response_id": value.get("id"), "finish_reason": value["status"],
                    "output_messages": output_messages}
        choice = value["choices"][0]
        if choice.get("finish_reason") in ("length", "content_filter"):
            raise ValueError(f"model response incomplete: {choice['finish_reason']}")
        content = choice["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("model returned no textual candidate/strategy")
        return {"text": content, "usage": value.get("usage", {}),
                "provider": config.provider, "model": config.model,
                "finish_reason": choice.get("finish_reason")}
