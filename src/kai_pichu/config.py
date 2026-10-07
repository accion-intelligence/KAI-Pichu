from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class ModelConfig(ConfigModel):
    provider: Literal["chat_completions", "responses", "anthropic", "replay"] = "chat_completions"
    model: str = ""
    base_url: str = ""
    api_key_env: str = Field(default="KAI_PICHU_API_KEY",
        description="Environment variable holding the bearer token; empty string means the endpoint needs no key")
    # The coder returns complete replacement text for every file it touches,
    # and for a reasoning provider this budget is shared with thinking tokens.
    # 256k leaves room for both; providers cap it at their own limit.
    max_tokens: int = Field(default=262144, ge=1)
    max_continuations: int = Field(default=2, ge=0, le=5, description=(
        "When a reply is cut off by max_tokens, ask the model to continue it up to this many times "
        "and splice the pieces; each continuation is a model call. 0 disables continuation"))
    temperature: float | None = Field(default=0.2, ge=0)
    timeout_seconds: float = Field(default=120, gt=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    responses: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_endpoint(self) -> ModelConfig:
        if self.provider == "anthropic":
            if not self.model or not self.api_key_env:
                raise ValueError("provider anthropic requires a model and a non-empty api_key_env")
            if self.base_url:
                self._check_url()
        elif self.provider != "replay":
            if not self.model or not self.base_url:
                raise ValueError("live providers require an explicit model and http(s) base_url")
            self._check_url()
        reserved = {"messages", "input", "model", "stream", "max_tokens", "max_output_tokens", "store", "system"}
        if set(self.extra_body) & reserved:
            raise ValueError("extra_body cannot replace model input, system prompt, token limits, stream or store")
        return self

    def _check_url(self) -> None:
        url = urlparse(self.base_url)
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError("base_url must be an http(s) URL")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url cannot contain credentials, a query or a fragment")


class Budget(ConfigModel):
    rounds: int = Field(default=10, ge=1)
    llm_calls: int = Field(default=30, ge=1)
    seconds: float = Field(default=3600, gt=0)
    evaluation_seconds: float = Field(default=300, gt=0)
    acceptance_repeats: int = Field(default=2, ge=1)


class Resources(ConfigModel):
    gpu_policy: Literal["exclusive", "unchecked"] = "exclusive"
    gpu_device: int | None = Field(default=None, ge=0)
    poll_seconds: float = Field(default=1, ge=0.1, le=10)


class ProfileConfig(ConfigModel):
    enabled: bool = False
    case_id: str | None = None
    ncu: str = "ncu"
    # An explicit metric list overrides sections, preserving existing configs.
    metrics: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=lambda: [
        "SpeedOfLight", "LaunchStats", "Occupancy", "MemoryWorkloadAnalysis",
        "SchedulerStats", "WarpStateStats", "ComputeWorkloadAnalysis", "InstructionStats", "SourceCounters",
    ])
    cases_per_round: int | None = Field(default=None, ge=1, description=(
        "How many search cases to profile each round; default every case. When fewer, cases rotate "
        "round by round so each one is captured in turn"))
    import_source: bool = Field(default=True, description=(
        "Pass --import-source yes so per-instruction counters and warp-stall samples map to source lines "
        "when the kernel was built with -lineinfo"))
    report_reader: str | None = None
    ncu_report_dir: str | None = None
    query_rounds: int = Field(default=4, ge=0, le=10)
    queries_per_round: int = Field(default=3, ge=1, le=8)
    query_seconds: float = Field(default=60, gt=0)
    max_rows: int = Field(default=30, ge=1, le=100)
    max_result_chars: int = Field(default=16000, ge=4000, le=100000)

    @model_validator(mode="after")
    def validate_capture(self) -> ProfileConfig:
        if not self.metrics and not self.sections:
            raise ValueError("profiling requires sections or explicit metrics")
        for name in self.metrics + self.sections:
            if not name or name != name.strip() or any(c in name for c in "\n\r,"):
                raise ValueError("profile metric/section names must be nonempty individual identifiers")
        return self


class OptimizeConfig(ConfigModel):
    schema_version: Literal[1] = 1
    coder: ModelConfig
    judge: ModelConfig | None = None
    budget: Budget = Field(default_factory=Budget)
    resources: Resources = Field(default_factory=Resources)
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    max_context_chars: int = Field(default=120000, ge=1000)

    @model_validator(mode="before")
    @classmethod
    def rename_legacy_role(cls, data: Any) -> Any:
        """Read configs written before the coder role was named `generator`, including old runs' config.json."""
        if isinstance(data, dict) and "generator" in data:
            if "coder" in data:
                raise ValueError("config names both coder and its legacy name generator; keep only coder")
            data = {("coder" if key == "generator" else key): value for key, value in data.items()}
        return data


def load_config(path: Path) -> OptimizeConfig:
    return OptimizeConfig.model_validate(yaml.safe_load(path.read_text()))


def missing_api_keys(config: OptimizeConfig) -> list[str]:
    """Names of configured key variables that are unset or empty; replay and keyless endpoints are skipped."""
    names = {model.api_key_env for model in (config.coder, config.judge)
             if model is not None and model.provider != "replay" and model.api_key_env}
    return sorted(name for name in names if not os.environ.get(name))


def require_api_keys(config: OptimizeConfig) -> None:
    missing = missing_api_keys(config)
    if missing:
        raise ValueError("API key environment variable not set: " + ", ".join(missing)
                         + "; export it before running, or set api_key_env to an empty string for an endpoint without authentication")


def example_config() -> dict[str, Any]:
    return OptimizeConfig(coder=ModelConfig(
        model="YOUR_MODEL_NAME", base_url="http://localhost:8000/v1",
    )).model_dump(exclude_none=True)
