"""Load and validate configs/experiments.toml into frozen dataclasses."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DEFAULT_CONFIG = Path("configs/experiments.toml")

Strategy = Literal["fixed", "structure"]
RetrievalMode = Literal["top_k", "token_budget"]


class ConfigError(ValueError):
    """Raised for any invalid or missing configuration."""


@dataclass(frozen=True)
class Settings:
    document: Path
    answer_key: Path
    results_dir: Path
    cache_dir: Path
    k: int
    retrieve_n: int
    token_budget: int


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    dimensions: int
    document_template: str
    query_template: str


@dataclass(frozen=True)
class LLMConfig:
    name: str
    provider: str
    model: str
    api_key_env: str
    base_url: str | None
    temperature: float


@dataclass(frozen=True)
class FixedParams:
    size_chars: int
    overlap_chars: int


@dataclass(frozen=True)
class StructureParams:
    min_tokens: int
    max_tokens: int
    overlap_tokens: int
    prefix: bool


@dataclass(frozen=True)
class RunConfig:
    id: str
    strategy: Strategy
    params: FixedParams | StructureParams
    retrieval: RetrievalMode
    generate: bool


@dataclass(frozen=True)
class Config:
    settings: Settings
    embedding: EmbeddingConfig
    llm: LLMConfig
    llm_profiles: dict[str, LLMConfig]
    runs: dict[str, RunConfig]

    def run(self, run_id: str) -> RunConfig:
        try:
            return self.runs[run_id]
        except KeyError:
            known = ", ".join(self.runs)
            raise ConfigError(f"Unknown run '{run_id}'. Known runs: {known}") from None


def _require(table: dict[str, Any], key: str, kind: type | tuple[type, ...], where: str) -> Any:
    if key not in table:
        raise ConfigError(f"[{where}] missing required key '{key}'")
    value = table[key]
    # bool is a subclass of int; don't let `k = true` pass as an int.
    if isinstance(value, bool) and kind is not bool:
        raise ConfigError(f"[{where}] '{key}' must be {_kind_name(kind)}, got a boolean")
    if not isinstance(value, kind):
        raise ConfigError(
            f"[{where}] '{key}' must be {_kind_name(kind)}, got {type(value).__name__}"
        )
    return value


def _kind_name(kind: type | tuple[type, ...]) -> str:
    if isinstance(kind, tuple):
        return " or ".join(k.__name__ for k in kind)
    return kind.__name__


def _positive(value: int, key: str, where: str) -> int:
    if value <= 0:
        raise ConfigError(f"[{where}] '{key}' must be > 0, got {value}")
    return value


def _check_unknown(table: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(table) - allowed
    if unknown:
        raise ConfigError(f"[{where}] unknown key(s): {', '.join(sorted(unknown))}")


def _parse_settings(raw: dict[str, Any]) -> Settings:
    where = "settings"
    _check_unknown(
        raw,
        {"document", "answer_key", "results_dir", "cache_dir", "k", "retrieve_n", "token_budget"},
        where,
    )
    k = _positive(_require(raw, "k", int, where), "k", where)
    retrieve_n = _positive(_require(raw, "retrieve_n", int, where), "retrieve_n", where)
    if retrieve_n < k:
        raise ConfigError(f"[{where}] retrieve_n ({retrieve_n}) must be >= k ({k})")
    return Settings(
        document=Path(_require(raw, "document", str, where)),
        answer_key=Path(_require(raw, "answer_key", str, where)),
        results_dir=Path(_require(raw, "results_dir", str, where)),
        cache_dir=Path(_require(raw, "cache_dir", str, where)),
        k=k,
        retrieve_n=retrieve_n,
        token_budget=_positive(_require(raw, "token_budget", int, where), "token_budget", where),
    )


def _parse_embedding(raw: dict[str, Any]) -> EmbeddingConfig:
    where = "embedding"
    _check_unknown(
        raw, {"provider", "model", "dimensions", "document_template", "query_template"}, where
    )
    provider = _require(raw, "provider", str, where)
    if provider != "gemini":
        raise ConfigError(f"[{where}] unsupported provider '{provider}' (supported: gemini)")
    templates = {}
    for key in ("document_template", "query_template"):
        template = _require(raw, key, str, where)
        if "{text}" not in template:
            raise ConfigError(f"[{where}] '{key}' must contain '{{text}}'")
        templates[key] = template
    return EmbeddingConfig(
        provider=provider,
        model=_require(raw, "model", str, where),
        dimensions=_positive(_require(raw, "dimensions", int, where), "dimensions", where),
        **templates,
    )


def _parse_llms(generation: dict[str, Any], profiles: dict[str, Any]) -> dict[str, LLMConfig]:
    _check_unknown(generation, {"active", "temperature"}, "generation")
    temperature = _require(generation, "temperature", (int, float), "generation")
    if not profiles:
        raise ConfigError("[llm] at least one [llm.<name>] profile is required")
    parsed = {}
    for name, raw in profiles.items():
        where = f"llm.{name}"
        _check_unknown(raw, {"provider", "model", "api_key_env", "base_url"}, where)
        provider = _require(raw, "provider", str, where)
        if provider not in ("gemini", "openai_compat"):
            raise ConfigError(
                f"[{where}] unsupported provider '{provider}' (supported: gemini, openai_compat)"
            )
        base_url = raw.get("base_url")
        if provider == "openai_compat" and not base_url:
            raise ConfigError(f"[{where}] 'base_url' is required for openai_compat")
        parsed[name] = LLMConfig(
            name=name,
            provider=provider,
            model=_require(raw, "model", str, where),
            api_key_env=_require(raw, "api_key_env", str, where),
            base_url=base_url,
            temperature=float(temperature),
        )
    return parsed


def _parse_run(raw: dict[str, Any]) -> RunConfig:
    run_id = raw.get("id", "<missing id>")
    where = f"runs.{run_id}"
    common = {"id", "strategy", "retrieval", "generate"}
    _require(raw, "id", str, where)
    strategy = _require(raw, "strategy", str, where)
    retrieval = _require(raw, "retrieval", str, where)
    if retrieval not in ("top_k", "token_budget"):
        raise ConfigError(f"[{where}] retrieval must be 'top_k' or 'token_budget'")

    params: FixedParams | StructureParams
    if strategy == "fixed":
        _check_unknown(raw, common | {"size_chars", "overlap_chars"}, where)
        size = _positive(_require(raw, "size_chars", int, where), "size_chars", where)
        overlap = _require(raw, "overlap_chars", int, where)
        if not 0 <= overlap < size:
            raise ConfigError(f"[{where}] overlap_chars must be in [0, size_chars), got {overlap}")
        params = FixedParams(size_chars=size, overlap_chars=overlap)
    elif strategy == "structure":
        _check_unknown(
            raw, common | {"min_tokens", "max_tokens", "overlap_tokens", "prefix"}, where
        )
        min_t = _require(raw, "min_tokens", int, where)
        max_t = _positive(_require(raw, "max_tokens", int, where), "max_tokens", where)
        overlap = _require(raw, "overlap_tokens", int, where)
        if not 0 <= min_t < max_t:
            raise ConfigError(f"[{where}] need 0 <= min_tokens < max_tokens")
        if not 0 <= overlap < max_t:
            raise ConfigError(f"[{where}] overlap_tokens must be in [0, max_tokens)")
        params = StructureParams(
            min_tokens=min_t,
            max_tokens=max_t,
            overlap_tokens=overlap,
            prefix=_require(raw, "prefix", bool, where),
        )
    else:
        raise ConfigError(f"[{where}] strategy must be 'fixed' or 'structure', got '{strategy}'")

    return RunConfig(
        id=run_id,
        strategy=strategy,  # type: ignore[arg-type]
        params=params,
        retrieval=retrieval,  # type: ignore[arg-type]
        generate=_require(raw, "generate", bool, where),
    )


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc

    _check_unknown(raw, {"settings", "embedding", "generation", "llm", "runs"}, "top level")
    for section in ("settings", "embedding", "generation", "llm", "runs"):
        if section not in raw:
            raise ConfigError(f"missing [{section}] section")

    profiles = _parse_llms(raw["generation"], raw["llm"])
    active = _require(raw["generation"], "active", str, "generation")
    if active not in profiles:
        raise ConfigError(
            f"[generation] active = '{active}' has no [llm.{active}] profile "
            f"(available: {', '.join(profiles)})"
        )

    runs: dict[str, RunConfig] = {}
    for raw_run in raw["runs"]:
        run = _parse_run(raw_run)
        if run.id in runs:
            raise ConfigError(f"duplicate run id '{run.id}'")
        runs[run.id] = run
    if not runs:
        raise ConfigError("no [[runs]] defined")

    return Config(
        settings=_parse_settings(raw["settings"]),
        embedding=_parse_embedding(raw["embedding"]),
        llm=profiles[active],
        llm_profiles=profiles,
        runs=runs,
    )


def require_env(name: str) -> str:
    """Return an env var, or fail with a hint. Call only when an API call is about to happen."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"Environment variable {name} is not set. Copy .env.example to .env and add it."
        )
    return value
