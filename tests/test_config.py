from pathlib import Path

import pytest

from rag_chunking.config import (
    DEFAULT_CONFIG,
    ConfigError,
    FixedParams,
    StructureParams,
    load_config,
    require_env,
)

REAL = DEFAULT_CONFIG.read_text(encoding="utf-8")


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "experiments.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_real_config_loads():
    config = load_config()
    assert len(config.runs) == 8
    assert isinstance(config.run("A-500").params, FixedParams)
    params = config.run("B-min100").params
    assert isinstance(params, StructureParams) and params.prefix
    assert config.llm.name == "gemini"


def test_unknown_run_lists_known_runs():
    with pytest.raises(ConfigError, match="Known runs: A-500"):
        load_config().run("nope")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("overlap_chars = 50", "overlap_chars = 500", "overlap_chars"),
        ('strategy = "fixed"', 'strategy = "magic"', "strategy must be"),
        ("min_tokens = 100", "min_tokens = 600", "min_tokens < max_tokens"),
        ('retrieval = "top_k"', 'retrieval = "best"', "retrieval must be"),
        ("k = 3", "k = 9", "retrieve_n"),
        ("k = 3", "k = true", "boolean"),
        ('active = "gemini"', 'active = "gpt"', "no \\[llm.gpt\\] profile"),
        ("{text}", "{txt}", "must contain"),
        ("prefix = true", "prefix = true\ncolour = 1", "unknown key"),
    ],
)
def test_broken_config_fails_readably(tmp_path, old, new, message):
    assert old in REAL
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, REAL.replace(old, new, 1)))


def test_duplicate_run_id(tmp_path):
    broken = REAL.replace('id = "A-300"', 'id = "A-500"')
    with pytest.raises(ConfigError, match="duplicate run id"):
        load_config(_write(tmp_path, broken))


def test_invalid_toml(tmp_path):
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(_write(tmp_path, "[settings\n"))


def test_require_env(monkeypatch):
    monkeypatch.delenv("RAG_TEST_KEY", raising=False)
    with pytest.raises(ConfigError, match=r"\.env"):
        require_env("RAG_TEST_KEY")
    monkeypatch.setenv("RAG_TEST_KEY", "abc")
    assert require_env("RAG_TEST_KEY") == "abc"
