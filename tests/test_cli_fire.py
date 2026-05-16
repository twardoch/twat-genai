# this_file: tests/test_cli_fire.py
"""Tests for the canonical Fire CLI entry point (__main__.py).

Covers SPEC §6 acceptance criteria:
- twat-genai --help → stdout, exit 0
- twat-genai version → semver
- each leaf --help → exit 0
"""

from __future__ import annotations

import subprocess
import sys
import re


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run python -m twat_genai with given args."""
    return subprocess.run(
        [sys.executable, "-m", "twat_genai", *args],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Help / discoverability
# ---------------------------------------------------------------------------


def test_help_exits_zero_and_shows_commands() -> None:
    """twat-genai --help exits 0 and lists subcommands."""
    result = _run("--help")
    assert result.returncode == 0, f"--help non-zero: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "image" in combined, "Expected 'image' in help output"
    assert "video" in combined, "Expected 'video' in help output"


def test_version_leaf_prints_semver() -> None:
    """twat-genai version prints a semver string."""
    result = _run("version")
    assert result.returncode == 0, f"version non-zero: {result.stderr}"
    combined = (result.stdout + result.stderr).strip()
    assert re.search(r"\d+\.\d+\.\d+", combined), f"No semver in output: {combined!r}"


def test_image_help_exits_zero() -> None:
    """twat-genai image --help exits 0."""
    result = _run("image", "--help")
    assert result.returncode == 0, f"image --help non-zero: {result.stderr}"


def test_video_help_exits_zero() -> None:
    """twat-genai video --help exits 0."""
    result = _run("video", "--help")
    assert result.returncode == 0, f"video --help non-zero: {result.stderr}"


def test_list_providers_help_exits_zero() -> None:
    """twat-genai list-providers --help exits 0."""
    result = _run("list-providers", "--help")
    assert result.returncode == 0, f"list-providers --help non-zero: {result.stderr}"


def test_list_models_help_exits_zero() -> None:
    """twat-genai list-models --help exits 0."""
    result = _run("list-models", "--help")
    assert result.returncode == 0, f"list-models --help non-zero: {result.stderr}"


# ---------------------------------------------------------------------------
# Leaf execution (no API calls needed)
# ---------------------------------------------------------------------------


def test_list_providers_returns_known_providers() -> None:
    """list-providers prints at least the four known provider names."""
    result = _run("list-providers")
    assert result.returncode == 0, f"list-providers failed: {result.stderr}"
    output = result.stdout + result.stderr
    for expected in ("fal", "chutes", "openai", "gemini"):
        assert expected in output.lower(), f"Provider {expected!r} missing from output"


def test_list_models_returns_fal_models() -> None:
    """list-models prints FAL model identifiers."""
    result = _run("list-models")
    assert result.returncode == 0, f"list-models failed: {result.stderr}"
    output = result.stdout + result.stderr
    assert "fal-ai" in output, "Expected fal-ai model endpoints in output"


def test_commands_dict_has_expected_keys() -> None:
    """COMMANDS dict contains the canonical leaf names."""
    from twat_genai.__main__ import COMMANDS

    expected = {"version", "image", "video", "list-providers", "list-models"}
    assert expected <= set(COMMANDS.keys()), f"Missing keys: {expected - set(COMMANDS.keys())}"


def test_cmd_helpers_are_callable() -> None:
    """Each cmd_* helper is importable and callable."""
    from twat_genai.__main__ import (
        cmd_image,
        cmd_list_models,
        cmd_list_providers,
        cmd_version,
        cmd_video,
    )

    for fn in (cmd_image, cmd_video, cmd_list_providers, cmd_list_models, cmd_version):
        assert callable(fn), f"{fn} is not callable"
