#!/usr/bin/env -S uv run
# /// script
# dependencies = []
# ///
"""Prompt handling and expansion utilities for twat-genai."""


def split_top_level(s: str, delimiter: str = ";") -> list[str]:
    """
    Split a string by the given delimiter only at the top level (i.e. not inside braces).

    Args:
        s: Input string to split
        delimiter: Character to split on (default: semicolon)

    Returns:
        List of split strings
    """
    parts = []
    buf = []
    depth = 0
    for char in s:
        if char == "{":
            depth += 1
        elif char == "}" and depth:
            depth -= 1
        if char == delimiter and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    parts.append("".join(buf))
    return [part.strip() for part in parts if part.strip()]


def expand_prompts(s: str) -> list[str]:
    """
    Recursively expand a prompt string with alternatives inside braces.

    Example:
        "a {red; blue} house with {white; black} windows"
        -> ["a red house with white windows",
            "a red house with black windows",
            "a blue house with white windows",
            "a blue house with black windows"]

    Args:
        s: Input prompt string with alternatives in braces

    Returns:
        List of expanded prompts
    """
    open_index = s.find("{")
    if open_index == -1:
        return [s]

    depth = 0
    close_index = -1
    for i, char in enumerate(s[open_index:], start=open_index):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                close_index = i
                break

    if close_index == -1:
        msg = "Unbalanced braces in prompt string."
        raise ValueError(msg)

    prefix = s[:open_index]
    brace_content = s[open_index + 1 : close_index]
    suffix = s[close_index + 1 :]

    alternatives = split_top_level(brace_content, delimiter=";")
    expanded = []

    for alt in alternatives:
        for alt_exp in expand_prompts(alt):
            for suffix_exp in expand_prompts(suffix):
                expanded.append(f"{prefix}{alt_exp}{suffix_exp}")

    return expanded


def normalize_prompts(prompts: str | list[str]) -> list[str]:
    """
    Normalize and expand a prompt or list of prompts.

    Args:
        prompts: Single prompt string or list of prompts

    Returns:
        List of expanded and normalized prompts
    """
    if isinstance(prompts, str):
        raw_prompts = split_top_level(prompts, delimiter=";")
    else:
        raw_prompts = prompts

    final_prompts: list[str] = []
    for raw in raw_prompts:
        final_prompts.extend(expand_prompts(raw.strip()))

    return final_prompts
