#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pydantic", "loguru"]
# ///
"""
Midjourney prompt parsing and handling utilities.

This module provides functionality to parse and process Midjourney-style prompts including:
- Basic text prompts
- Image prompts (URLs)
- Multi-prompts with weights (using :: separator)
- Permutation prompts (using {} for alternatives)
- Parameter handling (--param value)
"""

from __future__ import annotations

from loguru import logger
from pydantic import AnyHttpUrl, BaseModel, Field, parse_obj_as

# Constants for prompt parsing
ALLOWED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
MULTI_PROMPT_SEPARATOR = "::"
PERMUTATION_START = "{"
PERMUTATION_END = "}"
PERMUTATION_SEPARATOR = ","
PARAM_PREFIX = "--"


class ImagePrompt(BaseModel):
    """An image URL used as part of a prompt."""

    url: AnyHttpUrl = Field(
        ..., description="Direct image URL ending with allowed extension"
    )
    weight: float = Field(
        default=1.0, description="Image weight/influence (--iw parameter)"
    )

    @property
    def is_valid(self) -> bool:
        """Check if URL ends with allowed image extension."""
        return str(self.url).lower().endswith(ALLOWED_IMAGE_EXTENSIONS)


class PromptParameter(BaseModel):
    """A parameter that modifies prompt behavior."""

    name: str = Field(..., description="Parameter name without -- prefix")
    value: str | None = Field(None, description="Parameter value if any")


class PromptPart(BaseModel):
    """A weighted part of a multi-prompt."""

    text: str = Field(..., description="Text content of this prompt part")
    weight: float = Field(default=1.0, description="Relative weight of this part")


class MidjourneyPrompt(BaseModel):
    """Complete parsed Midjourney prompt structure."""

    image_prompts: list[ImagePrompt] = Field(
        default_factory=list, description="Image URLs used in prompt"
    )
    text_parts: list[PromptPart] = Field(..., description="Text portions of the prompt")
    parameters: list[PromptParameter] = Field(
        default_factory=list, description="Prompt parameters"
    )
    raw_prompt: str = Field(..., description="Original unparsed prompt")

    def to_string(self) -> str:
        """Convert parsed prompt back to string format."""
        parts = []

        # Add image prompts
        for img in self.image_prompts:
            parts.append(str(img.url))
            if img.weight != 1.0:
                parts.append(f"--iw {img.weight}")

        # Add text parts with weights
        text_parts = []
        for part in self.text_parts:
            if part.weight == 1.0:
                text_parts.append(part.text)
            else:
                text_parts.append(f"{part.text}::{part.weight}")
        parts.append(" ".join(text_parts))

        # Add parameters
        for param in self.parameters:
            if param.value:
                parts.append(f"--{param.name} {param.value}")
            else:
                parts.append(f"--{param.name}")

        return " ".join(parts)


def split_top_level(s: str, delimiter: str = ",") -> list[str]:
    """Split string by delimiter only at top level (not inside braces)."""
    parts = []
    current = []
    depth = 0

    for char in s:
        if char == PERMUTATION_START:
            depth += 1
        elif char == PERMUTATION_END and depth > 0:
            depth -= 1

        if char == delimiter and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        parts.append("".join(current).strip())

    return [p for p in parts if p]


def expand_permutations(prompt: str) -> list[str]:
    """Expand permutation groups in prompt into all combinations."""
    if PERMUTATION_START not in prompt:
        return [prompt]

    start_idx = prompt.find(PERMUTATION_START)
    if start_idx == -1:
        return [prompt]

    # Find matching end brace
    depth = 0
    end_idx = -1
    for i, char in enumerate(prompt[start_idx:], start=start_idx):
        if char == PERMUTATION_START:
            depth += 1
        elif char == PERMUTATION_END:
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if end_idx == -1:
        msg = f"Unmatched brace in prompt: {prompt}"
        raise ValueError(msg)

    prefix = prompt[:start_idx]
    options = prompt[start_idx + 1 : end_idx]
    suffix = prompt[end_idx + 1 :]

    # Split options and handle escaped commas
    parts = []
    current = []
    escaped = False
    for char in options:
        if char == "\\":
            escaped = True
            continue
        if char == PERMUTATION_SEPARATOR and not escaped:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        escaped = False
    if current:
        parts.append("".join(current).strip())

    # Recursively expand remaining permutations
    results = []
    for option in parts:
        for suffix_expanded in expand_permutations(suffix):
            results.append(f"{prefix}{option.strip()}{suffix_expanded}")

    return results


def parse_parameters(text: str) -> tuple[str, list[PromptParameter]]:
    """Extract parameters from prompt text."""
    parts = text.split()
    params = []
    prompt_parts = []

    i = 0
    while i < len(parts):
        part = parts[i]
        if part.startswith(PARAM_PREFIX):
            name = part[2:]  # Remove --
            value = None
            if i + 1 < len(parts) and not parts[i + 1].startswith(PARAM_PREFIX):
                value = parts[i + 1]
                i += 1
            params.append(PromptParameter(name=name, value=value))
        else:
            prompt_parts.append(part)
        i += 1

    return " ".join(prompt_parts), params


def parse_multi_prompt(text: str) -> list[PromptPart]:
    """Parse text into weighted prompt parts."""
    if MULTI_PROMPT_SEPARATOR not in text:
        return [PromptPart(text=text)]

    parts = []
    for part in text.split(MULTI_PROMPT_SEPARATOR):
        part = part.strip()
        if not part:
            continue

        # Check for weight
        weight = 1.0
        if " " in part:
            try:
                text, weight_str = part.rsplit(" ", 1)
                weight = float(weight_str)
            except ValueError:
                text = part
        else:
            text = part

        parts.append(PromptPart(text=text, weight=weight))

    return parts


def parse_prompt(prompt: str) -> MidjourneyPrompt:
    """Parse a complete Midjourney prompt into structured format."""
    # Split into parts and extract parameters
    text, parameters = parse_parameters(prompt)

    # Extract image prompts
    parts = text.split()
    image_prompts = []
    text_parts = []

    for part in parts:
        if any(part.lower().endswith(ext) for ext in ALLOWED_IMAGE_EXTENSIONS):
            try:
                # Ensure URL starts with http/https and parse using Pydantic
                url = (
                    part
                    if part.startswith(("http://", "https://"))
                    else f"https://{part}"
                )
                parsed_url = parse_obj_as(AnyHttpUrl, url)
                image_prompts.append(ImagePrompt(url=parsed_url))
            except Exception as e:
                logger.warning(f"Invalid image URL {part}: {e}")
        else:
            text_parts.append(part)

    # Parse remaining text as multi-prompt
    text = " ".join(text_parts)
    prompt_parts = parse_multi_prompt(text)

    return MidjourneyPrompt(
        image_prompts=image_prompts,
        text_parts=prompt_parts,
        parameters=parameters,
        raw_prompt=prompt,
    )


def normalize_prompts(prompts: str | list[str]) -> list[str]:
    """Normalize and expand a prompt or list of prompts."""
    raw_prompts = (
        split_top_level(prompts, delimiter=";") if isinstance(prompts, str) else prompts
    )

    final_prompts = []
    for raw in raw_prompts:
        final_prompts.extend(expand_permutations(raw.strip()))

    return final_prompts
