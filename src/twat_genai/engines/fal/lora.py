#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pydantic"]
# ///
"""LoRA handling and processing utilities."""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from fal.config import CombinedLoraSpecEntry, LoraLib, LoraSpecEntry
from twat.paths import PathManager


def get_lora_lib() -> LoraLib:
    """Get the LoRA library from the appropriate location.

    Priority:
    1. User-provided location (via environment variable)
    2. Central path management
    3. Bundled default library
    """
    paths = PathManager.for_package("twat_genai")

    # Check for user-provided location
    import os

    user_path = os.getenv("TWAT_GENAI_LORA_LIB")
    if user_path:
        lib_path = Path(user_path)
        if lib_path.exists():
            return LoraLib.model_validate_json(lib_path.read_text())

    # Check central path management
    lib_path = paths.genai.lora_dir / "loras.json"
    if lib_path.exists():
        return LoraLib.model_validate_json(lib_path.read_text())

    # Fall back to bundled library
    bundled_path = Path(__file__).parent.parent.parent / "__main___loras.json"
    return LoraLib.model_validate_json(bundled_path.read_text())


# Initialize LoRA library
LORA_LIB = get_lora_lib()


def parse_lora_phrase(phrase: str) -> LoraSpecEntry | CombinedLoraSpecEntry:
    """
    Parse a LoRA phrase which may include an optional scale.

    A phrase is either:
      - A key from LORA_LIB (the prompt portion) or
      - A URL/path optionally followed by a colon and a numeric scale

    Args:
        phrase: LoRA specification phrase

    Returns:
        Parsed LoRA specification

    Raises:
        ValueError: If the phrase has invalid format
    """
    phrase = phrase.strip()
    if phrase in LORA_LIB.root:
        entries = []
        for record in LORA_LIB.root[phrase].root:
            entries.append(
                LoraSpecEntry(path=record.url, scale=record.scale, prompt=phrase + ",")
            )
        return CombinedLoraSpecEntry(entries=entries, factory_key=phrase)

    if ":" in phrase:
        identifier, scale_str = phrase.split(":", 1)
        identifier = identifier.strip()
        try:
            scale = float(scale_str.strip())
        except ValueError:
            msg = f"Invalid scale value in LoRA phrase: {phrase}"
            raise ValueError(msg)
    else:
        identifier = phrase
        scale = 1.0

    return LoraSpecEntry(path=identifier, scale=scale, prompt="")


def normalize_lora_spec(
    spec: str | list | tuple | None,
) -> list[LoraSpecEntry | CombinedLoraSpecEntry]:
    """
    Normalize various LoRA specification formats into a unified list.

    Args:
        spec: LoRA specification in various formats

    Returns:
        List of normalized LoRA specifications

    Raises:
        ValueError: If the specification format is invalid
    """
    if spec is None:
        return []

    normalized: list[LoraSpecEntry | CombinedLoraSpecEntry] = []

    match spec:
        case list() | tuple() as items:
            for item in items:
                match item:
                    case dict(path=path, scale=scale, prompt=prompt):
                        normalized.append(
                            LoraSpecEntry(path=path, scale=float(scale), prompt=prompt)
                        )
                    case dict() as d:
                        if "path" not in d:
                            msg = "LoRA spec dictionary must have a 'path'."
                            raise ValueError(msg)
                        normalized.append(
                            LoraSpecEntry(
                                path=d["path"],
                                scale=float(d.get("scale", 1.0)),
                                prompt=d.get("prompt", ""),
                            )
                        )
                    case str() as phrase:
                        normalized.append(parse_lora_phrase(phrase))
                    case list() as sublist:
                        combined = [
                            parse_lora_phrase(sub)
                            for sub in sublist
                            if isinstance(sub, str)
                        ]
                        normalized.append(CombinedLoraSpecEntry(entries=combined))
                    case _:
                        msg = f"Unsupported LoRA spec item type: {type(item)}"
                        raise ValueError(msg)
        case str() as s:
            if s in LORA_LIB.root:
                return [parse_lora_phrase(s)]
            phrases = [phrase.strip() for phrase in s.split(";") if phrase.strip()]
            return [parse_lora_phrase(phrase) for phrase in phrases]
        case _:
            msg = f"Unsupported LoRA spec type: {type(spec)}"
            raise ValueError(msg)

    return normalized


async def build_lora_arguments(
    lora_spec: str | list | tuple | None, prompt: str
) -> tuple[list[dict[str, str | float]], str]:
    """
    Build the list of inference LoRA dictionaries and a final prompt.

    Args:
        lora_spec: LoRA specification
        prompt: Base prompt to augment

    Returns:
        Tuple of (LoRA argument list, final prompt)
    """
    entries = normalize_lora_spec(lora_spec)
    lora_list: list[dict[str, str | float]] = []
    prompt_prefixes: list[str] = []

    def process_entry(entry: LoraSpecEntry | CombinedLoraSpecEntry) -> None:
        if isinstance(entry, LoraSpecEntry):
            lora_list.append({"path": entry.path, "scale": entry.scale})
            if entry.prompt:
                prompt_prefixes.append(entry.prompt.rstrip(","))
        else:
            for sub_entry in entry.entries:
                process_entry(sub_entry)

    for entry in entries:
        process_entry(entry)

    logger.debug(f"Using LoRA configuration: {lora_list}")

    final_prompt = (
        f"{', '.join(prompt_prefixes)}, {prompt}".strip() if prompt_prefixes else prompt
    )
    return lora_list, final_prompt
