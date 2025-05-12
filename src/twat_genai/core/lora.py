#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pydantic"]
# ///
# this_file: src/twat_genai/core/lora.py

"""Core LoRA definitions and models."""

from __future__ import annotations

from pydantic import BaseModel, RootModel

# --- LoRA Models ---


class LoraRecord(BaseModel):
    """Single LoRA record with URL and scale."""

    url: str
    scale: float = 1.0


class LoraRecordList(RootModel[list[LoraRecord]]):
    """List of LoRA records."""


class LoraLib(RootModel[dict[str, LoraRecordList]]):
    """Library of LoRA configurations, mapping prompt keywords to lists of records."""


class LoraSpecEntry(BaseModel):
    """Single LoRA specification entry for inference.

    Represents one LoRA applied with a specific scale and optional prompt trigger.
    Path can be a URL or a local identifier.
    """

    path: str
    scale: float = 1.0
    prompt: str = ""


class CombinedLoraSpecEntry(BaseModel):
    """Combined specification composed of multiple LoraSpecEntry items.

    Used when a single keyword maps to multiple underlying LoRA files.
    """

    entries: list[LoraSpecEntry | CombinedLoraSpecEntry]
    factory_key: str | None = None  # Store the original keyword if created from LoraLib
