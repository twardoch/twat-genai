#!/usr/bin/env -S uv run
# /// script
# dependencies = ["Pillow"]
# ///
"""Image handling and size definitions for twat-genai."""

from enum import Enum
from pathlib import Path
from typing import Optional

from PIL import Image


class ImageSizes(str, Enum):
    """Standard image size presets."""

    SQ = "square_hd"
    SQL = "square"
    SDV = "portrait_4_3"
    HDV = "portrait_16_9"
    SD = "landscape_4_3"
    HD = "landscape_16_9"


class ImageFormats(str, Enum):
    """Supported image formats."""

    JPG = "jpeg"
    PNG = "png"
    PIL = "pil"


async def save_image(
    image: Image.Image,
    output_path: Path,
    format: ImageFormats = ImageFormats.JPG,
    quality: int = 95,
) -> None:
    """Save an image to disk with the specified format and quality."""
    image.save(output_path, format=format.value, quality=quality)


def validate_image_size(size_str: str) -> tuple[int, int] | None:
    """
    Validate and parse a custom image size string.
    Returns tuple of (width, height) if valid, None if invalid.
    Format: 'width,height' with integers.
    """
    if "," not in size_str:
        return None
    try:
        w, h = (int(x.strip()) for x in size_str.split(",", 1))
        return w, h
    except (ValueError, TypeError):
        return None
