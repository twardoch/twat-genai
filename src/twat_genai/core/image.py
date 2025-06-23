#!/usr/bin/env -S uv run
# /// script
# dependencies = ["Pillow"]
# ///
"""Image handling and size definitions for twat-genai."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image
from loguru import logger


class ImageSizes(str, Enum):
    """Standard image size presets."""

    SQ = "square_hd"
    SQL = "square"
    SDV = "portrait_4_3"
    HDV = "portrait_16_9"
    SD = "landscape_4_3"
    HD = "landscape_16_9"
    PORT = "portrait"
    LAND = "landscape"
    WIDE = "wide"
    ULTRA = "ultra"


class ImageFormats(str, Enum):
    """Supported image formats."""

    JPG = "jpeg"
    PNG = "png"
    PIL = "pil"


async def save_image(
    image: Image.Image,
    path: str | Path,
    img_format: ImageFormats | None = None,
    quality: int = 95,
) -> None:
    """Save an image to disk."""
    save_kwargs = {"quality": quality}
    format_str: str | None = None
    if img_format:
        format_str = img_format.value
        save_kwargs["format"] = format_str
    else:
        pass

    try:
        logger.debug(
            f"Saving image to {path} with format '{format_str}' and args: {save_kwargs}"
        )
        image.save(path, **save_kwargs)
        logger.info(f"Saved image to {path}")
    except KeyError:
        logger.warning(
            f"Unknown format enum {img_format} ('{format_str}'), saving without explicit format."
        )
        save_kwargs.pop("format", None)
        try:
            image.save(path, format=None, quality=quality)
        except Exception as e_inner:
            logger.error(
                f"Failed to save image {path} even without format: {e_inner}",
                exc_info=True,
            )
            msg = f"Failed to save image {path}"
            raise RuntimeError(msg) from e_inner

    except Exception as e:
        logger.error(f"Failed to save image to {path}: {e}", exc_info=True)
        msg = f"Failed to save image to {path}"
        raise RuntimeError(msg) from e


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


# Removed the placeholder load_image added in previous step
# def load_image(source: str | Path | Image.Image) -> Image.Image:
#     pass
