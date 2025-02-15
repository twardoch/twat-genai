"""Core data models for image generation."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from pydantic import BaseModel

if TYPE_CHECKING:
    from PIL import Image


class ModelTypes(str, Enum):
    """Available model types."""

    TEXT = "fal-ai/flux-lora"
    IMAGE = "fal-ai/flux-lora/image-to-image"
    CANNY = "fal-ai/flux-lora-canny"
    DEPTH = "fal-ai/flux-lora-depth"


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


class ImageSizeWH(BaseModel):
    """Custom image dimensions."""

    width: int
    height: int


class ImageResult(BaseModel):
    """Result of an image generation operation."""

    request_id: str
    timestamp: str
    result: dict[str, Any]
    image_info: dict[str, Any]
    image: Image.Image | None = None
    original_prompt: str | None = None
    job_params: dict[str, Any] | None = None

    model_config = {"arbitrary_types_allowed": True}


# Type aliases
ImageSize = Union[ImageSizes, ImageSizeWH]
OutputDir = Optional[Path]
