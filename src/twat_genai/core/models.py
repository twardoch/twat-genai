"""Core data models for image generation."""

from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image
from pydantic import BaseModel


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
    image: Optional[Image.Image] = None
    original_prompt: Optional[str] = None
    job_params: Optional[dict[str, Any]] = None

    model_config = {"arbitrary_types_allowed": True}


# Type aliases
ImageSize = Union[ImageSizes, ImageSizeWH]
OutputDir = Optional[Path] 