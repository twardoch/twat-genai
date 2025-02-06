#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pydantic", "Pillow"]
# ///
"""Core configuration models and type definitions for twat-genai."""

from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image
from pydantic import BaseModel

# Type aliases
Prompts = list[str]
OutputDir = Optional[Path]
GuidanceScale = float
NumInferenceSteps = int
URLStr = str
RequestID = str
JsonDict = dict[str, Any]


class ImageSizeWH(BaseModel):
    """Width and height for a custom image size."""

    width: int
    height: int


class ImageInput(BaseModel):
    """Represents an image input that can be a URL, file path, or PIL Image."""

    url: Optional[str] = None
    path: Optional[Path] = None
    pil_image: Optional[Image.Image] = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def is_valid(self) -> bool:
        """Check if exactly one input type is provided."""
        return (
            sum(1 for x in (self.url, self.path, self.pil_image) if x is not None) == 1
        )


class ImageResult(BaseModel):
    """Result of a single image generation."""

    request_id: str
    timestamp: str
    result: JsonDict
    image_info: dict[str, Any]
    image: Optional[Image.Image] = None
    original_prompt: Optional[str] = None
    job_params: Optional[dict[str, Any]] = None

    model_config = {"arbitrary_types_allowed": True}
