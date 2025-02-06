#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pydantic", "Pillow"]
# ///
"""Core configuration models and type definitions for twat-genai."""

from pathlib import Path
from typing import Any, Optional

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

    url: str | None = None
    path: Path | None = None
    pil_image: Image.Image | None = None

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
    image: Image.Image | None = None
    original_prompt: str | None = None
    job_params: dict[str, Any] | None = None

    model_config = {"arbitrary_types_allowed": True}
