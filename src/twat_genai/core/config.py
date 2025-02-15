#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pydantic", "Pillow"]
# ///
"""Core configuration models and type definitions for twat-genai."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from PIL import Image

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

    async def to_url(self) -> str:
        """Convert the input to a URL format.

        This is an abstract method that should be implemented by specific engine handlers.
        """
        msg = "to_url() must be implemented by a specific engine handler"
        raise NotImplementedError(
            msg
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
