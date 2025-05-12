#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pydantic", "Pillow"]
# ///
"""Core configuration models and type definitions for twat-genai."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from pydantic import BaseModel
from PIL import Image  # Import PIL Image directly, not conditionally

# Import core types used elsewhere, define EngineConfig in base.py

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
ImageSizeNames = Optional[list[str]]


class ImageSizeWH(BaseModel):
    """Width and height for a custom image size."""

    width: int
    height: int


# Type alias for combined size representation
ImageSize = Union[ImageSizeNames, ImageSizeWH]


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

    async def to_url(self, client=None) -> str:
        """Convert the input to a URL format.

        If the input is already a URL, return it directly.
        For path and PIL image handling, the client object should be provided.

        Args:
            client: Optional API client for uploading images if needed.

        Returns:
            str: URL to the image.

        Raises:
            ValueError: If no valid input exists.
        """
        if self.url:
            return self.url

        # These cases require a client to handle uploads
        if client is None:
            msg = "Client required to convert path or PIL image to URL."
            raise ValueError(msg)

        # Actual implementation handled in specific engine adapter
        msg = "Implementation should be provided by engine adapter"
        raise ValueError(msg)


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


# EngineConfig is defined in engines/base.py to avoid circular imports
# class EngineConfig(BaseModel):
#     guidance_scale: float = 3.5
#     num_inference_steps: int = 28
#     image_size: ImageSize = ImageSizes.SQ
#     enable_safety_checker: bool = False
