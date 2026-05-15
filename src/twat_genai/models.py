"""Provider-neutral request and response models for generative media.

# this_file: src/twat_genai/models.py
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class MediaKind(str, Enum):
    """Generated media categories handled by twat-genai."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    SPEECH = "speech"


class ProviderName(str, Enum):
    """Known provider families exposed through the shared provider layer."""

    FAL = "fal"
    CHUTES = "chutes"
    OPENAI = "openai"
    GEMINI = "gemini"


class MediaFile(BaseModel):
    """A generated or input media file represented as URL, path, or bytes."""

    kind: MediaKind
    url: str | None = None
    path: Path | None = None
    data: bytes | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _require_location(self) -> MediaFile:
        if self.url is None and self.path is None and self.data is None:
            msg = "MediaFile requires at least one of url, path, or data."
            raise ValueError(msg)
        return self


class ImageGenerationRequest(BaseModel):
    """Provider-neutral text-to-image generation request."""

    prompt: str = Field(..., min_length=1)
    provider: ProviderName = ProviderName.FAL
    model: str | None = None
    negative_prompt: str | None = None
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    n: int = Field(default=1, ge=1)
    seed: int | None = Field(default=None, ge=0)
    guidance_scale: float | None = None
    inference_steps: int | None = Field(default=None, ge=1)
    output_format: Literal["png", "jpeg", "jpg", "webp"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageEditRequest(ImageGenerationRequest):
    """Provider-neutral image editing request."""

    input_images: list[MediaFile] = Field(default_factory=list)
    mask: MediaFile | None = None

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _require_input_image(self) -> ImageEditRequest:
        if not self.input_images:
            msg = "ImageEditRequest requires at least one input image."
            raise ValueError(msg)
        return self


class VideoGenerationRequest(BaseModel):
    """Provider-neutral video generation request."""

    prompt: str = Field(..., min_length=1)
    provider: ProviderName = ProviderName.CHUTES
    model: str | None = None
    first_frame: MediaFile | None = None
    last_frame: MediaFile | None = None
    resolution: str | None = None
    fps: int | None = Field(default=None, ge=1)
    num_frames: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BackgroundRemovalRequest(BaseModel):
    """Provider-neutral background removal request."""

    image: MediaFile
    provider: ProviderName = ProviderName.CHUTES
    output_format: Literal["png", "webp"] = "png"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    """Normalized response for thin provider clients."""

    provider: ProviderName
    kind: MediaKind
    files: list[MediaFile] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None
    model: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Return True when the provider response does not contain an error."""

        return self.error is None
