#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pydantic"]
# ///
"""FAL-specific configuration and models."""

from enum import Enum
from pathlib import Path
from typing import Any, Union

from pydantic import BaseModel, RootModel

from ...core.config import ImageInput


class ModelTypes(str, Enum):
    """Available FAL model types."""

    TEXT = "fal-ai/flux-lora"
    IMAGE = "fal-ai/flux-lora/image-to-image"
    CANNY = "fal-ai/flux-lora-canny"
    DEPTH = "fal-ai/flux-lora-depth"


class ImageToImageConfig(BaseModel):
    """Configuration for image-to-image operations."""

    model_type: ModelTypes
    input_image: ImageInput
    strength: float = 0.75  # Only used for standard image-to-image
    negative_prompt: str = ""


class LoraRecord(BaseModel):
    """Single LoRA record with URL and scale."""

    url: str
    scale: float = 1.0


class LoraRecordList(RootModel[list[LoraRecord]]):
    """List of LoRA records."""

    pass


class LoraLib(RootModel[dict[str, LoraRecordList]]):
    """Library of LoRA configurations."""

    pass


class LoraSpecEntry(BaseModel):
    """Single LoRA specification for inference."""

    path: str
    scale: float = 1.0
    prompt: str = ""


class CombinedLoraSpecEntry(BaseModel):
    """Combined specification of multiple LoRA entries."""

    entries: list[Union[LoraSpecEntry, "CombinedLoraSpecEntry"]]
    factory_key: str | None = None


class FALJobConfig(BaseModel):
    """Configuration for a FAL image generation job."""

    prompt: str
    original_prompt: str
    model: ModelTypes = ModelTypes.TEXT
    lora_spec: str | list | tuple | None = None
    output_dir: Path | None = None
    filename_suffix: str | None = None
    filename_prefix: str | None = None
    image_config: ImageToImageConfig | None = None

    async def to_fal_arguments(self) -> dict[str, Any]:
        """Convert job config to FAL API arguments."""
        from .lora import build_lora_arguments  # Avoid circular import

        lora_list, final_prompt = await build_lora_arguments(
            self.lora_spec, self.prompt
        )

        args = {
            "loras": lora_list,
            "prompt": final_prompt,
            "num_images": 1,
            "output_format": "jpeg",
            "enable_safety_checker": False,
        }

        if self.model != ModelTypes.TEXT and self.image_config:
            image_url = await self.image_config.input_image.to_url()
            args["image_url"] = image_url
            if self.model == ModelTypes.IMAGE:
                args["strength"] = self.image_config.strength
            if self.image_config.negative_prompt:
                args["negative_prompt"] = self.image_config.negative_prompt

        return args
