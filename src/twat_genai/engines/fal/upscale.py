# this_file: src/twat_genai/engines/fal/upscale.py
"""Handles image upscaling using various FAL models."""

from __future__ import annotations

from dataclasses import dataclass  # Keep dataclass if needed later
from typing import Any, TYPE_CHECKING
from pathlib import Path
from loguru import logger

# from fal.config import ModelTypes  # Import the unified ModelTypes
from twat_genai.engines.fal.config import ModelTypes  # Corrected import
from twat_genai.core.config import ImageResult

if TYPE_CHECKING:
    from twat_genai.engines.fal.client import FalApiClient
    from twat_genai.engines.fal.config import UpscaleConfig

# TODO: Re-evaluate if a separate enum is needed or if ModelTypes suffices.
# For now, keep the mapping structure for parameters.


# Mapping of upscale ModelTypes to their default parameters
# Based on superres/tools.py TOOL_DEFAULT_PARAMS
UPSCALE_TOOL_DEFAULT_PARAMS: dict[ModelTypes, dict[str, Any]] = {
    ModelTypes.UPSCALER_DRCT: {"upscaling_factor": 4},
    ModelTypes.UPSCALER_IDEOGRAM: {"resemblance": 50, "detail": 50},
    ModelTypes.UPSCALER_RECRAFT_CREATIVE: {"sync_mode": True},
    ModelTypes.UPSCALER_RECRAFT_CLARITY: {"sync_mode": True},
    ModelTypes.UPSCALER_CCSR: {"scale": 2, "steps": 50, "color_fix_type": "adain"},
    ModelTypes.UPSCALER_ESRGAN: {
        "model": "RealESRGAN_x4plus",
        "scale": 2,
        "face": False,
        "tile": 0,
    },
    ModelTypes.UPSCALER_AURA_SR: {
        "model_type": "SD_1_5",
        "scale": 2,
        "creativity": 0.5,
        "detail": 1.0,
        "shape_preservation": 0.25,
        "prompt_suffix": " high quality, highly detailed, high resolution, sharp",
        "negative_prompt": "blurry, low resolution, bad, ugly, low quality",
    },
    ModelTypes.UPSCALER_CLARITY: {
        "scale": 2,
        "creativity": 0.35,
        "resemblance": 0.6,
        "prompt": "masterpiece, best quality, highres",
        "negative_prompt": "(worst quality, low quality, normal quality:2)",
    },
}

# Maximum input image dimensions for each tool
# Based on superres/tools.py TOOL_MAX_INPUT_SIZES
UPSCALE_TOOL_MAX_INPUT_SIZES: dict[ModelTypes, int] = {
    ModelTypes.UPSCALER_DRCT: 2048,
    ModelTypes.UPSCALER_IDEOGRAM: 1024,
    ModelTypes.UPSCALER_RECRAFT_CREATIVE: 2048,
    ModelTypes.UPSCALER_RECRAFT_CLARITY: 2048,
    ModelTypes.UPSCALER_CCSR: 2048,
    ModelTypes.UPSCALER_ESRGAN: 2048,
    ModelTypes.UPSCALER_AURA_SR: 1024,
    ModelTypes.UPSCALER_CLARITY: 1024,
}


# Dataclasses for specific tool parameters (e.g., Fooocus)
@dataclass
class ImagePrompt:
    """Represents an image prompt for Fooocus upscaling"""

    type: str = "ImagePrompt"
    image_url: str = ""
    stop_at: float = 0.5
    weight: float = 1.0


@dataclass
class LoraWeight:
    """Represents a LoRA weight configuration"""

    path: str
    scale: float = 0.1
