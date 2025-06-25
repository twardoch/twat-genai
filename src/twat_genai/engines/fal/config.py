#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pydantic"]
# ///
"""FAL-specific configuration and models."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from twat_genai.core.config import ImageInput

if TYPE_CHECKING:
    pass


class ModelTypes(str, Enum):
    """Available FAL model types."""

    TEXT = "fal-ai/flux-lora"
    IMAGE = "fal-ai/flux-lora/image-to-image"
    CANNY = "fal-ai/flux-lora-canny"
    DEPTH = "fal-ai/flux-lora-depth"

    # Upscalers from superres
    UPSCALER_DRCT = "fal-ai/drct-super-resolution"
    UPSCALER_IDEOGRAM = "fal-ai/ideogram/upscale"
    UPSCALER_RECRAFT_CREATIVE = "fal-ai/recraft-creative-upscale"
    UPSCALER_RECRAFT_CLARITY = "fal-ai/recraft-clarity-upscale"
    UPSCALER_CCSR = "fal-ai/ccsr"
    UPSCALER_ESRGAN = "fal-ai/esrgan"
    UPSCALER_AURA_SR = "fal-ai/aura-sr"
    UPSCALER_CLARITY = "fal-ai/clarity-upscaler"

    # Outpainters
    OUTPAINT_BRIA = "fal-ai/bria/expand"
    OUTPAINT_FLUX = "fal-ai/flux-lora/inpainting"


class ImageToImageConfig(BaseModel):
    """Configuration for image-to-image operations."""

    model_type: ModelTypes
    input_image: ImageInput
    strength: float = 0.75  # Only used for standard image-to-image
    negative_prompt: str = ""


# --- Upscaling Config ---
class UpscaleConfig(BaseModel):
    """Configuration specific to upscaling operations."""

    input_image: ImageInput  # Required for upscaling
    prompt: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    scale: float | None = None  # General scale, default depends on model

    # --- Tool-specific parameters ---
    # Ideogram
    ideogram_resemblance: int | None = None
    ideogram_detail: int | None = None
    ideogram_expand_prompt: bool | None = None

    # Recraft
    recraft_sync_mode: bool | None = None

    # Fooocus
    fooocus_styles: list[str] | None = None
    fooocus_performance: (
        Literal["Speed", "Quality", "Extreme Speed", "Lightning"] | None
    ) = None
    fooocus_guidance_scale: float | None = None
    fooocus_sharpness: float | None = None
    fooocus_uov_method: (
        Literal[
            "Vary (Subtle)",
            "Vary (Strong)",
            "Upscale (1.5x)",
            "Upscale (2x)",
            "Upscale (Fast 2x)",
        ]
        | None
    ) = None

    # ESRGAN
    esrgan_model: (
        Literal[
            "RealESRGAN_x4plus",
            "RealESRGAN_x2plus",
            "RealESRGAN_x4plus_anime_6B",
            "RealESRGAN_x4_v3",
            "RealESRGAN_x4_wdn_v3",
            "RealESRGAN_x4_anime_v3",
        ]
        | None
    ) = None
    esrgan_tile: int | None = None
    esrgan_face: bool | None = None

    # Clarity / AuraSR (Shared params)
    clarity_creativity: float | None = None  # (0.0 - 1.0)
    clarity_resemblance: float | None = None  # (0.0 - 1.0)
    clarity_guidance_scale: float | None = None
    clarity_num_inference_steps: int | None = None

    # CCSR
    ccsr_scale: int | None = None
    ccsr_tile_diffusion: Literal["none", "mix", "gaussian"] | None = None
    ccsr_color_fix_type: Literal["none", "wavelet", "adain"] | None = None
    ccsr_steps: int | None = None


# --- Outpainting Config ---
class OutpaintConfig(BaseModel):
    """Configuration specific to outpainting operations."""

    input_image: ImageInput  # Required for outpainting
    prompt: str
    target_width: int
    target_height: int
    num_images: int = 1
    outpaint_tool: Literal["bria", "flux"] = (
        "bria"  # Default to bria for backward compatibility
    )
    # Extra parameters for flux outpainting
    guidance_scale: float | None = None  # Flux-specific
    num_inference_steps: int | None = None  # Flux-specific
    negative_prompt: str | None = None  # Flux-specific
    enable_safety_checker: bool | None = None  # Flux-specific
    border_thickness_factor: float = (
        0.05  # Border thickness for GenFill post-processing
    )

    # Note: original_image_size and original_image_location are typically calculated
    # just before the API call based on the input_image dimensions and target size.
    # They are not stored here directly.


# --- Lora Handling (Moved to core/lora.py) ---
# TODO: Confirm all LoRA logic/models are removed from here and __main__.py

# --- Constants from former engines/fal/upscale.py ---
# Maximum input image dimensions for each tool
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
