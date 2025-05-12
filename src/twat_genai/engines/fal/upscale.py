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


async def run_upscale(
    client: FalApiClient,
    config: UpscaleConfig,
    model_type: ModelTypes,
    output_dir: Path | None = None,
    filename_suffix: str | None = None,
    filename_prefix: str | None = None,
) -> ImageResult:
    """
    Assembles parameters from UpscaleConfig and defaults, then calls the appropriate
    FalApiClient method to perform image upscaling.

    Args:
        client: The FalApiClient instance.
        config: The UpscaleConfig object containing specific parameters.
        model_type: The specific upscale model to use.
        output_dir: Directory to save the result image and metadata.
        filename_suffix: Optional suffix for generated filenames.
        filename_prefix: Optional prefix for generated filenames.

    Returns:
        An ImageResult object containing information about the generated image.

    Raises:
        ValueError: If the model_type is not a valid upscale model.
        RuntimeError: If the image upload or API call fails.
    """
    if not model_type.name.startswith("UPSCALER_"):
        msg = f"Invalid model type provided to run_upscale: {model_type}"
        logger.error(msg)
        raise ValueError(msg)

    # 1. Get the default parameters for the model type
    default_params = UPSCALE_TOOL_DEFAULT_PARAMS.get(model_type, {})
    logger.debug(f"Default params for {model_type.name}: {default_params}")

    # 2. Prepare arguments from UpscaleConfig, excluding None values and input_image
    config_args = {
        k: v
        for k, v in config.model_dump(exclude={"input_image"}).items()
        if v is not None
    }
    logger.debug(f"Config args (non-None): {config_args}")

    # 3. Remap config keys to FAL API expected keys if necessary
    #    (Currently mapping based on UpscaleConfig field names)
    #    This requires careful mapping between UpscaleConfig fields and FAL API params.
    #    Example mapping (adjust as needed based on actual API requirements):
    api_args_from_config = {
        "prompt": config_args.get("prompt"),
        "negative_prompt": config_args.get("negative_prompt"),
        "seed": config_args.get("seed"),
        # "scale": config_args.get("scale"),  # General scale - Handled below for CCSR
        # Ideogram
        "resemblance": config_args.get("ideogram_resemblance"),
        "detail": config_args.get("ideogram_detail"),
        "expand_prompt": config_args.get("ideogram_expand_prompt"),
        # Recraft
        "sync_mode": config_args.get("recraft_sync_mode"),
        # Fooocus (Assuming FAL model takes these directly - CHECK API)
        "styles": config_args.get("fooocus_styles"),
        "performance": config_args.get("fooocus_performance"),
        "guidance_scale": config_args.get("fooocus_guidance_scale"),
        "sharpness": config_args.get("fooocus_sharpness"),
        "uov_method": config_args.get("fooocus_uov_method"),
        # ESRGAN
        "model": config_args.get("esrgan_model"),
        "tile": config_args.get("esrgan_tile"),
        "face": config_args.get("esrgan_face"),
        # Clarity / Aura SR
        "creativity": config_args.get("clarity_creativity"),
        # "resemblance": config_args.get("clarity_resemblance"), # REMOVED - Duplicate key with Ideogram
        # "guidance_scale": config_args.get("clarity_guidance_scale"), # Duplicate key?
        # "num_inference_steps": config_args.get("clarity_num_inference_steps"), # Duplicate key?
        # CCSR
        "scale": config_args.get("ccsr_scale")
        if config_args.get("ccsr_scale") is not None
        else config_args.get("scale"),  # Use general scale if ccsr_scale not set
        "tile_diffusion": config_args.get("ccsr_tile_diffusion"),
        "color_fix_type": config_args.get("ccsr_color_fix_type"),
        "steps": config_args.get("ccsr_steps"),
    }
    # Remove None values after mapping
    api_args_from_config = {
        k: v for k, v in api_args_from_config.items() if v is not None
    }
    logger.debug(f"API args from config (mapped & non-None): {api_args_from_config}")

    # 4. Merge defaults and config arguments, prioritizing config
    #    Start with defaults, update with mapped config args
    final_api_args = default_params.copy()
    final_api_args.update(api_args_from_config)
    logger.debug(f"Final API args merged: {final_api_args}")

    # 5. Get the image URL (upload if necessary)
    #    The config.input_image is already a FALImageInput which handles upload
    try:
        image_url = await config.input_image.to_url(client)  # Pass client for upload
        logger.info(f"Using image URL for upscale: {image_url}")
    except Exception as e:
        logger.error(f"Failed to get image URL: {e}", exc_info=True)
        msg = f"Failed to get image URL: {e}"
        raise RuntimeError(msg) from e

    # 6. Call the client's process_upscale method
    logger.info(f"Calling client.process_upscale for {model_type.name}")
    try:
        result = await client.process_upscale(
            model_type=model_type,
            image_url=image_url,
            output_dir=output_dir,
            filename_suffix=filename_suffix,
            filename_prefix=filename_prefix,
            **final_api_args,  # Pass the merged arguments
        )
        logger.success(
            f"Upscale job completed for {model_type.name}. Request ID: {result.request_id}"
        )
        return result
    except Exception as e:
        logger.error(f"FalApiClient.process_upscale failed: {e}", exc_info=True)
        msg = f"Upscale process failed: {e}"
        raise RuntimeError(msg) from e
