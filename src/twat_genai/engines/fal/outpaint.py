# this_file: src/twat_genai/engines/fal/outpaint.py
"""Handles image outpainting using the FAL BRIA model."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from loguru import logger

from twat_genai.core.config import ImageResult
from twat_genai.core.image_utils import create_outpaint_mask

if TYPE_CHECKING:
    from twat_genai.engines.fal.client import FalApiClient
    from twat_genai.engines.fal.config import OutpaintConfig


async def run_outpaint(
    client: FalApiClient,
    config: OutpaintConfig,
    original_width: int,
    original_height: int,
    output_dir: Path | None = None,
    filename_suffix: str | None = None,
    filename_prefix: str | None = None,
) -> ImageResult:
    """
    Handle image outpainting based on the configuration, choosing the appropriate outpaint tool.

    Args:
        client: The FalApiClient instance to use for API calls.
        config: The OutpaintConfig object containing the configuration.
        original_width: The width of the original input image.
        original_height: The height of the original input image.
        output_dir: Directory to save the result image and metadata.
        filename_suffix: Optional suffix for generated filenames.
        filename_prefix: Optional prefix for generated filenames.

    Returns:
        An ImageResult object containing information about the generated image.

    Raises:
        ValueError: If required parameters are missing or invalid.
        RuntimeError: If the image upload or API call fails.
    """
    logger.info(
        f"Starting outpaint process with {config.outpaint_tool} tool, "
        f"target size {config.target_width}x{config.target_height}"
    )

    # Choose the appropriate outpainting implementation based on the tool
    if config.outpaint_tool == "flux":
        return await run_flux_outpaint(
            client=client,
            config=config,
            original_width=original_width,
            original_height=original_height,
            output_dir=output_dir,
            filename_suffix=filename_suffix,
            filename_prefix=filename_prefix,
        )
    elif config.outpaint_tool == "bria":
        # Use the original implementation for bria
        # The validation of dimensions is handled in both implementations
        if (
            original_width > config.target_width
            or original_height > config.target_height
        ):
            msg = (
                f"Original image ({original_width}x{original_height}) cannot be larger "
                f"than target size ({config.target_width}x{config.target_height}) for outpainting."
            )
            logger.error(msg)
            raise ValueError(msg)

        try:
            # Calculate where the original image goes in the target
            offset_x = (config.target_width - original_width) // 2
            offset_y = (config.target_height - original_height) // 2
            original_image_location = [offset_x, offset_y]
            original_image_size = [original_width, original_height]

            # Get the image URL
            image_url = await config.input_image.to_url(client)
            logger.info(f"Using image URL for bria outpaint: {image_url}")

            # Assemble arguments for the outpaint API
            api_args = {
                "image_url": image_url,
                "prompt": config.prompt,
                "original_image_location": original_image_location,
                "original_image_size": original_image_size,
                "canvas_size": [config.target_width, config.target_height],
                "num_outputs": config.num_images,
            }

            # Call the client process_outpaint method
            logger.debug(f"Calling BRIA outpaint API with arguments: {api_args}")
            return await client.process_outpaint(
                prompt=config.prompt,
                image_url=image_url,
                lora_spec=None,  # No LoRA for BRIA
                output_dir=output_dir,
                filename_suffix=filename_suffix,
                filename_prefix=filename_prefix,
                outpaint_tool="bria",
                target_width=config.target_width,
                target_height=config.target_height,
                original_image_location=original_image_location,
                original_image_size=original_image_size,
                num_images=config.num_images,
            )
        except Exception as e:
            logger.error(f"Outpaint process failed: {e}", exc_info=True)
            msg = f"Outpaint process failed: {e}"
            raise RuntimeError(msg) from e
    else:
        # Should never happen due to validation in OutpaintConfig
        msg = f"Invalid outpaint tool: {config.outpaint_tool}. Valid options are 'bria' or 'flux'."
        logger.error(msg)
        raise ValueError(msg)


async def run_flux_outpaint(
    client: FalApiClient,
    config: OutpaintConfig,
    original_width: int,
    original_height: int,
    output_dir: Path | None = None,
    filename_suffix: str | None = None,
    filename_prefix: str | None = None,
) -> ImageResult:
    """
    Uses the flux-lora/inpainting endpoint to perform outpainting by creating
    a mask image where the original image area is white and the rest is black.

    Args:
        client: The FalApiClient instance.
        config: The OutpaintConfig object containing specific parameters.
        original_width: The width of the original input image.
        original_height: The height of the original input image.
        output_dir: Directory to save the result image and metadata.
        filename_suffix: Optional suffix for generated filenames.
        filename_prefix: Optional prefix for generated filenames.

    Returns:
        An ImageResult object containing information about the generated image.

    Raises:
        ValueError: If required parameters are missing or invalid.
        RuntimeError: If the image upload or API call fails.
    """
    logger.info(
        f"Starting flux outpaint process with target size {config.target_width}x{config.target_height}"
    )

    # 1. Validate dimensions
    if original_width > config.target_width or original_height > config.target_height:
        msg = (
            f"Original image ({original_width}x{original_height}) cannot be larger "
            f"than target size ({config.target_width}x{config.target_height}) for outpainting."
        )
        logger.error(msg)
        raise ValueError(msg)

    # 2. Create a mask image (black with white rectangle where original image goes)
    mask_path, original_image_location = create_outpaint_mask(
        image_width=original_width,
        image_height=original_height,
        target_width=config.target_width,
        target_height=config.target_height,
    )

    # 3. Get the image URLs
    try:
        image_url = await config.input_image.to_url(client)
        logger.info(f"Using image URL for flux outpaint: {image_url}")

        # Upload the mask image using client.upload_image (previously upload_file)
        mask_url = await client.upload_image(mask_path)
        logger.info(f"Using mask URL for flux outpaint: {mask_url}")
    except Exception as e:
        logger.error(f"Failed to get image or mask URL: {e}", exc_info=True)
        msg = f"Failed to prepare images for flux outpainting: {e}"
        raise RuntimeError(msg) from e

    # 4. Call the client's process_outpaint method with flux as the outpaint_tool
    logger.info("Calling client.process_outpaint for flux outpainting")
    try:
        # Use the new client API that handles outpainting properly
        result = await client.process_outpaint(
            prompt=config.prompt,
            image_url=image_url,
            lora_spec=None,  # No LoRA for this call - we'll let the client handle it
            output_dir=output_dir,
            filename_suffix=filename_suffix or "_flux_outpaint",
            filename_prefix=filename_prefix,
            outpaint_tool="flux",  # Use the flux outpaint tool
            target_width=config.target_width,
            target_height=config.target_height,
            # Additional parameters specific to flux outpainting
            mask_url=mask_url,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            negative_prompt=config.negative_prompt,
            enable_safety_checker=config.enable_safety_checker,
            num_images=config.num_images,
        )
        logger.success(f"Flux outpaint job completed. Request ID: {result.request_id}")
        return result
    except Exception as e:
        logger.error(f"FalApiClient.process_outpaint failed: {e}", exc_info=True)
        msg = f"Flux outpaint process failed: {e}"
        raise RuntimeError(msg) from e
