#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["fal-client", "rich"]
# ///
# this_file: src/twat_genai/engines/fal/client.py

"""
API client for interacting with the FAL.ai API.
Derived from the superres tool's client.
"""

from __future__ import annotations

import logging  # TODO: Switch to loguru?
from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable
import httpx
from slugify import slugify
from loguru import logger
from datetime import datetime, timezone
import asyncio
import json
import time

import fal_client
from fal_client.client import FalClientError  # Add direct import for FalClientError
from twat_genai.core.config import ImageResult

# Import ModelTypes, the unified enum
# from fal.config import ModelTypes
from twat_genai.engines.fal.config import ModelTypes
from twat_genai.engines.fal.lora import build_lora_arguments

# Logging setup
# TODO: Adapt logging to use loguru
log = logging.getLogger("twat_genai.engines.fal.client")

# Constants
METHOD_KEY = "method"  # Key for method parameter


# --- Helper Functions (adapted from src/twat_genai/__main__.py) ---


async def _download_image_helper(url: str, output_path: Path) -> None:
    """
    Internal helper to download an image from a URL and save it to disk.
    Uses httpx.
    """
    try:
        async with httpx.AsyncClient() as client:
            logger.debug(f"Downloading result image from {url} to {output_path}")
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure dir exists
            output_path.write_bytes(response.content)
            logger.info(f"Saved result image to: {output_path}")
    except httpx.RequestError as e:
        logger.error(
            f"HTTP error downloading result image from {url}: {e}", exc_info=True
        )
        msg = f"Failed to download result image from {url}: {e}"
        raise RuntimeError(msg) from e
    except Exception as e:
        logger.error(f"Error saving result image {output_path}: {e}", exc_info=True)
        # Don't re-raise, allow get_result to continue if download fails
        # We still want the metadata if possible.


async def _submit_fal_job(model_endpoint: str, arguments: dict[str, Any]) -> str:
    """
    Internal helper to submit an asynchronous job to FAL.

    Args:
        model_endpoint: The full FAL model endpoint string (e.g., 'fal-ai/flux-lora')
        arguments: The arguments dictionary for the FAL job.

    Returns:
        The FAL request ID.
    """
    logger.debug(f"Submitting job to {model_endpoint} with args: {arguments}")
    handler = await fal_client.submit_async(model_endpoint, arguments=arguments)
    logger.info(f"Submitted job to {model_endpoint} with ID: {handler.request_id}")
    return handler.request_id


# --- FalApiClient Class ---


class FalApiClient:
    """
    Client for interacting with the Fal.ai API.
    Encapsulates job submission and result handling for different operations.
    """

    def __init__(self):
        """Initialize the API client."""
        # TODO: Integrate with FALEngine key handling? Ensure FAL_KEY is checked.
        pass  # No state needed currently, methods use fal_client directly

    async def upload_image(self, image_path: str | Path) -> str:
        """
        Upload an image to the Fal.ai storage.

        Args:
            image_path: Path to the image file

        Returns:
            str: URL of the uploaded image
        """
        try:
            path_obj = Path(image_path)
            logger.debug(f"Uploading image from: {path_obj}")
            response = await fal_client.upload_file_async(path_obj)
            logger.info(f"Image uploaded to: {response}")
            return response
        except Exception as e:
            logger.error(f"Failed to upload image {image_path}: {e}", exc_info=True)
            msg = f"Failed to upload image: {e!s}"
            raise RuntimeError(msg) from e

    async def process_upscale(
        self,
        image_url: str,
        output_dir: Path | None = None,
        filename_suffix: str | None = None,
        filename_prefix: str | None = None,
        upscaler: Literal[
            "drct",
            "ideogram",
            "recraft_creative",
            "recraft_clarity",
            "ccsr",
            "esrgan",
            "aura_sr",
            "clarity",
        ] = "clarity",
        **kwargs: Any,  # Upscale-specific parameters
    ) -> ImageResult:
        """
        Process an upscale job.

        Args:
            image_url: URL of the image to upscale
            output_dir: Optional directory to save the result
            filename_suffix: Optional suffix for the result filename
            filename_prefix: Optional prefix for the result filename
            upscaler: Which upscaler to use (default: "clarity")
            **kwargs: Additional parameters specific to the chosen upscaler
                     See UpscaleConfig for details on model-specific parameters
        """
        # Map upscaler choice to ModelTypes enum
        upscaler_map = {
            "drct": ModelTypes.UPSCALER_DRCT,
            "ideogram": ModelTypes.UPSCALER_IDEOGRAM,
            "recraft_creative": ModelTypes.UPSCALER_RECRAFT_CREATIVE,
            "recraft_clarity": ModelTypes.UPSCALER_RECRAFT_CLARITY,
            "ccsr": ModelTypes.UPSCALER_CCSR,
            "esrgan": ModelTypes.UPSCALER_ESRGAN,
            "aura_sr": ModelTypes.UPSCALER_AURA_SR,
            "clarity": ModelTypes.UPSCALER_CLARITY,
        }

        model_type = upscaler_map.get(upscaler)
        if not model_type:
            msg = f"Invalid upscaler choice: {upscaler}. Valid choices: {list(upscaler_map.keys())}"
            raise ValueError(msg)

        logger.info(f"Processing upscale job with {upscaler} ({model_type.value})")

        # Upscale requires an image but no prompt or LoRAs
        endpoint = model_type.value

        # Build the arguments for this specific upscaler
        fal_args = {
            "image_url": image_url,
            # Add any model-specific parameters from kwargs
            **{k: v for k, v in kwargs.items() if v is not None},
        }

        logger.debug(f"Final FAL arguments for upscale: {fal_args}")

        # Submit job and get result
        try:
            request_id = await _submit_fal_job(endpoint, fal_args)

            # Prepare job_params for metadata logging
            job_params = {
                "model": endpoint,
                "upscaler": upscaler,
                "input_image_url": image_url,  # Include input image URL for filename generation
                **kwargs,  # Include all passed config params
            }

            return await self._get_fal_result(
                request_id=request_id,
                model_endpoint=endpoint,
                output_dir=output_dir,
                filename_suffix=filename_suffix,
                filename_prefix=filename_prefix,
                original_prompt=None,  # No prompt for upscale
                job_params=job_params,
            )
        except Exception as e:
            logger.error(f"Upscale process failed: {e}", exc_info=True)
            msg = f"Upscale process failed: {e}"
            raise RuntimeError(msg) from e

    async def process_outpaint(
        self,
        prompt: str,
        image_url: str,
        lora_spec: Any | None = None,
        output_dir: Path | None = None,
        filename_suffix: str | None = None,
        filename_prefix: str | None = None,
        outpaint_tool: Literal["bria", "flux"] = "bria",
        target_width: int | None = None,
        target_height: int | None = None,
        canvas_size: list[int] | None = None,
        **kwargs: Any,  # Outpaint-specific parameters
    ) -> ImageResult:
        """
        Process an outpaint job to expand an image.

        Args:
            prompt: Text prompt describing the expanded areas
            image_url: URL of the image to outpaint
            lora_spec: Optional LoRA specification (only used with flux outpainting)
            output_dir: Optional directory to save the result
            filename_suffix: Optional suffix for the result filename
            filename_prefix: Optional prefix for the result filename
            outpaint_tool: Which outpainter to use ("bria" or "flux")
            target_width: Target width for the expanded image
            target_height: Target height for the expanded image
            canvas_size: List of two integers representing the canvas size
            **kwargs: Additional parameters specific to the chosen outpainter
        """
        # Map outpaint tool choice to ModelTypes enum
        outpaint_map = {
            "bria": ModelTypes.OUTPAINT_BRIA,
            "flux": ModelTypes.OUTPAINT_FLUX,
        }

        model_type = outpaint_map.get(outpaint_tool)
        if not model_type:
            msg = f"Invalid outpaint tool choice: {outpaint_tool}. Valid choices: {list(outpaint_map.keys())}"
            raise ValueError(msg)

        logger.info(
            f"Processing outpaint job with {outpaint_tool} ({model_type.value})"
        )

        endpoint = model_type.value

        # Prepare tool-specific args based on the chosen tool
        if outpaint_tool == "bria":
            # Bria requires target_width and target_height
            if not target_width or not target_height:
                msg = "target_width and target_height are required for bria outpainting"
                raise ValueError(msg)

            # If canvas_size is not explicitly provided, create it from target dimensions
            if not canvas_size and target_width and target_height:
                canvas_size = [target_width, target_height]

            # Set up args for Bria outpainting
            fal_args = {
                "image_url": image_url,
                "prompt": prompt,
                "target_width": target_width,
                "target_height": target_height,
                "canvas_size": canvas_size,  # Always include canvas_size
                # Add general kwargs for additional parameters
                **{k: v for k, v in kwargs.items() if v is not None},
            }

        elif outpaint_tool == "flux":
            # Flux requires canvas_size instead of target dimensions
            if not canvas_size:
                # If no canvas_size but we have target dimensions, create it
                if target_width and target_height:
                    canvas_size = [target_width, target_height]
                else:
                    msg = "canvas_size is required for flux outpainting"
                    raise ValueError(msg)

            # Set up args for Flux outpainting
            fal_args = {
                "image_url": image_url,
                "prompt": prompt,
                "canvas_size": canvas_size,
                # Add general kwargs for additional parameters
                **{k: v for k, v in kwargs.items() if v is not None},
            }

            # Add LoRA parameters if specified (only supported by Flux)
            if lora_spec:
                from twat_genai.engines.fal.lora import build_lora_arguments

                lora_args = build_lora_arguments(lora_spec)
                fal_args.update(lora_args)

        else:
            # This should never happen due to the validation above
            msg = f"Unexpected outpaint tool: {outpaint_tool}"
            raise ValueError(msg)

        logger.debug(f"Final FAL arguments for outpaint: {fal_args}")

        # Submit job and get result
        try:
            request_id = await _submit_fal_job(endpoint, fal_args)

            # Prepare job_params for metadata logging
            job_params = {
                "model": endpoint,
                "outpaint_tool": outpaint_tool,
                "prompt": prompt,
                "input_image_url": image_url,  # Include input URL for filename generation
                "target_width": target_width,
                "target_height": target_height,
                "canvas_size": canvas_size,
                **kwargs,  # Include all passed config params
                # Include lora info if used
                "lora": lora_spec if lora_spec else None,
            }

            return await self._get_fal_result(
                request_id=request_id,
                model_endpoint=endpoint,
                output_dir=output_dir,
                filename_suffix=filename_suffix,
                filename_prefix=filename_prefix,
                original_prompt=prompt,
                job_params=job_params,
            )
        except Exception as e:
            logger.error(f"Outpaint process failed: {e!r}", exc_info=True)
            msg = f"Outpaint process failed: {e!r}"
            raise RuntimeError(msg) from e

    async def process_depth(
        self,
        prompt: str,
        image_url: str,
        lora_spec: Any | None,
        output_dir: Path | None = None,
        filename_suffix: str | None = None,
        filename_prefix: str | None = None,
        **kwargs: Any,  # Base config + I2I specific (neg_prompt)
    ) -> ImageResult:
        """
        Process a depth-based image-to-image job.

        This is essentially a wrapper around _process_generic that handles the model selection
        and proper parameter passing.
        """
        return await self._process_generic(
            model_type=ModelTypes.DEPTH,
            prompt=prompt,
            lora_spec=lora_spec,
            output_dir=output_dir,
            filename_suffix=filename_suffix,
            filename_prefix=filename_prefix,
            image_url=image_url,
            input_image_url=image_url,  # Include input URL for filename generation
            **kwargs,
        )

    async def _process_generic(
        self,
        model_type: ModelTypes,
        prompt: str,
        lora_spec: Any | None,
        output_dir: Path | None = None,
        filename_suffix: str | None = None,
        filename_prefix: str | None = None,
        image_url: str | None = None,  # Optional for I2I, Canny, Depth
        input_image_url: str | None = None,  # For filename generation
        **kwargs: Any,  # Base config (image_size, steps, etc.) + I2I specific (strength, negative_prompt)
    ) -> ImageResult:
        """
        Generic method to process various image generation jobs.

        This provides a single implementation for TTI, I2I, and control model variants.
        The model_type parameter determines which endpoint to call.

        Args:
            model_type: ModelTypes enum value for this request
            prompt: Text prompt guiding generation
            lora_spec: LoRA specification string
            output_dir: Optional directory to save results
            filename_suffix: Optional suffix for result filenames
            filename_prefix: Optional prefix for result filenames
            image_url: Optional input image URL (required for I2I and control models)
            input_image_url: Optional input image URL for filename generation
            **kwargs: Additional parameters specific to the generation method
        """
        endpoint = model_type.value
        logger.info(f"Processing job with model {model_type.name} ({endpoint})")

        # Validate input for I2I and controlnet models
        if model_type in {ModelTypes.IMAGE, ModelTypes.CANNY, ModelTypes.DEPTH}:
            if not image_url:
                msg = f"input_image is required for {model_type.name} mode"
                raise ValueError(msg)

        # Prepare base arguments
        fal_args = {"prompt": prompt}

        # Add image URL for image-to-image and controlnet variants
        if image_url:
            fal_args["image_url"] = image_url

        # Add strength parameter for I2I if provided (default handled by API)
        if "strength" in kwargs:
            strength = kwargs.pop("strength")
            if strength is not None:
                fal_args["strength"] = strength

        # Add negative_prompt if provided
        if "negative_prompt" in kwargs:
            negative_prompt = kwargs.pop("negative_prompt")
            if negative_prompt:  # Only add if not empty
                fal_args["negative_prompt"] = negative_prompt

        # Handle image size (preset or dimensions)
        if "image_size" in kwargs:
            image_size = kwargs.pop("image_size")
            if image_size is not None:
                # Import required types here to avoid circular imports
                from twat_genai.core.image import ImageSizes, ImageSizeWH

                if isinstance(image_size, ImageSizeWH):
                    fal_args["width"] = image_size.width
                    fal_args["height"] = image_size.height
                elif isinstance(image_size, ImageSizes):
                    # Get the dimensions using the value property or preset table
                    if hasattr(image_size, "dimensions"):
                        width, height = image_size.dimensions
                    else:
                        # Fallback to known presets if dimensions not available
                        size_map = {
                            ImageSizes.SQ: (1024, 1024),  # Square
                            ImageSizes.PORT: (768, 1024),  # Portrait
                            ImageSizes.LAND: (1024, 768),  # Landscape
                            ImageSizes.WIDE: (1216, 832),  # Wide
                            ImageSizes.ULTRA: (1344, 768),  # Ultra wide
                        }
                        width, height = size_map.get(image_size, (1024, 1024))

                    fal_args["width"] = width
                    fal_args["height"] = height

        # Add remaining standard parameters if provided
        standard_params = ["guidance_scale", "num_inference_steps"]
        for param in standard_params:
            if param in kwargs:
                value = kwargs.pop(param)
                if value is not None:
                    fal_args[param] = value

        # Add any remaining kwargs as direct parameters
        # This allows for future extension without code changes
        for k, v in kwargs.items():
            if v is not None:
                fal_args[k] = v

        # Add LoRA parameters if specified
        if lora_spec:
            from twat_genai.engines.fal.lora import build_lora_arguments

            # build_lora_arguments is async and expects prompt
            lora_list, final_prompt = await build_lora_arguments(lora_spec, prompt)
            if lora_list:
                fal_args["loras"] = lora_list # FAL expects "loras" key
            fal_args["prompt"] = final_prompt # Update prompt with LoRA triggers

        logger.debug(f"Final FAL arguments: {fal_args}")

        # Submit job and get result
        try:
            request_id = await _submit_fal_job(endpoint, fal_args)

            # Prepare job_params for metadata logging
            job_params = {
                "model": endpoint,
                "prompt": prompt,
                "input_image_url": input_image_url
                or image_url,  # Include input URL for filename generation
                **kwargs,  # Include all passed config params
                # Include original args that may have been transformed
                "lora": lora_spec if lora_spec else None,
            }

            return await self._get_fal_result(
                request_id=request_id,
                model_endpoint=endpoint,
                output_dir=output_dir,
                filename_suffix=filename_suffix,
                filename_prefix=filename_prefix,
                original_prompt=prompt,
                job_params=job_params,
            )
        except Exception as e:
            logger.error(f"Job processing failed: {e}", exc_info=True)
            msg = f"Processing {model_type.name} job failed: {e}"
            raise RuntimeError(msg) from e

    async def process_tti(
        self,
        prompt: str,
        lora_spec: Any | None,
        output_dir: Path | None = None,
        filename_suffix: str | None = None,
        filename_prefix: str | None = None,
        **kwargs: Any,  # Base config params (image_size, steps, etc.)
    ) -> ImageResult:
        """Process a Text-to-Image job."""
        return await self._process_generic(
            model_type=ModelTypes.TEXT,
            prompt=prompt,
            lora_spec=lora_spec,
            output_dir=output_dir,
            filename_suffix=filename_suffix,
            filename_prefix=filename_prefix,
            image_url=None,  # No image for TTI
            **kwargs,
        )

    async def process_i2i(
        self,
        prompt: str,
        image_url: str,
        lora_spec: Any | None,
        output_dir: Path | None = None,
        filename_suffix: str | None = None,
        filename_prefix: str | None = None,
        **kwargs: Any,  # Base config + I2I specific (strength, neg_prompt)
    ) -> ImageResult:
        """
        Process an image-to-image job.

        This is essentially a wrapper around _process_generic that handles the model selection
        and proper parameter passing.
        """
        return await self._process_generic(
            model_type=ModelTypes.IMAGE,
            prompt=prompt,
            lora_spec=lora_spec,
            output_dir=output_dir,
            filename_suffix=filename_suffix,
            filename_prefix=filename_prefix,
            image_url=image_url,
            input_image_url=image_url,  # Include input URL for filename generation
            **kwargs,
        )

    async def process_canny(
        self,
        prompt: str,
        image_url: str,
        lora_spec: Any | None,
        output_dir: Path | None = None,
        filename_suffix: str | None = None,
        filename_prefix: str | None = None,
        **kwargs: Any,  # Base config + I2I specific (neg_prompt)
    ) -> ImageResult:
        """
        Process a canny edge detection image-to-image job.

        This is essentially a wrapper around _process_generic that handles the model selection
        and proper parameter passing.
        """
        return await self._process_generic(
            model_type=ModelTypes.CANNY,
            prompt=prompt,
            lora_spec=lora_spec,
            output_dir=output_dir,
            filename_suffix=filename_suffix,
            filename_prefix=filename_prefix,
            image_url=image_url,
            input_image_url=image_url,  # Include input URL for filename generation
            **kwargs,
        )

    async def process_genfill(
        self,
        prompt: str,
        image_url: str,  # The outpainted image URL
        mask_url: str,  # The border mask URL
        output_dir: Path | None = None,
        filename_suffix: str | None = None,
        filename_prefix: str | None = None,
        num_images: int = 1,
        **kwargs: Any,  # Should include things like negative_prompt if needed
    ) -> ImageResult:
        """
        Process an image using the Bria GenFill model.

        Args:
            prompt: The original prompt used for outpainting.
            image_url: URL of the image to process (the result from outpainting).
            mask_url: URL of the border mask to use for genfill.
            output_dir: Directory to save the final image and metadata.
            filename_suffix: Suffix to add to the generated filename.
            filename_prefix: Prefix for the generated filename.
            num_images: Number of images to generate (usually 1 for genfill).
            **kwargs: Additional arguments for the genfill API.

        Returns:
            ImageResult containing info about the processed image.
        """
        model_endpoint = "fal-ai/bria/genfill"
        job_params = {
            "prompt": prompt,
            "image_url": image_url,
            "mask_url": mask_url,
            "num_images": num_images,
            # Explicitly include known/expected optional args from kwargs
            # Add more as needed based on the genfill API specifics
            "negative_prompt": kwargs.get("negative_prompt", ""),
        }
        # Filter out None values if the API doesn't like them
        job_params = {k: v for k, v in job_params.items() if v is not None}

        try:
            logger.info(
                f"Submitting GenFill job to {model_endpoint} with params: {job_params}"
            )
            request_id = await _submit_fal_job(model_endpoint, job_params)
            logger.info(f"Submitted GenFill job {request_id}")

            return await self._get_fal_result(
                request_id=request_id,
                model_endpoint=model_endpoint,
                output_dir=output_dir,
                filename_suffix=filename_suffix,
                filename_prefix=filename_prefix,
                original_prompt=prompt,
                job_params=job_params,
            )
        except Exception as e:
            logger.error(
                f"Error processing GenFill job for {model_endpoint}: {e}",
                exc_info=True,
            )
            msg = f"Failed GenFill job: {e!s}"
            raise RuntimeError(msg) from e

    async def _get_fal_result(
        self,
        request_id: str,
        model_endpoint: str,  # Need the endpoint to poll status/result
        output_dir: Path | None = None,
        filename_suffix: str | None = None,
        filename_prefix: str | None = None,
        original_prompt: str | None = None,
        job_params: dict[str, Any] | None = None,
    ) -> ImageResult:
        """
        Get the result of a FAL job and process it into an ImageResult.

        Args:
            request_id: The FAL API request ID
            model_endpoint: The model endpoint being used
            output_dir: Optional directory to save the result
            filename_suffix: Optional suffix for the result filename
            filename_prefix: Optional prefix for the result filename
            original_prompt: Optional prompt text used for the generation
            job_params: Optional parameters dictionary used in job submission

        Returns:
            An ImageResult object containing information about the generated image

        Raises:
            RuntimeError: If the request fails or returns an error
        """
        logger.debug(f"Waiting for FAL result for request: {request_id}")

        # Imports kept local to avoid unnecessary imports when not using this
        import fal_client

        try:
            result = await fal_client.result_async(model_endpoint, request_id)
            logger.debug(f"Received result for request {request_id}: {result}")

            # Extract image result data
            result_extractor = self._get_result_extractor(model_endpoint)
            image_dicts = result_extractor(result)

            if not image_dicts:
                msg = "No valid image results returned from FAL API."
                raise ValueError(msg)

            # Download and save the result images if output_dir is provided
            if output_dir:
                timestamp = time.strftime("%Y%m%d%H%M%S")
                base_filename = (
                    f"{filename_prefix or ''}{timestamp}{filename_suffix or ''}"
                )
                base_filename = slugify(base_filename)

                # Ensure the output directory exists
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                for i, img_dict in enumerate(image_dicts):
                    if not img_dict.get("url"):
                        logger.warning(f"No image URL in result {i}, skipping save.")
                        continue

                    # Derive file extension from mimetype, with fallback to jpg
                    mimetype = img_dict.get("mimetype", "image/jpeg")
                    extension = mimetype.split("/")[-1].split("+")[
                        0
                    ]  # e.g. 'image/jpeg+xml' -> 'jpeg'
                    if extension not in {"jpeg", "jpg", "png", "webp", "gif"}:
                        logger.warning(
                            f"Unrecognized mimetype {mimetype}, defaulting to jpg."
                        )
                        extension = "jpg"

                    # Create numbered filenames if multiple results
                    index_suffix = f"_{i}" if len(image_dicts) > 1 else ""
                    output_filename = f"{base_filename}{index_suffix}.{extension}"
                    output_path = output_dir / output_filename

                    # Save metadata alongside the image
                    metadata_filename = f"{base_filename}{index_suffix}_metadata.json"
                    metadata_path = output_dir / metadata_filename

                    # Try to download and save the image
                    try:
                        # Make a new request to download the image to avoid chunked transfer issues
                        await _download_image_helper(img_dict["url"], output_path)
                        logger.info(f"Saved image to: {output_path}")

                        # Save metadata (if we have job parameters)
                        metadata = {
                            "request_id": request_id,
                            "model": model_endpoint,
                            "created_at": datetime.now().isoformat(),
                            "prompt": original_prompt,
                            "parameters": job_params or {},
                            "output_file": str(output_path),
                            "fal_result": img_dict,
                        }
                        with open(metadata_path, "w") as f:
                            json.dump(metadata, f, indent=2, default=str)
                        logger.debug(f"Saved metadata to: {metadata_path}")

                        # Update the image_dict with the local file path for the caller
                        img_dict["path"] = str(output_path)
                        img_dict["metadata_path"] = str(metadata_path)

                    except Exception as e:
                        logger.error(f"Failed to save image: {e}", exc_info=True)

            # Return the first image info, with the complete list in all_images
            return ImageResult(
                request_id=request_id,
                timestamp=datetime.now().strftime("%Y%m%d%H%M%S"),
                result=result
                if isinstance(result, dict)
                else {"raw_result": str(result)},
                image_info=image_dicts[0],
                original_prompt=original_prompt,
                job_params=job_params,
            )

        except FalClientError as e:
            # Handle FAL API errors specifically
            try:
                # Try to access the error details, which might be a list of dicts or a string
                error_details = e.args[0] if e.args else str(e)
                logger.error(f"FAL API error: {error_details!r}")
            except Exception as nested:
                # If there's an issue accessing/formatting the error, use a safer repr
                logger.error(f"FAL API error (details unavailable): {repr(e)}")

            # Re-raise with a more readable message
            raise RuntimeError(f"FAL API error: {repr(e)}") from e

        except Exception as e:
            # For other exceptions, just log and re-raise
            logger.error(
                f"Error during FAL result processing: {repr(e)}", exc_info=True
            )
            raise RuntimeError(f"Error processing FAL result: {repr(e)}") from e

    def _get_result_extractor(
        self, model_endpoint: str
    ) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
        """Return the appropriate result extraction function based on model endpoint."""
        # TODO: Implement specific extractors for different model types
        if model_endpoint == ModelTypes.OUTPAINT_BRIA.value:
            # return self._extract_outpaint_info # Placeholder
            logger.warning(
                f"Using generic extractor for outpaint model {model_endpoint}"
            )
            return FalApiClient._extract_generic_image_info
        elif (
            model_endpoint.startswith("fal-ai/drct")
            or model_endpoint.startswith("fal-ai/ideogram/upscale")
            or model_endpoint.startswith("fal-ai/recraft")
            or model_endpoint.startswith("fal-ai/ccsr")
            or model_endpoint.startswith("fal-ai/esrgan")
            or model_endpoint.startswith("fal-ai/aura-sr")
            or model_endpoint.startswith("fal-ai/clarity-upscaler")
        ):
            # return self._extract_upscale_info # Placeholder
            logger.warning(
                f"Using generic extractor for upscale model {model_endpoint}"
            )
            return FalApiClient._extract_generic_image_info
        else:  # Default to generic TTI/I2I extractor
            # return self._extract_tti_i2i_info # Placeholder
            return FalApiClient._extract_generic_image_info

    @staticmethod
    def _extract_generic_image_info(
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Extract image information from common API response patterns (images, image, url).
        Now an instance method.

        Args:
            result: The API response dictionary

        Returns:
            List of dictionaries containing image information.
            Returns empty list if no image info found.
        """
        extracted_list = []
        try:
            # Define a helper function to create the info dict
            def _create_info_dict(img_data: dict[str, Any] | str) -> dict[str, Any]:
                if isinstance(img_data, str):  # Handle cases where only URL is given
                    return {
                        "url": img_data,
                        "content_type": "image/png",  # Assume PNG if not specified
                        "file_name": "output.png",
                        "file_size": 0,
                        "width": None,
                        "height": None,
                        "seed": result.get("seed"),  # Try to get top-level seed
                        "file_data": None,
                    }
                # Extract potential seed from image data or top level
                seed = img_data.get("seed", result.get("seed"))
                return {
                    "url": img_data.get("url", ""),
                    "content_type": img_data.get("content_type", "image/png"),
                    "file_name": img_data.get("file_name", "output.png"),
                    "file_size": img_data.get("file_size", 0),
                    "width": img_data.get("width"),
                    "height": img_data.get("height"),
                    "seed": seed,
                    "file_data": img_data.get("file_data"),  # Base64 data if available
                }

            # Handle responses with multiple images
            if "images" in result and isinstance(result["images"], list):
                extracted_list = [_create_info_dict(img) for img in result["images"]]
            elif "image" in result:
                if isinstance(result["image"], list):
                    extracted_list = [_create_info_dict(img) for img in result["image"]]
                elif isinstance(result["image"], dict) or isinstance(
                    result["image"], str
                ):
                    # Single image dictionary or URL string
                    extracted_list = [_create_info_dict(result["image"])]
            elif "url" in result and isinstance(result["url"], str):
                # Single image URL string at top level
                extracted_list = [_create_info_dict(result["url"])]
            else:
                logger.warning(
                    f"No standard image key (images, image, url) found in API response: {result}"
                )
                return []  # Return empty list

            # Filter out entries without a URL
            valid_extracted_list = [info for info in extracted_list if info.get("url")]
            if not valid_extracted_list:
                logger.warning(
                    f"Extractor found image keys but no valid URLs in result: {result}"
                )

            return valid_extracted_list

        except Exception as e:
            # Log error but let _get_fal_result handle raising the final RuntimeError
            logger.error(f"Internal error extracting image info: {e}", exc_info=True)
            return []  # Return empty list to signal failure

    # --- Placeholder for specific extractors --- #
    # def _extract_outpaint_info(self, result: dict[str, Any]) -> list[dict[str, Any]]:
    #     # Implementation specific to outpaint result structure
    #     base_info = self._extract_generic_image_info(result)
    #     # Extract seed or other specific fields
    #     seed = result.get("seed")
    #     for info in base_info:
    #         info["seed"] = seed # Add/overwrite seed
    #     return base_info

    # def _extract_upscale_info(self, result: dict[str, Any]) -> list[dict[str, Any]]:
    #     # Implementation specific to various upscaler result structures
    #     return self._extract_generic_image_info(result)

    # def _extract_tti_i2i_info(self, result: dict[str, Any]) -> list[dict[str, Any]]:
    #     # May be same as generic, or extract specific TTI/I2I metadata
    #     return self._extract_generic_image_info(result)
