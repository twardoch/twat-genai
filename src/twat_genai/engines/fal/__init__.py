#!/usr/bin/env -S uv run
# /// script
# dependencies = ["fal-client", "python-dotenv"]
# ///
"""FAL image generation engine implementation."""

from __future__ import annotations

import os

import asyncio  # Unused
import tempfile
import time
import shutil
from pathlib import Path  # Make sure Path is imported at the top level

# from enum import Enum # Unused
from typing import TYPE_CHECKING, Any  # Removed Dict, Optional

from dotenv import load_dotenv
from loguru import logger
from PIL import Image  # Import Pillow
from slugify import slugify

# Internal imports
from twat_genai.engines.fal.client import FalApiClient  # Import only the client
from twat_genai.engines.fal.config import (
    # FALJobConfig, # Removed unused import
    ImageToImageConfig,
    ModelTypes,
    UpscaleConfig,
    OutpaintConfig,
)
from twat_genai.engines.base import EngineConfig, ImageGenerationEngine
from twat_genai.core.config import ImageResult, ImageInput
from twat_genai.core import image_utils  # Import the module

if TYPE_CHECKING:
    pass  # No Path import needed here as it's imported at module level

load_dotenv()


class FALEngine(ImageGenerationEngine):
    """FAL image generation engine implementation.

    Handles Text-to-Image, Image-to-Image, Upscaling, and Outpainting via FAL.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        """
        Initialize the FAL engine.

        Args:
            output_dir: Directory to save generated images and metadata.
        """
        self.output_dir = output_dir
        self.api_key: str | None = None
        self.client: FalApiClient | None = None

    async def initialize(self) -> None:
        """Initialize the engine, verify API key, and create client."""
        self.api_key = os.getenv("FAL_KEY")
        if not self.api_key:
            msg = "FAL_KEY environment variable not set. Please set it with your FAL API key."
            logger.error(msg)
            raise ValueError(msg)
        logger.debug("FAL_KEY found. Initializing FalApiClient.")
        self.client = FalApiClient()  # Instantiate our client

    async def _prepare_image_input(
        self,
        config_input: ImageInput,
        model_type: ModelTypes,
    ) -> tuple[str, int, int, Path | None]:
        """Handles image downloading, resizing, uploading, and returns URL and dimensions.

        Returns:
            tuple[str, int, int, Path | None]: Uploaded image URL, original width, original height, and optional local path
        """
        if not self.client:
            msg = "FALEngine not initialized. Call initialize() first."
            raise RuntimeError(msg)

        # The input image path to return
        original_input_path: Path | None = config_input.path

        # Try to convert ImageInput to FALImageInput for better handling
        try:
            from twat_genai.engines.fal.models import FALImageInput

            # If it's already a FALImageInput, use it
            if isinstance(config_input, FALImageInput):
                fal_input = config_input
            else:
                # Otherwise convert base ImageInput to FALImageInput
                fal_input = FALImageInput.from_base(config_input)

            # Get the URL using the FALImageInput implementation
            image_url = await fal_input.to_url(client=self.client)

            # Download the image to get dimensions
            temp_download_path = await image_utils.download_image_to_temp(image_url)
            temp_resized_path = None

            try:
                # Get original dimensions
                with Image.open(temp_download_path) as img:
                    original_width, original_height = img.size
                    logger.debug(
                        f"Original image dimensions: {original_width}x{original_height}"
                    )

                # Resize image if it's an upscaler model and needs resizing
                image_path_final = temp_download_path
                if model_type.name.startswith("UPSCALER_"):
                    logger.debug(
                        f"Model {model_type.name} is upscaler, checking dimensions..."
                    )
                    temp_resized_path = image_utils.resize_image_if_needed(
                        image_path=temp_download_path, model_type=model_type
                    )
                    if temp_resized_path:
                        logger.info(
                            f"Image resized for upscaler to: {temp_resized_path}"
                        )
                        image_path_final = temp_resized_path

                        # Re-upload if resized
                        if temp_resized_path != temp_download_path:
                            logger.info("Re-uploading resized image to FAL storage...")
                            image_url = await self.client.upload_image(image_path_final)
                            logger.info(
                                f"Resized image uploaded successfully: {image_url}"
                            )
                    else:
                        logger.debug("Upscaler input within limits, no resize needed.")

                # Clean up temporary files
                try:
                    if temp_download_path:
                        Path(temp_download_path).unlink()
                    if temp_resized_path and temp_resized_path != temp_download_path:
                        Path(temp_resized_path).unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp files: {e}")

                if not original_width or not original_height:
                    msg = "Internal error: Original image dimensions not obtained."
                    logger.error(msg)
                    raise RuntimeError(msg)

                return image_url, original_width, original_height, original_input_path

            except Exception as e:
                logger.error(f"Error processing image: {e}", exc_info=True)
                msg = f"Failed to process image: {e!s}"
                raise RuntimeError(msg) from e

        except ImportError as e:
            logger.error(f"Failed to import FALImageInput: {e}", exc_info=True)
            msg = f"Error importing FALImageInput: {e!s}"
            raise RuntimeError(msg) from e

        except Exception as e:
            logger.error(f"Failed to prepare image: {e}", exc_info=True)
            msg = f"Error preparing image input: {e!s}"
            raise RuntimeError(msg) from e

    async def generate(
        self,
        prompt: str,
        config: EngineConfig,
        **kwargs: Any,
    ) -> ImageResult:
        """
        Generate an image using FAL, handling different model types.

        Args:
            prompt: Text prompt for generation (used by TTI, Upscale, Outpaint)
            config: Base engine configuration (guidance_scale, num_inference_steps, image_size)
            **kwargs: Additional parameters including:
                model (ModelTypes): The specific FAL model/operation to use.
                image_config (ImageToImageConfig): Config for I2I, Canny, Depth.
                upscale_config (UpscaleConfig): Config for upscaling operations.
                outpaint_config (OutpaintConfig): Config for outpainting operations.
                lora_spec: LoRA configuration (used by TTI, I2I, Canny, Depth).
                filename_suffix: Optional suffix for generated filenames.
                filename_prefix: Optional prefix for generated filenames.

        Returns:
            ImageResult containing information about the generated image.
        """
        if not self.client:
            await self.initialize()  # Ensure client is initialized
            if not self.client:
                msg = "Failed to initialize FALEngine client."
                raise RuntimeError(msg)

        model: ModelTypes = kwargs.get("model", ModelTypes.TEXT)
        logger.info(f"Starting FAL generation job for model type: {model.name}")

        # --- Handle Upscaling ---
        if model.name.startswith("UPSCALER_"):
            upscale_config: UpscaleConfig | None = kwargs.get("upscale_config")
            if not upscale_config:
                msg = "upscale_config is required for upscaling models."
                raise ValueError(msg)

            # Prepare image input URL (download/resize/upload)
            (
                image_url,
                original_width,
                original_height,
                input_path,
            ) = await self._prepare_image_input(
                upscale_config.input_image, model_type=model
            )

            # Prepare upscale-specific arguments
            upscale_args = upscale_config.model_dump(
                exclude_none=True, exclude={"input_image"}
            )
            upscale_args["prompt"] = prompt  # Add main prompt if provided

            # Default output_dir to input image subfolder if no explicit output_dir provided
            output_dir = self.output_dir
            if not output_dir and input_path:
                output_dir = input_path.parent / input_path.stem
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(
                    f"Using input image basename for output directory: {output_dir}"
                )

            logger.debug(
                f"Calling client.process_upscale for {model.name} with args: {upscale_args}"
            )
            return await self.client.process_upscale(
                image_url=image_url,
                output_dir=output_dir,
                filename_suffix=kwargs.get("filename_suffix"),
                filename_prefix=kwargs.get("filename_prefix")
                or (input_path.stem if input_path else None),
                upscaler=model.name.replace("UPSCALER_", "").lower(),
                **upscale_args,
            )

        # --- Handle Outpainting ---
        elif model in [ModelTypes.OUTPAINT_BRIA, ModelTypes.OUTPAINT_FLUX]:
            outpaint_config: OutpaintConfig | None = kwargs.get("outpaint_config")
            if not outpaint_config:
                msg = "outpaint_config is required for outpainting model."
                raise ValueError(msg)

            # Prepare image input URL and get actual dimensions
            (
                image_url,
                original_width,
                original_height,
                input_path,
            ) = await self._prepare_image_input(
                outpaint_config.input_image, model_type=model
            )

            # Default output_dir to input image subfolder if no explicit output_dir provided
            output_dir = self.output_dir
            if not output_dir and input_path:
                output_dir = input_path.parent / input_path.stem
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(
                    f"Using input image basename for output directory: {output_dir}"
                )

            # Validate target dimensions against original dimensions
            if (
                original_width > outpaint_config.target_width
                or original_height > outpaint_config.target_height
            ):
                msg = (
                    f"Original image ({original_width}x{original_height}) cannot be larger "
                    f"than target size ({outpaint_config.target_width}x{outpaint_config.target_height}) for outpainting."
                )
                logger.error(msg)
                raise ValueError(msg)

            # Different handling depending on outpaint_tool
            if model == ModelTypes.OUTPAINT_BRIA:
                # Calculate placement (center the original image)
                offset_x = (outpaint_config.target_width - original_width) // 2
                offset_y = (outpaint_config.target_height - original_height) // 2
                original_image_location = [offset_x, offset_y]
                original_image_size = [original_width, original_height]
                logger.debug(
                    f"Calculated original_image_location: {original_image_location}, "
                    f"original_image_size: {original_image_size}"
                )

                # Prepare outpaint-specific arguments
                outpaint_args = outpaint_config.model_dump(
                    exclude_none=True, exclude={"input_image"}
                )
                # Overwrite/add calculated values for the API call
                outpaint_args["image_url"] = image_url
                outpaint_args["original_image_location"] = original_image_location
                outpaint_args["original_image_size"] = original_image_size
                # Ensure canvas size matches target size from config
                outpaint_args["canvas_width"] = outpaint_config.target_width
                outpaint_args["canvas_height"] = outpaint_config.target_height
                # Add canvas_size parameter (required by the API)
                outpaint_args["canvas_size"] = [
                    outpaint_config.target_width,
                    outpaint_config.target_height,
                ]
                # Ensure num_outputs matches config
                outpaint_args["num_outputs"] = outpaint_config.num_images

                logger.debug(
                    f"Calling client.process_outpaint for BRIA with calculated args: {outpaint_args}"
                )
                bria_outpaint_result = await self.client.process_outpaint(
                    prompt=prompt,
                    image_url=image_url,
                    lora_spec=None,  # No LoRA for BRIA
                    output_dir=output_dir,
                    filename_suffix=kwargs.get("filename_suffix"),
                    filename_prefix=kwargs.get("filename_prefix"),
                    outpaint_tool="bria",
                    target_width=outpaint_config.target_width,
                    target_height=outpaint_config.target_height,
                    num_images=outpaint_config.num_images,
                    original_image_location=original_image_location,
                    original_image_size=original_image_size,
                    # Pass the calculated canvas_size
                    canvas_size=outpaint_args.get("canvas_size"),
                )

                # --- BRIA GenFill Post-processing --- #
                if outpaint_config.border_thickness_factor > 0:
                    logger.info("Applying GenFill post-processing for BRIA outpaint.")
                    genfill_mask_path: str | None = None
                    # We need the result from the initial outpaint call
                    outpaint_result = (
                        bria_outpaint_result  # Store the result temporarily
                    )

                    try:
                        # Get URL and dimensions of the outpainted image
                        outpainted_image_url = outpaint_result.image_info.get("url")
                        outpainted_width = outpaint_result.image_info.get(
                            "width", outpaint_config.target_width
                        )
                        outpainted_height = outpaint_result.image_info.get(
                            "height", outpaint_config.target_height
                        )

                        if not outpainted_image_url:
                            logger.warning(
                                "No URL found in outpaint result, skipping GenFill."
                            )
                            return outpaint_result  # Return original if URL missing

                        # Calculate border thickness in pixels
                        min_dim = min(outpainted_width, outpainted_height)
                        border_thickness = round(
                            min_dim * outpaint_config.border_thickness_factor
                        )
                        logger.debug(
                            f"Calculated GenFill border thickness: {border_thickness}px"
                        )

                        if border_thickness <= 0:
                            logger.info(
                                "Calculated border thickness is zero or less, skipping GenFill."
                            )
                            return outpaint_result

                        # Create the border mask
                        from twat_genai.core.image_utils import (
                            create_genfill_border_mask,
                        )

                        genfill_mask_path = create_genfill_border_mask(
                            original_width=original_width,  # Width of the *original* paste area
                            original_height=original_height,  # Height of the *original* paste area
                            target_width=outpainted_width,  # Width of the *outpainted* image
                            target_height=outpainted_height,  # Height of the *outpainted* image
                            border_thickness=border_thickness,
                        )

                        # --- Save Debug GenFill Mask if Verbose ---
                        is_verbose = kwargs.get("verbose", False)
                        if is_verbose and self.output_dir and genfill_mask_path:
                            try:
                                # Construct debug mask filename
                                # Reuse prefix/timestamp logic from flux verbose save if applicable
                                # Or create a specific naming convention here
                                filename_prefix = (
                                    kwargs.get("filename_prefix") or "genfill"
                                )
                                timestamp_short = time.strftime("%Y%m%d%H%M%S")
                                main_suffix = (
                                    kwargs.get("filename_suffix") or ""
                                ) + "_bria_outpaint"
                                debug_mask_stem = f"{filename_prefix}{timestamp_short}{main_suffix}_genfill_debug_mask"
                                debug_mask_filename = slugify(debug_mask_stem) + ".png"
                                debug_mask_save_path = (
                                    self.output_dir / debug_mask_filename
                                )

                                # Copy the temporary mask file
                                shutil.copy2(genfill_mask_path, debug_mask_save_path)
                                logger.info(
                                    f"Saved debug GenFill mask to: {debug_mask_save_path}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to save debug GenFill mask: {e}"
                                )
                        # --- End Debug GenFill Mask Save ---

                        # Upload the mask
                        logger.debug("Uploading GenFill border mask...")
                        genfill_mask_url = await self.client.upload_image(
                            genfill_mask_path
                        )
                        logger.info(f"Uploaded GenFill mask: {genfill_mask_url}")

                        # Call the genfill process
                        genfill_suffix = (
                            kwargs.get("filename_suffix") or ""
                        ) + "_genfill"
                        genfill_result = await self.client.process_genfill(
                            prompt=prompt,  # Use original prompt
                            image_url=outpainted_image_url,
                            mask_url=genfill_mask_url,
                            output_dir=self.output_dir,
                            filename_suffix=genfill_suffix,
                            filename_prefix=kwargs.get("filename_prefix"),
                            num_images=outpaint_config.num_images,
                            # Pass other relevant params like negative_prompt from outpaint_config
                            negative_prompt=outpaint_config.negative_prompt,
                        )
                        logger.info("GenFill post-processing complete.")
                        return genfill_result

                    except Exception as e:
                        logger.error(
                            f"Error during GenFill post-processing: {e}", exc_info=True
                        )
                        # Fallback to returning the original outpaint result
                        logger.warning(
                            "GenFill failed, returning original outpaint result."
                        )
                        return outpaint_result
                    finally:
                        # Clean up the temporary genfill mask file
                        if genfill_mask_path:
                            try:
                                Path(genfill_mask_path).unlink(missing_ok=True)
                                logger.debug(
                                    f"Deleted temp GenFill mask: {genfill_mask_path}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to clean up GenFill mask file: {e}"
                                )
                else:
                    # No border thickness specified, return original BRIA result
                    logger.debug(
                        "Border thickness factor is zero or less, skipping GenFill."
                    )
                    return bria_outpaint_result

            elif model == ModelTypes.OUTPAINT_FLUX:
                # For flux-based outpainting, use the inpainting endpoint with custom assets
                from twat_genai.core.image_utils import (
                    create_flux_inpainting_assets,
                    download_image_to_temp,  # Keep for cleanup reference if needed
                )

                big_image_path: str | None = None
                big_mask_path: str | None = None

                try:
                    # 1. Create the 'big image' and 'big mask' assets
                    logger.debug("Creating flux inpainting/outpainting assets...")
                    # We need the original input image URL, which _prepare_image_input gives us.
                    # The image_url variable holds this.
                    (
                        big_image_path,
                        big_mask_path,
                        _,
                    ) = await create_flux_inpainting_assets(
                        input_image_url=image_url,  # Use the URL from _prepare_image_input
                        target_width=outpaint_config.target_width,
                        target_height=outpaint_config.target_height,
                    )
                    logger.info(
                        f"Flux assets created: Big Image={big_image_path}, Big Mask={big_mask_path}"
                    )

                    # 2. Upload the generated assets
                    logger.debug("Uploading flux assets...")
                    big_image_url = await self.client.upload_image(big_image_path)
                    big_mask_url = await self.client.upload_image(big_mask_path)
                    logger.info(
                        f"Flux assets uploaded: Big Image URL={big_image_url}, Big Mask URL={big_mask_url}"
                    )

                    # --- Save Debug Mask if Verbose ---
                    is_verbose = kwargs.get("verbose", False)
                    if is_verbose and self.output_dir and big_mask_path:
                        try:
                            # Determine a name for the debug mask based on potential output
                            # Use filename_prefix if provided, otherwise generate from prompt
                            filename_prefix = kwargs.get("filename_prefix")
                            if not filename_prefix and prompt:
                                words = prompt.split()
                                filename_prefix = (
                                    "_".join(words[:2]).lower() + "_"
                                    if len(words) >= 2
                                    else (words[0].lower() + "_")
                                    if words
                                    else "image_"
                                )
                            elif not filename_prefix:
                                filename_prefix = "image_"

                            # Basic timestamp/ID part
                            timestamp_short = time.strftime("%Y%m%d%H%M%S")

                            # Suffix for the main output image
                            main_suffix = (
                                kwargs.get("filename_suffix") or "_flux_outpaint"
                            )

                            # Construct debug mask filename
                            debug_mask_stem = f"{filename_prefix}{timestamp_short}{main_suffix}_debug_mask"

                            # Slugify and add extension (mask is PNG)
                            debug_mask_filename = slugify(debug_mask_stem) + ".png"
                            debug_mask_save_path = self.output_dir / debug_mask_filename

                            # Copy the temporary mask file
                            shutil.copy2(big_mask_path, debug_mask_save_path)
                            logger.info(
                                f"Saved debug flux mask to: {debug_mask_save_path}"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to save debug flux mask: {e}")
                    # --- End Debug Mask Save ---

                    # 3. Prepare flux-specific arguments using the new assets
                    args = {
                        "prompt": prompt,
                        # Use the uploaded BIG image URL here
                        "image_url": big_image_url,
                        # Use the uploaded BIG mask URL here
                        "mask_url": big_mask_url,
                        "lora_spec": kwargs.get("lora_spec"),
                        "width": outpaint_config.target_width,  # Target size
                        "height": outpaint_config.target_height,  # Target size
                        "guidance_scale": outpaint_config.guidance_scale
                        or config.guidance_scale,
                        "num_inference_steps": outpaint_config.num_inference_steps
                        or config.num_inference_steps,
                        "negative_prompt": outpaint_config.negative_prompt,
                        "enable_safety_checker": outpaint_config.enable_safety_checker,
                        "num_images": outpaint_config.num_images,
                        # Strength might be relevant for inpainting?
                        # Check fal docs if needed.
                        "strength": 0.85,  # Default from flux inpaint docs
                    }

                    logger.debug(
                        f"Calling client.process_outpaint (as inpaint) for FLUX with args: {args}"
                    )
                    # Call process_outpaint, but provide the big_image and big_mask URLs
                    # The client method might need adjustment if it doesn't handle this case,
                    # but let's assume it uses image_url and mask_url appropriately.
                    return await self.client.process_outpaint(
                        prompt=prompt,
                        image_url=big_image_url,  # Use the big image URL
                        lora_spec=kwargs.get("lora_spec"),
                        output_dir=self.output_dir,
                        filename_suffix=kwargs.get("filename_suffix")
                        or "_flux_outpaint",
                        filename_prefix=kwargs.get("filename_prefix"),
                        outpaint_tool="flux",  # Still identify as flux
                        target_width=outpaint_config.target_width,
                        target_height=outpaint_config.target_height,
                        mask_url=big_mask_url,  # Use the big mask URL
                        guidance_scale=outpaint_config.guidance_scale,
                        num_inference_steps=outpaint_config.num_inference_steps,
                        negative_prompt=outpaint_config.negative_prompt,
                        enable_safety_checker=outpaint_config.enable_safety_checker,
                        num_images=outpaint_config.num_images,
                        # Pass other relevant args if the client method expects them
                        strength=args.get("strength"),
                    )
                finally:
                    # Clean up the temporary asset files
                    logger.debug("Cleaning up temporary flux asset files...")
                    files_to_delete = [big_image_path, big_mask_path]
                    # The input image download path is managed within create_flux_inpainting_assets
                    for file_path in files_to_delete:
                        if file_path:
                            try:
                                Path(file_path).unlink(missing_ok=True)
                                logger.debug(f"Deleted temp file: {file_path}")
                            except Exception as e:
                                logger.warning(
                                    f"Failed to clean up temporary asset file {file_path}: {e}"
                                )

        # --- Handle Text-to-Image ---
        elif model == ModelTypes.TEXT:
            # Check for image_config (should not be provided for TTI)
            if kwargs.get("image_config"):
                logger.warning(
                    "image_config provided for TEXT model, but will be ignored."
                )

            logger.debug(f"Calling client.process_tti for {model.name}")
            return await self.client.process_tti(
                prompt=prompt,
                lora_spec=kwargs.get("lora_spec"),
                output_dir=self.output_dir,
                filename_suffix=kwargs.get("filename_suffix"),
                filename_prefix=kwargs.get("filename_prefix"),
                guidance_scale=config.guidance_scale,
                num_inference_steps=config.num_inference_steps,
                image_size=config.image_size,
                negative_prompt=kwargs.get("negative_prompt"),
            )

        # --- Handle Image-to-Image (regular, Canny, Depth) ---
        elif model in [ModelTypes.IMAGE, ModelTypes.CANNY, ModelTypes.DEPTH]:
            image_config: ImageToImageConfig | None = kwargs.get("image_config")
            if not image_config or not image_config.input_image:
                msg = (
                    "image_config with input_image is required for image-based models."
                )
                raise ValueError(msg)

            # Prepare image input URL and get dimensions
            (
                image_url,
                original_width,
                original_height,
                input_path,
            ) = await self._prepare_image_input(
                image_config.input_image, model_type=model
            )

            # Default output_dir to input image subfolder if no explicit output_dir provided
            output_dir = self.output_dir
            if not output_dir and input_path:
                output_dir = input_path.parent / input_path.stem
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(
                    f"Using input image basename for output directory: {output_dir}"
                )

            # Select appropriate client method based on model type
            if model == ModelTypes.IMAGE:
                logger.debug(f"Calling client.process_i2i for {model.name}")
                return await self.client.process_i2i(
                    prompt=prompt,
                    image_url=image_url,
                    lora_spec=kwargs.get("lora_spec"),
                    output_dir=output_dir,
                    filename_suffix=kwargs.get("filename_suffix"),
                    filename_prefix=kwargs.get("filename_prefix")
                    or (input_path.stem if input_path else None),
                    strength=image_config.strength,
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=config.num_inference_steps,
                    image_size=config.image_size,
                    negative_prompt=image_config.negative_prompt or None,
                )
            elif model == ModelTypes.CANNY:
                logger.debug(f"Calling client.process_canny for {model.name}")
                return await self.client.process_canny(
                    prompt=prompt,
                    image_url=image_url,
                    lora_spec=kwargs.get("lora_spec"),
                    output_dir=output_dir,
                    filename_suffix=kwargs.get("filename_suffix"),
                    filename_prefix=kwargs.get("filename_prefix")
                    or (input_path.stem if input_path else None),
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=config.num_inference_steps,
                    image_size=config.image_size,
                    negative_prompt=image_config.negative_prompt or None,
                )
            elif model == ModelTypes.DEPTH:
                logger.debug(f"Calling client.process_depth for {model.name}")
                return await self.client.process_depth(
                    prompt=prompt,
                    image_url=image_url,
                    lora_spec=kwargs.get("lora_spec"),
                    output_dir=output_dir,
                    filename_suffix=kwargs.get("filename_suffix"),
                    filename_prefix=kwargs.get("filename_prefix")
                    or (input_path.stem if input_path else None),
                    guidance_scale=config.guidance_scale,
                    num_inference_steps=config.num_inference_steps,
                    image_size=config.image_size,
                    negative_prompt=image_config.negative_prompt or None,
                )

        # Should never reach here due to the exhaustive conditionals
        msg = f"Unsupported model type: {model}"
        raise ValueError(msg)

    async def shutdown(self) -> None:
        """Clean up resources."""
        logger.debug("Shutting down FALEngine.")
        # No specific cleanup needed for client currently
