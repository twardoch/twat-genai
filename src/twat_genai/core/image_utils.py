#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["pillow", "requests", "httpx", "loguru"]
# ///
# this_file: src/twat_genai/core/image_utils.py

"""
Image processing utilities including resizing and downloading.
Primarily adapted from the superres tool.
"""

import tempfile
from pathlib import Path

import httpx  # Changed from aiohttp
from loguru import logger  # Changed from logging
from PIL import Image, ImageDraw

# Import definitions from within the package
# UPSCALE_TOOL_MAX_INPUT_SIZES is now in engines.fal.config
from twat_genai.engines.fal.config import ModelTypes, UPSCALE_TOOL_MAX_INPUT_SIZES


def resize_image_if_needed(image_path: str, model_type: ModelTypes) -> str | None:
    """
    Resize the image if it exceeds the maximum dimensions for the specific upscale model.

    Args:
        image_path: Path to the image file
        model_type: The upscale ModelType to check dimensions against

    Returns:
        Optional[str]: Path to the resized image (temp file) if resizing occurred, None otherwise.
                       Returns None if the model_type is not an upscaler or has no size limit.
    """
    # Only resize for known upscale models with size limits
    if model_type not in UPSCALE_TOOL_MAX_INPUT_SIZES:
        logger.debug(
            f"No max size defined or not an upscaler: {model_type.name}, skipping resize check."
        )
        return None

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            max_size = UPSCALE_TOOL_MAX_INPUT_SIZES[model_type]

            # Check if resizing is needed
            if width <= max_size and height <= max_size:
                logger.debug(
                    f"Image size {width}x{height} is within limits for {model_type.name}, no resize needed."
                )
                return None  # No resize needed

            # Calculate new dimensions while maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            # Create resized image
            logger.debug(
                f"Resizing image from {width}x{height} to {new_width}x{new_height} for {model_type.name}"
            )
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save to a new temporary file
            # Keep original format if possible, default to JPEG
            img_format = img.format or "JPEG"
            suffix = f".{img_format.lower()}"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                resized.save(tmp.name, format=img_format, quality=95)
                logger.info(
                    f"Image resized ({new_width}x{new_height}) and saved to temporary file: {tmp.name}"
                )
                return tmp.name

    except FileNotFoundError:
        logger.error(f"Image file not found for resizing: {image_path}")
        raise
    except Exception as e:
        logger.error(
            f"Error resizing image {image_path} for {model_type.name}: {e}",
            exc_info=True,
        )
        msg = f"Error resizing image: {e!s}"
        raise RuntimeError(msg) from e


async def download_image_to_temp(url: str) -> str:
    """
    Download an image from a URL to a temporary file.

    Args:
        url: URL of the image to download

    Returns:
        str: Path to the downloaded image file
    """
    try:
        async with httpx.AsyncClient() as client:
            logger.debug(f"Downloading image from URL: {url}")
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

            # Try to determine suffix from content-type or URL
            content_type = response.headers.get("content-type", "").lower()
            if "jpeg" in content_type or "jpg" in content_type:
                suffix = ".jpg"
            elif "png" in content_type:
                suffix = ".png"
            else:
                # Fallback based on URL extension or default to jpg
                suffix = Path(url).suffix.lower() or ".jpg"
                if suffix not in [".jpg", ".jpeg", ".png"]:
                    suffix = ".jpg"

            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(response.content)
                logger.info(
                    f"Image downloaded from {url} to temporary file: {tmp.name}"
                )
                return tmp.name
    except httpx.RequestError as e:
        logger.error(f"HTTP error downloading image from {url}: {e}", exc_info=True)
        msg = f"Failed to download image from {url}: {e}"
        raise RuntimeError(msg) from e
    except Exception as e:
        logger.error(f"Error downloading image {url} to temp file: {e}", exc_info=True)
        msg = f"Failed to download image from {url}: {e!s}"
        raise RuntimeError(msg) from e


async def download_image(url: str, output_path: str | Path) -> None:
    """
    Download an image from a URL and save it to the specified path.

    Args:
        url: URL of the image to download
        output_path: Path where the image should be saved
    """
    try:
        async with httpx.AsyncClient() as client:
            logger.debug(f"Downloading image from {url} to {output_path}")
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            # Ensure output_path is a Path object
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure dir exists
            output_path_obj.write_bytes(response.content)
            logger.info(f"Image downloaded from {url} and saved to: {output_path_obj}")
    except httpx.RequestError as e:
        logger.error(
            f"HTTP error downloading image from {url} to {output_path}: {e}",
            exc_info=True,
        )
        msg = f"Failed to download image from {url} to {output_path}: {e}"
        raise RuntimeError(msg) from e
    except Exception as e:
        logger.error(f"Error downloading {url} to {output_path}: {e}", exc_info=True)
        msg = f"Failed to download image from {url} to {output_path}: {e!s}"
        raise RuntimeError(msg) from e


def create_outpaint_mask(
    image_width: int, image_height: int, target_width: int, target_height: int
) -> tuple[str, list[int]]:
    """
    Creates a mask image for flux-based outpainting.
    The mask is a black image of the target size with a white rectangle in the center
    representing the original image area.

    Args:
        image_width: Width of the original image
        image_height: Height of the original image
        target_width: Width of the target (outpainted) image
        target_height: Height of the target (outpainted) image

    Returns:
        tuple[str, list[int]]: Path to the mask image and position [x, y] of the original image
    """
    logger.debug(
        f"Creating outpaint mask for {image_width}x{image_height} -> {target_width}x{target_height}"
    )

    # Calculate the position of the original image in the target image (centered)
    offset_x = (target_width - image_width) // 2
    offset_y = (target_height - image_height) // 2
    position = [offset_x, offset_y]

    # Create a black canvas of the target size
    mask_img = Image.new("L", (target_width, target_height), color=0)

    # Draw a white rectangle in the center where the original image will be
    draw = ImageDraw.Draw(mask_img)
    draw.rectangle(
        ((offset_x, offset_y), (offset_x + image_width, offset_y + image_height)),
        fill=255,
    )

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        mask_img.save(tmp.name, format="PNG")
        logger.info(f"Created outpaint mask and saved to temporary file: {tmp.name}")
        return tmp.name, position


async def create_flux_inpainting_assets(
    input_image_url: str, target_width: int, target_height: int
) -> tuple[str, str, list[int]]:
    """
    Creates the 'big image' and 'big mask' needed for flux inpainting/outpainting.

    1. Downloads the input image.
    2. Creates a 'small mask' from the input's alpha channel (or black if no alpha).
       - Transparent pixels in input -> White in small mask.
       - Opaque pixels in input -> Black in small mask.
    3. Creates a 'big image' (target size, transparent) with the input pasted in the center.
    4. Creates a 'big mask' (target size, white) with the 'small mask' pasted in the center.

    Args:
        input_image_url: URL of the original input image.
        target_width: Width of the target (output) image.
        target_height: Height of the target (output) image.

    Returns:
        tuple[str, str, list[int]]: Paths to the temporary 'big image' file,
                                   'big mask' file, and the calculated [x, y] position
                                   of the original image within the target canvas.
    """
    temp_files_to_clean = []
    try:
        # 1. Download input image
        logger.debug(f"Downloading input image for flux assets: {input_image_url}")
        input_image_path = await download_image_to_temp(input_image_url)
        temp_files_to_clean.append(input_image_path)
        logger.debug(f"Input image downloaded to: {input_image_path}")

        with Image.open(input_image_path) as input_img:
            input_width, input_height = input_img.size
            logger.debug(f"Input image size: {input_width}x{input_height}")

            # Calculate center position
            offset_x = (target_width - input_width) // 2
            offset_y = (target_height - input_height) // 2
            position = [offset_x, offset_y]
            logger.debug(f"Calculated center position: {position}")

            # 2. Create 'small mask'
            small_mask_img = None
            if input_img.mode in ("RGBA", "LA") or (input_img.info.get("transparency")):
                try:
                    logger.debug(
                        "Input image has alpha channel. Extracting and inverting."
                    )
                    alpha = input_img.getchannel("A")
                    # Invert: Transparent (0) -> White (255), Opaque (255) -> Black (0)
                    small_mask_img = alpha.point(lambda p: 255 - p)
                    small_mask_img = small_mask_img.convert("L")  # Ensure grayscale
                except ValueError:
                    logger.warning(
                        "Could not get alpha channel, creating black small mask."
                    )
                    small_mask_img = Image.new(
                        "L", (input_width, input_height), color=0
                    )
            else:
                logger.debug(
                    "Input image has no alpha channel. Creating black small mask."
                )
                small_mask_img = Image.new("L", (input_width, input_height), color=0)

            # --- Crop the small mask --- #
            if (
                input_width > 4 and input_height > 4
            ):  # Ensure image is large enough to crop
                crop_box = (2, 2, input_width - 2, input_height - 2)
                logger.debug(f"Cropping small mask with box: {crop_box}")
                cropped_small_mask_img = small_mask_img.crop(crop_box)
                # The position for pasting needs adjustment due to the crop
                paste_position_big_mask = (position[0] + 2, position[1] + 2)
            else:
                logger.warning("Small mask too small to crop by 2px, using original.")
                cropped_small_mask_img = small_mask_img
                paste_position_big_mask = tuple(position)  # Use original position
            # --- End Crop ---

            # Save small mask temporarily (use the cropped version)
            with tempfile.NamedTemporaryFile(
                suffix="_small_mask_cropped.png", delete=False
            ) as tmp_small_mask:
                cropped_small_mask_img.save(tmp_small_mask.name, "PNG")
                small_mask_path = tmp_small_mask.name
                temp_files_to_clean.append(small_mask_path)
                logger.debug(f"Small mask saved to: {small_mask_path}")

            # 3. Create 'big image' (transparent canvas with input centered)
            logger.debug("Creating big image (transparent canvas).")
            big_image_img = Image.new(
                "RGBA", (target_width, target_height), (0, 0, 0, 0)
            )
            # Paste input image using its own alpha if available
            paste_mask = (
                input_img.split()[-1] if input_img.mode in ("RGBA", "LA") else None
            )
            big_image_img.paste(
                input_img.convert("RGBA"), tuple(position), mask=paste_mask
            )

            # Save big image
            with tempfile.NamedTemporaryFile(
                suffix="_big_image.png", delete=False
            ) as tmp_big_image:
                big_image_img.save(tmp_big_image.name, "PNG")
                big_image_path = tmp_big_image.name
                temp_files_to_clean.append(big_image_path)
                logger.info(f"Big image saved to temporary file: {big_image_path}")

            # 4. Create 'big mask' (white canvas with small mask centered)
            logger.debug("Creating big mask (white canvas).")
            big_mask_img = Image.new("L", (target_width, target_height), color=255)
            # Paste the *cropped* small mask at the *adjusted* position
            big_mask_img.paste(cropped_small_mask_img, paste_position_big_mask)

            # Save big mask
            with tempfile.NamedTemporaryFile(
                suffix="_big_mask.png", delete=False
            ) as tmp_big_mask:
                big_mask_img.save(tmp_big_mask.name, "PNG")
                big_mask_path = tmp_big_mask.name
                temp_files_to_clean.append(big_mask_path)
                logger.info(f"Big mask saved to temporary file: {big_mask_path}")

        # Return paths to the final assets and the position
        return big_image_path, big_mask_path, position

    except Exception as e:
        logger.error(f"Error creating flux inpainting assets: {e}", exc_info=True)
        # Clean up any files created before the error
        for f_path in temp_files_to_clean:
            try:
                Path(f_path).unlink(missing_ok=True)
            except Exception as cleanup_e:
                logger.warning(f"Failed to clean up temp file {f_path}: {cleanup_e}")
        msg = f"Error creating flux inpainting assets: {e!s}"
        raise RuntimeError(msg) from e


# Note: The original create_outpaint_mask is kept for now,
# but the FAL engine logic will be updated to call create_flux_inpainting_assets instead.

# Example usage (for testing, needs an async context):
# async def main():
#     img_url = "URL_TO_YOUR_TRANSPARENT_PNG_OR_OTHER_IMAGE"
#     tgt_w, tgt_h = 1920, 1080
#     try:
#         big_img_p, big_mask_p, pos = await create_flux_inpainting_assets(img_url, tgt_w, tgt_h)
#         print(f"Big Image Path: {big_img_p}")
#         print(f"Big Mask Path: {big_mask_p}")
#         print(f"Position: {pos}")
#         # Remember to clean up the files: Path(big_img_p).unlink(); Path(big_mask_p).unlink()
#     except Exception as e:
#         print(f"Error: {e}")
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())


def create_genfill_border_mask(
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
    border_thickness: int,
) -> str:
    """
    Creates a mask for the Bria GenFill post-processing step.
    The mask is black except for a white border around the original image area.
    The border extends partially into the original area and partially into the outpainted area.

    Args:
        original_width: Width of the original image area within the target canvas.
        original_height: Height of the original image area within the target canvas.
        target_width: Width of the target canvas (outpainted image size).
        target_height: Height of the target canvas (outpainted image size).
        border_thickness: The thickness of the border mask in pixels.

    Returns:
        str: Path to the temporary mask image file (PNG).
    """
    if border_thickness <= 0:
        msg = "Border thickness must be positive to create a genfill mask."
        raise ValueError(msg)

    logger.debug(
        f"Creating GenFill border mask: original={original_width}x{original_height}, "
        f"target={target_width}x{target_height}, border={border_thickness}px"
    )

    # Create a black canvas of the target size
    mask_img = Image.new("L", (target_width, target_height), color=0)
    draw = ImageDraw.Draw(mask_img)

    # Calculate the coordinates of the original image area (centered)
    offset_x = (target_width - original_width) // 2
    offset_y = (target_height - original_height) // 2
    orig_left = offset_x
    orig_top = offset_y
    orig_right = offset_x + original_width
    orig_bottom = offset_y + original_height

    # Calculate border dimensions (10% inside, 90% outside)
    inside_border = round(border_thickness * 0.10)
    outside_border = (
        border_thickness - inside_border
    )  # Ensure total thickness is correct

    # Calculate the coordinates for the white border rectangle
    # Outer bounds
    border_left = max(0, orig_left - outside_border)
    border_top = max(0, orig_top - outside_border)
    border_right = min(target_width, orig_right + outside_border)
    border_bottom = min(target_height, orig_bottom + outside_border)

    # Inner bounds (where the black hole will be punched)
    hole_left = orig_left + inside_border
    hole_top = orig_top + inside_border
    hole_right = orig_right - inside_border
    hole_bottom = orig_bottom - inside_border

    # Draw the outer white rectangle
    logger.debug(
        f"Drawing outer white border: ({border_left}, {border_top}) to ({border_right}, {border_bottom})"
    )
    draw.rectangle(((border_left, border_top), (border_right, border_bottom)), fill=255)

    # Draw the inner black rectangle (punching the hole)
    # Ensure the hole coordinates are valid (right > left, bottom > top)
    if hole_right > hole_left and hole_bottom > hole_top:
        logger.debug(
            f"Drawing inner black hole: ({hole_left}, {hole_top}) to ({hole_right}, {hole_bottom})"
        )
        draw.rectangle(((hole_left, hole_top), (hole_right, hole_bottom)), fill=0)
    else:
        logger.warning(
            "Border thickness too large relative to original image; inner hole not drawn."
        )

    # Save to a temporary file
    try:
        with tempfile.NamedTemporaryFile(
            suffix="_genfill_mask.png", delete=False
        ) as tmp:
            mask_img.save(tmp.name, format="PNG")
            logger.info(
                f"Created GenFill border mask and saved to temporary file: {tmp.name}"
            )
            return tmp.name
    except Exception as e:
        logger.error(f"Failed to save GenFill mask: {e}", exc_info=True)
        msg = f"Failed to save GenFill mask: {e!s}"
        raise RuntimeError(msg) from e
