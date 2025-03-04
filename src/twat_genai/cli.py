#!/usr/bin/env -S uv run
# /// script
# dependencies = ["fire", "loguru"]
# ///
"""Command-line interface for twat-genai."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import fire
from loguru import logger
from twat.paths import PathManager
from twat_genai.core.config import ImageInput, ImageResult, ImageSizeWH
from twat_genai.core.image import ImageSizes
from twat_genai.core.prompt import normalize_prompts
from twat_genai.engines.base import EngineConfig
from twat_genai.engines.fal import FALEngine
from twat_genai.engines.fal.config import ImageToImageConfig, ModelTypes


def parse_image_size(size_str: str) -> ImageSizes | ImageSizeWH:
    """Parse image size string into appropriate type."""
    try:
        return ImageSizes[size_str.upper()]
    except KeyError:
        if "," in size_str:
            try:
                w, h = (int(x.strip()) for x in size_str.split(",", 1))
                return ImageSizeWH(width=w, height=h)
            except (ValueError, TypeError) as err:
                msg = "For custom image sizes use 'width,height' with integers."
                raise ValueError(msg) from err
        valid_names = ", ".join(s.name for s in ImageSizes)
        msg = f"image_size must be one of: {valid_names} or in 'width,height' format."
        raise ValueError(msg)


def get_output_dir(user_dir: str | Path | None = None) -> Path:
    """Get the output directory for generated images.

    Priority:
    1. User-provided directory
    2. Central path management
    3. Default 'generated_images' in current directory
    """
    if user_dir:
        return Path(user_dir)

    paths = PathManager.for_package("twat_genai")
    if paths.genai.output_dir:
        return paths.genai.output_dir

    return Path("generated_images")


async def async_main(
    prompts: str | list[str],
    output_dir: str | Path = "generated_images",
    filename_suffix: str | None = None,
    filename_prefix: str | None = None,
    model: ModelTypes = ModelTypes.TEXT,
    image_config: ImageToImageConfig | None = None,
    lora: str | list | None = None,
    image_size: str = "SQ",
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
) -> list[ImageResult]:
    """
    Generate images using FAL.

    Args:
        prompts: Text prompts for generation
        output_dir: Directory to save generated images
        filename_suffix: Optional suffix for generated filenames
        filename_prefix: Optional prefix for generated filenames
        model: Model type to use
        image_config: Configuration for image-to-image operations
        lora: LoRA configuration
        image_size: Size of the output image
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of inference steps

    Returns:
        List of image generation results
    """
    output_dir_path = get_output_dir(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    config = EngineConfig(
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        image_size=parse_image_size(image_size),
    )

    final_prompts = normalize_prompts(prompts)
    logger.debug(f"Expanded prompts: {final_prompts}")

    async with FALEngine(output_dir_path) as engine:
        results = []
        for prompt in final_prompts:
            result = await engine.generate(
                prompt,
                config,
                model=model,
                image_config=image_config,
                lora_spec=lora,
                filename_suffix=filename_suffix,
                filename_prefix=filename_prefix,
            )
            results.append(result)

    return results


def cli(
    prompts: str | list[str],
    output_dir: str | Path = "generated_images",
    filename_suffix: str | None = None,
    filename_prefix: str | None = None,
    model: str | ModelTypes = "text",
    input_image: str | Path | None = None,
    image_url: str | None = None,
    strength: float = 0.75,
    negative_prompt: str = "",
    lora: str | list | None = None,
    image_size: str = "SQ",
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    verbose: bool = False,
) -> list[ImageResult]:
    """
    CLI entry point for image generation.

    Args:
        prompts: Text prompts for generation
        output_dir: Directory to save generated images
        filename_suffix: Optional suffix for generated filenames
        filename_prefix: Optional prefix for generated filenames
        model: Model type to use (text, image, canny, depth)
        input_image: Path to input image for image-to-image operations
        image_url: URL to input image for image-to-image operations
        strength: How much to preserve from original image (0-1)
        negative_prompt: What to avoid in the generation
        lora: LoRA configuration
        image_size: Size of the output image
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of inference steps
        verbose: Enable verbose logging output

    Returns:
        List of image generation results
    """
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")

    if isinstance(model, str):
        try:
            model = ModelTypes[model.upper()]
        except KeyError:
            valid_models = ", ".join(m.name.lower() for m in ModelTypes)
            msg = f"Invalid model type. Must be one of: {valid_models}"
            raise ValueError(msg)

    image_config = None
    if model != ModelTypes.TEXT:
        if not (input_image or image_url):
            msg = "input_image or image_url is required for image-to-image operations"
            raise ValueError(msg)
        image_config = ImageToImageConfig(
            model_type=model,
            input_image=ImageInput(
                url=image_url,
                path=Path(input_image) if input_image else None,
            ),
            strength=strength,
            negative_prompt=negative_prompt,
        )

    return asyncio.run(
        async_main(
            prompts=prompts,
            output_dir=output_dir,
            filename_suffix=filename_suffix,
            filename_prefix=filename_prefix,
            model=model,
            image_config=image_config,
            lora=lora,
            image_size=image_size,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
    )


if __name__ == "__main__":
    fire.Fire(cli)
