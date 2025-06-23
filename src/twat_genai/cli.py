#!/usr/bin/env -S uv run
# /// script
# dependencies = ["fire", "loguru", "twat"]
# ///
"""Command-line interface for twat-genai."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import get_args, Any, cast, Literal

import fire
from loguru import logger

# Try importing PathManager, handle optional dependency
try:
    from twat_os.paths import PathManager
except ImportError:
    PathManager = None
    logger.warning(
        "'twat' package not fully available. PathManager features disabled."
        " Output directory resolution will use defaults."
    )

from twat_genai.core.config import ImageInput, ImageResult, ImageSizeWH
from twat_genai.core.image import ImageSizes
from twat_genai.core.prompt import normalize_prompts
from twat_genai.engines.base import EngineConfig
from twat_genai.engines.fal import FALEngine
from twat_genai.engines.fal.config import (
    ImageToImageConfig,
    ModelTypes,
    UpscaleConfig,
    OutpaintConfig,
)

# Helper to map upscale tool names to ModelTypes
UPSCALE_TOOL_MAP = {
    tool.name.replace("UPSCALER_", "").lower(): tool
    for tool in ModelTypes
    if tool.name.startswith("UPSCALER_")
}


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
        raise ValueError(msg) from None


def get_output_dir(
    user_dir: str | Path | None = None,
    subdirectory: str | None = None,
    input_image_path: Path | None = None,
) -> Path:
    """Get the output directory for generated images.

    Priority:
    1. User-provided directory
    2. Input image parent directory + basename subfolder (if input_image_path is provided)
    3. Central path management (if twat is installed)
    4. Default 'generated_images' in current directory

    Args:
        user_dir: Base output directory path
        subdirectory: Optional subdirectory inside the output directory
        input_image_path: Input image path to use for determining default output dir

    Returns:
        Path: Resolved output directory path
    """
    if user_dir:
        base_dir = Path(user_dir).resolve()
    elif input_image_path:
        # Create output in a subfolder named after the input image's basename
        # in the same directory as the input image
        parent_dir = input_image_path.parent
        basename = input_image_path.stem
        base_dir = (parent_dir / basename).resolve()
    elif PathManager:
        try:
            paths = PathManager.for_package("twat_genai")
            if paths.genai.output_dir:
                base_dir = paths.genai.output_dir.resolve()
            else:
                # Default to 'generated_images' in current directory
                base_dir = Path("generated_images").resolve()
        except (AttributeError, Exception) as e:
            logger.warning(
                f"Error using PathManager for output directory: {e}. Using default directory."
            )
            # Default to 'generated_images' in current directory
            base_dir = Path("generated_images").resolve()
    else:
        # Default to 'generated_images' in current directory
        base_dir = Path("generated_images").resolve()

    # Handle subdirectory if provided
    if subdirectory:
        return base_dir / subdirectory
    return base_dir


class TwatGenAiCLI:
    """Command-line interface for twat-genai.

    This class provides a structured interface for various image generation tasks
    through dedicated methods. Each public method represents a subcommand.

    Shared parameters are defined in __init__ and are available across all commands.
    """

    def __init__(
        self,
        output_dir: str | Path = "generated_images",
        filename_suffix: str | None = None,
        filename_prefix: str | None = None,
        verbose: bool = False,
        image_size: str = "SQ",
        guidance_scale: float = 3.5,
        num_inference_steps: int = 28,
        negative_prompt: str = "",
        lora: str | None = None,
    ):
        """Initialize shared parameters for all commands.

        Args:
            output_dir: Directory to save generated images
            filename_suffix: Optional suffix for generated filenames
            filename_prefix: Optional prefix for generated filenames
            verbose: Enable verbose logging
            image_size: Output image size preset or custom dimensions
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            negative_prompt: Negative prompt for generation
            lora: LoRA specification string
        """
        # Configure logging
        logger.remove()
        logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")
        logger.debug(f"CLI initialized with shared params: {locals()}")

        # Store shared parameters
        self.verbose = verbose
        self.output_dir = output_dir
        self.filename_suffix = filename_suffix
        self.filename_prefix = filename_prefix
        self.image_size = image_size
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.negative_prompt = negative_prompt
        self.lora = lora

        # Prepare base config
        self.base_config = EngineConfig(
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            image_size=parse_image_size(image_size),
        )

    def _prepare_image_input(self, input_image: str | None) -> ImageInput | None:
        """Helper to parse and validate image input."""
        if not input_image:
            return None

        try:
            is_url = isinstance(input_image, str) and input_image.startswith(
                ("http:", "https:")
            )
            input_img_obj = ImageInput(
                url=input_image if is_url else None,
                path=Path(input_image) if not is_url else None,
            )
            if input_img_obj.path and not input_img_obj.path.exists():
                msg = f"Input image path does not exist: {input_img_obj.path}"
                raise FileNotFoundError(msg)
            logger.debug(f"Parsed input image: {input_img_obj}")

            # If no filename_prefix is specified and we have a local file path,
            # use the input image's basename as the prefix
            if self.filename_prefix is None and input_img_obj.path:
                self.filename_prefix = input_img_obj.path.stem
                logger.debug(
                    f"Using input image basename as filename prefix: {self.filename_prefix}"
                )

            return input_img_obj
        except Exception as e:
            logger.error(
                f"Failed to parse input_image argument '{input_image}': {e}",
                exc_info=True,
            )
            msg = f"Invalid input_image source: '{input_image}'. {e}"
            raise ValueError(msg) from e

    async def _run_generation(
        self,
        prompts: str | list[str],
        model: ModelTypes,
        image_config: ImageToImageConfig | None = None,
        upscale_config: UpscaleConfig | None = None,
        outpaint_config: OutpaintConfig | None = None,
        output_subdirectory: str | None = None,
        input_image_path: Path | None = None,
    ) -> list[ImageResult]:
        """Core generation logic shared across commands."""
        # Use input_image_path to determine output directory if not explicitly provided by user
        output_dir_path = get_output_dir(
            self.output_dir, output_subdirectory, input_image_path
        )
        output_dir_path.mkdir(parents=True, exist_ok=True)

        final_prompts = normalize_prompts(prompts)
        logger.debug(f"Expanded prompts: {final_prompts}")

        # Ensure at least one prompt exists for upscaling and other modes
        # that can work with empty prompts
        if not final_prompts:
            if model.name.startswith("UPSCALER_"):
                logger.debug(f"Using empty string as default prompt for {model.name}")
                final_prompts = [""]  # Use a single empty string prompt

        # If we still have no prompts, warn but don't fail (some models may work without prompts)
        if not final_prompts:
            logger.warning(f"No prompts available for model {model.name}")
            final_prompts = [""]  # Fallback to empty string

        async with FALEngine(output_dir_path) as engine:
            results = []
            for prompt in final_prompts:
                kwargs = {
                    "model": model,
                    "image_config": image_config,
                    "upscale_config": upscale_config,
                    "outpaint_config": outpaint_config,
                    "lora_spec": self.lora,
                    "filename_suffix": self.filename_suffix,
                    "filename_prefix": self.filename_prefix,
                    "verbose": self.verbose,
                }
                result = await engine.generate(prompt, self.base_config, **kwargs)
                results.append(result)

        return results

    async def text(self, prompts: str | list[str] = "", output: str | None = None) -> None:
        """Generate images from text prompts.

        Args:
            prompts: One or more text prompts for generation
            output: Optional subdirectory for output files
        """
        if not prompts:
            msg = "Prompts are required for text generation."
            raise ValueError(msg)

        results = await self._run_generation(
            prompts, ModelTypes.TEXT, output_subdirectory=output
        )
        self._print_results(results)

    async def image(
        self,
        input_image: str,
        prompts: str | list[str] = "",
        strength: float = 0.75,
        output: str | None = None,
    ) -> None:
        """Generate images using an existing image as a reference.

        Args:
            input_image: Path or URL to the source image
            prompts: One or more text prompts for generation
            strength: How much influence the prompt should have over the image (0-1)
            output: Optional subdirectory for output files
        """
        import asyncio

        input_img = self._prepare_image_input(input_image)
        if not input_img:
            msg = "Invalid input_image source."
            raise ValueError(msg)

        # Create I2I config
        i2i_config = ImageToImageConfig(
            input_image=input_img,
            strength=strength,
            negative_prompt=self.negative_prompt,
            model_type=ModelTypes.IMAGE,
        )

        results = await self._run_generation(
            prompts,
            ModelTypes.IMAGE,
                image_config=i2i_config,
                output_subdirectory=output,
                input_image_path=input_img.path,
            )
        self._print_results(results)

    async def canny(
        self,
        input_image: str,
        prompts: str | list[str] = "",
        output: str | None = None,
    ) -> None:
        """Generate images using the Canny edge detection version of the input image.

        Args:
            input_image: Path or URL to the source image
            prompts: One or more text prompts for generation
            output: Optional subdirectory for output files
        """
        import asyncio

        input_img = self._prepare_image_input(input_image)
        if not input_img:
            msg = "Invalid input_image source."
            raise ValueError(msg)

        # Create I2I config with strength=1.0 for controlnet
        i2i_config = ImageToImageConfig(
            input_image=input_img,
            strength=1.0,  # Typical for controlnet
            negative_prompt=self.negative_prompt,
            model_type=ModelTypes.CANNY,
        )

        results = await self._run_generation(
            prompts,
            ModelTypes.CANNY,
                image_config=i2i_config,
                output_subdirectory=output,
                input_image_path=input_img.path,
            )
        self._print_results(results)

    async def depth(
        self,
        input_image: str,
        prompts: str | list[str] = "",
        output: str | None = None,
    ) -> None:
        """Generate images using the depth map of the input image.

        Args:
            input_image: Path or URL to the source image
            prompts: One or more text prompts for generation
            output: Optional subdirectory for output files
        """
        import asyncio

        input_img = self._prepare_image_input(input_image)
        if not input_img:
            msg = "Invalid input_image source."
            raise ValueError(msg)

        # Create I2I config with strength=1.0 for controlnet
        i2i_config = ImageToImageConfig(
            input_image=input_img,
            strength=1.0,  # Typical for controlnet
            negative_prompt=self.negative_prompt,
            model_type=ModelTypes.DEPTH,
        )

        results = await self._run_generation(
            prompts,
            ModelTypes.DEPTH,
                image_config=i2i_config,
                output_subdirectory=output,
                input_image_path=input_img.path,
            )
        self._print_results(results)

    async def upscale(
        self,
        input_image: str,
        tool: Literal[
            "aura_sr",
            "ccsr",
            "clarity",
            "drct",
            "esrgan",
            "ideogram",
            "recraft_clarity",
            "recraft_creative",
        ],
        prompts: str | list[str] = "",
        scale: float | None = None,
        resemblance: float | None = None,
        ideogram_detail: int | None = None,
        esrgan_model: str | None = None,
        esrgan_tile: int | None = None,
        clarity_creativity: float | None = None,
        clarity_guidance_scale: float | None = None,
        clarity_num_inference_steps: int | None = None,
        ccsr_scale: int | None = None,
        ccsr_tile_diffusion: str | None = None,
        ccsr_color_fix_type: str | None = None,
        ccsr_steps: int | None = None,
        output: str | None = None,
    ) -> None:
        """Upscale an image using one of several available models.

        Args:
            input_image: Path or URL to the source image
            tool: Which upscaling model/algorithm to use
            prompts: Optional prompt for models that support it (clarity, etc.)
            scale: Scale factor for upscaling (model dependent)
            resemblance: How closely to match original (0.0-1.0, model dependent)
            ideogram_detail: Detail level for ideogram (1-5)
            esrgan_model: ESRGAN model name
            esrgan_tile: Tile size for ESRGAN
            clarity_creativity: Creativity for clarity upscaling (0.0-1.0)
            clarity_guidance_scale: Guidance scale for clarity
            clarity_num_inference_steps: Number of steps for clarity
            ccsr_scale: Scale factor for CCSR (2, 3, or 4)
            ccsr_tile_diffusion: Tile diffusion mode for CCSR
            ccsr_color_fix_type: Color fix type for CCSR
            ccsr_steps: Number of steps for CCSR
            output: Optional subdirectory for output files
        """
        import asyncio

        # Get upscaler from tool argument
        upscaler = UPSCALE_TOOL_MAP.get(tool.lower())
        if not upscaler:
            valid_tools = ", ".join(UPSCALE_TOOL_MAP.keys())
            msg = f"Invalid upscale tool: {tool}. Valid options: {valid_tools}"
            raise ValueError(msg)

        # Parse and validate input image
        input_img = self._prepare_image_input(input_image)
        if not input_img:
            msg = "Invalid input_image source."
            raise ValueError(msg)

        # Create UpscaleConfig based on the selected tool
        upscale_kwargs = {}

        # DRCT params
        if tool == "drct" and scale is not None:
            upscale_kwargs["scale"] = scale

        # Ideogram params
        elif tool == "ideogram":
            if ideogram_detail is not None:
                upscale_kwargs["ideogram_detail"] = ideogram_detail # Corrected key
            if resemblance is not None: # Added resemblance for ideogram
                upscale_kwargs["ideogram_resemblance"] = resemblance

        # Recraft params (common between clarity/creative)
        elif tool in ("recraft_clarity", "recraft_creative"):
            if scale is not None: # General scale might apply here
                upscale_kwargs["scale"] = scale
            # Add sync_mode if it's a relevant CLI param for recraft
            # Example: upscale_kwargs["recraft_sync_mode"] = some_cli_param

        # ESRGAN params
        elif tool == "esrgan":
            if esrgan_model is not None:
                upscale_kwargs["esrgan_model"] = esrgan_model # Corrected key
            if esrgan_tile is not None:
                upscale_kwargs["esrgan_tile"] = esrgan_tile # Corrected key
            # Add esrgan_face if it's a CLI param:
            # Example: upscale_kwargs["esrgan_face"] = some_cli_param_for_face

        # Aura SR params
        elif tool == "aura_sr":
            if scale is not None:
                upscale_kwargs["scale"] = scale
            if resemblance is not None:
                upscale_kwargs["clarity_resemblance"] = resemblance # Aura uses clarity resemblance
            if clarity_creativity is not None: # Aura uses clarity creativity
                upscale_kwargs["clarity_creativity"] = clarity_creativity
            # Add other Aura SR specific params from CLI if available

        # Clarity params
        elif tool == "clarity":
            if scale is not None: # Clarity can also take a general scale
                upscale_kwargs["scale"] = scale
            if clarity_creativity is not None:
                upscale_kwargs["clarity_creativity"] = clarity_creativity
            if resemblance is not None: # Added resemblance for clarity
                upscale_kwargs["clarity_resemblance"] = resemblance
            if clarity_guidance_scale is not None:
                upscale_kwargs["clarity_guidance_scale"] = clarity_guidance_scale
            if clarity_num_inference_steps is not None:
                upscale_kwargs["clarity_num_inference_steps"] = clarity_num_inference_steps

        # CCSR params
        elif tool == "ccsr":
            if ccsr_scale is not None:
                upscale_kwargs["ccsr_scale"] = ccsr_scale # Corrected key
            else: # Fallback to general scale for CCSR if ccsr_scale not given
                upscale_kwargs["scale"] = scale
            if ccsr_tile_diffusion is not None:
                upscale_kwargs["ccsr_tile_diffusion"] = ccsr_tile_diffusion # Corrected key
            if ccsr_color_fix_type is not None:
                upscale_kwargs["ccsr_color_fix_type"] = ccsr_color_fix_type # Corrected key
            if ccsr_steps is not None:
                upscale_kwargs["ccsr_steps"] = ccsr_steps # Corrected key

        upscale_config = UpscaleConfig(input_image=input_img, prompt=prompts if isinstance(prompts, str) else "; ".join(prompts), negative_prompt=self.negative_prompt or None, **upscale_kwargs)

        results = await self._run_generation(
            prompts,
            upscaler,
                upscale_config=upscale_config,
                output_subdirectory=output,
                input_image_path=input_img.path,
            )
        self._print_results(results)

    async def outpaint(
        self,
        input_image: str,
        prompts: str | list[str],
        target_width: int,
        target_height: int,
        tool: Literal["bria", "flux"] = "bria",
        guidance_scale: float | None = None,
        num_inference_steps: int | None = None,
        enable_safety_checker: bool | None = None,
        num_images: int = 1,
        border: float = 5,
        output: str | None = None,
    ) -> None:
        """Expand an image's canvas using outpainting.

        Args:
            input_image: Path or URL to the source image
            prompts: Text prompts for generating the expanded areas
            target_width: Desired width of the expanded image
            target_height: Desired height of the expanded image
            tool: Outpainting tool to use ("bria" or "flux")
            guidance_scale: Override default guidance scale
            num_inference_steps: Override default number of steps
            enable_safety_checker: Enable safety checker
            num_images: Number of result images to generate
            border: Border width percentage for Bria
            output: Optional subdirectory for output files
        """
        import asyncio

        # Parse and validate input image
        input_img = self._prepare_image_input(input_image)
        if not input_img:
            msg = "Invalid input_image source."
            raise ValueError(msg)

        # Validate target dimensions
        if target_width <= 0 or target_height <= 0:
            msg = "Target dimensions must be positive integers."
            raise ValueError(msg)

        # Determine model type from tool parameter
        outpaint_tool_literal = cast(Literal["bria", "flux"], tool.lower())
        model_type = (
            ModelTypes.OUTPAINT_BRIA if tool == "bria" else ModelTypes.OUTPAINT_FLUX
        )

        # Prepare kwargs for OutpaintConfig
        outpaint_kwargs = {
            "input_image": input_img,
            "prompt": prompts if isinstance(prompts, str) else "; ".join(prompts),
            "target_width": target_width,
            "target_height": target_height,
            "outpaint_tool": outpaint_tool_literal,
            "num_images": num_images,
            "negative_prompt": self.negative_prompt or None,
        }

        # Add tool-specific parameters
        if tool == "bria":
            outpaint_kwargs["border_thickness_factor"] = border / 100.0
        elif tool == "flux":
            if enable_safety_checker is not None:
                outpaint_kwargs["enable_safety_checker"] = enable_safety_checker
        else:
            valid_tools = '"bria" or "flux"'
            msg = f"Invalid outpaint tool: {tool}. Valid options: {valid_tools}"
            raise ValueError(msg)

        # Override guidance_scale and num_inference_steps if provided
        if guidance_scale is not None:
            outpaint_kwargs["guidance_scale"] = guidance_scale
        if num_inference_steps is not None:
            outpaint_kwargs["num_inference_steps"] = num_inference_steps

        # Create OutpaintConfig
        outpaint_config = OutpaintConfig(**outpaint_kwargs)

        results = await self._run_generation(
            prompts,
            model_type,
                outpaint_config=outpaint_config,
                output_subdirectory=output,
                input_image_path=input_img.path,
            )
        self._print_results(results)

    def _print_results(self, results: list[ImageResult]) -> None:
        """Print generation results in a consistent format."""
        if results:
            logger.info("Results:")
            for result in results:
                path = result.image_info.get("path")
                meta_path = result.image_info.get("metadata_path")
                req_id = result.request_id
                log_msg = f"  - Request ID: {req_id}\n    Image: {path or '(Not saved)'}\n    Metadata: {meta_path or '(Not saved)'}"
                logger.info(log_msg)
        else:
            logger.warning("No results were generated.")


if __name__ == "__main__":
    fire.Fire(TwatGenAiCLI)
