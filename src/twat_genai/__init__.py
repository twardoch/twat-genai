"""twat-genai: Generative AI image toolkit powered by fal.ai.

`fal.ai <https://fal.ai>`_ is a cloud platform that hosts GPU-accelerated
AI models as APIs.  ``twat-genai`` wraps those APIs behind a clean Python
interface and a ``twat-genai`` CLI command.

Supported operations
---------------------
text
    Generate images from a text prompt (text-to-image).
image
    Vary or restyle an existing image guided by a prompt (image-to-image).
canny / depth
    ControlNet-style generation: structure is constrained by the edge map or
    depth map extracted from a reference image.
upscale
    Enlarge a low-resolution image using one of several AI upscalers
    (aura_sr, ccsr, clarity, drct, esrgan, ideogram, recraft_*).
outpaint
    Expand an image's canvas beyond its original borders.

LoRA support
------------
LoRA (Low-Rank Adaptation) files are small weight patches that steer a base
model toward a particular style or subject without full retraining.  Pass a
LoRA spec string via ``--lora`` or the ``lora=`` parameter.

Quick start::

    from twat_genai import TwatGenAiCLI
    import asyncio, fire

    # CLI: twat-genai text "a red panda in watercolour style"
    # Python:
    cli = TwatGenAiCLI(output_dir="out")
    asyncio.run(cli._run_generation("a red panda", ModelTypes.TEXT))
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("twat-genai")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from twat_genai.cli import TwatGenAiCLI
from twat_genai.core.config import ImageInput, ImageResult, ImageSizeWH
from twat_genai.core.image import ImageFormats, ImageSizes
from twat_genai.engines.chutes import ChutesClient, ChutesEndpoints
from twat_genai.engines.gemini import GeminiImageClient, resolve_gemini_model
from twat_genai.engines.openai_image import OpenAIImageClient, resolve_openai_image_model
from twat_genai.engines.base import EngineConfig, ImageGenerationEngine
from twat_genai.engines.fal import FALEngine
from twat_genai.engines.fal.config import (
    ImageToImageConfig,
    ModelTypes,
    OutpaintConfig,
    UpscaleConfig,
)
from twat_genai.models import (
    BackgroundRemovalRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    MediaFile,
    MediaKind,
    ProviderName,
    ProviderResponse,
    VideoGenerationRequest,
)
from twat_genai.providers import ProviderError, default_output_stem


def main() -> None:
    """CLI entry point for twat-genai."""
    import fire  # type: ignore[import-untyped]

    fire.Fire(TwatGenAiCLI)


__all__ = [
    "BackgroundRemovalRequest",
    "ChutesClient",
    "ChutesEndpoints",
    "EngineConfig",
    "FALEngine",
    "GeminiImageClient",
    "ImageEditRequest",
    "ImageFormats",
    "ImageGenerationEngine",
    "ImageGenerationRequest",
    "ImageInput",
    "ImageResult",
    "ImageSizeWH",
    "ImageSizes",
    "ImageToImageConfig",
    "MediaFile",
    "MediaKind",
    "ModelTypes",
    "OpenAIImageClient",
    "OutpaintConfig",
    "ProviderError",
    "ProviderName",
    "ProviderResponse",
    "TwatGenAiCLI",
    "UpscaleConfig",
    "VideoGenerationRequest",
    "__version__",
    "default_output_stem",
    "main",
    "resolve_gemini_model",
    "resolve_openai_image_model",
]
