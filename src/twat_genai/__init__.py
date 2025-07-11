"""Main package for twat-genai."""

from importlib.metadata import PackageNotFoundError, version

from twat_genai.cli import TwatGenAiCLI
from twat_genai.core.config import ImageInput, ImageResult, ImageSizeWH
from twat_genai.core.image import ImageFormats, ImageSizes
from twat_genai.engines.base import EngineConfig, ImageGenerationEngine
from twat_genai.engines.fal import FALEngine
from twat_genai.engines.fal.config import (
    ImageToImageConfig,
    ModelTypes,  # This now also includes upscale/outpaint model types
    OutpaintConfig,
    UpscaleConfig,
)

try:
    __version__ = version("twat-genai")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"

__all__ = [
    "EngineConfig",
    "FALEngine",
    "ImageFormats",
    "ImageGenerationEngine",
    "ImageInput",
    "ImageResult",
    "ImageSizeWH",  # Defined in core.config
    "ImageSizes",   # Defined in core.image
    "ImageToImageConfig",
    "ModelTypes",   # Defined in engines.fal.config
    "OutpaintConfig",
    "TwatGenAiCLI",
    "UpscaleConfig",
    "__version__",
]
