"""twat-genai: AI image generation package using fal.ai models."""

from importlib import metadata

from twat_genai.__version__ import __version__
from twat_genai.cli import cli
from twat_genai.core.config import ImageInput, ImageResult, ImageSizeWH
from twat_genai.core.image import ImageFormats, ImageSizes
from twat_genai.core.prompt import expand_prompts, normalize_prompts
from twat_genai.engines.base import EngineConfig, ImageGenerationEngine
from twat_genai.engines.fal import FALEngine
from twat_genai.engines.fal.config import ImageToImageConfig, ModelTypes

__all__ = [
    "EngineConfig",
    "FALEngine",
    "ImageFormats",
    "ImageGenerationEngine",
    "ImageInput",
    "ImageResult",
    "ImageSizeWH",
    "ImageSizes",
    "ImageToImageConfig",
    "ModelTypes",
    "__version__",
    "cli",
    "expand_prompts",
    "normalize_prompts",
]
