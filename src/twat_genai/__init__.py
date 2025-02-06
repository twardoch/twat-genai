"""twat-genai: AI image generation package using fal.ai models."""

from importlib import metadata

from .__version__ import __version__
from .cli import cli
from .core.config import ImageInput, ImageResult, ImageSizeWH
from .core.image import ImageFormats, ImageSizes
from .core.prompt import expand_prompts, normalize_prompts
from .engines.base import EngineConfig, ImageGenerationEngine
from .engines.fal import FALEngine
from .engines.fal.config import ImageToImageConfig, ModelTypes

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
