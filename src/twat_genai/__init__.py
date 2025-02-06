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
    "cli",
    "ImageInput",
    "ImageResult",
    "ImageSizeWH",
    "ImageFormats",
    "ImageSizes",
    "expand_prompts",
    "normalize_prompts",
    "EngineConfig",
    "ImageGenerationEngine",
    "FALEngine",
    "ImageToImageConfig",
    "ModelTypes",
    "__version__",
]
