"""twat-genai: Generative AI toolkit powered by fal.ai."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("twat-genai")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from twat_genai.cli import TwatGenAiCLI
from twat_genai.core.config import ImageInput, ImageResult, ImageSizeWH
from twat_genai.core.image import ImageFormats, ImageSizes
from twat_genai.engines.base import EngineConfig, ImageGenerationEngine
from twat_genai.engines.fal import FALEngine
from twat_genai.engines.fal.config import (
    ImageToImageConfig,
    ModelTypes,
    OutpaintConfig,
    UpscaleConfig,
)


def main() -> None:
    """CLI entry point for twat-genai."""
    import fire  # type: ignore[import-untyped]

    fire.Fire(TwatGenAiCLI)


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
    "OutpaintConfig",
    "TwatGenAiCLI",
    "UpscaleConfig",
    "__version__",
    "main",
]
