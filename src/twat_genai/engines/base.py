#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pydantic"]
# ///
"""Base interface for image generation engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

# Import core types used by EngineConfig and the ABC
from twat_genai.core.image import ImageSizes
from twat_genai.core.config import ImageResult, ImageSizeWH


class EngineConfig(BaseModel):
    """Base configuration for image generation engines.

    This class defines the common configuration parameters applicable
    across different image generation engines.
    """

    guidance_scale: float = 3.5
    num_inference_steps: int = 28
    # Use the Union type alias directly
    image_size: ImageSizes | ImageSizeWH = ImageSizes.SQ
    enable_safety_checker: bool = False


class ImageGenerationEngine(ABC):
    """Abstract base class for image generation engines."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the engine and any required resources."""

    @abstractmethod
    async def generate(
        self, prompt: str, config: EngineConfig, **kwargs: Any
    ) -> ImageResult:
        """
        Generate an image from the given prompt and configuration.

        Args:
            prompt: Text prompt for image generation
            config: An instance of EngineConfig defined in this module.
            **kwargs: Additional engine-specific parameters (e.g., image_config,
                      upscale_config, lora_spec for specific engines).

        Returns:
            Generated image result (ImageResult object from core.config)
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources and shut down the engine."""

    async def __aenter__(self) -> ImageGenerationEngine:
        """Context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        await self.shutdown()
