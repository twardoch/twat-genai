#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pydantic"]
# ///
"""Base interface for image generation engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..core.image import ImageSizes

if TYPE_CHECKING:
    from ..core.config import ImageResult, ImageSizeWH


class EngineConfig(BaseModel):
    """Base configuration for image generation engines."""

    guidance_scale: float = 3.5
    num_inference_steps: int = 28
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
            config: Engine configuration
            **kwargs: Additional engine-specific parameters

        Returns:
            Generated image result
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
