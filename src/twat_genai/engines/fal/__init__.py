#!/usr/bin/env -S uv run
# /// script
# dependencies = ["fal-client", "python-dotenv"]
# ///
"""FAL image generation engine implementation."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from loguru import logger

from ...core.config import ImageResult
from ..base import EngineConfig, ImageGenerationEngine
from .client import get_result, submit_job
from .config import FALJobConfig, ImageToImageConfig, ModelTypes

load_dotenv()


class FALEngine(ImageGenerationEngine):
    """FAL image generation engine implementation."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """
        Initialize the FAL engine.

        Args:
            output_dir: Directory to save generated images
        """
        self.output_dir = output_dir
        self.api_key = os.getenv("FAL_KEY")

    async def initialize(self) -> None:
        """Initialize the engine and verify API key."""
        if not self.api_key:
            raise ValueError(
                "FAL_KEY environment variable not set. Please set it with your FAL API key."
            )

    async def generate(
        self,
        prompt: str,
        config: EngineConfig,
        **kwargs: Any,
    ) -> ImageResult:
        """
        Generate an image using FAL.

        Args:
            prompt: Text prompt for generation
            config: Engine configuration
            **kwargs: Additional parameters:
                model: FAL model type
                image_config: Configuration for image-to-image operations
                lora_spec: LoRA configuration
                filename_suffix: Optional suffix for generated filenames
                filename_prefix: Optional prefix for generated filenames

        Returns:
            Generated image result
        """
        model = kwargs.get("model", ModelTypes.TEXT)
        image_config = kwargs.get("image_config")
        lora_spec = kwargs.get("lora_spec")
        filename_suffix = kwargs.get("filename_suffix")
        filename_prefix = kwargs.get("filename_prefix")

        job_config = FALJobConfig(
            prompt=prompt,
            original_prompt=prompt,
            model=model,
            lora_spec=lora_spec,
            output_dir=self.output_dir,
            filename_suffix=filename_suffix,
            filename_prefix=filename_prefix,
            image_config=image_config,
        )

        request_id = await submit_job(job_config)

        job_params = {
            "prompt": prompt,
            "model": model.value,
            "lora_spec": lora_spec,
            "image_size": config.image_size,
            "guidance_scale": config.guidance_scale,
            "num_inference_steps": config.num_inference_steps,
        }

        if image_config:
            job_params.update(
                {
                    "image_config": image_config.model_dump(),
                }
            )

        return await get_result(
            request_id,
            self.output_dir,
            filename_suffix,
            filename_prefix,
            prompt,
            job_params,
        )

    async def shutdown(self) -> None:
        """Clean up resources."""
        pass  # No cleanup needed for FAL engine
