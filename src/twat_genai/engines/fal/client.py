#!/usr/bin/env -S uv run
# /// script
# dependencies = ["fal-client", "loguru", "httpx"]
# ///
"""FAL API client and request handling."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

import fal_client
import httpx
from loguru import logger

from ...core.config import ImageResult

if TYPE_CHECKING:
    from pathlib import Path

    from fal.config import FALJobConfig


async def submit_job(job: FALJobConfig) -> str:
    """
    Submit an asynchronous image generation job.

    Args:
        job: Job configuration

    Returns:
        FAL request ID
    """
    args = await job.to_fal_arguments()
    handler = await fal_client.submit_async(job.model.value, arguments=args)
    logger.debug(f"Submitted job with ID: {handler.request_id}")
    return handler.request_id


async def download_image(url: str, output_path: Path) -> None:
    """
    Download an image from a URL and save it to disk.

    Args:
        url: Image URL
        output_path: Path to save the image
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        logger.info(f"Saved image to: {output_path}")


async def get_result(
    request_id: str,
    output_dir: Path | None = None,
    filename_suffix: str | None = None,
    filename_prefix: str | None = None,
    original_prompt: str | None = None,
    job_params: dict[str, Any] | None = None,
) -> ImageResult:
    """
    Retrieve and process the result of a submitted job.

    Args:
        request_id: FAL request ID
        output_dir: Directory to save generated images
        filename_suffix: Optional suffix for generated filenames
        filename_prefix: Optional prefix for generated filenames
        original_prompt: Original prompt before processing
        job_params: Original job parameters

    Returns:
        Image generation result
    """
    status = await fal_client.status_async(
        "fal-ai/flux-lora", request_id, with_logs=True
    )
    while isinstance(status, fal_client.InProgress):
        await asyncio.sleep(1)
        status = await fal_client.status_async(
            "fal-ai/flux-lora", request_id, with_logs=True
        )

    result = await fal_client.result_async("fal-ai/flux-lora", request_id)
    timestamp = result.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    image_info = result["images"][0]
    image_url = image_info["url"]

    image_path = None
    metadata_path = None

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        request_id_prefix = request_id.split("-")[0] if request_id else ""

        stem = f"{filename_prefix or 'image_'}{timestamp}-{request_id_prefix}"
        if filename_suffix:
            stem = f"{stem}-{filename_suffix}"

        image_path = output_dir / f"{stem}.jpg"
        try:
            await download_image(image_url, image_path)
        except Exception as e:
            logger.error(f"Failed to save image {image_path}: {e}")
            image_path = None

        try:
            metadata_path = output_dir / f"{stem}.json"
            metadata = ImageResult(
                request_id=request_id,
                timestamp=timestamp,
                result=result,
                image_info={
                    "path": str(image_path) if image_path else None,
                    "url": image_url,
                    "index": 0,
                    "suffix": filename_suffix,
                    "metadata_path": str(metadata_path) if metadata_path else None,
                    "original_prompt": original_prompt,
                },
                image=None,
                original_prompt=original_prompt,
                job_params=job_params,
            )
            metadata_path.write_text(metadata.model_dump_json(indent=2))
            logger.info(f"Saved metadata to: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata {metadata_path}: {e}")
            metadata_path = None

    return ImageResult(
        request_id=request_id,
        timestamp=timestamp,
        result=result,
        image_info={
            "path": str(image_path) if image_path else None,
            "url": image_url,
            "index": 0,
            "suffix": filename_suffix,
            "metadata_path": str(metadata_path) if metadata_path else None,
            "original_prompt": original_prompt,
        },
        image=None,
        original_prompt=original_prompt,
        job_params=job_params,
    )
