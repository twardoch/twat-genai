#!/usr/bin/env -S uv run
# /// script
# dependencies = ["fal-client", "Pillow"]
# ///
"""FAL-specific model classes."""

import tempfile
from pathlib import Path

import fal_client

from ...core.config import ImageInput


class FALImageInput(ImageInput):
    """FAL-specific implementation of ImageInput."""

    @classmethod
    def from_base(cls, base: ImageInput) -> "FALImageInput":
        """Create a FALImageInput instance from a base ImageInput."""
        return cls(url=base.url, path=base.path, pil_image=base.pil_image)

    async def to_url(self) -> str:
        """Convert the input to a URL format using fal_client."""
        if self.url:
            return self.url
        elif self.path:
            return await fal_client.upload_file_async(self.path)
        elif self.pil_image:
            # For PIL images, save to a temporary file first
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                self.pil_image.save(tmp_path, format="JPEG", quality=95)
                return await fal_client.upload_file_async(tmp_path)
        else:
            msg = "No valid image input provided"
            raise ValueError(msg)
