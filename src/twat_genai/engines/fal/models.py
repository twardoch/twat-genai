#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pydantic", "fal-client>=0.5.9", "Pillow>=11.1.0"]
# ///
"""FAL-specific model classes."""

import tempfile
from pathlib import Path
from typing import Any

# Try importing fal_client, but make its usage conditional or handle absence
try:
    import fal_client
except ImportError:
    fal_client = None  # type: ignore
    # logger.warning("fal_client not found. Image upload functionality will be limited.") # Requires logger

from twat_genai.core.config import ImageInput  # Use absolute import


class FALImageInput(ImageInput):
    """FAL-specific implementation of ImageInput."""

    @classmethod
    def from_base(cls, base: ImageInput) -> "FALImageInput":
        """Create a FALImageInput instance from a base ImageInput."""
        return cls(url=base.url, path=base.path, pil_image=base.pil_image)

    async def to_url(self, client: Any = None) -> str:
        """Convert the input to a URL format using fal_client.

        Requires fal_client to be installed for path/PIL image upload.

        Args:
            client: Optional API client, not used in this implementation
                   as we directly use fal_client for uploads.

        Returns:
            URL string to the image.

        Raises:
            ImportError: If fal_client is not installed.
            ValueError: If no valid input is provided.
        """
        if self.url:
            return self.url

        if not fal_client:
            msg = (
                "fal_client package is required to upload local files or PIL images."
                " Please install it (`uv pip install fal-client`) or provide a URL."
            )
            raise ImportError(msg)

        if self.path:
            return await fal_client.upload_file_async(self.path)
        elif self.pil_image:
            # For PIL images, save to a temporary file first
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                try:
                    self.pil_image.save(tmp_path, format="JPEG", quality=95)
                    # logger.debug(f"Saved PIL image to temp file {tmp_path} for upload") # Requires logger
                    upload_url = await fal_client.upload_file_async(tmp_path)
                finally:
                    # Ensure temporary file is deleted even if upload fails
                    try:
                        tmp_path.unlink()
                        # logger.debug(f"Deleted temporary PIL image file {tmp_path}") # Requires logger
                    except OSError:
                        # logger.warning(f"Could not delete temp file {tmp_path}: {e}") # Requires logger
                        pass  # Log warning, but don't fail the operation
                return upload_url
        else:
            # This should not happen if is_valid is checked before calling
            msg = "No valid image input (URL, Path, PIL Image) provided to convert to URL."
            raise ValueError(msg)
