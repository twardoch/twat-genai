"""Gemini/Nano Banana image transform support.

# this_file: src/twat_genai/engines/gemini.py
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from twat_genai.models import ImageEditRequest, MediaKind, ProviderName, ProviderResponse
from twat_genai.providers import ProviderError, files_from_payload, media_file_to_data_url

GEMINI_MODEL_ALIASES = {
    "flash25": "gemini-2.5-flash-preview-image-generation",
    "pro3": "gemini-3-pro-image-preview",
    "flash31": "gemini-3.1-flash-image-preview",
    "nano": "gemini-3.1-flash-image-preview",
}


def resolve_gemini_model(model: str | None) -> str:
    """Resolve short Gemini/Nano Banana model names."""

    return GEMINI_MODEL_ALIASES.get(model or "flash31", model or "gemini-3.1-flash-image-preview")


class GeminiImageClient:
    """REST-style Gemini image transform client with explicit API-key config."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = client

    @staticmethod
    def transform_payload(
        request: ImageEditRequest, *, thinking: str = "MINIMAL", quality: str = "4K", aspect: str = "9:16"
    ) -> dict[str, Any]:
        """Build a Gemini generateContent payload from an image edit request."""

        parts: list[dict[str, Any]] = [{"text": request.prompt}]
        for media in request.input_images:
            data_url = media_file_to_data_url(media)
            if data_url.startswith("data:"):
                header, _, data = data_url.partition(",")
                mime_type = header.removeprefix("data:").split(";", 1)[0] or media.mime_type or "image/png"
                parts.append({"inlineData": {"mimeType": mime_type, "data": data}})
            else:
                parts.append({"fileData": {"mimeType": media.mime_type or "image/png", "fileUri": data_url}})
        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
                "imageConfig": {"aspectRatio": aspect, "imageSize": quality.upper()},
                "thinkingConfig": {"thinkingLevel": thinking},
            },
        }

    def transform_image(self, request: ImageEditRequest) -> ProviderResponse:
        """Transform an image through Gemini image generation."""

        if not self.api_key:
            msg = "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini image calls."
            raise ProviderError(msg)
        model = resolve_gemini_model(request.model)
        client = self.client or httpx.Client(timeout=self.timeout)
        close_client = self.client is None
        try:
            response = client.post(
                f"{self.base_url}/models/{model}:generateContent",
                params={"key": self.api_key},
                json=self.transform_payload(request),
            )
            response.raise_for_status()
            return files_from_payload(response.json(), provider=ProviderName.GEMINI, kind=MediaKind.IMAGE)
        finally:
            if close_client:
                client.close()
