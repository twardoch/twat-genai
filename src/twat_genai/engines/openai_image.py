"""OpenAI-compatible image generation/editing client.

# this_file: src/twat_genai/engines/openai_image.py
"""

from __future__ import annotations

import os
from pathlib import Path
import httpx

from twat_genai.models import ImageGenerationRequest, MediaFile, MediaKind, ProviderName, ProviderResponse
from twat_genai.providers import ProviderError, decode_data_url_or_fetch, files_from_payload

MODEL_ALIASES = {"gpt": "gpt-image-2", "nano": "gemini-3.1-flash-image"}


def resolve_openai_image_model(model: str | None) -> str:
    """Resolve reference-script model aliases without shell assumptions."""

    return MODEL_ALIASES.get(model or "gpt", model or "gpt-image-2")


class OpenAIImageClient:
    """HTTP client for OpenAI-compatible image endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("CLIPROXY_API_KEY")
        self.base_url = (base_url or os.getenv("CLIPROXY_API_ENDPOINT") or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout
        self.client = client

    @property
    def headers(self) -> dict[str, str]:
        if not self.api_key:
            msg = "OPENAI_API_KEY or CLIPROXY_API_KEY is required for OpenAI image calls."
            raise ProviderError(msg)
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def generation_payload(request: ImageGenerationRequest) -> dict[str, object]:
        """Build an OpenAI images.generate payload."""

        payload: dict[str, object] = {
            "model": resolve_openai_image_model(request.model),
            "prompt": request.prompt,
            "n": request.n,
        }
        if request.width and request.height:
            payload["size"] = f"{request.width}x{request.height}"
        if request.output_format:
            payload["output_format"] = "jpeg" if request.output_format == "jpg" else request.output_format
        payload.update(request.metadata)
        return {key: value for key, value in payload.items() if value is not None}

    def generate(self, request: ImageGenerationRequest) -> ProviderResponse:
        """Generate images through an OpenAI-compatible `/images/generations` endpoint."""

        client = self.client or httpx.Client(timeout=self.timeout)
        close_client = self.client is None
        try:
            response = client.post(
                f"{self.base_url}/images/generations",
                headers={**self.headers, "Content-Type": "application/json"},
                json=self.generation_payload(request),
            )
            response.raise_for_status()
            payload = response.json()
            normalized = files_from_payload(payload, provider=ProviderName.OPENAI, kind=MediaKind.IMAGE)
            if not normalized.files:
                normalized.files = [
                    file
                    for item in payload.get("data", [])
                    if (url := item.get("url"))
                    for file in [file_from_url(url, item)]
                ]
            return normalized
        finally:
            if close_client:
                client.close()

    def save_response_files(self, response: ProviderResponse, output_dir: Path, stem: str) -> list[Path]:
        """Persist URL/data-url image outputs and return saved paths."""

        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        client = self.client or httpx.Client(timeout=60.0)
        close_client = self.client is None
        try:
            for index, media in enumerate(response.files, start=1):
                if media.data is not None:
                    data = media.data
                elif media.url is not None:
                    data = decode_data_url_or_fetch(str(media.url), client=client)
                else:
                    continue
                suffix = ".png" if media.mime_type is None else f".{media.mime_type.rsplit('/', 1)[-1]}"
                path = output_dir / f"{stem}-{index}{suffix}"
                path.write_bytes(data)
                saved.append(path)
        finally:
            if close_client:
                client.close()
        return saved


def file_from_url(url: str, metadata: dict[str, object]) -> MediaFile:
    """Create a MediaFile without importing it in a list comprehension scope."""

    return MediaFile(kind=MediaKind.IMAGE, url=url, metadata=metadata)
