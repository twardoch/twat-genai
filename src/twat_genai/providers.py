"""Small provider helpers shared by the media provider clients.

# this_file: src/twat_genai/providers.py
"""

from __future__ import annotations

import base64
import mimetypes
from typing import Any

import httpx
from slugify import slugify

from twat_genai.models import MediaFile, MediaKind, ProviderName, ProviderResponse


class ProviderError(RuntimeError):
    """Raised when a provider request cannot be completed."""


def default_output_stem(prompt: str, max_words: int = 6) -> str:
    """Return a stable filename stem derived from the prompt."""

    words = prompt.strip().split()[:max_words]
    return slugify(" ".join(words), max_length=64, word_boundary=True) or "generated-media"


def media_file_to_data_url(media: MediaFile) -> str:
    """Convert a media file to a data URL for providers that need inline images."""

    if media.data is not None:
        mime_type = media.mime_type or "application/octet-stream"
        return f"data:{mime_type};base64,{base64.b64encode(media.data).decode()}"

    if media.path is not None:
        mime_type = media.mime_type or mimetypes.guess_type(media.path.name)[0] or "application/octet-stream"
        return f"data:{mime_type};base64,{base64.b64encode(media.path.read_bytes()).decode()}"

    if media.url is not None:
        return str(media.url)

    msg = "MediaFile has no usable content."
    raise ProviderError(msg)


def decode_data_url_or_fetch(url: str, *, client: httpx.Client | None = None) -> bytes:
    """Decode a data URL or fetch a remote URL."""

    if url.startswith("data:"):
        _, _, payload = url.partition(",")
        return base64.b64decode(payload)

    http_client = client or httpx.Client(timeout=60.0)
    close_client = client is None
    try:
        response = http_client.get(url)
        response.raise_for_status()
        return bytes(response.content)
    finally:
        if close_client:
            http_client.close()


def files_from_payload(
    payload: dict[str, Any],
    *,
    provider: ProviderName,
    kind: MediaKind,
    url_keys: tuple[str, ...] = ("url", "image_url", "video_url"),
) -> ProviderResponse:
    """Extract common media URL shapes into a normalized ProviderResponse."""

    files: list[MediaFile] = []
    candidates: list[Any] = []
    for key in ("images", "videos", "files", "data", "output"):
        value = payload.get(key)
        if isinstance(value, list):
            candidates.extend(value)
        elif isinstance(value, dict):
            candidates.append(value)
    candidates.append(payload)

    for item in candidates:
        if isinstance(item, str):
            files.append(MediaFile(kind=kind, url=item))
            continue
        if not isinstance(item, dict):
            continue
        for key in url_keys:
            url = item.get(key)
            if url:
                files.append(MediaFile(kind=kind, url=url, metadata=item))
                break

    return ProviderResponse(
        provider=provider,
        kind=kind,
        files=files,
        raw=payload,
        request_id=str(payload.get("request_id") or payload.get("id") or "") or None,
        model=str(payload.get("model") or "") or None,
    )
