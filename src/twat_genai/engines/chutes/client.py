"""Thin Chutes API client used by domain packages through twat_genai.

# this_file: src/twat_genai/engines/chutes/client.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, cast

import httpx

from twat_genai.models import (
    BackgroundRemovalRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    MediaKind,
    ProviderName,
    ProviderResponse,
    VideoGenerationRequest,
)
from twat_genai.providers import ProviderError, files_from_payload, media_file_to_data_url


@dataclass(frozen=True)
class ChutesEndpoints:
    """Default Chutes endpoints represented by reference/chutes_image scripts."""

    image: str = "https://image.chutes.ai/generate"
    hidream_generate: str = "https://chutes-hidream.chutes.ai/generate"
    hidream_edit: str = "https://chutes-hidream-edit.chutes.ai/generate"
    rembg: str = "https://desudesuka-rembg.chutes.ai/remove"
    skyreels_text_to_video: str = "https://kikakkz-skyreels-v2-14b-540p.chutes.ai/text2video"
    skyreels_image_to_video: str = "https://kikakkz-skyreels-v2-14b-540p.chutes.ai/image2video"
    wan_flf2v: str = "https://kikakkz-wan2-1-14b-flf2v.chutes.ai/flf2video"


class ChutesClient:
    """Minimal sync/async Chutes client with injectable HTTP transports."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoints: ChutesEndpoints | None = None,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("CHUTES_API_KEY") or os.getenv("CHUTES_API_TOKEN")
        self.endpoints = endpoints or ChutesEndpoints()
        self.timeout = timeout
        self.client = client
        self.async_client = async_client

    @property
    def headers(self) -> dict[str, str]:
        """Return Chutes authorization headers."""

        if not self.api_key:
            msg = "CHUTES_API_KEY or CHUTES_API_TOKEN is required for Chutes provider calls."
            raise ProviderError(msg)
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        client = self.client or httpx.Client(timeout=self.timeout)
        close_client = self.client is None
        try:
            response = client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            if response.headers.get("content-type", "").startswith("application/json"):
                return cast(dict[str, Any], response.json())
            return {"data": response.content.decode("latin1")}
        finally:
            if close_client:
                client.close()

    async def _apost_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        client = self.async_client or httpx.AsyncClient(timeout=self.timeout)
        close_client = self.async_client is None
        try:
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            if response.headers.get("content-type", "").startswith("application/json"):
                return cast(dict[str, Any], response.json())
            return {"data": response.content.decode("latin1")}
        finally:
            if close_client:
                await client.aclose()

    @staticmethod
    def image_payload(request: ImageGenerationRequest) -> dict[str, Any]:
        """Build the general Chutes image payload."""

        payload: dict[str, Any] = {"prompt": request.prompt}
        for source, target in (
            ("model", "model"),
            ("negative_prompt", "negative_prompt"),
            ("width", "width"),
            ("height", "height"),
            ("guidance_scale", "guidance_scale"),
            ("inference_steps", "inference_steps"),
            ("seed", "seed"),
        ):
            value = getattr(request, source)
            if value is not None:
                payload[target] = value
        return payload

    @staticmethod
    def hidream_edit_payload(request: ImageEditRequest) -> dict[str, Any]:
        """Build a HiDream edit payload from a neutral edit request."""

        image_data_url = media_file_to_data_url(request.input_images[0])
        _, _, image_b64 = image_data_url.partition(",")
        payload = ChutesClient.image_payload(request)
        payload["image_b64"] = image_b64 or image_data_url
        return payload

    @staticmethod
    def rembg_payload(request: BackgroundRemovalRequest) -> dict[str, Any]:
        """Build a Chutes rembg payload."""

        data_url = media_file_to_data_url(request.image)
        _, _, image_b64 = data_url.partition(",")
        return {"image_b64": image_b64 or data_url}

    @staticmethod
    def video_payload(request: VideoGenerationRequest) -> dict[str, Any]:
        """Build a Chutes SkyReels/WAN video payload."""

        payload: dict[str, Any] = {"prompt": request.prompt}
        for attr in ("resolution", "fps", "num_frames", "seed"):
            value = getattr(request, attr)
            if value is not None:
                payload[attr] = value
        if request.first_frame is not None:
            data_url = media_file_to_data_url(request.first_frame)
            payload["img_b64_first"] = data_url.partition(",")[2] or data_url
        if request.last_frame is not None:
            data_url = media_file_to_data_url(request.last_frame)
            payload["img_b64_last"] = data_url.partition(",")[2] or data_url
        return payload

    def generate_image(self, request: ImageGenerationRequest) -> ProviderResponse:
        """Generate an image through the general Chutes image endpoint."""

        payload = self.image_payload(request)
        raw = self._post_json(self.endpoints.image, payload)
        return files_from_payload(raw, provider=ProviderName.CHUTES, kind=MediaKind.IMAGE)

    async def generate_image_async(self, request: ImageGenerationRequest) -> ProviderResponse:
        """Generate an image asynchronously through the Chutes image endpoint."""

        payload = self.image_payload(request)
        raw = await self._apost_json(self.endpoints.image, payload)
        return files_from_payload(raw, provider=ProviderName.CHUTES, kind=MediaKind.IMAGE)

    def edit_hidream(self, request: ImageEditRequest) -> ProviderResponse:
        """Edit an image through Chutes HiDream Edit."""

        raw = self._post_json(self.endpoints.hidream_edit, self.hidream_edit_payload(request))
        return files_from_payload(raw, provider=ProviderName.CHUTES, kind=MediaKind.IMAGE)

    def remove_background(self, request: BackgroundRemovalRequest) -> ProviderResponse:
        """Remove an image background through the Chutes rembg endpoint."""

        raw = self._post_json(self.endpoints.rembg, self.rembg_payload(request))
        return files_from_payload(raw, provider=ProviderName.CHUTES, kind=MediaKind.IMAGE)

    def generate_video(self, request: VideoGenerationRequest) -> ProviderResponse:
        """Generate video through SkyReels or WAN depending on frame inputs."""

        payload = self.video_payload(request)
        endpoint = (
            self.endpoints.wan_flf2v
            if request.first_frame and request.last_frame
            else self.endpoints.skyreels_text_to_video
        )
        if request.first_frame and not request.last_frame:
            endpoint = self.endpoints.skyreels_image_to_video
        raw = self._post_json(endpoint, payload)
        return files_from_payload(raw, provider=ProviderName.CHUTES, kind=MediaKind.VIDEO)
