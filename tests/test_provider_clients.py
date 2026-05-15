"""Tests for provider-neutral media clients."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from twat_genai import (
    ChutesClient,
    ChutesEndpoints,
    GeminiImageClient,
    ImageEditRequest,
    ImageGenerationRequest,
    MediaFile,
    MediaKind,
    OpenAIImageClient,
    ProviderError,
    ProviderName,
    VideoGenerationRequest,
    resolve_gemini_model,
    resolve_openai_image_model,
)


def test_media_file_requires_location() -> None:
    """MediaFile rejects empty references."""

    with pytest.raises(ValueError, match="requires at least one"):
        MediaFile(kind=MediaKind.IMAGE)


def test_openai_model_aliases() -> None:
    """Reference-script aliases resolve to provider model IDs."""

    assert resolve_openai_image_model("gpt") == "gpt-image-2"
    assert resolve_openai_image_model("nano") == "gemini-3.1-flash-image"
    assert resolve_openai_image_model("custom") == "custom"


def test_gemini_model_aliases() -> None:
    """Nano/Gemini aliases resolve to full model IDs."""

    assert resolve_gemini_model("flash31") == "gemini-3.1-flash-image-preview"
    assert resolve_gemini_model("nano") == "gemini-3.1-flash-image-preview"
    assert resolve_gemini_model("models/custom") == "models/custom"


def test_chutes_image_payload_and_response() -> None:
    """ChutesClient builds the reusable image-generation payload."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.test/image"
        assert request.headers["authorization"] == "Bearer token"
        assert b'"model":"FLUX.1-dev"' in request.content
        return httpx.Response(200, json={"images": [{"url": "https://cdn.example/image.png"}], "id": "abc"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    chutes = ChutesClient(
        api_key="token",
        endpoints=ChutesEndpoints(image="https://example.test/image"),
        client=client,
    )

    response = chutes.generate_image(ImageGenerationRequest(prompt="a castle", model="FLUX.1-dev", width=1024))

    assert response.provider == ProviderName.CHUTES
    assert response.files[0].url == "https://cdn.example/image.png"
    assert response.request_id == "abc"


def test_chutes_video_selects_wan_for_first_and_last_frames(tmp_path: Path) -> None:
    """WAN endpoint is selected when both first and last frames are present."""

    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    first.write_bytes(b"first")
    last.write_bytes(b"last")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.test/wan"
        return httpx.Response(200, json={"videos": [{"url": "https://cdn.example/video.mp4"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    chutes = ChutesClient(
        api_key="token",
        endpoints=ChutesEndpoints(wan_flf2v="https://example.test/wan"),
        client=client,
    )
    request = VideoGenerationRequest(
        prompt="animate",
        first_frame=MediaFile(kind=MediaKind.IMAGE, path=first),
        last_frame=MediaFile(kind=MediaKind.IMAGE, path=last),
    )

    response = chutes.generate_video(request)

    assert response.files[0].url == "https://cdn.example/video.mp4"


def test_chutes_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chutes calls fail early when no key is configured."""

    monkeypatch.delenv("CHUTES_API_KEY", raising=False)
    monkeypatch.delenv("CHUTES_API_TOKEN", raising=False)
    with pytest.raises(ProviderError, match="CHUTES_API_KEY"):
        _ = ChutesClient().headers


def test_openai_generation_payload() -> None:
    """OpenAIImageClient builds /images/generations payloads."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://cliproxy.test/v1/images/generations"
        assert b'"model":"gpt-image-2"' in request.content
        assert b'"size":"1024x768"' in request.content
        return httpx.Response(200, json={"data": [{"url": "https://cdn.example/openai.png"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    openai_client = OpenAIImageClient(api_key="key", base_url="https://cliproxy.test/v1", client=client)
    response = openai_client.generate(
        ImageGenerationRequest(prompt="quote poster", model="gpt", width=1024, height=768)
    )

    assert response.provider == ProviderName.OPENAI
    assert response.files[0].url == "https://cdn.example/openai.png"


def test_gemini_transform_payload(tmp_path: Path) -> None:
    """GeminiImageClient sends inline image data with explicit API key."""

    image = tmp_path / "input.png"
    image.write_bytes(b"img")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "key=gemini-key" in str(request.url)
        assert b"inlineData" in request.content
        return httpx.Response(200, json={"images": [{"url": "https://cdn.example/gemini.png"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gemini = GeminiImageClient(api_key="gemini-key", base_url="https://gemini.test/v1beta", client=client)
    request = ImageEditRequest(
        prompt="match lighting", model="flash31", input_images=[MediaFile(kind=MediaKind.IMAGE, path=image)]
    )

    response = gemini.transform_image(request)

    assert response.provider == ProviderName.GEMINI
    assert response.files[0].url == "https://cdn.example/gemini.png"
