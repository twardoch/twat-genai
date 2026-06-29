# this_file: tests/test_image_utils.py
"""Tests for image utility functions."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from PIL import Image

from twat_genai.core.image_utils import (
    download_image_to_temp,
    download_image,
    resize_image_if_needed,
)
from twat_genai.engines.fal.config import ModelTypes  # For resize test

# --- Tests for download_image_to_temp ---


@pytest.mark.asyncio
async def test_download_image_to_temp_success(mocker):
    "Test successful download to a temporary file."
    mock_response = AsyncMock()
    mock_response.content = b"fake image data"
    mock_response.headers = {"content-type": "image/jpeg"}
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.get = AsyncMock(return_value=mock_response)

    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__.return_value = mock_client_instance

    mock_async_client = mocker.patch("httpx.AsyncClient", return_value=mock_context_manager)

    mock_named_temp_file = mocker.patch("tempfile.NamedTemporaryFile")
    mock_file_handle = MagicMock()
    mock_file_handle.name = "/tmp/fake_temp_file.jpg"
    mock_named_temp_file.return_value.__enter__.return_value = mock_file_handle

    temp_path = await download_image_to_temp("http://fake.url/img.jpg")

    mock_async_client.assert_called_once()
    mock_client_instance.get.assert_called_once_with("http://fake.url/img.jpg", follow_redirects=True)
    mock_named_temp_file.assert_called_once_with(suffix=".jpg", delete=False)
    mock_file_handle.write.assert_called_once_with(b"fake image data")
    assert temp_path == "/tmp/fake_temp_file.jpg"


@pytest.mark.asyncio
async def test_download_image_to_temp_http_error(mocker):
    "Test HTTP error during download to temp."
    import httpx as _httpx

    mock_client_instance = AsyncMock()
    mock_client_instance.get.side_effect = _httpx.RequestError("Fake HTTP Error")

    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__.return_value = mock_client_instance

    mocker.patch("httpx.AsyncClient", return_value=mock_context_manager)

    with pytest.raises(RuntimeError, match="Failed to download image"):
        await download_image_to_temp("http://fake.url/img.jpg")


# --- Tests for download_image (to specific path) ---


@pytest.mark.asyncio
async def test_download_image_success(mocker, tmp_path):
    "Test successful download to a specific path."
    output_path = tmp_path / "output.png"

    mock_response = AsyncMock()
    mock_response.content = b"fake png data"
    mock_response.headers = {"content-type": "image/png"}
    mock_response.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.get = AsyncMock(return_value=mock_response)

    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__.return_value = mock_client_instance

    mocker.patch("httpx.AsyncClient", return_value=mock_context_manager)

    await download_image("http://fake.url/img.png", output_path)

    mock_client_instance.get.assert_called_once_with("http://fake.url/img.png", follow_redirects=True)
    assert output_path.exists()
    assert output_path.read_bytes() == b"fake png data"


# --- Tests for resize_image_if_needed ---


# Fixture to create a dummy image file
@pytest.fixture
def dummy_image_file(tmp_path):
    file_path = tmp_path / "test_image.jpg"
    img = Image.new("RGB", (3000, 2000), color="red")  # Larger than max size
    img.save(file_path)
    return file_path


# Mock the UPSCALE_TOOL_MAX_INPUT_SIZES for testing
@pytest.fixture(autouse=True)
def mock_upscale_sizes(mocker):
    # Patch the location where UPSCALE_TOOL_MAX_INPUT_SIZES is now defined
    mocker.patch(
        "twat_genai.engines.fal.config.UPSCALE_TOOL_MAX_INPUT_SIZES",
        {ModelTypes.UPSCALER_DRCT: 2048, ModelTypes.UPSCALER_ESRGAN: 1024},
    )
    # Also need to update the import reference within image_utils.py if it's direct,
    # or ensure image_utils.py gets it from the new location.
    # For the test, we assume image_utils.py will correctly access the patched dict.
    # If image_utils.py imports it directly as:
    # `from twat_genai.engines.fal.config import UPSCALE_TOOL_MAX_INPUT_SIZES`
    # then this patch is correct.

    # If image_utils.py had its own copy or imported from the old upscale.py,
    # this patch target would need to change to where image_utils *sees* it.
    # Based on the plan, image_utils.py should be updated to import from engines.fal.config.


def test_resize_needed(dummy_image_file):
    "Test when resizing is needed."
    # Test with DRCT (max 2048), image is 3000x2000 -> should resize based on width
    resized_path_str = resize_image_if_needed(str(dummy_image_file), ModelTypes.UPSCALER_DRCT)
    assert resized_path_str is not None
    resized_path = Path(resized_path_str)
    assert resized_path.exists()
    with Image.open(resized_path) as img:
        assert img.width == 2048
        assert img.height == int(2000 * (2048 / 3000))
    resized_path.unlink()  # Clean up temp file


def test_resize_not_needed(dummy_image_file):
    "Test when resizing is not needed (image within limits)."
    # Create smaller image
    small_img_path = dummy_image_file.parent / "small.jpg"
    img = Image.new("RGB", (1000, 800), color="blue")
    img.save(small_img_path)

    resized_path_str = resize_image_if_needed(str(small_img_path), ModelTypes.UPSCALER_DRCT)
    assert resized_path_str is None


def test_resize_not_upscaler(dummy_image_file):
    "Test that non-upscaler models are skipped."
    # Text model is not in our mocked UPSCALE_TOOL_MAX_INPUT_SIZES
    resized_path_str = resize_image_if_needed(str(dummy_image_file), ModelTypes.TEXT)
    assert resized_path_str is None


def test_resize_file_not_found():
    "Test resizing when the input file does not exist."
    with pytest.raises(FileNotFoundError):
        resize_image_if_needed("nonexistent_file.jpg", ModelTypes.UPSCALER_DRCT)
