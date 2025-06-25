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
    # Mock httpx response
    mock_response = AsyncMock()
    mock_response.content = b"fake image data"  # This is fine for AsyncMock if not directly awaited
    mock_response.headers = {"content-type": "image/jpeg"} # This needs to be a direct dict
    mock_response.raise_for_status = MagicMock() # This is fine

    # Mock httpx client's get method
    async def mock_get(*args, **kwargs):
        return mock_response

    mock_client_instance = AsyncMock()
    mock_client_instance.get = mock_get

    mock_async_client_context_manager = AsyncMock()
    mock_async_client_context_manager.__aenter__.return_value = mock_client_instance # mock client instance

    mocker.patch("httpx.AsyncClient", return_value=mock_async_client_context_manager)

    # Mock tempfile
    mock_named_temp_file = mocker.patch("tempfile.NamedTemporaryFile")
    mock_file_handle = MagicMock()
    mock_file_handle.name = "/tmp/fake_temp_file.jpg"
    mock_named_temp_file.return_value.__enter__.return_value = mock_file_handle

    temp_path = await download_image_to_temp("http://fake.url/img.jpg")

    mock_async_client.assert_called_once()
    mock_client.get.assert_called_once_with(
        "http://fake.url/img.jpg", follow_redirects=True
    )
    mock_named_temp_file.assert_called_once_with(suffix=".jpg", delete=False)
    mock_file_handle.write.assert_called_once_with(b"fake image data")
    assert temp_path == "/tmp/fake_temp_file.jpg"


@pytest.mark.asyncio
async def test_download_image_to_temp_http_error(mocker):
    "Test HTTP error during download to temp."
    # Mock httpx response to raise error
    mock_response = AsyncMock()
    mock_response.raise_for_status.side_effect = mocker.patch(
        "httpx.RequestError", side_effect=ValueError("Fake HTTP Error")
    )()

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mocker.patch("httpx.AsyncClient", return_value=mock_client)

    with pytest.raises(
        RuntimeError, match="Failed to download image"
    ):  # Match the wrapped error
        await download_image_to_temp("http://fake.url/img.jpg")


# --- Tests for download_image (to specific path) ---


@pytest.mark.asyncio
async def test_download_image_success(mocker, tmp_path):
    "Test successful download to a specific path."
    output_path = tmp_path / "output.png"

    # Mock httpx response
    mock_response = AsyncMock()
    mock_response.content = b"fake png data" # direct assignment for read by write_bytes
    mock_response.headers = {"content-type": "image/png"} # direct dict
    mock_response.raise_for_status = MagicMock()

    # Mock httpx client's get method
    async def mock_get(*args, **kwargs):
        return mock_response

    mock_client_instance = AsyncMock()
    mock_client_instance.get = mock_get

    mock_async_client_context_manager = AsyncMock()
    mock_async_client_context_manager.__aenter__.return_value = mock_client_instance

    mocker.patch("httpx.AsyncClient", return_value=mock_async_client_context_manager)

    await download_image("http://fake.url/img.png", output_path)

    mock_client_instance.get.assert_called_once_with( # Check call on the instance
        "http://fake.url/img.png", follow_redirects=True
    )
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
    resized_path_str = resize_image_if_needed(
        str(dummy_image_file), ModelTypes.UPSCALER_DRCT
    )
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

    resized_path_str = resize_image_if_needed(
        str(small_img_path), ModelTypes.UPSCALER_DRCT
    )
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
