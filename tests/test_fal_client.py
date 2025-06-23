# this_file: tests/test_fal_client.py
"""Tests for the FAL API Client and its helpers."""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock # Added MagicMock
from pathlib import Path
from typing import Any, cast
from collections.abc import Callable

from twat_genai.core.config import ImageResult
from twat_genai.engines.fal.client import FalApiClient
from twat_genai.engines.fal.config import ModelTypes

# --- Tests for _extract_generic_image_info (Instance Method) ---


@pytest.mark.parametrize(
    "api_result, expected_info_list",
    [
        # Single image dict
        (
            {"image": {"url": "url1", "width": 10, "height": 10}},
            [
                {
                    "url": "url1",
                    "width": 10,
                    "height": 10,
                    "content_type": "image/png",
                    "file_name": "output.png",
                    "file_size": 0,
                    "seed": None,
                    "file_data": None,
                }
            ],
        ),
        # Single image dict with seed
        (
            {
                "image": {"url": "url1", "width": 10, "height": 10, "seed": 123},
                "seed": 456,
            },
            [
                {
                    "url": "url1",
                    "width": 10,
                    "height": 10,
                    "content_type": "image/png",
                    "file_name": "output.png",
                    "file_size": 0,
                    "seed": 123,
                    "file_data": None,
                }
            ],
        ),
        # Single image URL string with top-level seed
        (
            {"url": "url2", "seed": 789},
            [
                {
                    "url": "url2",
                    "content_type": "image/png",
                    "file_name": "output.png",
                    "file_size": 0,
                    "width": None,
                    "height": None,
                    "seed": 789,
                    "file_data": None,
                }
            ],
        ),
        # List of image dicts
        (
            {
                "images": [
                    {"url": "url3", "width": 20},
                    {"url": "url4", "height": 30, "seed": 111},
                ]
            },
            [
                {
                    "url": "url3",
                    "width": 20,
                    "height": None,
                    "content_type": "image/png",
                    "file_name": "output.png",
                    "file_size": 0,
                    "seed": None,
                    "file_data": None,
                },
                {
                    "url": "url4",
                    "width": None,
                    "height": 30,
                    "content_type": "image/png",
                    "file_name": "output.png",
                    "file_size": 0,
                    "seed": 111,
                    "file_data": None,
                },
            ],
        ),
        # List of image URL strings (seed extraction not possible here)
        (
            {"image": ["url5", "url6"]},
            [
                {
                    "url": "url5",
                    "content_type": "image/png",
                    "file_name": "output.png",
                    "file_size": 0,
                    "width": None,
                    "height": None,
                    "seed": None,
                    "file_data": None,
                },
                {
                    "url": "url6",
                    "content_type": "image/png",
                    "file_name": "output.png",
                    "file_size": 0,
                    "width": None,
                    "height": None,
                    "seed": None,
                    "file_data": None,
                },
            ],
        ),
    ],
)
def test_extract_generic_image_info_success(
    api_result: dict[str, Any], expected_info_list: list[dict[str, Any]]
) -> None:
    """Test successful extraction of image info from various result formats."""
    client = FalApiClient()
    extracted = client._extract_generic_image_info(api_result)
    assert extracted == expected_info_list


def test_extract_generic_image_info_no_image() -> None:
    """Test extraction when no image/url key is present."""
    client = FalApiClient()
    extracted = client._extract_generic_image_info({"some_other_key": "value"})
    assert extracted == []  # Should return empty list on failure


def test_extract_generic_image_info_empty_result() -> None:
    """Test extraction with an empty result dict."""
    client = FalApiClient()
    extracted = client._extract_generic_image_info({})
    assert extracted == []


def test_extract_generic_image_info_no_url_in_data() -> None:
    """Test extraction when image data lacks a URL."""
    client = FalApiClient()
    extracted = client._extract_generic_image_info({"image": {"width": 10}})
    assert extracted == []  # Expect empty list if no URL found


# --- Tests for upload_image ---


@pytest.mark.asyncio
async def test_upload_image_success(mocker: Mock) -> None:
    """Test successful image upload."""
    mock_upload = mocker.patch("fal_client.upload_file_async", new_callable=AsyncMock)
    mock_upload.return_value = "https://fake.fal.ai/uploaded.jpg"

    client = FalApiClient()
    fake_path = Path("fake/image.jpg")
    # We don't need the file to exist for this mocked test

    result_url = await client.upload_image(fake_path)

    mock_upload.assert_called_once_with(fake_path)
    assert result_url == "https://fake.fal.ai/uploaded.jpg"


@pytest.mark.asyncio
async def test_upload_image_failure(mocker: Mock) -> None:
    """Test image upload failure."""
    mock_upload = mocker.patch("fal_client.upload_file_async", new_callable=AsyncMock)
    mock_upload.side_effect = Exception("FAL upload failed")

    client = FalApiClient()
    fake_path = Path("fake/image.jpg")

    with pytest.raises(RuntimeError, match="Failed to upload image: FAL upload failed"):
        await client.upload_image(fake_path)

    mock_upload.assert_called_once_with(fake_path)


# --- Tests for _download_image_helper ---
# (Moved to test_image_utils.py as it's a general helper now)

# --- Tests for _submit_fal_job ---


@pytest.mark.asyncio
async def test_submit_fal_job_success(mocker: Mock) -> None:
    """Test successful job submission."""
    mock_submit = mocker.patch("fal_client.submit_async", new_callable=AsyncMock)
    mock_handler = AsyncMock()
    mock_handler.request_id = "req-123"
    mock_submit.return_value = mock_handler

    # Import within function scope if needed or ensure it's globally available
    try:
        from twat_genai.engines.fal.client import _submit_fal_job
    except ImportError:
        # Handle case where it might be moved into the class
        # This part depends on the final location of _submit_fal_job
        msg = "_submit_fal_job helper function not found"
        raise AssertionError(msg)

    endpoint = "fal-ai/test-model"
    args = {"prompt": "test"}
    request_id = await _submit_fal_job(endpoint, args)

    mock_submit.assert_called_once_with(endpoint, arguments=args)
    assert request_id == "req-123"


# --- Tests for _get_fal_result ---
# TODO: Add tests for _get_fal_result (complex due to polling, saving, parsing)
# Requires mocking fal_client.status_async, result_async, _download_image_helper

# --- Tests for process_upscale ---


@pytest.mark.asyncio
async def test_process_upscale_success(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    """Test successful process_upscale call with various params."""
    model_type = ModelTypes.UPSCALER_ESRGAN
    image_url = "https://fake.fal.ai/input_upscale.jpg"
    kwargs = {
        "prompt": "enhance photo",
        "negative_prompt": "blurry",
        "scale": 4,
        "seed": 12345,
        "esrgan_model": "RealESRGAN_x4plus",
        "esrgan_tile": 0,
        "esrgan_face": False,  # Example boolean
        # Add other relevant kwargs for the model if needed
    }
    output_dir = Path("/tmp/test_upscale_out")
    request_id = "req-upscale-789"

    # Mock submit and result
    mock_submit_job.return_value = request_id
    expected_result = ImageResult(
        request_id=request_id,
        timestamp="ts_upscale",
        result={},
        image_info={"url": "fake_upscaled_url"},
        original_prompt=kwargs["prompt"],
    )
    mock_get_result(mock_fal_client, expected_result)

    # Call method
    result = await mock_fal_client.process_upscale(
        model_type=model_type,
        image_url=image_url,
        output_dir=output_dir,
        filename_suffix="upscale_test",
        filename_prefix="test_up",
        **kwargs,
    )

    # Assertions
    expected_fal_args = {"image_url": image_url, **kwargs}
    # Remove None values if any were added implicitly (though unlikely here)
    expected_fal_args = {k: v for k, v in expected_fal_args.items() if v is not None}
    mock_submit_job.assert_called_once_with(model_type.value, expected_fal_args)

    expected_job_params = {
        "model": model_type.value,
        "input_image_url": image_url,
        **kwargs,
    }
    mock_fal_client._get_fal_result.assert_called_once_with(
        request_id=request_id,
        model_endpoint=model_type.value,
        output_dir=output_dir,
        filename_suffix="upscale_test",
        filename_prefix="test_up",
        original_prompt=kwargs["prompt"],
        job_params=expected_job_params,
    )
    assert result == expected_result


@pytest.mark.asyncio
async def test_process_upscale_invalid_model(mock_fal_client: FalApiClient) -> None:
    """Test process_upscale failure with non-upscaler model."""
    with pytest.raises(ValueError, match="Invalid model type for upscale"):
        await mock_fal_client.process_upscale(
            model_type=ModelTypes.TEXT,  # Invalid type
            image_url="fake_url",
        )


@pytest.mark.asyncio
async def test_process_upscale_submit_failure(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock
) -> None:
    """Test process_upscale failure during job submission."""
    mock_submit_job.side_effect = Exception("Upscale submit failed")

    with pytest.raises(
        RuntimeError, match="Upscale process failed: Upscale submit failed"
    ):
        await mock_fal_client.process_upscale(
            model_type=ModelTypes.UPSCALER_DRCT,
            image_url="fake_url",
            prompt="test",
        )


@pytest.mark.asyncio
async def test_process_upscale_get_result_failure(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    """Test process_upscale failure during result fetching."""
    request_id = "req-upscale-fail"
    mock_submit_job.return_value = request_id
    mock_get_result(mock_fal_client, None)  # Setup mock on instance
    mock_fal_client._get_fal_result.side_effect = Exception("Upscale get result failed")

    with pytest.raises(
        RuntimeError, match="Upscale process failed: Upscale get result failed"
    ):
        await mock_fal_client.process_upscale(
            model_type=ModelTypes.UPSCALER_AURA_SR,
            image_url="fake_url",
        )


@pytest.mark.asyncio
async def test_process_upscale_ideogram_params(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Mock
) -> None:
    """Test upscaling with Ideogram-specific parameters."""
    image_url = "https://fake.fal.ai/input.jpg"
    output_dir = Path("fake/output/dir")
    request_id = "req-123"
    model_type = ModelTypes.UPSCALE_IDEOGRAM

    # Mock submit and result
    mock = cast(AsyncMock, mock_submit_job)
    mock.return_value = request_id
    expected_result = ImageResult(
        request_id=request_id,
        timestamp="ts1",
        result={},
        image_info={"url": "fake_url"},
    )
    mock_get_result.return_value = expected_result

    # Call method
    result = await mock_fal_client.process_upscale(
        image_url=image_url,
        output_dir=output_dir,
        model_type=model_type,
        resemblance=0.8,
        detail=0.6,
        expand_prompt="test prompt",
    )

    # Assertions
    expected_fal_args = {
        "image_url": image_url,
        "resemblance": 0.8,
        "detail": 0.6,
        "expand_prompt": "test prompt",
    }
    mock.assert_called_once_with(model_type.value, expected_fal_args)
    assert result == expected_result


@pytest.mark.asyncio
async def test_process_upscale_ccsr_params(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    """Test process_upscale with CCSR-specific parameters."""
    model_type = ModelTypes.UPSCALER_CCSR
    image_url = "https://fake.fal.ai/input_upscale.jpg"
    kwargs = {
        "scale": 2,  # CCSR-specific
        "steps": 50,  # CCSR-specific
        "color_fix_type": "adain",  # CCSR-specific
        "tile_diffusion": True,  # CCSR-specific
    }
    request_id = "req-upscale-ccsr"

    # Mock submit and result
    mock_submit_job.return_value = request_id
    expected_result = ImageResult(
        request_id=request_id,
        timestamp="ts_upscale",
        result={},
        image_info={"url": "fake_upscaled_url"},
    )
    mock_get_result(mock_fal_client, expected_result)

    # Call method
    result = await mock_fal_client.process_upscale(
        model_type=model_type,
        image_url=image_url,
        **kwargs,
    )

    # Assertions
    expected_fal_args = {"image_url": image_url, **kwargs}
    mock_submit_job.assert_called_once_with(model_type.value, expected_fal_args)
    assert result == expected_result


@pytest.mark.asyncio
async def test_process_upscale_clarity_params(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    """Test process_upscale with Clarity/Aura SR-specific parameters."""
    model_type = ModelTypes.UPSCALER_CLARITY
    image_url = "https://fake.fal.ai/input_upscale.jpg"
    kwargs = {
        "scale": 2,
        "creativity": 0.35,  # Clarity-specific
        "resemblance": 0.6,  # Clarity-specific
        "prompt": "masterpiece, best quality, highres",
        "negative_prompt": "(worst quality, low quality, normal quality:2)",
    }
    request_id = "req-upscale-clarity"

    # Mock submit and result
    mock_submit_job.return_value = request_id
    expected_result = ImageResult(
        request_id=request_id,
        timestamp="ts_upscale",
        result={},
        image_info={"url": "fake_upscaled_url"},
        original_prompt=kwargs["prompt"],
    )
    mock_get_result(mock_fal_client, expected_result)

    # Call method
    result = await mock_fal_client.process_upscale(
        model_type=model_type,
        image_url=image_url,
        **kwargs,
    )

    # Assertions
    expected_fal_args = {"image_url": image_url, **kwargs}
    mock_submit_job.assert_called_once_with(model_type.value, expected_fal_args)
    assert result == expected_result


@pytest.mark.asyncio
async def test_process_upscale_recraft_params(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    """Test process_upscale with Recraft-specific parameters."""
    model_type = ModelTypes.UPSCALER_RECRAFT_CREATIVE
    image_url = "https://fake.fal.ai/input_upscale.jpg"
    kwargs = {
        "sync_mode": True,  # Recraft-specific
        "prompt": "enhance creative details",
    }
    request_id = "req-upscale-recraft"

    # Mock submit and result
    mock_submit_job.return_value = request_id
    expected_result = ImageResult(
        request_id=request_id,
        timestamp="ts_upscale",
        result={},
        image_info={"url": "fake_upscaled_url"},
        original_prompt=kwargs["prompt"],
    )
    mock_get_result(mock_fal_client, expected_result)

    # Call method
    result = await mock_fal_client.process_upscale(
        model_type=model_type,
        image_url=image_url,
        **kwargs,
    )

    # Assertions
    expected_fal_args = {"image_url": image_url, **kwargs}
    mock_submit_job.assert_called_once_with(model_type.value, expected_fal_args)
    assert result == expected_result


@pytest.mark.asyncio
async def test_process_upscale_drct_params(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    """Test process_upscale with DRCT-specific parameters."""
    model_type = ModelTypes.UPSCALER_DRCT
    image_url = "https://fake.fal.ai/input_upscale.jpg"
    kwargs = {
        "upscaling_factor": 4,  # DRCT-specific
        "prompt": "enhance details",
    }
    request_id = "req-upscale-drct"

    # Mock submit and result
    mock_submit_job.return_value = request_id
    expected_result = ImageResult(
        request_id=request_id,
        timestamp="ts_upscale",
        result={},
        image_info={"url": "fake_upscaled_url"},
        original_prompt=kwargs["prompt"],
    )
    mock_get_result(mock_fal_client, expected_result)

    # Call method
    result = await mock_fal_client.process_upscale(
        model_type=model_type,
        image_url=image_url,
        **kwargs,
    )

    # Assertions
    expected_fal_args = {"image_url": image_url, **kwargs}
    mock_submit_job.assert_called_once_with(model_type.value, expected_fal_args)
    assert result == expected_result


# --- Tests for process_i2i ---


@pytest.mark.asyncio
async def test_process_i2i_success(
    mock_fal_client: FalApiClient,
    mock_build_lora_arguments: AsyncMock,
    mock_submit_job: AsyncMock,
    mock_get_result: Callable,
) -> None:
    """Test successful process_i2i call."""
    prompt = "test i2i prompt"
    image_url = "https://fake.fal.ai/input.jpg"
    lora_spec = None
    kwargs = {
        "image_size": "landscape_hd",
        "guidance_scale": 5.0,
        "num_inference_steps": 25,
        "strength": 0.8,
        "negative_prompt": "bad quality",
    }
    output_dir = Path("/tmp/test_i2i_out")

    # Mock LoRA return
    mock_build_lora_arguments.return_value = ([], prompt)
    # Mock result return
    expected_result = ImageResult(
        request_id="req-test-456",
        timestamp="ts2",
        result={},
        image_info={"url": "fake_url_i2i"},
        original_prompt=prompt,
    )
    mock_submit_job.return_value = "req-test-456"
    mock_get_result(mock_fal_client, expected_result)

    # Call method
    result = await mock_fal_client.process_i2i(
        prompt=prompt,
        image_url=image_url,
        lora_spec=lora_spec,
        output_dir=output_dir,
        filename_suffix="i2i_test",
        filename_prefix="test_i2i",
        **kwargs,
    )

    # Assertions
    mock_build_lora_arguments.assert_called_once_with(lora_spec, prompt)
    expected_fal_args = {
        "loras": [],
        "prompt": prompt,
        "image_url": image_url,
        "strength": 0.8,
        "negative_prompt": "bad quality",
        "num_images": 1,
        "output_format": "jpeg",
        "enable_safety_checker": False,
        "image_size": "landscape_hd",
        "guidance_scale": 5.0,
        "num_inference_steps": 25,
    }
    mock_submit_job.assert_called_once_with(ModelTypes.IMAGE.value, expected_fal_args)

    expected_job_params = {
        "model": ModelTypes.IMAGE.value,
        "prompt": prompt,
        "lora_spec": lora_spec,
        "input_image_url": image_url,
        **kwargs,
    }
    mock_fal_client._get_fal_result.assert_called_once_with(
        request_id="req-test-456",
        model_endpoint=ModelTypes.IMAGE.value,
        output_dir=output_dir,
        filename_suffix="i2i_test",
        filename_prefix="test_i2i",
        original_prompt=prompt,
        job_params=expected_job_params,
    )
    assert result == expected_result


@pytest.mark.asyncio
async def test_process_i2i_missing_image_url(mock_fal_client: FalApiClient) -> None:
    """Test process_i2i failure when image_url is missing (should not happen via _process_generic check)."""
    # This tests the internal check within _process_generic
    with pytest.raises(ValueError, match="Image URL is required for model type IMAGE"):
        # We call _process_generic directly here to test its internal validation
        await mock_fal_client._process_generic(
            model_type=ModelTypes.IMAGE,
            prompt="test",
            lora_spec=None,
            image_url=None,  # Explicitly pass None here
        )


# --- Tests for process_canny / process_depth ---
# (Similar structure to process_i2i, potentially combine or add specific checks)
# TODO: Add tests for process_canny, process_depth

# --- Tests for process_outpaint ---


@pytest.mark.asyncio
async def test_process_outpaint_success(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Mock
) -> None:
    """Test successful outpainting."""
    image_url = "https://fake.fal.ai/input.jpg"
    output_dir = Path("fake/output/dir")
    request_id = "req-123"

    # Mock submit and result
    mock = cast(AsyncMock, mock_submit_job)
    mock.return_value = request_id
    expected_result = ImageResult(
        request_id=request_id,
        timestamp="ts1",
        result={},
        image_info={"url": "fake_url"},
    )
    mock_get_result.return_value = expected_result

    # Call method
    result = await mock_fal_client.process_outpaint(
        image_url=image_url, output_dir=output_dir, prompt="test prompt"
    )

    # Assertions
    expected_fal_args = {"image_url": image_url, "prompt": "test prompt"}
    mock.assert_called_once_with("outpaint", expected_fal_args)
    assert result == expected_result


@pytest.mark.asyncio
async def test_process_outpaint_missing_image_url(
    mock_fal_client: FalApiClient,
) -> None:
    """Test process_outpaint failure when image_url is missing in kwargs."""
    kwargs_no_url = {"prompt": "test"}
    with pytest.raises(ValueError, match="Missing required argument 'image_url'"):
        await mock_fal_client.process_outpaint(**kwargs_no_url)


@pytest.mark.asyncio
async def test_process_outpaint_submit_failure(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock
) -> None:
    """Test process_outpaint failure during job submission."""
    mock_submit_job.side_effect = Exception("Outpaint submit failed")
    kwargs = {"image_url": "fake_url", "prompt": "test"}

    with pytest.raises(
        RuntimeError, match="Outpainting process failed: Outpaint submit failed"
    ):
        await mock_fal_client.process_outpaint(**kwargs)


@pytest.mark.asyncio
async def test_process_outpaint_get_result_failure(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    """Test process_outpaint failure during result fetching."""
    request_id = "req-outpaint-fail"
    kwargs = {"image_url": "fake_url", "prompt": "test"}
    mock_submit_job.return_value = request_id
    mock_get_result(mock_fal_client, None)
    mock_fal_client._get_fal_result.side_effect = Exception(
        "Outpaint get result failed"
    )

    with pytest.raises(
        RuntimeError, match="Outpainting process failed: Outpaint get result failed"
    ):
        await mock_fal_client.process_outpaint(**kwargs)


# --- Mocks for process_* tests ---


@pytest.fixture
def mock_fal_client(mocker: Mock) -> FalApiClient:
    """Provides a FalApiClient instance with mocked internal methods."""
    client = FalApiClient()
    # Mock the helper methods that interact with the actual fal_client library or file system
    mocker.patch.object(client, "_get_fal_result", new_callable=AsyncMock)
    # Mock _submit_fal_job (even though it's top-level, process_* calls it)
    # TODO: Move _submit_fal_job into the class?
    mocker.patch(
        "twat_genai.engines.fal.client._submit_fal_job", new_callable=AsyncMock
    )
    # Mock LoRA building
    mocker.patch(
        "twat_genai.engines.fal.lora.build_lora_arguments", new_callable=AsyncMock
    )
    # Mock download helper used by _get_fal_result
    mocker.patch(
        "twat_genai.engines.fal.client._download_image_helper", new_callable=AsyncMock
    )
    return client


@pytest.fixture
def mock_build_lora_arguments(mocker: Mock) -> MagicMock: # Changed to MagicMock
    """Fixture to mock build_lora_arguments."""
    mock = mocker.patch(
        "twat_genai.engines.fal.lora.build_lora_arguments", new_callable=MagicMock # Changed to MagicMock
    )
    # Default return: no loras, original prompt
    mock.return_value = ([], "test prompt")
    return mock


@pytest.fixture
def mock_submit_job() -> AsyncMock:
    """Mock for _submit_fal_job."""
    return AsyncMock()


@pytest.fixture
def mock_get_result() -> Mock:
    """Mock for _get_fal_result."""
    mock = Mock()

    def _mock_get_result(client: FalApiClient, result: ImageResult) -> None:
        client._get_fal_result = Mock(return_value=result)

    mock.side_effect = _mock_get_result
    return mock


# --- Tests for process_tti ---


@pytest.mark.asyncio
async def test_process_tti_success(
    mock_fal_client: FalApiClient,  # Use the client with mocks
    mock_build_lora_arguments: AsyncMock,
    mock_submit_job: AsyncMock,
    mock_get_result: Callable,
) -> None:
    """Test successful process_tti call."""
    prompt = "test prompt"
    lora_spec = None
    kwargs = {
        "image_size": "square_hd",
        "guidance_scale": 7.0,
        "num_inference_steps": 30,
    }
    output_dir = Path("/tmp/test_out")

    # Setup mock return value for _get_fal_result
    expected_result = ImageResult(
        request_id="req-test-123",
        timestamp="ts",
        result={},
        image_info={"url": "fake_url"},  # Minimal valid info
        original_prompt=prompt,
    )
    mock_get_result(mock_fal_client, expected_result)

    # Call the method under test
    result = await mock_fal_client.process_tti(
        prompt=prompt,
        lora_spec=lora_spec,
        output_dir=output_dir,
        filename_suffix="tti_test",
        filename_prefix="test",
        **kwargs,
    )

    # Assertions
    mock_build_lora_arguments.assert_called_once_with(lora_spec, prompt)
    expected_fal_args = {
        "loras": [],
        "prompt": prompt,
        "num_images": 1,
        "output_format": "jpeg",
        "enable_safety_checker": False,
        "image_size": "square_hd",
        "guidance_scale": 7.0,
        "num_inference_steps": 30,
    }
    mock_submit_job.assert_called_once_with(ModelTypes.TEXT.value, expected_fal_args)

    expected_job_params = {
        "model": ModelTypes.TEXT.value,
        "prompt": prompt,
        "lora_spec": lora_spec,
        **kwargs,
    }
    mock_fal_client._get_fal_result.assert_called_once_with(
        request_id="req-test-123",
        model_endpoint=ModelTypes.TEXT.value,
        output_dir=output_dir,
        filename_suffix="tti_test",
        filename_prefix="test",
        original_prompt=prompt,
        job_params=expected_job_params,
    )
    assert result == expected_result


@pytest.mark.asyncio
async def test_process_tti_lora_failure(
    mock_fal_client: FalApiClient, mock_build_lora_arguments: AsyncMock
) -> None:
    """Test process_tti failure during LoRA building."""
    mock_build_lora_arguments.side_effect = Exception("LoRA build error")

    with pytest.raises(
        RuntimeError, match="Failed to build LoRA arguments: LoRA build error"
    ):
        await mock_fal_client.process_tti(prompt="test", lora_spec="invalid")


@pytest.mark.asyncio
async def test_process_tti_submit_failure(
    mock_fal_client: FalApiClient,
    mock_build_lora_arguments: AsyncMock,
    mock_submit_job: AsyncMock,
) -> None:
    """Test process_tti failure during job submission."""
    mock_submit_job.side_effect = Exception("Submit failed")

    with pytest.raises(RuntimeError, match="Generic process failed: Submit failed"):
        await mock_fal_client.process_tti(prompt="test", lora_spec=None)


@pytest.mark.asyncio
async def test_process_tti_get_result_failure(
    mock_fal_client: FalApiClient,
    mock_build_lora_arguments: AsyncMock,
    mock_submit_job: AsyncMock,
    mock_get_result: Callable,
) -> None:
    """Test process_tti failure during result fetching."""
    # Setup _get_fal_result mock to raise an error
    mock_get_result(mock_fal_client, None)  # Need instance
    mock_fal_client._get_fal_result.side_effect = Exception("Get result failed")

    with pytest.raises(RuntimeError, match="Generic process failed: Get result failed"):
        await mock_fal_client.process_tti(prompt="test", lora_spec=None)
