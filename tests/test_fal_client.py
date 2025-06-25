# this_file: tests/test_fal_client.py
"""Tests for the FAL API Client and its helpers."""

import pytest
from unittest.mock import AsyncMock, Mock
from pathlib import Path
from typing import Any, cast
from collections.abc import Callable

from twat_genai.engines.fal.client import FalApiClient
from twat_genai.engines.fal.config import ModelTypes
from twat_genai.core.config import ImageResult

# --- Tests for _extract_generic_image_info (Instance Method) ---


@pytest.mark.parametrize(
    "api_result, expected_info_list",
    [
        (
            {"image": {"url": "url1", "width": 10, "height": 10}},
            [
                {
                    "url": "url1", "width": 10, "height": 10, "content_type": "image/png",
                    "file_name": "output.png", "file_size": 0, "seed": None, "file_data": None,
                }
            ],
        ),
        (
            {
                "image": {"url": "url1", "width": 10, "height": 10, "seed": 123},
                "seed": 456,
            },
            [
                {
                    "url": "url1", "width": 10, "height": 10, "content_type": "image/png",
                    "file_name": "output.png", "file_size": 0, "seed": 123, "file_data": None,
                }
            ],
        ),
        (
            {"url": "url2", "seed": 789},
            [
                {
                    "url": "url2", "content_type": "image/png", "file_name": "output.png",
                    "file_size": 0, "width": None, "height": None, "seed": 789, "file_data": None,
                }
            ],
        ),
        (
            {
                "images": [
                    {"url": "url3", "width": 20},
                    {"url": "url4", "height": 30, "seed": 111},
                ]
            },
            [
                {
                    "url": "url3", "width": 20, "height": None, "content_type": "image/png",
                    "file_name": "output.png", "file_size": 0, "seed": None, "file_data": None,
                },
                {
                    "url": "url4", "width": None, "height": 30, "content_type": "image/png",
                    "file_name": "output.png", "file_size": 0, "seed": 111, "file_data": None,
                },
            ],
        ),
        (
            {"image": ["url5", "url6"]},
            [
                {
                    "url": "url5", "content_type": "image/png", "file_name": "output.png",
                    "file_size": 0, "width": None, "height": None, "seed": None, "file_data": None,
                },
                {
                    "url": "url6", "content_type": "image/png", "file_name": "output.png",
                    "file_size": 0, "width": None, "height": None, "seed": None, "file_data": None,
                },
            ],
        ),
    ],
)
def test_extract_generic_image_info_success(
    api_result: dict[str, Any], expected_info_list: list[dict[str, Any]]
) -> None:
    client = FalApiClient()
    extracted = client._extract_generic_image_info(api_result)
    assert extracted == expected_info_list

def test_extract_generic_image_info_no_image() -> None:
    client = FalApiClient()
    extracted = client._extract_generic_image_info({"some_other_key": "value"})
    assert extracted == []

def test_extract_generic_image_info_empty_result() -> None:
    client = FalApiClient()
    extracted = client._extract_generic_image_info({})
    assert extracted == []

def test_extract_generic_image_info_no_url_in_data() -> None:
    client = FalApiClient()
    extracted = client._extract_generic_image_info({"image": {"width": 10}})
    assert extracted == []

# --- Tests for upload_image ---

@pytest.mark.asyncio
async def test_upload_image_success(mocker: Mock) -> None:
    mock_upload = mocker.patch("fal_client.upload_file_async", new_callable=AsyncMock)
    mock_upload.return_value = "https://fake.fal.ai/uploaded.jpg"
    client = FalApiClient()
    fake_path = Path("fake/image.jpg")
    result_url = await client.upload_image(fake_path)
    mock_upload.assert_called_once_with(fake_path)
    assert result_url == "https://fake.fal.ai/uploaded.jpg"

@pytest.mark.asyncio
async def test_upload_image_failure(mocker: Mock) -> None:
    mock_upload = mocker.patch("fal_client.upload_file_async", new_callable=AsyncMock)
    mock_upload.side_effect = Exception("FAL upload failed")
    client = FalApiClient()
    fake_path = Path("fake/image.jpg")
    with pytest.raises(RuntimeError, match="Failed to upload image: FAL upload failed"):
        await client.upload_image(fake_path)
    mock_upload.assert_called_once_with(fake_path)

# --- Tests for _submit_fal_job ---

@pytest.mark.asyncio
async def test_submit_fal_job_success(mocker: Mock) -> None:
    mock_submit = mocker.patch("fal_client.submit_async", new_callable=AsyncMock)
    mock_handler = AsyncMock()
    mock_handler.request_id = "req-123"
    mock_submit.return_value = mock_handler
    from twat_genai.engines.fal.client import _submit_fal_job
    endpoint = "fal-ai/test-model"
    args = {"prompt": "test"}
    request_id = await _submit_fal_job(endpoint, args)
    mock_submit.assert_called_once_with(endpoint, arguments=args)
    assert request_id == "req-123"

# --- Tests for process_upscale ---

@pytest.mark.asyncio
async def test_process_upscale_success(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    upscaler_tool_name = "esrgan"
    model_type = ModelTypes.UPSCALER_ESRGAN
    image_url = "https://fake.fal.ai/input_upscale.jpg"
    kwargs = {
        "prompt": "enhance photo",
        "negative_prompt": "blurry",
        "scale": 4,
        "seed": 12345,
        "esrgan_model": "RealESRGAN_x4plus",
        "esrgan_tile": 0,
        "esrgan_face": False,
    }
    output_dir = Path("/tmp/test_upscale_out")
    request_id = "req-upscale-789"

    mock_submit_job.return_value = request_id
    expected_image_result = ImageResult(
        request_id=request_id, timestamp="ts_upscale", result={},
        image_info={"url": "fake_upscaled_url"}, original_prompt=None, job_params={}
    )
    configured_get_fal_result_mock = mock_get_result(mock_fal_client, return_value=expected_image_result)

    result = await mock_fal_client.process_upscale(
        upscaler=upscaler_tool_name,
        image_url=image_url,
        output_dir=output_dir,
        filename_suffix="upscale_test",
        filename_prefix="test_up",
        **kwargs,
    )

    actual_submit_call = mock_submit_job.call_args
    assert actual_submit_call[0][0] == model_type.value
    submitted_fal_args = actual_submit_call[0][1]
    assert submitted_fal_args["image_url"] == image_url
    assert submitted_fal_args["prompt"] == kwargs["prompt"]
    assert submitted_fal_args["esrgan_model"] == kwargs["esrgan_model"]
    assert submitted_fal_args["scale"] == kwargs["scale"]

    expected_job_params_for_get_result = {
        "model": model_type.value, "upscaler": upscaler_tool_name,
        "input_image_url": image_url, **kwargs,
    }
    configured_get_fal_result_mock.assert_called_once_with(
        request_id=request_id, model_endpoint=model_type.value, output_dir=output_dir,
        filename_suffix="upscale_test", filename_prefix="test_up",
        original_prompt=None, job_params=expected_job_params_for_get_result,
    )
    assert result == expected_image_result

@pytest.mark.asyncio
async def test_process_upscale_invalid_model(mock_fal_client: FalApiClient) -> None:
    with pytest.raises(ValueError, match="Invalid upscaler choice: text"):
        await mock_fal_client.process_upscale(
            upscaler="text",
            image_url="fake_url",
        )

@pytest.mark.asyncio
async def test_process_upscale_submit_failure(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock
) -> None:
    mock_submit_job.side_effect = Exception("Upscale submit failed")
    with pytest.raises(RuntimeError, match="Upscale process failed: Upscale submit failed"):
        await mock_fal_client.process_upscale(
            upscaler="drct", image_url="fake_url", prompt="test",
        )

@pytest.mark.asyncio
async def test_process_upscale_get_result_failure(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    request_id = "req-upscale-fail"
    upscaler_tool_name = "aura_sr"
    mock_submit_job.return_value = request_id
    configured_async_mock = mock_get_result(mock_fal_client, side_effect=Exception("Upscale get result failed"))
    with pytest.raises(RuntimeError, match="Upscale process failed: Upscale get result failed"):
        await mock_fal_client.process_upscale(
            upscaler=upscaler_tool_name, image_url="fake_url",
        )
    configured_async_mock.assert_called_once()

@pytest.mark.asyncio
async def test_process_upscale_ideogram_params(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    image_url = "https://fake.fal.ai/input.jpg"
    output_dir = Path("fake/output/dir")
    request_id = "req-123"
    model_type = ModelTypes.UPSCALER_IDEOGRAM
    upscaler_tool_name = "ideogram"

    mock_submit_job.return_value = request_id
    expected_image_result = ImageResult(
        request_id=request_id, timestamp="ts1", result={},
        image_info={"url": "fake_url"}, original_prompt=None,
    )
    configured_get_fal_result_mock = mock_get_result(mock_fal_client, return_value=expected_image_result)

    result = await mock_fal_client.process_upscale(
        image_url=image_url, output_dir=output_dir, upscaler=upscaler_tool_name,
        resemblance=0.8, detail=0.6, prompt="test prompt",
    )

    expected_fal_args = {
        "image_url": image_url, "resemblance": 0.8, "detail": 0.6, "prompt": "test prompt",
    }
    mock_submit_job.assert_called_once_with(ModelTypes.UPSCALER_IDEOGRAM.value, expected_fal_args)

    expected_job_params_for_get_result = {
        "model": ModelTypes.UPSCALER_IDEOGRAM.value, "upscaler": upscaler_tool_name,
        "input_image_url": image_url, "resemblance": 0.8, "detail": 0.6, "prompt": "test prompt",
    }
    configured_get_fal_result_mock.assert_called_once_with(
        request_id=request_id, model_endpoint=ModelTypes.UPSCALER_IDEOGRAM.value,
        output_dir=output_dir, filename_suffix=None, filename_prefix=None,
        original_prompt=None, job_params=expected_job_params_for_get_result
    )
    assert result == expected_image_result

@pytest.mark.asyncio
async def test_process_upscale_ccsr_params(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    upscaler_tool_name = "ccsr"
    model_type = ModelTypes.UPSCALER_CCSR
    image_url = "https://fake.fal.ai/input_upscale.jpg"
    kwargs = { "scale": 2, "steps": 50, "color_fix_type": "adain", "tile_diffusion": "mix",}
    request_id = "req-upscale-ccsr"

    mock_submit_job.return_value = request_id
    expected_image_result = ImageResult(
        request_id=request_id, timestamp="ts_upscale", result={},
        image_info={"url": "fake_upscaled_url"}, original_prompt=None,
    )
    configured_get_fal_result_mock = mock_get_result(mock_fal_client, return_value=expected_image_result)
    result = await mock_fal_client.process_upscale(
        upscaler=upscaler_tool_name, image_url=image_url, **kwargs,
    )
    expected_fal_args = {"image_url": image_url, **kwargs}
    mock_submit_job.assert_called_once_with(model_type.value, expected_fal_args)
    assert result == expected_image_result

@pytest.mark.asyncio
async def test_process_upscale_clarity_params(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    upscaler_tool_name = "clarity"
    model_type = ModelTypes.UPSCALER_CLARITY
    image_url = "https://fake.fal.ai/input_upscale.jpg"
    kwargs = {
        "scale": 2, "creativity": 0.35, "resemblance": 0.6,
        "prompt": "masterpiece, best quality, highres",
        "negative_prompt": "(worst quality, low quality, normal quality:2)",
    }
    request_id = "req-upscale-clarity"

    mock_submit_job.return_value = request_id
    expected_image_result = ImageResult( # Corrected variable name
        request_id=request_id, timestamp="ts_upscale", result={},
        image_info={"url": "fake_upscaled_url"}, original_prompt=None,
    )
    configured_get_fal_result_mock = mock_get_result(mock_fal_client, return_value=expected_image_result) # Use the new var

    result = await mock_fal_client.process_upscale( # Removed model_type from call
        upscaler=upscaler_tool_name,
        image_url=image_url,
        **kwargs,
    )

    expected_fal_args = {"image_url": image_url, **kwargs}
    mock_submit_job.assert_called_once_with(model_type.value, expected_fal_args)

    expected_job_params_for_get_result = {
        "model": model_type.value, "upscaler": upscaler_tool_name,
        "input_image_url": image_url, **kwargs,
    }
    configured_get_fal_result_mock.assert_called_once_with(
        request_id=request_id, model_endpoint=model_type.value, output_dir=None,
        filename_suffix=None, filename_prefix=None, original_prompt=None,
        job_params=expected_job_params_for_get_result,
    )
    assert result == expected_image_result

@pytest.mark.asyncio
async def test_process_upscale_recraft_params(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    upscaler_tool_name = "recraft_creative"
    model_type = ModelTypes.UPSCALER_RECRAFT_CREATIVE
    image_url = "https://fake.fal.ai/input_upscale.jpg"
    kwargs = { "sync_mode": True, "prompt": "enhance creative details",}
    request_id = "req-upscale-recraft"
    mock_submit_job.return_value = request_id
    expected_image_result = ImageResult(
        request_id=request_id, timestamp="ts_upscale", result={},
        image_info={"url": "fake_upscaled_url"}, original_prompt=None,
    )
    configured_get_fal_result_mock = mock_get_result(mock_fal_client, return_value=expected_image_result)
    result = await mock_fal_client.process_upscale(
        upscaler=upscaler_tool_name, image_url=image_url, **kwargs,
    )
    expected_fal_args = {"image_url": image_url, **kwargs}
    mock_submit_job.assert_called_once_with(model_type.value, expected_fal_args)
    expected_job_params_for_get_result = {
        "model": model_type.value, "upscaler": upscaler_tool_name,
        "input_image_url": image_url, **kwargs,
    }
    configured_get_fal_result_mock.assert_called_once_with(
        request_id=request_id, model_endpoint=model_type.value, output_dir=None,
        filename_suffix=None, filename_prefix=None, original_prompt=None,
        job_params=expected_job_params_for_get_result,
    )
    assert result == expected_image_result


@pytest.mark.asyncio
async def test_process_upscale_drct_params(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    upscaler_tool_name = "drct"
    model_type = ModelTypes.UPSCALER_DRCT
    image_url = "https://fake.fal.ai/input_upscale.jpg"
    kwargs = { "upscaling_factor": 4, "prompt": "enhance details",}
    request_id = "req-upscale-drct"
    mock_submit_job.return_value = request_id
    expected_image_result = ImageResult(
        request_id=request_id, timestamp="ts_upscale", result={},
        image_info={"url": "fake_upscaled_url"}, original_prompt=None,
    )
    configured_get_fal_result_mock = mock_get_result(mock_fal_client, return_value=expected_image_result)
    result = await mock_fal_client.process_upscale(
        upscaler=upscaler_tool_name, image_url=image_url, **kwargs,
    )
    expected_fal_args = {"image_url": image_url, **kwargs}
    mock_submit_job.assert_called_once_with(model_type.value, expected_fal_args)
    expected_job_params_for_get_result = {
        "model": model_type.value, "upscaler": upscaler_tool_name,
        "input_image_url": image_url, **kwargs,
    }
    configured_get_fal_result_mock.assert_called_once_with(
        request_id=request_id, model_endpoint=model_type.value, output_dir=None,
        filename_suffix=None, filename_prefix=None, original_prompt=None,
        job_params=expected_job_params_for_get_result,
    )
    assert result == expected_image_result

# --- Tests for process_i2i ---

@pytest.mark.asyncio
async def test_process_i2i_success(
    mock_fal_client: FalApiClient, mock_build_lora_arguments: AsyncMock,
    mock_submit_job: AsyncMock, mock_get_result: Callable,
) -> None:
    prompt = "test i2i prompt"
    image_url = "https://fake.fal.ai/input.jpg"
    lora_spec = None
    kwargs = {
        "image_size": "landscape_hd", "guidance_scale": 5.0, "num_inference_steps": 25,
        "strength": 0.8, "negative_prompt": "bad quality",
    }
    output_dir = Path("/tmp/test_i2i_out")
    mock_build_lora_arguments.return_value = ([], prompt)
    expected_image_result = ImageResult(
        request_id="req-test-456", timestamp="ts2", result={},
        image_info={"url": "fake_url_i2i"}, original_prompt=prompt, job_params={}
    )
    mock_submit_job.return_value = "req-test-456"
    configured_get_fal_result_mock = mock_get_result(mock_fal_client, return_value=expected_image_result)
    result = await mock_fal_client.process_i2i(
        prompt=prompt, image_url=image_url, lora_spec=lora_spec, output_dir=output_dir,
        filename_suffix="i2i_test", filename_prefix="test_i2i", **kwargs,
    )
    mock_build_lora_arguments.assert_called_once_with(lora_spec, prompt)
    expected_fal_args = {
        "loras": [], "prompt": prompt, "image_url": image_url, "strength": 0.8,
        "negative_prompt": "bad quality", "num_images": 1, "output_format": "jpeg",
        "enable_safety_checker": False, # "image_size": "landscape_hd", # This gets converted
        "width": 1024, "height": 768, # from "landscape_hd"
        "guidance_scale": 5.0, "num_inference_steps": 25,
    }
    # expected_fal_args.pop("image_size", None) # No longer needed as it's not added to fal_args if preset
    mock_submit_job.assert_called_once_with(ModelTypes.IMAGE.value, expected_fal_args)
    expected_job_params = {
        "model": ModelTypes.IMAGE.value, "prompt": prompt, "lora_spec": lora_spec,
        "input_image_url": image_url, "image_size": "landscape_hd", "guidance_scale": 5.0,
        "num_inference_steps": 25, "strength": 0.8, "negative_prompt": "bad quality",
    }
    configured_get_fal_result_mock.assert_called_once_with(
        request_id="req-test-456", model_endpoint=ModelTypes.IMAGE.value, output_dir=output_dir,
        filename_suffix="i2i_test", filename_prefix="test_i2i",
        original_prompt=prompt, job_params=expected_job_params,
    )
    assert result == expected_image_result

@pytest.mark.asyncio
async def test_process_i2i_missing_image_url(mock_fal_client: FalApiClient) -> None:
    with pytest.raises(ValueError, match="input_image is required for IMAGE mode"):
        await mock_fal_client._process_generic(
            model_type=ModelTypes.IMAGE, prompt="test", lora_spec=None, image_url=None,
        )

# --- Tests for process_outpaint ---

@pytest.mark.asyncio
async def test_process_outpaint_success(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    image_url = "https://fake.fal.ai/input.jpg"; prompt = "test prompt"
    output_dir = Path("fake/output/dir"); request_id = "req-outpaint-123"
    target_width = 1024; target_height = 1024; outpaint_tool = "bria"
    model_type = ModelTypes.OUTPAINT_BRIA
    # kwargs that process_outpaint might pass to job_params
    kwargs_for_job_params = {
        "lora_spec": None, "num_images": 1, "guidance_scale": None,
        "num_inference_steps": None, "enable_safety_checker": None, "strength": None, "mask_url": None,
        # Bria specific, but client.py process_outpaint passes them if they exist in its own kwargs
        "original_image_location": [ # This would be calculated if not passed
            (target_width - 512) // 2 if target_width > 512 else 0, # Assuming input 512 for calc
            (target_height - 512) // 2 if target_height > 512 else 0
        ],
        "original_image_size": [512,512] # Assuming input 512 for calc
    }


    mock_submit_job.return_value = request_id
    expected_image_result = ImageResult(
        request_id=request_id, timestamp="ts1", result={},
        image_info={"url": "fake_url_outpainted"}, original_prompt=prompt, job_params={}
    )
    configured_get_fal_result_mock = mock_get_result(mock_fal_client, return_value=expected_image_result)
    result = await mock_fal_client.process_outpaint(
        prompt=prompt, image_url=image_url, output_dir=output_dir,
        target_width=target_width, target_height=target_height, outpaint_tool=outpaint_tool, num_images=1,
        # Passing original_image_location and original_image_size as process_outpaint expects them if bria
        original_image_location=kwargs_for_job_params["original_image_location"],
        original_image_size=kwargs_for_job_params["original_image_size"]
    )
    actual_submit_call = mock_submit_job.call_args
    assert actual_submit_call[0][0] == model_type.value
    submitted_fal_args = actual_submit_call[0][1]
    assert submitted_fal_args["image_url"] == image_url; assert submitted_fal_args["prompt"] == prompt
    assert submitted_fal_args["target_width"] == target_width; assert submitted_fal_args["target_height"] == target_height
    assert submitted_fal_args["canvas_size"] == [target_width, target_height]
    assert submitted_fal_args["num_outputs"] == 1
    assert submitted_fal_args["original_image_location"] == kwargs_for_job_params["original_image_location"]
    assert submitted_fal_args["original_image_size"] == kwargs_for_job_params["original_image_size"]


    expected_job_params_for_get_result = {
        "model": model_type.value, "outpaint_tool": outpaint_tool, "prompt": prompt,
        "input_image_url": image_url, "target_width": target_width, "target_height": target_height,
        "canvas_size": [target_width, target_height],
        **kwargs_for_job_params
    }

    expected_job_params_for_get_result_filtered = {
        k: v for k, v in expected_job_params_for_get_result.items() if v is not None
    }
    # Remove num_images from the job_params dict if it's 1, as the client might not add it if it's the default
    if expected_job_params_for_get_result_filtered.get("num_images") == 1:
         pass # num_images is passed to FAL as num_outputs, but might not be in job_params if default

    configured_get_fal_result_mock.assert_called_once_with(
        request_id=request_id, model_endpoint=model_type.value,
        output_dir=output_dir, # Fixed: output_dir should be passed
        filename_suffix=None, filename_prefix=None, original_prompt=prompt,
        job_params=expected_job_params_for_get_result_filtered,
    )
    assert result == expected_image_result


async def test_process_outpaint_submit_failure(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock
) -> None:
    mock_submit_job.side_effect = Exception("Outpaint submit failed")
    prompt = "test"; image_url = "fake_url"; target_width = 1024; target_height = 1024
    with pytest.raises(RuntimeError, match="Outpaint process failed: Outpaint submit failed" ):
        await mock_fal_client.process_outpaint(
            prompt=prompt, image_url=image_url, target_width=target_width, target_height=target_height
        )

@pytest.mark.asyncio
async def test_process_outpaint_get_result_failure(
    mock_fal_client: FalApiClient, mock_submit_job: AsyncMock, mock_get_result: Callable
) -> None:
    request_id = "req-outpaint-fail"; prompt = "test"; image_url = "fake_url"
    target_width = 1024; target_height = 1024
    mock_submit_job.return_value = request_id
    configured_async_mock = mock_get_result(mock_fal_client, side_effect=Exception("Outpaint get result failed"))
    with pytest.raises(RuntimeError, match="Outpaint process failed: Outpaint get result failed" ):
        await mock_fal_client.process_outpaint(
            prompt=prompt, image_url=image_url, target_width=target_width, target_height=target_height
        )
    configured_async_mock.assert_called_once()

# --- Mocks for process_* tests ---

@pytest.fixture
def mock_fal_client(mocker: Mock) -> FalApiClient:
    client = FalApiClient()
    mocker.patch.object(client, "_get_fal_result", new_callable=AsyncMock)
    mocker.patch("twat_genai.engines.fal.client._submit_fal_job", new_callable=AsyncMock)
    mocker.patch("twat_genai.engines.fal.lora.build_lora_arguments", new_callable=AsyncMock)
    mocker.patch("twat_genai.engines.fal.client._download_image_helper", new_callable=AsyncMock)
    return client

@pytest.fixture
def mock_build_lora_arguments(mocker: Mock) -> AsyncMock:
    mock = mocker.patch("twat_genai.engines.fal.lora.build_lora_arguments", new_callable=AsyncMock)
    mock.return_value = ([], "test prompt")
    return mock

@pytest.fixture
def mock_submit_job() -> AsyncMock:
    return AsyncMock()

@pytest.fixture
def mock_get_result(mocker: Mock) -> Mock:
    def _configurator(client_instance: FalApiClient, return_value: ImageResult | None = None, side_effect: Exception | None = None) -> AsyncMock:
        async_mock = mocker.patch.object(client_instance, "_get_fal_result", new_callable=AsyncMock)
        if side_effect: async_mock.side_effect = side_effect
        elif return_value: async_mock.return_value = return_value
        else: async_mock.return_value = ImageResult(request_id="default_req", timestamp="default_ts", result={}, image_info={"url": "default_url"})
        return async_mock
    return _configurator

# --- Tests for process_tti ---

@pytest.mark.asyncio
async def test_process_tti_success(
    mock_fal_client: FalApiClient, mock_build_lora_arguments: AsyncMock,
    mock_submit_job: AsyncMock, mock_get_result: Callable,
) -> None:
    prompt = "test prompt"; lora_spec = None
    kwargs = {"image_size": "square_hd", "guidance_scale": 7.0, "num_inference_steps": 30,}
    output_dir = Path("/tmp/test_out")
    mock_build_lora_arguments.return_value = ([], prompt)
    expected_image_result = ImageResult(
        request_id="req-test-123", timestamp="ts", result={},
        image_info={"url": "fake_url"}, original_prompt=prompt, job_params={}
    )
    mock_submit_job.return_value = "req-test-123"
    configured_get_fal_result_mock = mock_get_result(mock_fal_client, return_value=expected_image_result)
    result = await mock_fal_client.process_tti(
        prompt=prompt, lora_spec=lora_spec, output_dir=output_dir,
        filename_suffix="tti_test", filename_prefix="test", **kwargs,
    )
    mock_build_lora_arguments.assert_called_once_with(lora_spec, prompt)
    expected_fal_args = {
        "loras": [], "prompt": prompt, "num_images": 1, "output_format": "jpeg",
        "enable_safety_checker": False, "width": 1024, "height": 1024,
        "guidance_scale": 7.0, "num_inference_steps": 30,
    }
    mock_submit_job.assert_called_once_with(ModelTypes.TEXT.value, expected_fal_args)
    expected_job_params = {
        "model": ModelTypes.TEXT.value, "prompt": prompt, "lora_spec": lora_spec,
        "image_size": "square_hd", "guidance_scale": 7.0, "num_inference_steps": 30,
        "input_image_url": None,
    }
    configured_get_fal_result_mock.assert_called_once_with(
        request_id="req-test-123", model_endpoint=ModelTypes.TEXT.value, output_dir=output_dir,
        filename_suffix="tti_test", filename_prefix="test",
        original_prompt=prompt, job_params=expected_job_params,
    )
    assert result == expected_image_result

@pytest.mark.asyncio
async def test_process_tti_lora_failure(
    mock_fal_client: FalApiClient, mock_build_lora_arguments: AsyncMock
) -> None:
    mock_build_lora_arguments.side_effect = Exception("LoRA build error")
    with pytest.raises(RuntimeError, match="Processing TEXT job failed: LoRA build error"):
        await mock_fal_client.process_tti(prompt="test", lora_spec="invalid")

@pytest.mark.asyncio
async def test_process_tti_submit_failure(
    mock_fal_client: FalApiClient, mock_build_lora_arguments: AsyncMock, mock_submit_job: AsyncMock,
) -> None:
    mock_build_lora_arguments.return_value = ([], "test")
    mock_submit_job.side_effect = Exception("Submit failed")
    with pytest.raises(RuntimeError, match="Processing TEXT job failed: Submit failed"):
        await mock_fal_client.process_tti(prompt="test", lora_spec=None)

@pytest.mark.asyncio
async def test_process_tti_get_result_failure(
    mock_fal_client: FalApiClient, mock_build_lora_arguments: AsyncMock,
    mock_submit_job: AsyncMock, mock_get_result: Callable,
) -> None:
    prompt = "test"; lora_spec = None; request_id = "req-tti-fail"
    mock_build_lora_arguments.return_value = ([], prompt)
    mock_submit_job.return_value = request_id
    configured_async_mock = mock_get_result(mock_fal_client, side_effect=Exception("Get result failed"))
    with pytest.raises(RuntimeError, match="Processing TEXT job failed: Get result failed"):
        await mock_fal_client.process_tti(prompt=prompt, lora_spec=lora_spec)
    configured_async_mock.assert_called_once()
