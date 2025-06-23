#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pytest", "pytest-asyncio", "twat-genai"]
# ///
# this_file: tests/test_cli.py
"""Tests for the command-line interface."""

import pytest
from unittest.mock import MagicMock, AsyncMock, Mock
from pathlib import Path

# Import the class to test
from twat_genai.cli import TwatGenAiCLI, parse_image_size # Keep parse_image_size if used directly by tests
from twat_genai.engines.fal.config import ModelTypes
from twat_genai.engines.base import EngineConfig
from twat_genai.core.config import ImageSizeWH, ImageResult # Added ImageResult
from twat_genai.core.image import ImageSizes

# --- Fixtures ---


@pytest.fixture
def mock_async_main(mocker: Mock) -> AsyncMock:
    """Mocks the core async_main function."""
    return mocker.patch("twat_genai.cli.async_main", new_callable=AsyncMock)


@pytest.fixture
def mock_fal_engine(mocker: Mock) -> MagicMock:
    """Mocks the FALEngine class and its methods."""
    # Patch the class FALEngine
    mock_engine_class = mocker.patch("twat_genai.engines.fal.FALEngine")

    # This is the instance that will be returned when FALEngine() is called
    mock_instance = mock_engine_class.return_value

    # Mock methods needed by the CLI tests on this instance
    mock_instance.initialize = AsyncMock()
    # Make generate an AsyncMock that returns a dummy ImageResult-like object
    dummy_image_result = MagicMock(spec=ImageResult) # Create a mock that looks like ImageResult
    dummy_image_result.image_info = {} # Ensure image_info attribute exists
    mock_instance.generate = AsyncMock(return_value=dummy_image_result)

    # Mock the async context manager behavior
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock()

    # Patch deeper calls. _submit_fal_job is an async function.
    # FalApiClient._get_fal_result is an async method.
    mocker.patch("twat_genai.engines.fal.client._submit_fal_job", AsyncMock(return_value="dummy_request_id"))
    mocker.patch("twat_genai.engines.fal.client.FalApiClient._get_fal_result", AsyncMock(return_value=dummy_image_result))

    return mock_instance


# --- Test Cases ---


@pytest.mark.asyncio
async def test_cli_text_basic(mock_fal_engine: MagicMock, mocker: Mock) -> None:
    """Test basic text-to-image call."""
    mocker.patch("os.getenv", return_value="fake_fal_key")  # Mock FAL_KEY
    prompts = "a test prompt"
    output_dir = "test_output"
    shared_params = {"output_dir": output_dir, "verbose": True}
    cli_instance = TwatGenAiCLI(**shared_params)

    # Call the cli function directly (it orchestrates calls to engine)
    await cli_instance.text(prompts=prompts) # model="text" is implicit in the method

    # Assert FALEngine was initialized and called
    # The mock_fal_engine fixture returns the *instance* of the mocked FALEngine class
    # __aenter__ is part of the async context manager protocol
    mock_fal_engine.__aenter__.assert_awaited_once()

    # Check the arguments passed to engine.generate
    # We expect cli() to create configs and call generate
    # The first positional arg is prompt, the second is config
    call_args, call_kwargs = mock_fal_engine.generate.await_args
    assert call_args[0] == prompts
    assert isinstance(call_args[1], EngineConfig)
    # Check kwargs passed to generate
    assert call_kwargs.get("model") == ModelTypes.TEXT
    assert call_kwargs.get("filename_suffix") is None
    assert call_kwargs.get("filename_prefix") is None
    assert call_kwargs.get("image_config") is None
    assert call_kwargs.get("upscale_config") is None
    assert call_kwargs.get("outpaint_config") is None
    assert call_kwargs.get("lora_spec") is None

    # Check config values (defaults)
    engine_config: EngineConfig = call_args[1]
    assert engine_config.guidance_scale == 3.5
    assert engine_config.num_inference_steps == 28
    assert engine_config.image_size == ImageSizes.SQ

    mock_fal_engine.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_cli_text_with_options(mock_fal_engine: MagicMock, mocker: Mock) -> None:
    """Test text-to-image call with various options."""
    mocker.patch("os.getenv", return_value="fake_fal_key")  # Mock FAL_KEY
    prompts = ["prompt 1", "prompt 2"]
    output_dir = Path("custom_dir")
    image_size = "1024,768"
    gs = 5.0
    steps = 40
    lora = "style1:0.8"
    suffix = "_abc"
    prefix = "xyz_"
    neg = "ugly"

    cli_instance = TwatGenAiCLI(
        output_dir=output_dir,
        image_size=image_size,
        guidance_scale=gs,
        num_inference_steps=steps,
        lora=lora,
        filename_suffix=suffix,
        filename_prefix=prefix,
        negative_prompt=neg,
        verbose=True,
    )
    await cli_instance.text(prompts=prompts)

    # Assert generate called twice (once per prompt)
    assert mock_fal_engine.generate.await_count == 2

    # Check the arguments of the first call
    call_args, call_kwargs = mock_fal_engine.generate.await_args_list[0]
    assert call_args[0] == prompts[0]
    assert isinstance(call_args[1], EngineConfig)
    assert call_kwargs.get("model") == ModelTypes.TEXT
    assert call_kwargs.get("filename_suffix") == suffix
    assert call_kwargs.get("filename_prefix") == prefix
    assert call_kwargs.get("lora_spec") == lora

    # Check config values for the first call
    engine_config: EngineConfig = call_args[1]
    assert engine_config.guidance_scale == gs
    assert engine_config.num_inference_steps == steps
    assert isinstance(engine_config.image_size, ImageSizeWH)
    assert engine_config.image_size.width == 1024
    assert engine_config.image_size.height == 768

    # Check second call prompt
    call_args_2, _ = mock_fal_engine.generate.await_args_list[1]
    assert call_args_2[0] == prompts[1]

    mock_fal_engine.__aenter__.assert_awaited_once()
    mock_fal_engine.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_cli_missing_prompt_for_text() -> None:
    """Test error when prompts are missing for text model."""
    cli_instance = TwatGenAiCLI() # Default params are fine
    # The error is raised synchronously before any await
    with pytest.raises(ValueError, match="Prompts are required for text generation."):
        await cli_instance.text(prompts="") # await is still needed as it's an async method
    with pytest.raises(ValueError, match="Prompts are required for text generation."):
        await cli_instance.text(prompts=[])


def test_cli_invalid_image_size() -> None:
    """Test error handling for invalid image_size format."""
    # Expect ValueError directly from parse_image_size during TwatGenAiCLI.__init__
    with pytest.raises(ValueError, match="image_size must be one of: .* or in 'width,height' format."):
        TwatGenAiCLI(image_size="invalid_size")
    with pytest.raises(ValueError, match="image_size must be one of: .* or in 'width,height' format."):
        TwatGenAiCLI(image_size="100x200")
    with pytest.raises(ValueError, match="For custom image sizes use 'width,height' with integers."):
        TwatGenAiCLI(image_size="100,abc")

    # Test a valid custom size to ensure it doesn't raise
    try:
        TwatGenAiCLI(image_size="100,200")
    except ValueError:
        pytest.fail("Valid custom image_size '100,200' unexpectedly raised ValueError")

    # Test valid ImageSizes enum name
    try:
        TwatGenAiCLI(image_size="SQ")
    except ValueError:
        pytest.fail("Valid ImageSizes enum name 'SQ' unexpectedly raised ValueError")


# TODO: Add tests for image, canny, depth models
# TODO: Add tests for upscale model with various upscale args
# TODO: Add tests for outpaint model
# TODO: Add tests for input_image parsing (URL, existing path, non-existing path)
# TODO: Add tests for invalid model name
