#!/usr/bin/env -S uv run
# /// script
# dependencies = ["pytest", "pytest-asyncio", "twat-genai"]
# ///
# this_file: tests/test_cli.py
"""Tests for the command-line interface."""

import pytest
from unittest.mock import MagicMock, AsyncMock, Mock
from pathlib import Path

# Module to test
from twat_genai.cli import TwatGenAiCLI # Import the class
from twat_genai.engines.fal.config import ModelTypes
from twat_genai.engines.base import EngineConfig
from twat_genai.core.config import ImageSizeWH
from twat_genai.core.image import ImageSizes

# --- Fixtures ---


@pytest.fixture
def mock_async_main(mocker: Mock) -> AsyncMock:
    """Mocks the core async_main function."""
    return mocker.patch("twat_genai.cli.async_main", new_callable=AsyncMock)


@pytest.fixture
def mock_fal_engine(mocker: Mock) -> MagicMock:
    """Mocks the FALEngine class and its methods."""
    mock_engine_class = mocker.patch("twat_genai.cli.FALEngine", autospec=True) # Patch where it's used by CLI
    mock_instance = mock_engine_class.return_value
    mock_instance.generate = AsyncMock()
    # Mock the async context manager methods
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock()
    return mock_engine_class # Return the mocked class


# --- Test Cases ---


@pytest.mark.asyncio
async def test_cli_text_basic(mock_fal_engine: MagicMock) -> None:
    """Test basic text-to-image call."""
    prompts = "a test prompt"
    output_dir = "test_output"

    # Instantiate CLI and call the 'text' method
    cli_instance = TwatGenAiCLI(output_dir=output_dir)
    cli_instance.text(prompts=prompts)

    # Assert FALEngine was initialized and called
    mock_fal_engine.assert_called_once_with(Path(output_dir).resolve()) # FALEngine now resolves path
    mock_fal_engine.return_value.__aenter__.assert_awaited_once()

    # Check the arguments passed to engine.generate
    # The generate method is on the instance returned by the mocked class
    generate_mock = mock_fal_engine.return_value.generate
    call_args, call_kwargs = generate_mock.await_args
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
async def test_cli_text_with_options(mock_fal_engine: MagicMock) -> None:
    """Test text-to-image call with various options."""
    prompts = ["prompt 1", "prompt 2"]
    output_dir = Path("custom_dir")
    image_size = "1024,768"
    gs = 5.0
    steps = 40
    lora = "style1:0.8"
    suffix = "_abc"
    prefix = "xyz_"
    neg = "ugly"

    cli_module(
        prompts=prompts,
        output_dir=output_dir,
        model="text",
        image_size=image_size,
        guidance_scale=gs,
        num_inference_steps=steps,
        lora=lora,
        filename_suffix=suffix,
        filename_prefix=prefix,
        negative_prompt=neg,  # Note: neg prompt goes into image_config for img models, ignored here?
        verbose=True,  # Test verbose flag
    )

    # Assert generate called twice (once per prompt)
    generate_mock = mock_fal_engine.return_value.generate
    assert generate_mock.await_count == 2

    # Check the arguments of the first call
    call_args, call_kwargs = generate_mock.await_args_list[0]
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
    call_args_2, _ = generate_mock.await_args_list[1]
    assert call_args_2[0] == prompts[1]

    # FALEngine is instantiated once for the CLI instance, context manager entered once per call to _run_generation
    # Since _run_generation is called once (handling multiple prompts internally), __aenter__ and __aexit__ called once.
    mock_fal_engine.assert_called_once() # Class instantiation
    mock_fal_engine.return_value.__aenter__.assert_awaited_once()
    mock_fal_engine.return_value.__aexit__.assert_awaited_once()


def test_cli_missing_prompt_for_text() -> None:
    """Test error when prompts are missing for text model."""
    cli_instance = TwatGenAiCLI()
    with pytest.raises(ValueError, match="Prompts are required for text generation."):
        cli_instance.text(prompts="")
    with pytest.raises(ValueError, match="Prompts are required for text generation."):
        cli_instance.text(prompts=[])


def test_cli_invalid_image_size() -> None:
    """Test error handling for invalid image_size format."""
    with pytest.raises(ValueError, match="image_size must be one of"):
        TwatGenAiCLI(image_size="invalid_size").text(prompts="test")
    with pytest.raises(ValueError, match="For custom image sizes use"):
        TwatGenAiCLI(image_size="100x200").text(prompts="test")
    with pytest.raises(ValueError, match="For custom image sizes use"):
        TwatGenAiCLI(image_size="100,abc").text(prompts="test")


# TODO: Add tests for image, canny, depth models
# TODO: Add tests for upscale model with various upscale args
# TODO: Add tests for outpaint model
# TODO: Add tests for input_image parsing (URL, existing path, non-existing path)
# TODO: Add tests for invalid model name
