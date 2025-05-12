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
from twat_genai import cli as cli_module

# from twat_genai import cli # New import
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
    mock_engine_class = mocker.patch("twat_genai.engines.fal.FALEngine", autospec=True)
    mock_instance = mock_engine_class.return_value
    mock_instance.generate = AsyncMock()
    # Mock the async context manager methods
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock()
    return mock_instance  # Return the mocked instance


# --- Test Cases ---


@pytest.mark.asyncio
async def test_cli_text_basic(mock_fal_engine: MagicMock) -> None:
    """Test basic text-to-image call."""
    prompts = "a test prompt"
    output_dir = "test_output"

    # Call the cli function directly (it orchestrates calls to engine)
    cli_module(prompts=prompts, output_dir=output_dir, model="text")

    # Assert FALEngine was initialized and called
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


def test_cli_missing_prompt_for_text() -> None:
    """Test error when prompts are missing for text model."""
    with pytest.raises(ValueError, match="Prompts are required for text model."):
        cli_module(model="text", prompts="")
    with pytest.raises(ValueError, match="Prompts are required for text model."):
        cli_module(model="text", prompts=[])


def test_cli_invalid_image_size() -> None:
    """Test error handling for invalid image_size format."""
    # Expect SystemExit because the main cli function catches ValueError and exits
    with pytest.raises(SystemExit):
        cli_module(prompts="test", image_size="invalid_size")
    with pytest.raises(SystemExit):
        cli_module(prompts="test", image_size="100x200")
    with pytest.raises(SystemExit):
        cli_module(prompts="test", image_size="100,abc")


# TODO: Add tests for image, canny, depth models
# TODO: Add tests for upscale model with various upscale args
# TODO: Add tests for outpaint model
# TODO: Add tests for input_image parsing (URL, existing path, non-existing path)
# TODO: Add tests for invalid model name
