"""Integration tests for the twat-genai package."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import asyncio

from twat_genai import FALEngine
from twat_genai.engines.fal.config import ModelTypes
from twat_genai.engines.base import EngineConfig
from twat_genai.core.config import ImageInput
from twat_genai.core.image import ImageSizes


@pytest.mark.asyncio
async def test_engine_initialization():
    """Test engine initialization."""
    output_dir = Path("/tmp/test_output")
    output_dir.mkdir(exist_ok=True)
    
    with patch('os.getenv') as mock_getenv:
        mock_getenv.return_value = "fake_api_key"
        
        engine = FALEngine(output_dir=output_dir)
        await engine.initialize()
        
        assert engine.output_dir == output_dir
        assert engine.api_key == "fake_api_key"


@pytest.mark.asyncio
async def test_engine_text_generation():
    """Test text-to-image generation setup."""
    output_dir = Path("/tmp/test_output")
    output_dir.mkdir(exist_ok=True)
    
    with patch('os.getenv') as mock_getenv:
        mock_getenv.return_value = "fake_api_key"
        
        engine = FALEngine(output_dir=output_dir)
        await engine.initialize()
        
        config = EngineConfig(num_inference_steps=20)
        
        # Just test that the config is properly set
        assert config.num_inference_steps == 20
        assert engine.api_key == "fake_api_key"


@pytest.mark.asyncio
async def test_engine_context_manager():
    """Test engine as context manager."""
    output_dir = Path("/tmp/test_output")
    output_dir.mkdir(exist_ok=True)
    
    with patch('os.getenv') as mock_getenv:
        mock_getenv.return_value = "fake_api_key"
        
        async with FALEngine(output_dir=output_dir) as engine:
            assert engine.output_dir == output_dir
            assert engine.api_key == "fake_api_key"


def test_model_types_enum():
    """Test ModelTypes enum has required values."""
    assert hasattr(ModelTypes, 'TEXT')
    assert hasattr(ModelTypes, 'IMAGE')
    assert hasattr(ModelTypes, 'UPSCALER_ESRGAN')
    assert hasattr(ModelTypes, 'OUTPAINT_BRIA')


def test_engine_config_validation():
    """Test engine configuration validation."""
    config = EngineConfig(
        num_inference_steps=25,
        guidance_scale=7.5,
        image_size=ImageSizes.SQ
    )
    
    assert config.num_inference_steps == 25
    assert config.guidance_scale == 7.5
    assert config.image_size == ImageSizes.SQ


def test_image_input_validation():
    """Test image input validation."""
    # Test URL input
    url_input = ImageInput(url="https://example.com/image.jpg")
    assert url_input.url == "https://example.com/image.jpg"
    
    # Test path input
    path_input = ImageInput(path=Path("/tmp/test.jpg"))
    assert path_input.path == Path("/tmp/test.jpg")