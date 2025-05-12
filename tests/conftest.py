# this_file: tests/conftest.py
"""Pytest fixtures for twat-genai tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_fal_api_client(mocker):
    """Provides a mocked instance of FalApiClient."""
    # Mock the class itself
    mock_client_class = mocker.patch(
        "twat_genai.engines.fal.client.FalApiClient", autospec=True
    )

    # Create an instance of the mock class
    mock_instance = mock_client_class.return_value

    # Mock async methods used by FALEngine
    mock_instance.upload_image = AsyncMock(
        return_value="https://fake.fal.ai/uploaded_image.jpg"
    )
    mock_instance.process_upscale = AsyncMock(
        return_value=MagicMock()
    )  # Return dummy ImageResult mock
    mock_instance.process_outpaint = AsyncMock(
        return_value=MagicMock()
    )  # Return dummy ImageResult mock
    # Add mocks for process_tti, process_i2i etc. if FALEngine calls those directly later

    # Mock static method (if needed directly in tests)
    # We mock the class, so staticmethod should be mocked too, but can override
    # mocker.patch("twat_genai.engines.fal.client.FalApiClient.extract_image_info", return_value={...})

    return mock_instance


# Add other common fixtures here if needed
