"""Tests for Flash 3.1 Image Service."""
from unittest.mock import MagicMock
import pytest

from nanobanana_mcp_server.config.settings import Flash31ImageConfig, ThinkingLevel
from nanobanana_mcp_server.services.flash31_image_service import Flash31ImageService


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    client.extract_images.return_value = [b"fake_image_data"]
    return client


@pytest.fixture
def flash31_config():
    return Flash31ImageConfig()


@pytest.fixture
def mock_storage_service():
    storage = MagicMock()
    stored = MagicMock()
    stored.id = "test-id"
    stored.full_path = "/tmp/test.png"
    stored.size_bytes = 1024
    stored.thumbnail_size_bytes = 256
    stored.width = 512
    stored.height = 512
    stored.expires_at = None
    storage.store_image.return_value = stored
    storage.get_thumbnail_base64.return_value = None
    return storage


@pytest.fixture
def service(mock_gemini_client, flash31_config, mock_storage_service):
    return Flash31ImageService(mock_gemini_client, flash31_config, mock_storage_service)


def test_service_init(service, flash31_config):
    assert service.config == flash31_config
    assert service.config.model_name == "gemini-3.1-flash-image-preview"


def test_generate_images_calls_client(service, mock_gemini_client):
    images, metadata = service.generate_images(
        prompt="A cat sitting on a mat",
        n=1,
        use_storage=False,
    )
    mock_gemini_client.generate_content.assert_called_once()
    assert len(images) == 1
    assert len(metadata) == 1
    assert metadata[0]["model"] == "gemini-3.1-flash-image-preview"
    assert metadata[0]["model_tier"] == "flash_31"


def test_generate_images_with_thinking(service, mock_gemini_client):
    images, metadata = service.generate_images(
        prompt="A complex scene",
        thinking_level=ThinkingLevel.HIGH,
        include_thoughts=True,
        use_storage=False,
    )
    assert metadata[0]["thinking_level"] == "high"
    assert metadata[0]["include_thoughts"] is True


def test_generate_images_with_grounding(service, mock_gemini_client):
    images, metadata = service.generate_images(
        prompt="A real building",
        enable_grounding=True,
        use_storage=False,
    )
    assert metadata[0]["grounding_enabled"] is True
