"""Tests for GeminiClient Flash 3.1 parameter handling."""
from unittest.mock import MagicMock
import pytest

from nanobanana_mcp_server.config.settings import (
    AuthMethod,
    Flash31ImageConfig,
    ServerConfig,
)
from nanobanana_mcp_server.services.gemini_client import GeminiClient


@pytest.fixture
def flash31_client():
    server_config = MagicMock(spec=ServerConfig)
    server_config.auth_method = AuthMethod.API_KEY
    server_config.gemini_api_key = "test-key"
    flash31_config = Flash31ImageConfig()
    return GeminiClient(server_config, flash31_config)


def test_filter_parameters_flash31_thinking(flash31_client):
    """Flash 3.1 should pass through thinking_level."""
    config = {"thinking_level": "high", "include_thoughts": True}
    filtered = flash31_client._filter_parameters(config)
    assert "thinking_level" in filtered
    assert filtered["thinking_level"] == "high"
    assert "include_thoughts" in filtered


def test_filter_parameters_flash31_grounding(flash31_client):
    """Flash 3.1 should pass through enable_grounding."""
    config = {"enable_grounding": True}
    filtered = flash31_client._filter_parameters(config)
    assert "enable_grounding" in filtered
