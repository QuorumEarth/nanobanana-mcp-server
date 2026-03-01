"""Tests for Flash 3.1 configuration."""
import pytest
from nanobanana_mcp_server.config.settings import (
    Flash31ImageConfig,
    ModelTier,
    ThinkingLevel,
)


def test_model_tier_flash_31_exists():
    tier = ModelTier.FLASH_31
    assert tier.value == "flash_31"


def test_thinking_level_minimal_exists():
    level = ThinkingLevel.MINIMAL
    assert level.value == "minimal"


def test_flash31_config_defaults():
    config = Flash31ImageConfig()
    assert config.model_name == "gemini-3.1-flash-image-preview"
    assert config.max_resolution == 1024
    assert config.supports_thinking is True
    assert config.supports_grounding is True
    assert config.supports_media_resolution is False
    assert config.max_input_images == 14
    assert config.default_thinking_level == ThinkingLevel.MINIMAL
    assert config.enable_search_grounding is False
    assert config.request_timeout == 75
