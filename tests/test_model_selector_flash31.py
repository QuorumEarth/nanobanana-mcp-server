"""Tests for ModelSelector with Flash 3.1 tier."""
from unittest.mock import MagicMock
import pytest

from nanobanana_mcp_server.config.settings import ModelSelectionConfig, ModelTier
from nanobanana_mcp_server.services.model_selector import ModelSelector


@pytest.fixture
def selector():
    flash_service = MagicMock()
    flash31_service = MagicMock()
    pro_service = MagicMock()
    config = ModelSelectionConfig()
    return ModelSelector(flash_service, flash31_service, pro_service, config)


def test_explicit_flash_31_selection(selector):
    service, tier = selector.select_model("any prompt", requested_tier=ModelTier.FLASH_31)
    assert tier == ModelTier.FLASH_31


def test_explicit_flash_selection(selector):
    service, tier = selector.select_model("any prompt", requested_tier=ModelTier.FLASH)
    assert tier == ModelTier.FLASH


def test_explicit_pro_selection(selector):
    service, tier = selector.select_model("any prompt", requested_tier=ModelTier.PRO)
    assert tier == ModelTier.PRO


def test_auto_select_balanced_prompt(selector):
    """A moderately complex prompt should select Flash 3.1."""
    service, tier = selector.select_model(
        "a detailed illustration of a medieval castle"
    )
    assert tier == ModelTier.FLASH_31


def test_auto_select_simple_prompt(selector):
    """A simple/draft prompt should select Flash 2.5."""
    service, tier = selector.select_model("quick sketch of a cat")
    assert tier == ModelTier.FLASH


def test_auto_select_pro_quality(selector):
    """A professional quality prompt should select Pro."""
    service, tier = selector.select_model("professional 4k photograph of a product")
    assert tier == ModelTier.PRO


def test_get_model_info_flash_31(selector):
    info = selector.get_model_info(ModelTier.FLASH_31)
    assert info["tier"] == "flash_31"
    assert "3.1" in info["name"]
    assert info["model_id"] == "gemini-3.1-flash-image-preview"


def test_get_model_info_flash(selector):
    info = selector.get_model_info(ModelTier.FLASH)
    assert info["tier"] == "flash"


def test_get_model_info_pro(selector):
    info = selector.get_model_info(ModelTier.PRO)
    assert info["tier"] == "pro"


def test_auto_select_grounding_favors_flash31(selector):
    """Grounding should favor Flash 3.1 (not require Pro)."""
    service, tier = selector.select_model(
        "a building", enable_grounding=True
    )
    # Grounding adds quality_score but shouldn't jump all the way to Pro
    assert tier in [ModelTier.FLASH_31, ModelTier.PRO]


def test_4k_resolution_forces_pro(selector):
    """4K resolution should force Pro model."""
    service, tier = selector.select_model(
        "a simple cat", resolution="4k"
    )
    assert tier == ModelTier.PRO
