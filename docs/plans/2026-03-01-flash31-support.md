# Gemini 3.1 Flash Image Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `gemini-3.1-flash-image-preview` model support with all new features (512px resolution, thinking levels, Google Search grounding, expanded aspect ratios, up to 14 reference images) as a new `FLASH_31` model tier.

**Architecture:** New `FLASH_31` model tier sits between existing `FLASH` (2.5) and `PRO` tiers. A new `Flash31ImageService` follows the `ProImageService` pattern. The `ModelSelector` gains three-tier auto-selection with a balanced score for the middle tier. The `GeminiClient` is updated to handle Flash 3.1-specific parameters (grounding tools, thinking config, 512px resolution).

**Tech Stack:** Python 3.11+, google-genai >= 1.57.0, FastMCP, Pydantic, pytest

---

### Task 1: Update Configuration Enums and Add Flash31ImageConfig

**Files:**
- Modify: `nanobanana_mcp_server/config/settings.py`
- Test: `tests/test_config_flash31.py`

**Step 1: Write the failing test**

```python
# tests/test_config_flash31.py
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
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/chriskantrowitz/Library/Application Support/Claude/mcp-servers/nanobanana-mcp-server" && python -m pytest tests/test_config_flash31.py -v`
Expected: FAIL with ImportError (Flash31ImageConfig doesn't exist yet)

**Step 3: Implement changes to settings.py**

In `nanobanana_mcp_server/config/settings.py`:

Add `FLASH_31` to `ModelTier` enum:
```python
class ModelTier(str, Enum):
    """Model selection options."""
    FLASH = "flash"        # Speed-optimized (Gemini 2.5 Flash)
    FLASH_31 = "flash_31"  # Balanced (Gemini 3.1 Flash) - NEW
    PRO = "pro"            # Quality-optimized (Gemini 3 Pro)
    AUTO = "auto"          # Automatic selection
```

Add `MINIMAL` to `ThinkingLevel` enum:
```python
class ThinkingLevel(str, Enum):
    """Gemini thinking levels for advanced reasoning."""
    MINIMAL = "minimal"  # Minimal reasoning (Flash 3.1 default)
    LOW = "low"          # Less reasoning
    HIGH = "high"        # Maximum reasoning (Pro default)
```

Add new `Flash31ImageConfig` dataclass after `FlashImageConfig`:
```python
@dataclass
class Flash31ImageConfig(BaseModelConfig):
    """Gemini 3.1 Flash Image configuration (balanced speed + features)."""
    model_name: str = "gemini-3.1-flash-image-preview"
    max_resolution: int = 1024
    default_thinking_level: ThinkingLevel = ThinkingLevel.MINIMAL
    supports_thinking: bool = True
    supports_grounding: bool = True
    supports_media_resolution: bool = False
    enable_search_grounding: bool = False
    max_input_images: int = 14
    request_timeout: int = 75
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/chriskantrowitz/Library/Application Support/Claude/mcp-servers/nanobanana-mcp-server" && python -m pytest tests/test_config_flash31.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nanobanana_mcp_server/config/settings.py tests/test_config_flash31.py
git commit -m "feat: add Flash 3.1 model tier, MINIMAL thinking level, and Flash31ImageConfig"
```

---

### Task 2: Update Constants for Flash 3.1

**Files:**
- Modify: `nanobanana_mcp_server/config/constants.py`

**Step 1: Update constants**

In `nanobanana_mcp_server/config/constants.py`:

Add Flash 3.1 specific constants:
```python
# Max input images per model
MAX_INPUT_IMAGES = 3           # Legacy default (Flash 2.5)
MAX_INPUT_IMAGES_FLASH_31 = 14  # Flash 3.1 supports up to 14

# Flash 3.1 expanded aspect ratios (in addition to standard ones)
FLASH_31_ASPECT_RATIOS = [
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
    "9:16", "16:9", "21:9",
    "1:4", "4:1", "1:8", "8:1",  # New in Flash 3.1
]
```

**Step 2: Commit**

```bash
git add nanobanana_mcp_server/config/constants.py
git commit -m "feat: add Flash 3.1 constants for max input images and aspect ratios"
```

---

### Task 3: Create Flash31ImageService

**Files:**
- Create: `nanobanana_mcp_server/services/flash31_image_service.py`
- Test: `tests/test_flash31_image_service.py`

**Step 1: Write the failing test**

```python
# tests/test_flash31_image_service.py
"""Tests for Flash 3.1 Image Service."""
from unittest.mock import MagicMock, patch
import pytest

from nanobanana_mcp_server.config.settings import Flash31ImageConfig, ThinkingLevel
from nanobanana_mcp_server.services.flash31_image_service import Flash31ImageService


@pytest.fixture
def mock_gemini_client():
    client = MagicMock()
    # Mock extract_images to return fake image bytes
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
    storage.get_thumbnail_base64.return_value = None  # No thumbnail
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
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/chriskantrowitz/Library/Application Support/Claude/mcp-servers/nanobanana-mcp-server" && python -m pytest tests/test_flash31_image_service.py -v`
Expected: FAIL (module doesn't exist)

**Step 3: Create Flash31ImageService**

Create `nanobanana_mcp_server/services/flash31_image_service.py`:

```python
"""Gemini 3.1 Flash Image service for balanced speed and quality generation."""

import base64
import logging
from typing import Any

from fastmcp.utilities.types import Image as MCPImage

from ..config.settings import Flash31ImageConfig, ThinkingLevel
from ..core.progress_tracker import ProgressContext
from ..utils.image_utils import validate_image_format
from .gemini_client import GeminiClient
from .image_storage_service import ImageStorageService


class Flash31ImageService:
    """Service for image generation using Gemini 3.1 Flash Image model.

    Balanced between Flash 2.5 (speed) and Pro (quality), with features:
    - Thinking/reasoning support (minimal/high)
    - Google Search grounding
    - Up to 14 reference images
    - Expanded aspect ratios (1:4, 4:1, 1:8, 8:1)
    - 512px resolution option
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        config: Flash31ImageConfig,
        storage_service: ImageStorageService | None = None,
    ):
        self.gemini_client = gemini_client
        self.config = config
        self.storage_service = storage_service
        self.logger = logging.getLogger(__name__)

    def generate_images(
        self,
        prompt: str,
        n: int = 1,
        resolution: str | None = None,
        thinking_level: ThinkingLevel | None = None,
        include_thoughts: bool = False,
        enable_grounding: bool | None = None,
        negative_prompt: str | None = None,
        system_instruction: str | None = None,
        input_images: list[tuple[str, str]] | None = None,
        aspect_ratio: str | None = None,
        use_storage: bool = True,
    ) -> tuple[list[MCPImage], list[dict[str, Any]]]:
        """
        Generate images using Gemini 3.1 Flash Image.

        Args:
            prompt: Main generation prompt
            n: Number of images to generate
            resolution: Output resolution ('high', '1k', '512px')
            thinking_level: Reasoning depth (MINIMAL or HIGH)
            include_thoughts: Include model's reasoning in response
            enable_grounding: Enable Google Search grounding
            negative_prompt: Things to avoid
            system_instruction: Optional system-level guidance
            input_images: List of (base64, mime_type) tuples for conditioning
            aspect_ratio: Output aspect ratio
            use_storage: Store images and return resource links with thumbnails

        Returns:
            Tuple of (image_blocks_or_resource_links, metadata_list)
        """
        if thinking_level is None:
            thinking_level = self.config.default_thinking_level
        if enable_grounding is None:
            enable_grounding = self.config.enable_search_grounding

        with ProgressContext(
            "flash31_image_generation",
            f"Generating {n} image(s) with Gemini 3.1 Flash...",
            {"prompt": prompt[:100], "count": n}
        ) as progress:
            progress.update(5, "Configuring Flash 3.1 parameters...")

            self.logger.info(
                f"Flash 3.1 generation: prompt='{prompt[:50]}...', n={n}, "
                f"thinking={thinking_level.value}, grounding={enable_grounding}"
            )

            progress.update(10, "Preparing generation request...")

            contents = []

            if system_instruction:
                contents.append(system_instruction)

            enhanced_prompt = self._enhance_prompt(prompt, negative_prompt)
            contents.append(enhanced_prompt)

            # Add input images (Flash 3.1 supports up to 14)
            if input_images:
                images_b64, mime_types = zip(*input_images, strict=False)
                image_parts = self.gemini_client.create_image_parts(
                    list(images_b64), list(mime_types)
                )
                contents = image_parts + contents

            progress.update(20, "Sending requests to Gemini 3.1 Flash API...")

            all_images = []
            all_metadata = []

            for i in range(n):
                try:
                    progress.update(
                        20 + (i * 70 // n),
                        f"Generating image {i + 1}/{n}..."
                    )

                    gen_config = {
                        "thinking_level": thinking_level.value,
                        "include_thoughts": include_thoughts,
                        "enable_grounding": enable_grounding,
                    }

                    if resolution:
                        gen_config["resolution"] = resolution

                    response = self.gemini_client.generate_content(
                        contents,
                        config=gen_config,
                        aspect_ratio=aspect_ratio,
                    )
                    images = self.gemini_client.extract_images(response)

                    for j, image_bytes in enumerate(images):
                        metadata = {
                            "model": self.config.model_name,
                            "model_tier": "flash_31",
                            "response_index": i + 1,
                            "image_index": j + 1,
                            "thinking_level": thinking_level.value,
                            "include_thoughts": include_thoughts,
                            "grounding_enabled": enable_grounding,
                            "mime_type": f"image/{self.config.default_image_format}",
                            "prompt": prompt,
                            "enhanced_prompt": enhanced_prompt,
                            "negative_prompt": negative_prompt,
                            "aspect_ratio": aspect_ratio,
                        }

                        if resolution:
                            metadata["resolution"] = resolution

                        if use_storage and self.storage_service:
                            stored_info = self.storage_service.store_image(
                                image_bytes,
                                f"image/{self.config.default_image_format}",
                                metadata
                            )

                            thumbnail_b64 = self.storage_service.get_thumbnail_base64(
                                stored_info.id
                            )
                            if thumbnail_b64:
                                thumbnail_bytes = base64.b64decode(thumbnail_b64)
                                thumbnail_image = MCPImage(data=thumbnail_bytes, format="jpeg")
                                all_images.append(thumbnail_image)

                            metadata.update({
                                "storage_id": stored_info.id,
                                "full_image_uri": f"file://images/{stored_info.id}",
                                "full_path": stored_info.full_path,
                                "thumbnail_uri": f"file://images/{stored_info.id}/thumbnail",
                                "size_bytes": stored_info.size_bytes,
                                "thumbnail_size_bytes": stored_info.thumbnail_size_bytes,
                                "width": stored_info.width,
                                "height": stored_info.height,
                                "expires_at": stored_info.expires_at,
                                "is_stored": True,
                            })

                            all_metadata.append(metadata)

                            self.logger.info(
                                f"Generated Flash 3.1 image {i + 1}.{j + 1} - "
                                f"stored as {stored_info.id} "
                                f"({stored_info.size_bytes} bytes, {stored_info.width}x{stored_info.height})"
                            )
                        else:
                            mcp_image = MCPImage(
                                data=image_bytes,
                                format=self.config.default_image_format
                            )
                            all_images.append(mcp_image)
                            all_metadata.append(metadata)

                            self.logger.info(
                                f"Generated Flash 3.1 image {i + 1}.{j + 1} "
                                f"(size: {len(image_bytes)} bytes)"
                            )

                except Exception as e:
                    self.logger.error(f"Failed to generate Flash 3.1 image {i + 1}: {e}")
                    raise

            progress.update(100, f"Generated {len(all_images)} image(s)")

            if not all_images:
                self.logger.warning("No images were generated by Flash 3.1 model")

            return all_images, all_metadata

    def edit_image(
        self,
        instruction: str,
        base_image_b64: str,
        mime_type: str = "image/png",
        thinking_level: ThinkingLevel | None = None,
        include_thoughts: bool = False,
        use_storage: bool = True,
    ) -> tuple[list[MCPImage], int]:
        """
        Edit images with Flash 3.1 model.

        Args:
            instruction: Natural language editing instruction
            base_image_b64: Base64 encoded source image
            mime_type: MIME type of source image
            thinking_level: Reasoning depth
            include_thoughts: Include model's reasoning
            use_storage: Store edited images

        Returns:
            Tuple of (edited_images, count)
        """
        if thinking_level is None:
            thinking_level = self.config.default_thinking_level

        with ProgressContext(
            "flash31_image_editing",
            "Editing image with Gemini 3.1 Flash...",
            {"instruction": instruction[:100]}
        ) as progress:
            try:
                progress.update(10, "Configuring Flash 3.1 editing parameters...")

                self.logger.info(
                    f"Flash 3.1 edit: instruction='{instruction[:50]}...', "
                    f"thinking={thinking_level.value}"
                )

                validate_image_format(mime_type)

                progress.update(20, "Preparing edit request...")

                enhanced_instruction = (
                    f"{instruction}\n\n"
                    "Maintain the original image's quality and style. "
                    "Make precise edits as described."
                )

                image_parts = self.gemini_client.create_image_parts(
                    [base_image_b64], [mime_type]
                )
                contents = [*image_parts, enhanced_instruction]

                progress.update(40, "Sending edit request to Gemini 3.1 Flash API...")

                gen_config = {
                    "thinking_level": thinking_level.value,
                    "include_thoughts": include_thoughts,
                }

                response = self.gemini_client.generate_content(
                    contents,
                    config=gen_config
                )
                image_bytes_list = self.gemini_client.extract_images(response)

                progress.update(70, "Processing edited images...")

                mcp_images = []
                for i, image_bytes in enumerate(image_bytes_list):
                    metadata = {
                        "model": self.config.model_name,
                        "model_tier": "flash_31",
                        "instruction": instruction,
                        "thinking_level": thinking_level.value,
                        "source_mime_type": mime_type,
                        "result_mime_type": f"image/{self.config.default_image_format}",
                        "edit_index": i + 1,
                    }

                    if use_storage and self.storage_service:
                        stored_info = self.storage_service.store_image(
                            image_bytes,
                            f"image/{self.config.default_image_format}",
                            metadata
                        )

                        thumbnail_b64 = self.storage_service.get_thumbnail_base64(
                            stored_info.id
                        )
                        if thumbnail_b64:
                            thumbnail_bytes = base64.b64decode(thumbnail_b64)
                            thumbnail_image = MCPImage(data=thumbnail_bytes, format="jpeg")
                            mcp_images.append(thumbnail_image)

                        self.logger.info(
                            f"Edited image {i + 1} with Flash 3.1 - stored as {stored_info.id} "
                            f"({stored_info.size_bytes} bytes)"
                        )
                    else:
                        mcp_image = MCPImage(
                            data=image_bytes,
                            format=self.config.default_image_format
                        )
                        mcp_images.append(mcp_image)

                        self.logger.info(
                            f"Edited image {i + 1} with Flash 3.1 (size: {len(image_bytes)} bytes)"
                        )

                progress.update(
                    100, f"Successfully edited image, generated {len(mcp_images)} result(s)"
                )
                return mcp_images, len(mcp_images)

            except Exception as e:
                self.logger.error(f"Failed to edit image with Flash 3.1: {e}")
                raise

    def _enhance_prompt(
        self,
        prompt: str,
        negative_prompt: str | None
    ) -> str:
        """Enhance prompt for Flash 3.1 model."""
        enhanced = prompt

        if negative_prompt:
            enhanced += f"\n\nAvoid: {negative_prompt}"

        return enhanced
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/chriskantrowitz/Library/Application Support/Claude/mcp-servers/nanobanana-mcp-server" && python -m pytest tests/test_flash31_image_service.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nanobanana_mcp_server/services/flash31_image_service.py tests/test_flash31_image_service.py
git commit -m "feat: add Flash31ImageService for Gemini 3.1 Flash Image model"
```

---

### Task 4: Update GeminiClient for Flash 3.1 Parameters

**Files:**
- Modify: `nanobanana_mcp_server/services/gemini_client.py`
- Test: `tests/test_gemini_client_flash31.py`

**Step 1: Write the failing test**

```python
# tests/test_gemini_client_flash31.py
"""Tests for GeminiClient Flash 3.1 parameter handling."""
from unittest.mock import MagicMock, patch
import pytest

from nanobanana_mcp_server.config.settings import Flash31ImageConfig, ServerConfig
from nanobanana_mcp_server.services.gemini_client import GeminiClient


@pytest.fixture
def flash31_client():
    server_config = MagicMock(spec=ServerConfig)
    server_config.auth_method = MagicMock()
    server_config.auth_method.value = "api_key"
    # Use string comparison for AuthMethod
    from nanobanana_mcp_server.config.settings import AuthMethod
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
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/chriskantrowitz/Library/Application Support/Claude/mcp-servers/nanobanana-mcp-server" && python -m pytest tests/test_gemini_client_flash31.py -v`
Expected: FAIL (Flash31ImageConfig not handled in _filter_parameters)

**Step 3: Update GeminiClient**

In `nanobanana_mcp_server/services/gemini_client.py`:

1. Add import for `Flash31ImageConfig`:
```python
from ..config.settings import (
    AuthMethod,
    BaseModelConfig,
    Flash31ImageConfig,
    FlashImageConfig,
    GeminiConfig,
    ProImageConfig,
    ServerConfig,
)
```

2. Update `_filter_parameters()` to handle Flash 3.1:
```python
def _filter_parameters(self, config: dict[str, Any]) -> dict[str, Any]:
    if not config:
        return {}

    filtered = {}

    # Common parameters (supported by all models)
    for param in ["temperature", "top_p", "top_k", "max_output_tokens"]:
        if param in config:
            filtered[param] = config[param]

    # Flash 3.1 parameters
    if isinstance(self.gemini_config, Flash31ImageConfig):
        # Flash 3.1 supports thinking_level and include_thoughts
        for param in ["thinking_level", "include_thoughts", "enable_grounding"]:
            if param in config:
                filtered[param] = config[param]

    # Pro-specific parameters
    elif isinstance(self.gemini_config, ProImageConfig):
        if "thinking_level" in config:
            self.logger.info("Note: thinking_level is not supported by gemini-3-pro-image-preview, ignoring")

    else:
        # Flash 2.5 model - warn if advanced parameters are used
        pro_params = ["thinking_level", "media_resolution", "output_resolution", "include_thoughts", "enable_grounding"]
        used_pro_params = [p for p in pro_params if p in config]
        if used_pro_params:
            self.logger.warning(
                f"Advanced parameters ignored for Flash 2.5 model: {used_pro_params}"
            )

    return filtered
```

3. Update `generate_content()` to handle Flash 3.1 grounding and thinking config. In the config building section, after the `image_config_kwargs` block, add grounding tool support and thinking config:

```python
# Handle resolution mapping including 512px for Flash 3.1
resolution = config.get("resolution") if config else None
if resolution:
    resolution_map = {
        "4k": "4K",
        "2k": "2K",
        "1k": "1K",
        "high": "1K",
        "512px": "512px",  # Flash 3.1 supports 512px
    }
    image_size = resolution_map.get(resolution.lower())
    if image_size:
        image_config_kwargs["image_size"] = image_size
        self.logger.info(f"Setting image_size={image_size} for resolution={resolution}")

# Handle grounding for Flash 3.1
enable_grounding = filtered_config.pop("enable_grounding", False)
if enable_grounding and isinstance(self.gemini_config, Flash31ImageConfig):
    from google.genai import types as gx_tools
    config_kwargs["tools"] = [gx_tools.Tool(google_search=gx_tools.GoogleSearch())]
    self.logger.info("Enabled Google Search grounding for Flash 3.1")

# Handle thinking config for Flash 3.1
thinking_level = filtered_config.pop("thinking_level", None)
include_thoughts = filtered_config.pop("include_thoughts", False)
if thinking_level and isinstance(self.gemini_config, Flash31ImageConfig):
    from google.genai import types as gx_thinking
    config_kwargs["thinking_config"] = gx_thinking.ThinkingConfig(
        thinking_budget=thinking_level,
        include_thoughts=include_thoughts,
    )
    self.logger.info(f"Set thinking_config: level={thinking_level}, include_thoughts={include_thoughts}")
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/chriskantrowitz/Library/Application Support/Claude/mcp-servers/nanobanana-mcp-server" && python -m pytest tests/test_gemini_client_flash31.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nanobanana_mcp_server/services/gemini_client.py tests/test_gemini_client_flash31.py
git commit -m "feat: update GeminiClient to handle Flash 3.1 thinking, grounding, and 512px resolution"
```

---

### Task 5: Update ModelSelector for Three-Tier Selection

**Files:**
- Modify: `nanobanana_mcp_server/services/model_selector.py`
- Test: `tests/test_model_selector_flash31.py`

**Step 1: Write the failing test**

```python
# tests/test_model_selector_flash31.py
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
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/chriskantrowitz/Library/Application Support/Claude/mcp-servers/nanobanana-mcp-server" && python -m pytest tests/test_model_selector_flash31.py -v`
Expected: FAIL (ModelSelector doesn't accept flash31_service)

**Step 3: Update ModelSelector**

In `nanobanana_mcp_server/services/model_selector.py`:

1. Add import:
```python
from .flash31_image_service import Flash31ImageService
```

2. Update `__init__` to accept Flash 3.1 service:
```python
def __init__(
    self,
    flash_service: ImageService,
    flash31_service: Flash31ImageService,
    pro_service: ProImageService,
    selection_config: ModelSelectionConfig
):
    self.flash_service = flash_service
    self.flash31_service = flash31_service
    self.pro_service = pro_service
    self.config = selection_config
    self.logger = logging.getLogger(__name__)
```

3. Update `select_model()` to handle FLASH_31:
```python
def select_model(
    self,
    prompt: str,
    requested_tier: ModelTier | None = None,
    **kwargs
) -> tuple[ImageService | Flash31ImageService | ProImageService, ModelTier]:
    if requested_tier == ModelTier.FLASH:
        self.logger.info("Explicit Flash model selection")
        return self.flash_service, ModelTier.FLASH

    if requested_tier == ModelTier.FLASH_31:
        self.logger.info("Explicit Flash 3.1 model selection")
        return self.flash31_service, ModelTier.FLASH_31

    if requested_tier == ModelTier.PRO:
        self.logger.info("Explicit Pro model selection")
        return self.pro_service, ModelTier.PRO

    # Auto selection
    if requested_tier == ModelTier.AUTO or requested_tier is None:
        tier = self._auto_select(prompt, **kwargs)
        service_map = {
            ModelTier.FLASH: self.flash_service,
            ModelTier.FLASH_31: self.flash31_service,
            ModelTier.PRO: self.pro_service,
        }
        service = service_map[tier]
        self.logger.info(
            f"Auto-selected {tier.value.upper()} model for prompt: '{prompt[:50]}...'"
        )
        return service, tier

    self.logger.warning(
        f"Unknown model tier '{requested_tier}', falling back to Flash 3.1"
    )
    return self.flash31_service, ModelTier.FLASH_31
```

4. Update `_auto_select()` for three-tier scoring:
```python
def _auto_select(self, prompt: str, **kwargs) -> ModelTier:
    quality_score = 0
    speed_score = 0

    prompt_lower = prompt.lower()

    quality_score = sum(
        1 for keyword in self.config.auto_quality_keywords
        if keyword in prompt_lower
    )

    speed_score = sum(
        1 for keyword in self.config.auto_speed_keywords
        if keyword in prompt_lower
    )

    # Strong quality indicators (weighted heavily)
    strong_quality_keywords = ["4k", "professional", "production", "high-res", "hd"]
    strong_quality_matches = sum(
        1 for keyword in strong_quality_keywords
        if keyword in prompt_lower
    )
    quality_score += strong_quality_matches * 2

    # Resolution parameter analysis
    resolution = kwargs.get("resolution", "").lower()
    if resolution == "4k":
        self.logger.info("4K resolution requested - Pro model required")
        return ModelTier.PRO
    elif resolution in ["high", "2k"]:
        quality_score += 3

    # Batch size
    n = kwargs.get("n", 1)
    if n > 2:
        speed_score += 1

    # Multi-image conditioning
    input_images = kwargs.get("input_images")
    if input_images and len(input_images) > 1:
        quality_score += 1

    # Thinking level hint
    thinking_level = kwargs.get("thinking_level", "").lower()
    if thinking_level == "high":
        quality_score += 1

    # Grounding hint
    enable_grounding = kwargs.get("enable_grounding", False)
    if enable_grounding:
        quality_score += 2

    self.logger.debug(
        f"Model selection scores - Quality: {quality_score}, Speed: {speed_score}"
    )

    # Three-tier decision:
    # Pro: quality_score >= 4 (strong quality signals)
    # Flash 2.5: speed_score > quality_score (explicit speed preference)
    # Flash 3.1: everything else (default balanced tier)
    if quality_score >= 4:
        self.logger.info(f"Selected PRO model (quality_score={quality_score})")
        return ModelTier.PRO
    elif speed_score > quality_score:
        self.logger.info(f"Selected FLASH model (speed_score={speed_score} > quality_score={quality_score})")
        return ModelTier.FLASH
    else:
        self.logger.info(f"Selected FLASH_31 model (balanced: quality={quality_score}, speed={speed_score})")
        return ModelTier.FLASH_31
```

5. Add Flash 3.1 to `get_model_info()`:
```python
def get_model_info(self, tier: ModelTier) -> dict:
    if tier == ModelTier.PRO:
        return {
            "tier": "pro",
            "name": "Gemini 3 Pro Image",
            "model_id": "gemini-3-pro-image-preview",
            "max_resolution": "4K (3840px)",
            "features": ["4K resolution", "Google Search grounding", "Advanced reasoning", "High-quality text rendering"],
            "best_for": "Professional assets, production-ready images",
            "emoji": "\U0001f3c6"
        }
    elif tier == ModelTier.FLASH_31:
        return {
            "tier": "flash_31",
            "name": "Gemini 3.1 Flash Image",
            "model_id": "gemini-3.1-flash-image-preview",
            "max_resolution": "1024px",
            "features": [
                "Thinking/reasoning support",
                "Google Search grounding",
                "Up to 14 reference images",
                "Expanded aspect ratios (1:4, 4:1, 1:8, 8:1)",
                "512px resolution option",
                "Character consistency",
                "International text rendering",
            ],
            "best_for": "Balanced quality and speed, complex prompts",
            "emoji": "\U0001f680"
        }
    else:  # FLASH
        return {
            "tier": "flash",
            "name": "Gemini 2.5 Flash Image",
            "model_id": "gemini-2.5-flash-image",
            "max_resolution": "1024px",
            "features": ["Very fast generation", "Low latency", "High-volume support"],
            "best_for": "Rapid prototyping, quick iterations",
            "emoji": "\u26a1"
        }
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/chriskantrowitz/Library/Application Support/Claude/mcp-servers/nanobanana-mcp-server" && python -m pytest tests/test_model_selector_flash31.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add nanobanana_mcp_server/services/model_selector.py tests/test_model_selector_flash31.py
git commit -m "feat: update ModelSelector with three-tier selection (Flash/Flash 3.1/Pro)"
```

---

### Task 6: Update Service Registry

**Files:**
- Modify: `nanobanana_mcp_server/services/__init__.py`

**Step 1: Update service registry**

In `nanobanana_mcp_server/services/__init__.py`:

1. Add imports:
```python
from ..config.settings import Flash31ImageConfig
from .flash31_image_service import Flash31ImageService
```

2. Add global variables:
```python
_flash31_gemini_client: GeminiClient | None = None
_flash31_image_service: Flash31ImageService | None = None
```

3. Update `initialize_services()` to create Flash 3.1 services:
```python
# Add to global declarations at top of function:
_flash31_gemini_client, \
_flash31_image_service, \

# Add after pro_config = ProImageConfig():
flash31_config = Flash31ImageConfig()

# Add after _pro_gemini_client creation:
_flash31_gemini_client = GeminiClient(server_config, flash31_config)

# Add after _pro_image_service creation:
_flash31_image_service = Flash31ImageService(
    _flash31_gemini_client,
    flash31_config,
    _image_storage_service
)

# Update ModelSelector to include Flash 3.1:
_model_selector = ModelSelector(
    _file_image_service,      # Flash 2.5 service
    _flash31_image_service,   # Flash 3.1 service (NEW)
    _pro_image_service,       # Pro service
    selection_config
)
```

4. Add getter function:
```python
def get_flash31_image_service() -> Flash31ImageService:
    """Get the Flash 3.1 image service instance."""
    if _flash31_image_service is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _flash31_image_service
```

**Step 2: Commit**

```bash
git add nanobanana_mcp_server/services/__init__.py
git commit -m "feat: register Flash 3.1 services in service registry"
```

---

### Task 7: Update generate_image Tool

**Files:**
- Modify: `nanobanana_mcp_server/tools/generate_image.py`

**Step 1: Update tool parameters and routing**

In `nanobanana_mcp_server/tools/generate_image.py`:

1. Update `model_tier` parameter description:
```python
model_tier: Annotated[
    str | None,
    Field(
        description="Model tier: 'flash' (speed, 1024px), 'flash_31' (balanced, thinking+grounding), "
        "'pro' (quality, up to 4K), or 'auto' (smart selection). "
        "Default: 'auto' - automatically selects based on prompt analysis."
    ),
] = "auto",
```

2. Update `aspect_ratio` to include Flash 3.1 expanded ratios:
```python
aspect_ratio: Annotated[
    Literal["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9", "1:4", "4:1", "1:8", "8:1"] | None,
    Field(
        description="Optional output aspect ratio. Extended ratios (1:4, 4:1, 1:8, 8:1) "
        "available with flash_31 model. Standard: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9."
    ),
] = None,
```

3. Update `thinking_level` description:
```python
thinking_level: Annotated[
    str | None,
    Field(
        description="Reasoning depth: 'minimal' (Flash 3.1 default), 'low', 'high'. "
        "Applies to Flash 3.1 and Pro models."
    ),
] = None,
```

4. Add new `include_thoughts` parameter (after `enable_grounding`):
```python
include_thoughts: Annotated[
    bool,
    Field(
        description="Include model's reasoning/thoughts in response (Flash 3.1 only). "
        "Default: false."
    ),
] = False,
```

5. Update `enable_grounding` description:
```python
enable_grounding: Annotated[
    bool,
    Field(
        description="Enable Google Search grounding for factual accuracy. "
        "Supported by Flash 3.1 and Pro models. Default: false."
    ),
] = False,
```

6. Update `resolution` description:
```python
resolution: Annotated[
    str | None,
    Field(
        description="Output resolution: 'high', '4k', '2k', '1k', '512px'. "
        "4K/2K only with 'pro'. 512px only with 'flash_31'. Default: 'high'."
    ),
] = "high",
```

7. Update ThinkingLevel validation to accept "minimal":
```python
# Validate thinking level
try:
    if thinking_level:
        _ = ThinkingLevel(thinking_level)
except ValueError:
    logger.warning(f"Invalid thinking_level '{thinking_level}', using model default")
    thinking_level = None
```

8. Update MAX_INPUT_IMAGES validation to be model-aware. After model selection:
```python
# Model-aware input image validation
if input_image_paths:
    from ..config.constants import MAX_INPUT_IMAGES, MAX_INPUT_IMAGES_FLASH_31
    max_images = MAX_INPUT_IMAGES_FLASH_31 if selected_tier == ModelTier.FLASH_31 else MAX_INPUT_IMAGES
    if len(input_image_paths) > max_images:
        raise ValidationError(f"Maximum {max_images} input images allowed for {selected_tier.value} model")
```

9. Add Flash 3.1 routing branch (after the PRO branch at line 279):
```python
if selected_tier == ModelTier.PRO:
    # ... existing Pro code ...
elif selected_tier == ModelTier.FLASH_31:
    # Use Flash 3.1 service for balanced generation
    logger.info(f"Using FLASH_31 model: {model_info['model_id']}")
    thumbnail_images, metadata = selected_service.generate_images(
        prompt=prompt,
        n=n,
        resolution=resolution,
        thinking_level=ThinkingLevel(thinking_level) if thinking_level else None,
        include_thoughts=include_thoughts,
        enable_grounding=enable_grounding,
        negative_prompt=negative_prompt,
        system_instruction=system_instruction,
        input_images=input_images,
        aspect_ratio=aspect_ratio,
        use_storage=True,
    )
else:
    # Flash 2.5 (existing code)
    ...
```

10. Update summary section to show Flash 3.1 info:
```python
# Add Flash 3.1 specific information
if selected_tier == ModelTier.FLASH_31:
    if thinking_level:
        summary_lines.append(f"\U0001f9e0 **Thinking Level**: {thinking_level}")
    if enable_grounding:
        summary_lines.append("\U0001f50d **Grounding**: Enabled (Google Search)")
    if include_thoughts:
        summary_lines.append("\U0001f4ad **Thoughts**: Included in response")
```

**Step 2: Run full test suite**

Run: `cd "/Users/chriskantrowitz/Library/Application Support/Claude/mcp-servers/nanobanana-mcp-server" && python -m pytest -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add nanobanana_mcp_server/tools/generate_image.py
git commit -m "feat: update generate_image tool with Flash 3.1 parameters and routing"
```

---

### Task 8: Run Full Test Suite and Fix Issues

**Step 1: Run all tests**

Run: `cd "/Users/chriskantrowitz/Library/Application Support/Claude/mcp-servers/nanobanana-mcp-server" && python -m pytest -v --tb=short`

**Step 2: Run linting**

Run: `cd "/Users/chriskantrowitz/Library/Application Support/Claude/mcp-servers/nanobanana-mcp-server" && ruff check . && ruff format --check .`

**Step 3: Fix any failures found**

Address any test failures or linting issues.

**Step 4: Commit fixes**

```bash
git add -A
git commit -m "fix: resolve test failures and linting issues for Flash 3.1 integration"
```

---

### Task 9: Update CLAUDE.md Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Add Flash 3.1 documentation alongside existing Pro documentation. Key sections to update:
- Project Overview: Add Flash 3.1 capabilities alongside Pro
- ModelTier enum: Document FLASH_31 value
- Configuration: Document Flash31ImageConfig
- Model Selection Logic: Document three-tier selection
- Architecture: Document Flash31ImageService

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with Flash 3.1 model documentation"
```

---

## Verification

After all tasks are complete:

1. **Unit tests**: `python -m pytest -v` - all pass
2. **Linting**: `ruff check . && ruff format --check .` - clean
3. **Manual test**: Start the server with `fastmcp dev mcp_dev.py:app` and test:
   - Generate with `model_tier="flash_31"` explicitly
   - Generate with `model_tier="auto"` and a balanced prompt (should auto-select Flash 3.1)
   - Generate with `enable_grounding=True` on Flash 3.1
   - Generate with `thinking_level="high"` on Flash 3.1
   - Generate with `aspect_ratio="1:4"` (new Flash 3.1 ratio)
   - Generate with `model_tier="flash"` to verify Flash 2.5 still works
   - Generate with `model_tier="pro"` to verify Pro still works

## Execution

**Recommended: Team agents** (as requested by user) - Use TeamCreate to create a team with parallel agents working on independent tasks. Tasks 1-2 (config), Task 3 (service), and Task 4 (client) can run in parallel. Tasks 5-7 depend on earlier tasks. Task 8-9 are sequential finalization.
