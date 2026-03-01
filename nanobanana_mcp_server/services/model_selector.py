"""Intelligent model selection service for routing requests to optimal models."""

import logging

from ..config.settings import ModelSelectionConfig, ModelTier
from .flash31_image_service import Flash31ImageService
from .image_service import ImageService
from .pro_image_service import ProImageService


class ModelSelector:
    """
    Intelligent model selection and routing service.

    Routes image generation/editing requests to the appropriate model
    (Flash, Flash 3.1, or Pro) based on prompt analysis, explicit user
    preference, or automatic selection logic.
    """

    def __init__(
        self,
        flash_service: ImageService,
        flash31_service: Flash31ImageService,
        pro_service: ProImageService,
        selection_config: ModelSelectionConfig,
    ):
        self.flash_service = flash_service
        self.flash31_service = flash31_service
        self.pro_service = pro_service
        self.config = selection_config
        self.logger = logging.getLogger(__name__)

    def select_model(
        self, prompt: str, requested_tier: ModelTier | None = None, **kwargs
    ) -> tuple[ImageService | Flash31ImageService | ProImageService, ModelTier]:
        """
        Select appropriate model based on requirements.

        Args:
            prompt: User's image generation/edit prompt
            requested_tier: Explicit model tier request (or None for auto)
            **kwargs: Additional context (n, resolution, input_images, etc.)

        Returns:
            Tuple of (selected_service, selected_tier)
        """
        # Explicit selection takes precedence
        if requested_tier == ModelTier.FLASH:
            self.logger.info("Explicit Flash model selection")
            return self.flash_service, ModelTier.FLASH

        if requested_tier == ModelTier.FLASH_31:
            self.logger.info("Explicit Flash 3.1 model selection")
            return self.flash31_service, ModelTier.FLASH_31

        if requested_tier == ModelTier.PRO:
            self.logger.info("Explicit Pro model selection")
            return self.pro_service, ModelTier.PRO

        # Auto selection logic
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

        # Fallback to Flash 3.1 for unknown values
        self.logger.warning(f"Unknown model tier '{requested_tier}', falling back to Flash 3.1")
        return self.flash31_service, ModelTier.FLASH_31

    def _auto_select(self, prompt: str, **kwargs) -> ModelTier:
        """
        Automatic model selection based on prompt and context analysis.

        Three-tier decision:
        - Pro: Strong quality signals (4K, professional, production)
        - Flash 2.5: Explicit speed preference (quick, draft, sketch)
        - Flash 3.1: Everything else (default balanced tier)

        Args:
            prompt: User's prompt text
            **kwargs: Additional context

        Returns:
            Selected ModelTier (FLASH, FLASH_31, or PRO)
        """
        quality_score = 0
        speed_score = 0

        prompt_lower = prompt.lower()

        # Analyze prompt for quality indicators
        quality_score = sum(
            1 for keyword in self.config.auto_quality_keywords if keyword in prompt_lower
        )

        # Analyze prompt for speed indicators
        speed_score = sum(
            1 for keyword in self.config.auto_speed_keywords if keyword in prompt_lower
        )

        # Strong quality indicators (weighted heavily)
        strong_quality_keywords = ["4k", "professional", "production", "high-res", "hd"]
        strong_quality_matches = sum(
            1 for keyword in strong_quality_keywords if keyword in prompt_lower
        )
        quality_score += strong_quality_matches * 2

        # Resolution parameter analysis
        resolution = kwargs.get("resolution", "").lower() if kwargs.get("resolution") else ""
        if resolution == "4k":
            self.logger.info("4K resolution requested - Pro model required")
            return ModelTier.PRO
        elif resolution in ["high", "2k"]:
            quality_score += 3

        # Batch size consideration
        n = kwargs.get("n", 1)
        if n > 2:
            speed_score += 1

        # Multi-image conditioning
        input_images = kwargs.get("input_images")
        if input_images and len(input_images) > 1:
            quality_score += 1

        # Thinking level hint
        thinking_level = kwargs.get("thinking_level", "")
        if thinking_level and thinking_level.lower() == "high":
            quality_score += 1

        # Enable grounding hint
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
            self.logger.info(
                f"Selected FLASH model (speed_score={speed_score} > quality_score={quality_score})"
            )
            return ModelTier.FLASH
        else:
            self.logger.info(
                f"Selected FLASH_31 model (balanced: quality={quality_score}, speed={speed_score})"
            )
            return ModelTier.FLASH_31

    def get_model_info(self, tier: ModelTier) -> dict:
        """Get information about a specific model tier."""
        if tier == ModelTier.PRO:
            return {
                "tier": "pro",
                "name": "Gemini 3 Pro Image",
                "model_id": "gemini-3-pro-image-preview",
                "max_resolution": "4K (3840px)",
                "features": [
                    "4K resolution",
                    "Google Search grounding",
                    "Advanced reasoning",
                    "High-quality text rendering",
                ],
                "best_for": "Professional assets, production-ready images",
                "emoji": "🏆",
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
                "emoji": "🚀",
            }
        else:  # FLASH
            return {
                "tier": "flash",
                "name": "Gemini 2.5 Flash Image",
                "model_id": "gemini-2.5-flash-image",
                "max_resolution": "1024px",
                "features": ["Very fast generation", "Low latency", "High-volume support"],
                "best_for": "Rapid prototyping, quick iterations",
                "emoji": "⚡",
            }
