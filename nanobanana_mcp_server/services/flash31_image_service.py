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
        """Generate images using Gemini 3.1 Flash Image."""
        if thinking_level is None:
            thinking_level = self.config.default_thinking_level
        if enable_grounding is None:
            enable_grounding = self.config.enable_search_grounding

        with ProgressContext(
            "flash31_image_generation",
            f"Generating {n} image(s) with Gemini 3.1 Flash...",
            {"prompt": prompt[:100], "count": n},
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

            all_images: list[MCPImage] = []
            all_metadata: list[dict[str, Any]] = []

            for i in range(n):
                try:
                    progress.update(
                        20 + (i * 70 // n),
                        f"Generating image {i + 1}/{n}...",
                    )

                    gen_config: dict[str, Any] = {
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
                        metadata: dict[str, Any] = {
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
                                metadata,
                            )

                            thumbnail_b64 = self.storage_service.get_thumbnail_base64(
                                stored_info.id
                            )
                            if thumbnail_b64:
                                thumbnail_bytes = base64.b64decode(thumbnail_b64)
                                thumbnail_image = MCPImage(data=thumbnail_bytes, format="jpeg")
                                all_images.append(thumbnail_image)

                            metadata.update(
                                {
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
                                }
                            )

                            all_metadata.append(metadata)

                            self.logger.info(
                                f"Generated Flash 3.1 image {i + 1}.{j + 1} - "
                                f"stored as {stored_info.id} "
                                f"({stored_info.size_bytes} bytes, "
                                f"{stored_info.width}x{stored_info.height})"
                            )
                        else:
                            mcp_image = MCPImage(
                                data=image_bytes,
                                format=self.config.default_image_format,
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
        """Edit images with Flash 3.1 model."""
        if thinking_level is None:
            thinking_level = self.config.default_thinking_level

        with ProgressContext(
            "flash31_image_editing",
            "Editing image with Gemini 3.1 Flash...",
            {"instruction": instruction[:100]},
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

                image_parts = self.gemini_client.create_image_parts([base_image_b64], [mime_type])
                contents = [*image_parts, enhanced_instruction]

                progress.update(40, "Sending edit request to Gemini 3.1 Flash API...")

                gen_config = {
                    "thinking_level": thinking_level.value,
                    "include_thoughts": include_thoughts,
                }

                response = self.gemini_client.generate_content(contents, config=gen_config)
                image_bytes_list = self.gemini_client.extract_images(response)

                progress.update(70, "Processing edited images...")

                mcp_images: list[MCPImage] = []
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
                            metadata,
                        )

                        thumbnail_b64 = self.storage_service.get_thumbnail_base64(stored_info.id)
                        if thumbnail_b64:
                            thumbnail_bytes = base64.b64decode(thumbnail_b64)
                            thumbnail_image = MCPImage(data=thumbnail_bytes, format="jpeg")
                            mcp_images.append(thumbnail_image)

                        self.logger.info(
                            f"Edited image {i + 1} with Flash 3.1 - "
                            f"stored as {stored_info.id} "
                            f"({stored_info.size_bytes} bytes)"
                        )
                    else:
                        mcp_image = MCPImage(
                            data=image_bytes,
                            format=self.config.default_image_format,
                        )
                        mcp_images.append(mcp_image)

                        self.logger.info(
                            f"Edited image {i + 1} with Flash 3.1 (size: {len(image_bytes)} bytes)"
                        )

                progress.update(
                    100,
                    f"Successfully edited image, generated {len(mcp_images)} result(s)",
                )
                return mcp_images, len(mcp_images)

            except Exception as e:
                self.logger.error(f"Failed to edit image with Flash 3.1: {e}")
                raise

    def _enhance_prompt(
        self,
        prompt: str,
        negative_prompt: str | None,
    ) -> str:
        """Enhance prompt for Flash 3.1 model."""
        enhanced = prompt

        if negative_prompt:
            enhanced += f"\n\nAvoid: {negative_prompt}"

        return enhanced
