"""
Tier-2 VLM fallback using Google Gemini.

When the local ONNX model is under-confident (<threshold) or the image is
Out-Of-Distribution, we send the raw image bytes to Gemini for zero-shot
multimodal plant identification.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

from src.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a botanist and plant identification expert.
Identify the plant in the image as precisely as possible.
You must respond with a valid JSON object (no markdown, no code blocks) with exactly these keys:
{
  "plant_name": "<common name>",
  "scientific_name": "<genus species>",
  "confidence": <float 0.0-1.0>,
  "description": "<one sentence description>",
  "is_plant": <true|false>,
  "ood_reason": "<empty string if is_plant is true, else reason why it is not a recognisable plant>"
}
Focus on: Colocasia / Taro leaves, Green Sorrel / Sour Sorrel, Dill leaves,
Fenugreek leaves, Spinach, and similar leafy plants. If the image is clearly
not a plant, set is_plant to false.
"""


def _client_available() -> bool:
    """Check whether the google-genai SDK is installed and an API key exists."""
    try:
        import google.genai  # noqa: F401
        return bool(GEMINI_API_KEY)
    except ImportError:
        return False


class VLMFallback:
    """Wrapper around the Gemini Interactions API for plant identification."""

    @property
    def is_ready(self) -> bool:
        return _client_available()

    def identify(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> dict[str, Any]:
        """
        Send the image to Gemini and parse the structured JSON response.

        Returns a dict with keys:
            plant_name, scientific_name, confidence, description,
            is_plant, ood_reason, raw_response

        Raises:
            RuntimeError: If the API key is missing or SDK unavailable.
            ValueError: If Gemini returns unparseable JSON.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Gemini API key (GEMINI_API_KEY) is not set or google-genai SDK "
                "is not installed. VLM fallback unavailable."
            )

        from google import genai  # type: ignore

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Build multimodal input: system instruction + image bytes inline
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        interaction = client.interactions.create(
            model=GEMINI_MODEL,
            system_instruction=_SYSTEM_PROMPT,
            input=[
                {
                    "type": "image",
                    "data": b64_image,
                    "mime_type": mime_type,
                },
                {
                    "type": "text",
                    "text": (
                        "Identify this plant. Return only the JSON object as specified."
                    ),
                },
            ],
            store=False,
        )

        raw_text: str = interaction.output_text or ""
        logger.debug("Gemini raw response: %s", raw_text)

        # Strip accidental markdown fences
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            parsed: dict[str, Any] = json.loads(clean)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini JSON: %s", raw_text)
            raise ValueError(
                f"Gemini returned non-JSON response: {raw_text[:200]}"
            ) from exc

        parsed["raw_response"] = raw_text
        return parsed


# Module-level singleton
vlm_fallback = VLMFallback()
