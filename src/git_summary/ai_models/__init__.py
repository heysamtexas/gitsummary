"""AI model management for git-summary.

This module provides the ModelManager class for handling AI model metadata,
cost estimation, and configuration discovery. It follows the same pattern
as the PersonaManager for consistency.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from git_summary.config import Config

logger = logging.getLogger(__name__)


@dataclass
class AIModel:
    """AI model metadata."""

    name: str
    display_name: str
    provider: str
    description: str
    context_limit: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    capabilities: list[str]
    tier: str
    status: str = "stable"
    yaml_path: str | None = None


@dataclass
class AIProvider:
    """AI provider metadata."""

    name: str
    api_key_env: str
    models: list[AIModel]


class ModelManager:
    """Manage AI models and their metadata.

    Provides a centralized way to manage AI model information, cost estimation,
    and API key status checking. Follows the same YAML-based pattern as PersonaManager
    for consistency.
    """

    def __init__(self) -> None:
        """Initialize the ModelManager."""
        self._providers: dict[str, AIProvider] = {}
        self._models: dict[str, AIModel] = {}
        self._load_built_in_models()

    def _load_built_in_models(self) -> None:
        """Load built-in model definitions from YAML files."""
        # Get the directory where this module is located
        models_dir = Path(__file__).parent / "built_in"

        if not models_dir.exists():
            logger.warning(f"Built-in models directory not found: {models_dir}")
            return

        # Load all YAML files in the built_in directory
        for yaml_file in models_dir.glob("*.yaml"):
            try:
                self._load_provider_file(yaml_file)
            except Exception as e:
                logger.error(f"Failed to load model file {yaml_file}: {e}")

    def _load_provider_file(self, yaml_file: Path) -> None:
        """Load a provider YAML file."""
        try:
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            provider_name = data["provider"]
            api_key_env = data["api_key_env"]
            models_data = data["models"]

            models = []
            for model_data in models_data:
                model = AIModel(
                    name=model_data["name"],
                    display_name=model_data.get("display_name", model_data["name"]),
                    provider=provider_name,
                    description=model_data["description"],
                    context_limit=model_data["context_limit"],
                    input_cost_per_1k=model_data["pricing"]["input_per_1k"],
                    output_cost_per_1k=model_data["pricing"]["output_per_1k"],
                    capabilities=model_data.get("capabilities", []),
                    tier=model_data["tier"],
                    status=model_data.get("status", "stable"),
                    yaml_path=str(yaml_file),
                )
                models.append(model)
                self._models[model.name] = model

            provider = AIProvider(
                name=provider_name,
                api_key_env=api_key_env,
                models=models,
            )
            self._providers[provider_name.lower()] = provider

            logger.debug(f"Loaded {len(models)} models from {provider_name}")

        except Exception as e:
            logger.error(f"Error loading provider file {yaml_file}: {e}")
            raise

    def get_available_models(
        self, provider: str | None = None, capability: str | None = None
    ) -> list[AIModel]:
        """Get list of available models with optional filtering.

        Args:
            provider: Filter by provider name (case-insensitive)
            capability: Filter by capability (e.g., "chat", "vision")

        Returns:
            List of AIModel objects matching the criteria
        """
        models = list(self._models.values())

        if provider:
            models = [m for m in models if m.provider.lower() == provider.lower()]

        if capability:
            models = [
                m
                for m in models
                if capability.lower() in [c.lower() for c in m.capabilities]
            ]

        # Sort by provider, then by tier, then by name
        tier_order = {"low": 0, "medium": 1, "high": 2}
        models.sort(
            key=lambda m: (
                m.provider,
                tier_order.get(m.tier.lower(), 99),
                m.name,
            )
        )

        return models

    def get_model_info(self, model_name: str) -> AIModel | None:
        """Get detailed information about a specific model.

        Args:
            model_name: Name of the model to get info for

        Returns:
            AIModel object if found, None otherwise
        """
        return self._models.get(model_name)

    def get_providers(self) -> list[str]:
        """Get list of all provider names.

        Returns:
            List of provider names
        """
        return list(self._providers.keys())

    def estimate_cost(
        self, model_name: str, input_tokens: int, output_tokens: int
    ) -> dict[str, Any]:
        """Estimate cost for using a specific model.

        Args:
            model_name: Name of the model
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Dictionary with cost estimation details

        Raises:
            ValueError: If model is not found
        """
        model = self._models.get(model_name)
        if not model:
            raise ValueError(f"Model not found: {model_name}")

        input_cost = (input_tokens / 1000) * model.input_cost_per_1k
        output_cost = (output_tokens / 1000) * model.output_cost_per_1k
        total_cost = input_cost + output_cost

        return {
            "model": model_name,
            "provider": model.provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_per_1k": model.input_cost_per_1k,
            "output_cost_per_1k": model.output_cost_per_1k,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "currency": "USD",
        }

    def check_api_key_status(self, provider: str) -> bool:
        """Check if API key is configured for a provider.

        Args:
            provider: Provider name (case-insensitive)

        Returns:
            True if API key is configured, False otherwise
        """
        provider_lower = provider.lower()
        if provider_lower not in self._providers:
            return False

        provider_obj = self._providers[provider_lower]

        # Check environment variable first
        if os.getenv(provider_obj.api_key_env):
            return True

        # Check config file
        config = Config()
        api_keys = config.list_ai_api_keys()
        return api_keys.get(provider_lower, False)

    def get_models_with_status(self) -> list[dict[str, Any]]:
        """Get all models with their API key configuration status.

        Returns:
            List of dictionaries with model info and status
        """
        models = self.get_available_models()
        result = []

        for model in models:
            status = self.check_api_key_status(model.provider)
            result.append(
                {
                    "model": model,
                    "configured": status,
                    "status_text": "✓ Configured" if status else "✗ Not configured",
                }
            )

        return result

    def get_cost_tier_display(self, model: AIModel) -> str:
        """Get display-friendly cost tier.

        Args:
            model: AIModel to get tier for

        Returns:
            Capitalized tier string
        """
        return model.tier.title()
