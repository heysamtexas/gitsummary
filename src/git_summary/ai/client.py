"""LiteLLM client wrapper for git-summary AI features.

This module provides a clean interface around LiteLLM with our conventions,
supporting multiple AI providers with built-in fallbacks and error handling.
"""

import logging
import os
from typing import Any

import litellm
from litellm import acompletion, completion

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limits are exceeded."""

    pass


class LLMModelNotAvailableError(LLMError):
    """Raised when the requested model is not available."""

    pass


class LLMClient:
    """Clean wrapper around LiteLLM with our conventions.

    Provides a simplified interface for generating AI summaries with
    automatic fallback support and proper error handling.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        fallback_model: str = "anthropic/claude-3-haiku-20240307",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        timeout: int = 30,
    ):
        """Initialize the LLM client.

        Args:
            model: Primary model to use (e.g., "gpt-4o-mini", "anthropic/claude-3-5-sonnet-20241022")
            fallback_model: Fallback model if primary fails
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-1.0)
            timeout: Request timeout in seconds
        """
        self.model = model
        self.fallback_model = fallback_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        # Configure LiteLLM
        litellm.set_verbose = False  # We handle our own logging
        litellm.drop_params = True  # Drop unsupported parameters gracefully

        # Validate API keys are available
        self._validate_api_keys()

        logger.info(
            f"Initialized LLM client with model: {model}, fallback: {fallback_model}"
        )

    def _validate_api_keys(self) -> None:
        """Validate that required API keys are available."""
        required_keys = []

        # Check which providers we're using
        if any(
            provider in self.model.lower() for provider in ["gpt", "openai"]
        ) and not os.getenv("OPENAI_API_KEY"):
            required_keys.append("OPENAI_API_KEY")

        if (
            "anthropic" in self.model.lower() or "claude" in self.model.lower()
        ) and not os.getenv("ANTHROPIC_API_KEY"):
            required_keys.append("ANTHROPIC_API_KEY")

        if (
            ("gemini" in self.model.lower() or "google" in self.model.lower())
            and not os.getenv("GOOGLE_API_KEY")
            and not os.getenv("GEMINI_API_KEY")
        ):
            required_keys.append("GOOGLE_API_KEY or GEMINI_API_KEY")

        if required_keys:
            logger.warning(
                f"Missing API keys: {', '.join(required_keys)}. Some models may not work."
            )

    async def generate_summary(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate AI summary with automatic fallback.

        Args:
            system_prompt: System instruction for the AI
            user_content: User content to analyze
            max_tokens: Override default max tokens
            temperature: Override default temperature

        Returns:
            Generated summary text

        Raises:
            LLMError: If both primary and fallback models fail
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Use provided values or defaults
        tokens = max_tokens or self.max_tokens
        temp = temperature or self.temperature

        # Try primary model first
        try:
            logger.debug(f"Attempting to generate summary with model: {self.model}")
            response = await acompletion(
                model=self.model,
                messages=messages,
                max_tokens=tokens,
                temperature=temp,
                timeout=self.timeout,
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMError("Empty response from model")

            logger.info(
                f"Successfully generated summary with {self.model} ({len(content)} chars)"
            )
            return str(content).strip()

        except Exception as e:
            logger.warning(f"Primary model {self.model} failed: {e}")

            # Try fallback model
            try:
                logger.info(f"Attempting fallback to model: {self.fallback_model}")
                response = await acompletion(
                    model=self.fallback_model,
                    messages=messages,
                    max_tokens=tokens,
                    temperature=temp,
                    timeout=self.timeout,
                )

                content = response.choices[0].message.content
                if not content:
                    raise LLMError("Empty response from fallback model")

                logger.info(
                    f"Successfully generated summary with fallback {self.fallback_model}"
                )
                return str(content).strip()

            except Exception as fallback_error:
                logger.error(
                    f"Fallback model {self.fallback_model} also failed: {fallback_error}"
                )
                raise LLMError(
                    f"Both primary ({e}) and fallback ({fallback_error}) models failed"
                )

    def generate_summary_sync(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Synchronous version of generate_summary for non-async contexts.

        Args:
            system_prompt: System instruction for the AI
            user_content: User content to analyze
            max_tokens: Override default max tokens
            temperature: Override default temperature

        Returns:
            Generated summary text

        Raises:
            LLMError: If both primary and fallback models fail
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Use provided values or defaults
        tokens = max_tokens or self.max_tokens
        temp = temperature or self.temperature

        # Try primary model first
        try:
            logger.debug(f"Attempting to generate summary with model: {self.model}")
            response = completion(
                model=self.model,
                messages=messages,
                max_tokens=tokens,
                temperature=temp,
                timeout=self.timeout,
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMError("Empty response from model")

            logger.info(
                f"Successfully generated summary with {self.model} ({len(content)} chars)"
            )
            return str(content).strip()

        except Exception as e:
            logger.warning(f"Primary model {self.model} failed: {e}")

            # Try fallback model
            try:
                logger.info(f"Attempting fallback to model: {self.fallback_model}")
                response = completion(
                    model=self.fallback_model,
                    messages=messages,
                    max_tokens=tokens,
                    temperature=temp,
                    timeout=self.timeout,
                )

                content = response.choices[0].message.content
                if not content:
                    raise LLMError("Empty response from fallback model")

                logger.info(
                    f"Successfully generated summary with fallback {self.fallback_model}"
                )
                return str(content).strip()

            except Exception as fallback_error:
                logger.error(
                    f"Fallback model {self.fallback_model} also failed: {fallback_error}"
                )
                raise LLMError(
                    f"Both primary ({e}) and fallback ({fallback_error}) models failed"
                )

    def estimate_cost(self, system_prompt: str, user_content: str) -> dict[str, Any]:
        """Estimate cost for the given prompts.

        Args:
            system_prompt: System instruction
            user_content: User content

        Returns:
            Dict with cost estimation info
        """
        # Simple token estimation (rough approximation)
        total_text = system_prompt + user_content
        estimated_input_tokens = (
            len(total_text) // 4
        )  # Rough estimate: 4 chars per token
        estimated_output_tokens = self.max_tokens

        # Basic cost estimates (these would need to be updated with current pricing)
        cost_estimates = {
            "gpt-4o-mini": {"input": 0.150, "output": 0.600},  # per 1M tokens
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        }

        # Find matching cost estimate
        model_key = None
        for key in cost_estimates:
            if (
                key.replace("-", "").replace(".", "")
                in self.model.replace("-", "").replace(".", "").lower()
            ):
                model_key = key
                break

        if model_key:
            costs = cost_estimates[model_key]
            estimated_cost = (estimated_input_tokens / 1_000_000) * costs["input"] + (
                estimated_output_tokens / 1_000_000
            ) * costs["output"]
        else:
            estimated_cost = 0.01  # Unknown model, small default

        return {
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_cost_usd": round(estimated_cost, 4),
            "model": self.model,
            "currency": "USD",
        }

    def get_available_models(self) -> list[dict[str, str]]:
        """Get list of available models with descriptions.

        Returns:
            List of model info dicts
        """
        return [
            {
                "name": "gpt-4o-mini",
                "provider": "OpenAI",
                "description": "Fast, cost-effective model for most tasks",
                "cost_tier": "Low",
            },
            {
                "name": "gpt-4o",
                "provider": "OpenAI",
                "description": "High-quality model for complex analysis",
                "cost_tier": "Medium",
            },
            {
                "name": "anthropic/claude-3-haiku-20240307",
                "provider": "Anthropic",
                "description": "Fast, efficient model for quick summaries",
                "cost_tier": "Low",
            },
            {
                "name": "anthropic/claude-3-5-sonnet-20241022",
                "provider": "Anthropic",
                "description": "Advanced model for detailed analysis",
                "cost_tier": "Medium",
            },
            {
                "name": "ollama/llama3.1",
                "provider": "Local (Ollama)",
                "description": "Local model for private usage",
                "cost_tier": "Free",
            },
        ]
