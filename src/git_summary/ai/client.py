"""LiteLLM client wrapper for git-summary AI features.

This module provides a clean interface around LiteLLM with our conventions,
supporting multiple AI providers with built-in fallbacks and error handling.
"""

import logging
import os
from typing import Any

import litellm
from litellm import acompletion, completion

from git_summary.ai_models import ModelManager
from git_summary.config import Config

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
        model: str = "anthropic/claude-3-7-sonnet-latest",
        fallback_model: str = "groq/llama-3.3-70b-versatile",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        timeout: int = 30,
    ):
        """Initialize the LLM client.

        Args:
            model: Primary model to use (e.g., "anthropic/claude-3-7-sonnet-latest", "groq/llama-3.1-70b-versatile")
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

        # Initialize ModelManager instance
        self._model_manager = ModelManager()

        # Set up API keys from config
        self._setup_api_keys()

        # Validate API keys are available
        self._validate_api_keys()

        logger.info(
            f"Initialized LLM client with model: {model}, fallback: {fallback_model}"
        )

    def _setup_api_keys(self) -> None:
        """Set up API keys from config file, falling back to environment variables."""
        config = Config()

        # Set OpenAI API key if available in config and not already in environment
        if not os.getenv("OPENAI_API_KEY"):
            openai_key = config.get_ai_api_key("openai")
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
                logger.debug("Set OPENAI_API_KEY from config")

        # Set Anthropic API key if available in config and not already in environment
        if not os.getenv("ANTHROPIC_API_KEY"):
            anthropic_key = config.get_ai_api_key("anthropic")
            if anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_key
                logger.debug("Set ANTHROPIC_API_KEY from config")

        # Set Google API key if available in config and not already in environment
        if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            google_key = config.get_ai_api_key("google")
            if google_key:
                os.environ["GOOGLE_API_KEY"] = google_key
                logger.debug("Set GOOGLE_API_KEY from config")

        # Set Groq API key if available in config and not already in environment
        if not os.getenv("GROQ_API_KEY"):
            groq_key = config.get_ai_api_key("groq")
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key
                logger.debug("Set GROQ_API_KEY from config")

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

        if ("groq" in self.model.lower()) and not os.getenv("GROQ_API_KEY"):
            required_keys.append("GROQ_API_KEY")

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

        # Use ModelManager for cost estimation
        try:
            cost_info = self._model_manager.estimate_cost(
                self.model, estimated_input_tokens, estimated_output_tokens
            )
            return {
                "estimated_input_tokens": estimated_input_tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "estimated_cost_usd": cost_info["total_cost"],
                "model": self.model,
                "currency": "USD",
            }
        except ValueError:
            # Fallback for unknown models
            return {
                "estimated_input_tokens": estimated_input_tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "estimated_cost_usd": 0.01,  # Unknown model, small default
                "model": self.model,
                "currency": "USD",
            }

    def get_available_models(self) -> list[dict[str, str]]:
        """Get list of available models with descriptions.

        Returns:
            List of model info dicts (backward compatibility format)
        """
        models = self._model_manager.get_available_models()

        # Convert to the expected format for backward compatibility
        result = []
        for model in models:
            # Format context limit for display
            if model.context_limit >= 1_000_000:
                context_window = f"{model.context_limit // 1_000_000}M"
            elif model.context_limit >= 1000:
                context_window = f"{model.context_limit // 1000}K"
            else:
                context_window = str(model.context_limit)

            result.append(
                {
                    "name": model.name,
                    "provider": model.provider,
                    "description": model.description,
                    "cost_tier": model.tier.title(),
                    "context_window": context_window,
                }
            )

        return result
