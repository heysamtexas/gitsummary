"""LiteLLM client wrapper for git-summary AI features.

This module provides a clean interface around LiteLLM with our conventions,
supporting multiple AI providers with built-in fallbacks and error handling.
"""

import logging
import os
from typing import Any

import litellm
from litellm import acompletion, completion

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
        litellm.set_verbose = False  # type: ignore[attr-defined]  # We handle our own logging
        litellm.drop_params = True  # Drop unsupported parameters gracefully

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

        # Basic cost estimates (these would need to be updated with current pricing)
        cost_estimates = {
            "gpt-4o-mini": {"input": 0.150, "output": 0.600},  # per 1M tokens
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
            "claude-3-7-sonnet": {"input": 3.00, "output": 15.00},  # estimated
            "llama-3.1-70b": {"input": 0.59, "output": 0.79},  # Groq pricing
            "llama-3.1-8b": {"input": 0.05, "output": 0.08},  # Groq pricing
            "mixtral-8x7b": {"input": 0.24, "output": 0.24},  # Groq pricing
            "gemini-1.5-pro": {"input": 3.50, "output": 10.50},  # Google pricing
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
                "name": "anthropic/claude-3-7-sonnet-latest",
                "provider": "Anthropic",
                "description": "Latest Claude 3.7 Sonnet with large context window (default)",
                "cost_tier": "Medium",
                "context_window": "200K",
            },
            {
                "name": "groq/llama-3.3-70b-versatile",
                "provider": "Groq",
                "description": "High-quality model with fast inference (fallback)",
                "cost_tier": "Low",
                "context_window": "131K",
            },
            {
                "name": "groq/llama-3.1-8b-instant",
                "provider": "Groq",
                "description": "Very fast model for quick summaries",
                "cost_tier": "Low",
                "context_window": "131K",
            },
            {
                "name": "groq/mixtral-8x7b-32768",
                "provider": "Groq",
                "description": "Balanced performance model",
                "cost_tier": "Low",
                "context_window": "32K",
            },
            {
                "name": "gpt-4o-mini",
                "provider": "OpenAI",
                "description": "Fast, cost-effective model for most tasks",
                "cost_tier": "Low",
                "context_window": "128K",
            },
            {
                "name": "gpt-4o",
                "provider": "OpenAI",
                "description": "High-quality model for complex analysis",
                "cost_tier": "Medium",
                "context_window": "128K",
            },
            {
                "name": "anthropic/claude-3-haiku-20240307",
                "provider": "Anthropic",
                "description": "Fast, efficient model for quick summaries",
                "cost_tier": "Low",
                "context_window": "200K",
            },
            {
                "name": "anthropic/claude-3-5-sonnet-20241022",
                "provider": "Anthropic",
                "description": "Advanced model for detailed analysis",
                "cost_tier": "Medium",
                "context_window": "200K",
            },
            {
                "name": "google/gemini-1.5-pro",
                "provider": "Google",
                "description": "Large context model for comprehensive analysis",
                "cost_tier": "Medium",
                "context_window": "2M",
            },
            {
                "name": "groq/moonshotai/kimi-k2-instruct",
                "provider": "Groq (Moonshot AI)",
                "description": "Preview model for evaluation (may be discontinued)",
                "cost_tier": "Low",
                "context_window": "131K",
            },
            {
                "name": "ollama/llama3.1",
                "provider": "Local (Ollama)",
                "description": "Local model for private usage",
                "cost_tier": "Free",
                "context_window": "128K",
            },
        ]
