"""Tests for AI client wrapper."""

from unittest.mock import Mock, patch

import pytest

from git_summary.ai.client import LLMClient, LLMError


class TestLLMClient:
    """Test the LiteLLM client wrapper."""

    def setup_method(self):
        """Set up test client."""
        self.client = LLMClient(
            model="gpt-4o-mini", fallback_model="anthropic/claude-3-haiku-20240307"
        )

    def test_client_initialization(self):
        """Test client initialization."""
        assert self.client.model == "gpt-4o-mini"
        assert self.client.fallback_model == "anthropic/claude-3-haiku-20240307"
        assert self.client.max_tokens == 2000
        assert self.client.temperature == 0.7

    @patch("git_summary.ai.client.acompletion")
    @pytest.mark.asyncio
    async def test_generate_summary_success(self, mock_acompletion):
        """Test successful summary generation."""
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a test summary."
        mock_acompletion.return_value = mock_response

        result = await self.client.generate_summary(
            "You are a helpful assistant.", "Summarize this test content."
        )

        assert result == "This is a test summary."
        mock_acompletion.assert_called_once()

    @patch("git_summary.ai.client.acompletion")
    @pytest.mark.asyncio
    async def test_generate_summary_with_fallback(self, mock_acompletion):
        """Test fallback when primary model fails."""
        # First call fails, second succeeds
        mock_acompletion.side_effect = [
            Exception("Primary model failed"),
            Mock(choices=[Mock(message=Mock(content="Fallback summary"))]),
        ]

        result = await self.client.generate_summary(
            "You are a helpful assistant.", "Summarize this test content."
        )

        assert result == "Fallback summary"
        assert mock_acompletion.call_count == 2

    @patch("git_summary.ai.client.acompletion")
    @pytest.mark.asyncio
    async def test_generate_summary_both_models_fail(self, mock_acompletion):
        """Test error when both primary and fallback models fail."""
        mock_acompletion.side_effect = Exception("Both models failed")

        with pytest.raises(LLMError, match="Both primary"):
            await self.client.generate_summary(
                "You are a helpful assistant.", "Summarize this test content."
            )

    @patch("git_summary.ai.client.completion")
    def test_generate_summary_sync(self, mock_completion):
        """Test synchronous summary generation."""
        # Mock successful response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Sync summary result."
        mock_completion.return_value = mock_response

        result = self.client.generate_summary_sync(
            "You are a helpful assistant.", "Summarize this test content."
        )

        assert result == "Sync summary result."
        mock_completion.assert_called_once()

    def test_estimate_cost(self):
        """Test cost estimation functionality."""
        cost_info = self.client.estimate_cost(
            "You are a helpful assistant.", "This is test content to summarize."
        )

        assert "estimated_input_tokens" in cost_info
        assert "estimated_output_tokens" in cost_info
        assert "estimated_cost_usd" in cost_info
        assert "model" in cost_info
        assert cost_info["model"] == "gpt-4o-mini"

    def test_get_available_models(self):
        """Test getting available models list."""
        models = self.client.get_available_models()

        assert len(models) > 0
        assert all("name" in model for model in models)
        assert all("provider" in model for model in models)
        assert all("cost_tier" in model for model in models)

        # Check that our default models are in the list
        model_names = [m["name"] for m in models]
        assert "gpt-4o-mini" in model_names

    def test_client_with_custom_parameters(self):
        """Test client with custom parameters."""
        custom_client = LLMClient(
            model="gpt-4o", max_tokens=1000, temperature=0.3, timeout=60
        )

        assert custom_client.model == "gpt-4o"
        assert custom_client.max_tokens == 1000
        assert custom_client.temperature == 0.3
        assert custom_client.timeout == 60
