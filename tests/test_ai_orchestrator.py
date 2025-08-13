"""Tests for AI Summary Orchestrator Pipeline."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from git_summary.ai.client import LLMClient
from git_summary.ai.context import RichContext
from git_summary.ai.orchestrator import ActivitySummarizer
from git_summary.ai.personas import PersonaManager, TechnicalAnalystPersona
from git_summary.github_client import GitHubClient
from git_summary.models import Actor, Repository


class MockGitHubEvent:
    """Mock GitHub event for testing."""

    def __init__(
        self,
        event_type: str,
        created_at: datetime,
        payload: dict[str, Any] | None = None,
        repo_name: str = "test/repo",
    ):
        self.id = "123456"
        self.type = event_type
        self.created_at = created_at.isoformat()
        self.actor = Actor(id=1, login="testuser", url="", avatar_url="")
        self.repo = Repository(id=1, name=repo_name, url="")
        self.payload = payload or {}
        self.public = True


def create_mock_event(
    event_type: str,
    created_at: datetime,
    payload: dict[str, Any] | None = None,
    repo_name: str = "test/repo",
) -> MockGitHubEvent:
    """Create a mock GitHub event for testing."""
    return MockGitHubEvent(event_type, created_at, payload, repo_name)


class TestActivitySummarizer:
    """Test the ActivitySummarizer orchestrator."""

    def setup_method(self):
        """Set up test dependencies."""
        self.github_client = Mock(spec=GitHubClient)
        self.llm_client = Mock(spec=LLMClient)
        self.persona_manager = Mock(spec=PersonaManager)

        # Set up default mock responses
        self.llm_client.model = "gpt-4o-mini"
        self.llm_client.generate_summary = AsyncMock(return_value="Test summary")
        self.llm_client.estimate_cost = Mock(
            return_value={"total_cost": 0.001, "input_tokens": 100, "output_tokens": 50}
        )

        # Set up persona manager
        self.mock_persona = Mock(spec=TechnicalAnalystPersona)
        self.mock_persona.name = "Tech Analyst"
        self.mock_persona.get_system_prompt.return_value = "You are a technical analyst"
        self.mock_persona.get_summary_instructions.return_value = "Analyze this data"
        self.persona_manager.get_persona.return_value = self.mock_persona
        self.persona_manager.list_personas.return_value = [self.mock_persona]

    def test_summarizer_initialization_with_defaults(self):
        """Test that summarizer initializes with default components."""
        summarizer = ActivitySummarizer(self.github_client)

        assert summarizer.github_client is self.github_client
        assert isinstance(summarizer.llm_client, LLMClient)
        assert isinstance(summarizer.persona_manager, PersonaManager)
        assert summarizer.default_token_budget == 8000

    def test_summarizer_initialization_with_custom_components(self):
        """Test initialization with custom components."""
        summarizer = ActivitySummarizer(
            self.github_client,
            llm_client=self.llm_client,
            persona_manager=self.persona_manager,
            default_token_budget=5000,
        )

        assert summarizer.github_client is self.github_client
        assert summarizer.llm_client is self.llm_client
        assert summarizer.persona_manager is self.persona_manager
        assert summarizer.default_token_budget == 5000

    @pytest.mark.asyncio
    async def test_generate_summary_success(self):
        """Test successful summary generation."""
        summarizer = ActivitySummarizer(
            self.github_client,
            llm_client=self.llm_client,
            persona_manager=self.persona_manager,
        )

        # Mock context gathering
        mock_context = RichContext(
            total_events=2,
            repositories=["test/repo"],
            commits=[{"sha": "abc123", "message": "Test commit"}],
            commit_messages=["Test commit"],
            estimated_tokens=100,
        )
        mock_context.date_range = (datetime.now(UTC), datetime.now(UTC))

        with patch.object(
            summarizer.context_engine, "gather_context", return_value=mock_context
        ):
            events = [
                create_mock_event("PushEvent", datetime.now(UTC)),
                create_mock_event("IssuesEvent", datetime.now(UTC)),
            ]

            result = await summarizer.generate_summary(
                events, persona_name="tech_analyst"
            )

            # Verify the response structure
            assert "summary" in result
            assert "persona_used" in result
            assert "model_used" in result
            assert "metadata" in result
            assert "context" in result  # include_context_details=True by default

            assert result["summary"] == "Test summary"
            assert result["persona_used"] == "tech_analyst"
            assert result["model_used"] == "gpt-4o-mini"
            assert result["metadata"]["total_events"] == 2
            assert result["metadata"]["tokens_used"] == 100

    @pytest.mark.asyncio
    async def test_generate_summary_without_context_details(self):
        """Test summary generation without detailed context."""
        summarizer = ActivitySummarizer(
            self.github_client,
            llm_client=self.llm_client,
            persona_manager=self.persona_manager,
        )

        mock_context = RichContext(total_events=1, estimated_tokens=50)
        mock_context.date_range = (datetime.now(UTC), datetime.now(UTC))

        with patch.object(
            summarizer.context_engine, "gather_context", return_value=mock_context
        ):
            events = [create_mock_event("PushEvent", datetime.now(UTC))]

            result = await summarizer.generate_summary(
                events, include_context_details=False
            )

            assert "summary" in result
            assert "metadata" in result
            assert "context" not in result  # Should not include detailed context

    @pytest.mark.asyncio
    async def test_generate_summary_persona_not_found(self):
        """Test error handling when persona is not found."""
        self.persona_manager.get_persona.return_value = None

        summarizer = ActivitySummarizer(
            self.github_client,
            llm_client=self.llm_client,
            persona_manager=self.persona_manager,
        )

        mock_context = RichContext(total_events=1)
        with patch.object(
            summarizer.context_engine, "gather_context", return_value=mock_context
        ):
            events = [create_mock_event("PushEvent", datetime.now(UTC))]

            with pytest.raises(ValueError, match="Persona 'nonexistent' not found"):
                await summarizer.generate_summary(events, persona_name="nonexistent")

    @pytest.mark.asyncio
    async def test_generate_summary_llm_failure(self):
        """Test error handling when LLM generation fails."""
        self.llm_client.generate_summary.side_effect = Exception("LLM API error")

        summarizer = ActivitySummarizer(
            self.github_client,
            llm_client=self.llm_client,
            persona_manager=self.persona_manager,
        )

        mock_context = RichContext(total_events=1)
        with patch.object(
            summarizer.context_engine, "gather_context", return_value=mock_context
        ):
            events = [create_mock_event("PushEvent", datetime.now(UTC))]

            with pytest.raises(RuntimeError, match="Failed to generate summary"):
                await summarizer.generate_summary(events)

    def test_generate_summary_sync(self):
        """Test synchronous summary generation."""
        summarizer = ActivitySummarizer(
            self.github_client,
            llm_client=self.llm_client,
            persona_manager=self.persona_manager,
        )

        mock_context = RichContext(total_events=1, estimated_tokens=50)
        mock_context.date_range = (datetime.now(UTC), datetime.now(UTC))

        with patch.object(
            summarizer.context_engine, "gather_context", return_value=mock_context
        ):
            events = [create_mock_event("PushEvent", datetime.now(UTC))]

            result = summarizer.generate_summary_sync(events)

            assert "summary" in result
            assert result["summary"] == "Test summary"

    @pytest.mark.asyncio
    async def test_estimate_cost_success(self):
        """Test successful cost estimation."""
        summarizer = ActivitySummarizer(
            self.github_client,
            llm_client=self.llm_client,
            persona_manager=self.persona_manager,
        )

        mock_context = RichContext(
            total_events=3,
            commits=[{"sha": "abc123"}],
            pull_requests=[{"number": 1}],
            issues=[{"number": 2}],
            estimated_tokens=150,
        )

        with patch.object(
            summarizer.context_engine, "gather_context", return_value=mock_context
        ):
            events = [
                create_mock_event("PushEvent", datetime.now(UTC)),
                create_mock_event("PullRequestEvent", datetime.now(UTC)),
                create_mock_event("IssuesEvent", datetime.now(UTC)),
            ]

            result = await summarizer.estimate_cost(events)

            assert "cost_estimate" in result
            assert "context_tokens" in result
            assert "total_events" in result
            assert "event_breakdown" in result
            assert "persona_used" in result
            assert "model" in result

            assert result["cost_estimate"]["total_cost"] == 0.001
            assert result["context_tokens"] == 150
            assert result["total_events"] == 3
            assert result["event_breakdown"]["commits"] == 1
            assert result["event_breakdown"]["pull_requests"] == 1
            assert result["event_breakdown"]["issues"] == 1

    @pytest.mark.asyncio
    async def test_estimate_cost_persona_not_found(self):
        """Test cost estimation error when persona not found."""
        self.persona_manager.get_persona.return_value = None

        summarizer = ActivitySummarizer(
            self.github_client,
            llm_client=self.llm_client,
            persona_manager=self.persona_manager,
        )

        mock_context = RichContext(total_events=1)
        with patch.object(
            summarizer.context_engine, "gather_context", return_value=mock_context
        ):
            events = [create_mock_event("PushEvent", datetime.now(UTC))]

            with pytest.raises(ValueError, match="Persona 'invalid' not found"):
                await summarizer.estimate_cost(events, persona_name="invalid")

    def test_get_available_personas(self):
        """Test getting available personas."""
        summarizer = ActivitySummarizer(
            self.github_client,
            llm_client=self.llm_client,
            persona_manager=self.persona_manager,
        )

        personas = summarizer.get_available_personas()

        assert personas == ["tech_analyst"]
        self.persona_manager.list_personas.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_token_budget(self):
        """Test using custom token budget."""
        summarizer = ActivitySummarizer(
            self.github_client,
            llm_client=self.llm_client,
            persona_manager=self.persona_manager,
        )

        mock_context = RichContext(total_events=1, estimated_tokens=200)
        mock_context.date_range = (datetime.now(UTC), datetime.now(UTC))

        with patch.object(summarizer.context_engine, "gather_context") as mock_gather:
            mock_gather.return_value = mock_context
            events = [create_mock_event("PushEvent", datetime.now(UTC))]

            await summarizer.generate_summary(events, token_budget=1000)

            # Verify that the custom budget was passed
            mock_gather.assert_called_once()
            call_args = mock_gather.call_args
            # Budget is passed as second positional argument
            budget = call_args.args[1]
            assert budget.max_tokens == 1000

    def test_format_context_for_llm(self):
        """Test context formatting for LLM consumption."""
        summarizer = ActivitySummarizer(self.github_client)

        rich_context = RichContext(
            total_events=2,
            repositories=["repo1", "repo2"],
            commits=[{"sha": "abc123", "message": "Test"}],
            commit_messages=["Test commit"],
            pull_requests=[{"number": 1, "title": "PR"}],
            pr_titles=["Test PR"],
            estimated_tokens=100,
        )
        rich_context.date_range = (datetime.now(UTC), datetime.now(UTC))

        formatted = summarizer._format_context_for_llm(rich_context)

        assert formatted["total_events"] == 2
        assert formatted["repositories"] == ["repo1", "repo2"]
        assert formatted["commits"] == [{"sha": "abc123", "message": "Test"}]
        assert formatted["commit_messages"] == ["Test commit"]
        assert formatted["pull_requests"] == [{"number": 1, "title": "PR"}]
        assert formatted["pr_titles"] == ["Test PR"]
        assert formatted["estimated_tokens"] == 100
        assert "date_range" in formatted

    @pytest.mark.asyncio
    async def test_empty_events_list(self):
        """Test handling of empty events list."""
        summarizer = ActivitySummarizer(
            self.github_client,
            llm_client=self.llm_client,
            persona_manager=self.persona_manager,
        )

        mock_context = RichContext(total_events=0, estimated_tokens=0)

        with patch.object(
            summarizer.context_engine, "gather_context", return_value=mock_context
        ):
            result = await summarizer.generate_summary([])

            assert result["metadata"]["total_events"] == 0
            assert result["metadata"]["tokens_used"] == 0
            assert result["metadata"]["event_breakdown"]["commits"] == 0
