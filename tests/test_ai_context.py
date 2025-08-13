"""Tests for AI context gathering engine."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import pytest

from git_summary.ai.context import ContextGatheringEngine, RichContext, TokenBudget
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


class TestTokenBudget:
    """Test token budget management."""

    def test_budget_initialization(self):
        """Test budget initialization."""
        budget = TokenBudget(max_tokens=1000)
        assert budget.max_tokens == 1000
        assert budget.used_tokens == 0
        assert budget.reserved_tokens == 0
        assert budget.remaining == 1000
        assert budget.utilization == 0.0

    def test_can_allocate(self):
        """Test allocation checking."""
        budget = TokenBudget(max_tokens=1000)
        assert budget.can_allocate(500) is True
        assert budget.can_allocate(1000) is True
        assert budget.can_allocate(1001) is False

    def test_reserve_tokens(self):
        """Test token reservation."""
        budget = TokenBudget(max_tokens=1000)

        # Successful reservation
        assert budget.reserve(300) is True
        assert budget.reserved_tokens == 300
        assert budget.remaining == 700

        # Another reservation
        assert budget.reserve(400) is True
        assert budget.reserved_tokens == 700
        assert budget.remaining == 300

        # Failed reservation (exceeds budget)
        assert budget.reserve(400) is False
        assert budget.reserved_tokens == 700

    def test_consume_reserved(self):
        """Test consuming reserved tokens."""
        budget = TokenBudget(max_tokens=1000)
        budget.reserve(500)

        # Consume some reserved tokens
        budget.consume_reserved(200)
        assert budget.used_tokens == 200
        assert budget.reserved_tokens == 300
        assert budget.remaining == 500

        # Try to consume more than reserved
        with pytest.raises(ValueError, match="Cannot consume"):
            budget.consume_reserved(400)

    def test_consume_direct(self):
        """Test direct token consumption."""
        budget = TokenBudget(max_tokens=1000)

        # Successful consumption
        assert budget.consume(300) is True
        assert budget.used_tokens == 300
        assert budget.remaining == 700

        # Failed consumption (exceeds budget)
        assert budget.consume(800) is False
        assert budget.used_tokens == 300

    def test_utilization_calculation(self):
        """Test utilization percentage calculation."""
        budget = TokenBudget(max_tokens=1000)

        # Empty budget
        assert budget.utilization == 0.0

        # Partial usage
        budget.consume(300)
        budget.reserve(200)
        assert budget.utilization == 0.5  # (300 + 200) / 1000

        # Full usage
        budget.consume(500)
        assert budget.utilization == 1.0

        # Zero max tokens edge case
        zero_budget = TokenBudget(max_tokens=0)
        assert zero_budget.utilization == 0.0


class TestRichContext:
    """Test rich context data structure."""

    def setup_method(self):
        """Set up test context."""
        self.context = RichContext()
        self.mock_event = create_mock_event("PushEvent", datetime.now(UTC))

    def test_context_initialization(self):
        """Test context initialization."""
        assert self.context.total_events == 0
        assert self.context.date_range is None
        assert len(self.context.repositories) == 0
        assert len(self.context.commits) == 0
        assert self.context.estimated_tokens == 0

    def test_add_commit(self):
        """Test adding commit information."""
        commit_details = {
            "sha": "abc123",
            "message": "Fix critical bug in authentication",
            "author": {"name": "Test User"},
            "stats": {"additions": 15, "deletions": 3},
            "files": ["auth.py", "test_auth.py"],
        }

        self.context.add_commit(self.mock_event, commit_details)

        assert len(self.context.commits) == 1
        assert len(self.context.commit_messages) == 1

        commit = self.context.commits[0]
        assert commit["sha"] == "abc123"
        assert commit["message"] == "Fix critical bug in authentication"
        assert commit["additions"] == 15
        assert commit["deletions"] == 3
        assert commit["files_changed"] == 2
        assert commit["repository"] == "test/repo"

        assert self.context.commit_messages[0] == "Fix critical bug in authentication"

    def test_add_pull_request(self):
        """Test adding pull request information."""
        pr_details = {
            "number": 42,
            "title": "Add user authentication system",
            "state": "merged",
            "user": {"login": "developer"},
            "merged": True,
            "additions": 150,
            "deletions": 20,
            "changed_files": 8,
            "comments": 5,
            "commits": 3,
        }

        self.context.add_pull_request(self.mock_event, pr_details)

        assert len(self.context.pull_requests) == 1
        assert len(self.context.pr_titles) == 1

        pr = self.context.pull_requests[0]
        assert pr["number"] == 42
        assert pr["title"] == "Add user authentication system"
        assert pr["state"] == "merged"
        assert pr["merged"] is True
        assert pr["additions"] == 150
        assert pr["changed_files"] == 8

        assert self.context.pr_titles[0] == "Add user authentication system"

    def test_add_issue(self):
        """Test adding issue information."""
        issue_details = {
            "number": 15,
            "title": "Login system throws 500 error",
            "state": "open",
            "user": {"login": "reporter"},
            "labels": [{"name": "bug"}, {"name": "priority-high"}],
            "comments": 3,
            "assignees": [{"login": "assignee1"}, {"login": "assignee2"}],
        }

        self.context.add_issue(self.mock_event, issue_details)

        assert len(self.context.issues) == 1
        assert len(self.context.issue_titles) == 1

        issue = self.context.issues[0]
        assert issue["number"] == 15
        assert issue["title"] == "Login system throws 500 error"
        assert issue["state"] == "open"
        assert issue["author"] == "reporter"
        assert issue["labels"] == ["bug", "priority-high"]
        assert issue["comments"] == 3
        assert issue["assignees"] == ["assignee1", "assignee2"]

        assert self.context.issue_titles[0] == "Login system throws 500 error"

    def test_add_release(self):
        """Test adding release information."""
        release_details = {
            "tag_name": "v2.1.0",
            "name": "Version 2.1.0 - Major Authentication Update",
            "author": {"login": "maintainer"},
            "prerelease": False,
            "draft": False,
            "body": "## Features\n- New OAuth2 support\n- Enhanced security\n\n## Bug Fixes\n- Fixed login errors",
            "assets": ["binary1.zip", "binary2.tar.gz"],
        }

        self.context.add_release(self.mock_event, release_details)

        assert len(self.context.releases) == 1
        assert len(self.context.release_notes) == 1

        release = self.context.releases[0]
        assert release["tag_name"] == "v2.1.0"
        assert release["name"] == "Version 2.1.0 - Major Authentication Update"
        assert release["author"] == "maintainer"
        assert release["prerelease"] is False
        assert release["assets_count"] == 2

        assert "OAuth2 support" in self.context.release_notes[0]

    def test_estimate_token_usage(self):
        """Test token usage estimation."""
        # Add some content
        self.context.commit_messages = ["Fix bug", "Add feature", "Update docs"]
        self.context.issue_titles = ["Bug report", "Feature request"]
        self.context.pr_titles = ["Fix authentication"]

        estimated = self.context.estimate_token_usage()

        # Should estimate based on character count / 4
        assert estimated > 0
        assert self.context.estimated_tokens == estimated

        # Add more content and verify increase
        self.context.commit_messages.append("Another commit with a much longer message")
        new_estimated = self.context.estimate_token_usage()
        assert new_estimated > estimated

    def test_prioritize_content_within_budget(self):
        """Test content prioritization when it fits within budget."""
        # Add minimal content
        self.context.commit_messages = ["Short commit"]
        self.context.pr_titles = ["Short PR"]

        budget = TokenBudget(max_tokens=10000)  # Large budget
        prioritized = self.context.prioritize_content(budget)

        # Should return the same content when budget is sufficient
        assert prioritized.commit_messages == self.context.commit_messages
        assert prioritized.pr_titles == self.context.pr_titles

    def test_prioritize_content_exceeds_budget(self):
        """Test content prioritization when content exceeds budget."""
        # Add lots of content
        self.context.commits = [
            {"repo": "test", "message": "commit " + str(i)} for i in range(100)
        ]
        self.context.commit_messages = [
            f"Very long commit message number {i} with lots of details"
            for i in range(100)
        ]
        self.context.pull_requests = [
            {"number": i, "title": f"PR {i}"} for i in range(50)
        ]
        self.context.pr_titles = [
            f"Pull request title {i} with detailed description" for i in range(50)
        ]
        self.context.releases = [{"tag": "v1.0.0", "notes": "Release notes"}]
        self.context.release_notes = ["Important release with new features"]

        budget = TokenBudget(
            max_tokens=200
        )  # Very small budget to force prioritization
        prioritized = self.context.prioritize_content(budget)

        # Should have fewer items
        assert len(prioritized.commits) < len(self.context.commits)
        assert len(prioritized.pull_requests) < len(self.context.pull_requests)

        # Releases should be preserved (highest priority)
        assert len(prioritized.releases) == len(self.context.releases)
        assert len(prioritized.release_notes) == len(self.context.release_notes)

        # Should fit within budget
        assert prioritized.estimate_token_usage() <= budget.remaining


class TestContextGatheringEngine:
    """Test the context gathering engine."""

    def setup_method(self):
        """Set up test engine."""
        self.github_client = Mock(spec=GitHubClient)
        self.engine = ContextGatheringEngine(self.github_client, default_budget=5000)

    def test_engine_initialization(self):
        """Test engine initialization."""
        assert self.engine.github_client is self.github_client
        assert self.engine.default_budget == 5000

    @pytest.mark.asyncio
    async def test_gather_context_empty_events(self):
        """Test gathering context from empty event list."""
        context = await self.engine.gather_context([])

        assert context.total_events == 0
        assert context.date_range is None
        assert len(context.repositories) == 0
        assert context.estimated_tokens == 0

    @pytest.mark.asyncio
    async def test_gather_context_basic_events(self):
        """Test gathering context from basic events."""
        now = datetime.now(UTC)
        events = [
            create_mock_event(
                "PushEvent",
                now,
                {"commits": [{"sha": "abc123", "message": "Initial commit"}]},
                "owner/repo1",
            ),
            create_mock_event(
                "IssuesEvent",
                now,
                {"issue": {"number": 1, "title": "Bug report", "state": "open"}},
                "owner/repo2",
            ),
        ]

        context = await self.engine.gather_context(events)

        assert context.total_events == 2
        assert context.date_range is not None
        assert len(context.repositories) == 2
        assert "owner/repo1" in context.repositories
        assert "owner/repo2" in context.repositories
        assert len(context.commits) == 1
        assert len(context.issues) == 1

    @pytest.mark.asyncio
    async def test_gather_context_with_custom_budget(self):
        """Test gathering context with custom budget."""
        events = [
            create_mock_event(
                "PushEvent",
                datetime.now(UTC),
                {"commits": [{"sha": "abc123", "message": "Commit"}]},
            )
        ]

        custom_budget = TokenBudget(max_tokens=100)
        context = await self.engine.gather_context(events, budget=custom_budget)

        assert context.total_events == 1
        # Budget should have been considered in prioritization
        assert context.estimated_tokens <= custom_budget.max_tokens

    @pytest.mark.asyncio
    async def test_process_push_events(self):
        """Test processing push events specifically."""
        events = [
            create_mock_event(
                "PushEvent",
                datetime.now(UTC),
                {
                    "commits": [
                        {
                            "sha": "abc123",
                            "message": "Fix authentication bug",
                            "author": {"name": "Dev1"},
                        },
                        {
                            "sha": "def456",
                            "message": "Add unit tests",
                            "author": {"name": "Dev2"},
                        },
                    ]
                },
            )
        ]

        context = await self.engine.gather_context(events)

        assert len(context.commits) == 2
        assert len(context.commit_messages) == 2
        assert "Fix authentication bug" in context.commit_messages
        assert "Add unit tests" in context.commit_messages

        # Check commit details
        commit1 = context.commits[0]
        assert commit1["sha"] == "abc123"
        assert commit1["message"] == "Fix authentication bug"

    @pytest.mark.asyncio
    async def test_process_pr_events(self):
        """Test processing pull request events."""
        events = [
            create_mock_event(
                "PullRequestEvent",
                datetime.now(UTC),
                {
                    "pull_request": {
                        "number": 42,
                        "title": "Implement OAuth2 authentication",
                        "state": "merged",
                        "user": {"login": "developer"},
                        "merged": True,
                        "additions": 150,
                        "deletions": 25,
                        "changed_files": 8,
                    }
                },
            )
        ]

        context = await self.engine.gather_context(events)

        assert len(context.pull_requests) == 1
        assert len(context.pr_titles) == 1
        assert "Implement OAuth2 authentication" in context.pr_titles

        pr = context.pull_requests[0]
        assert pr["number"] == 42
        assert pr["merged"] is True
        assert pr["additions"] == 150

    @pytest.mark.asyncio
    async def test_process_mixed_event_types(self):
        """Test processing multiple event types together."""
        now = datetime.now(UTC)
        events = [
            create_mock_event(
                "PushEvent", now, {"commits": [{"sha": "abc123", "message": "Fix bug"}]}
            ),
            create_mock_event(
                "PullRequestEvent",
                now,
                {
                    "pull_request": {
                        "number": 1,
                        "title": "Add feature",
                        "state": "open",
                    }
                },
            ),
            create_mock_event(
                "IssuesEvent",
                now,
                {"issue": {"number": 1, "title": "Bug report", "state": "closed"}},
            ),
            create_mock_event(
                "ReleaseEvent",
                now,
                {
                    "release": {
                        "tag_name": "v1.0.0",
                        "name": "First Release",
                        "body": "Initial release",
                    }
                },
            ),
        ]

        context = await self.engine.gather_context(events)

        assert context.total_events == 4
        assert len(context.commits) == 1
        assert len(context.pull_requests) == 1
        assert len(context.issues) == 1
        assert len(context.releases) == 1

        # Check that all types were processed
        assert len(context.commit_messages) == 1
        assert len(context.pr_titles) == 1
        assert len(context.issue_titles) == 1
        assert len(context.release_notes) == 1

    @pytest.mark.asyncio
    async def test_process_events_with_missing_payload(self):
        """Test handling events with missing or empty payloads."""
        events = [
            create_mock_event("PushEvent", datetime.now(UTC), None),
            create_mock_event("PushEvent", datetime.now(UTC), {}),
            create_mock_event("PushEvent", datetime.now(UTC), {"commits": []}),
        ]

        context = await self.engine.gather_context(events)

        assert context.total_events == 3
        assert len(context.commits) == 0  # No valid commits to process
        assert len(context.commit_messages) == 0
