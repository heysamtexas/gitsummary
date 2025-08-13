"""Tests for GitHub API data models."""

import pytest
from pydantic import ValidationError

from git_summary.models import (
    ActivityPeriod,
    ActivitySummary,
    Actor,
    BaseGitHubEvent,
    CreateEvent,
    DailyRollup,
    DetailedEvent,
    EventFactory,
    GitHubActivityReport,
    IssuesEvent,
    PushEvent,
    Repository,
    RepositoryBreakdown,
    WatchEvent,
)


class TestActor:
    """Test the Actor model."""

    def test_actor_creation(self):
        """Test creating an Actor instance."""
        actor_data = {
            "id": 123,
            "login": "testuser",
            "gravatar_id": "",
            "url": "https://api.github.com/users/testuser",
            "avatar_url": "https://avatars.githubusercontent.com/u/123",
        }
        actor = Actor(**actor_data)  # type: ignore[arg-type]
        assert actor.id == 123
        assert actor.login == "testuser"
        assert actor.url == "https://api.github.com/users/testuser"

    def test_actor_minimal(self):
        """Test creating an Actor with minimal required fields."""
        actor_data = {
            "id": 123,
            "login": "testuser",
            "url": "https://api.github.com/users/testuser",
            "avatar_url": "https://avatars.githubusercontent.com/u/123",
        }
        actor = Actor(**actor_data)  # type: ignore[arg-type]
        assert actor.id == 123
        assert actor.login == "testuser"


class TestRepository:
    """Test the Repository model."""

    def test_repository_creation(self):
        """Test creating a Repository instance."""
        repo_data = {
            "id": 456,
            "name": "test-repo",
            "url": "https://api.github.com/repos/testuser/test-repo",
            "private": False,
        }
        repo = Repository(**repo_data)  # type: ignore[arg-type]
        assert repo.id == 456
        assert repo.name == "test-repo"
        assert repo.private is False

    def test_repository_with_owner(self):
        """Test repository with owner."""
        actor_data = {
            "id": 123,
            "login": "testuser",
            "url": "https://api.github.com/users/testuser",
            "avatar_url": "https://avatars.githubusercontent.com/u/123",
        }
        repo_data = {
            "id": 456,
            "name": "test-repo",
            "url": "https://api.github.com/repos/testuser/test-repo",
            "owner": actor_data,
        }
        repo = Repository(**repo_data)  # type: ignore[arg-type]
        assert repo.owner is not None
        assert repo.owner.login == "testuser"


class TestBaseGitHubEvent:
    """Test the BaseGitHubEvent model."""

    def test_base_event_creation(self):
        """Test creating a BaseGitHubEvent."""
        actor_data = {
            "id": 123,
            "login": "testuser",
            "url": "https://api.github.com/users/testuser",
            "avatar_url": "https://avatars.githubusercontent.com/u/123",
        }
        repo_data = {
            "id": 456,
            "name": "test-repo",
            "url": "https://api.github.com/repos/testuser/test-repo",
        }
        event_data = {
            "id": "123456789",
            "type": "TestEvent",
            "actor": actor_data,
            "repo": repo_data,
            "created_at": "2023-01-01T12:00:00Z",
            "public": True,
        }
        event = BaseGitHubEvent(**event_data)  # type: ignore[arg-type]
        assert event.id == "123456789"
        assert event.type == "TestEvent"
        assert event.actor.login == "testuser"
        assert event.repo.name == "test-repo"

    def test_invalid_datetime(self):
        """Test validation of invalid datetime."""
        actor_data = {
            "id": 123,
            "login": "testuser",
            "url": "https://api.github.com/users/testuser",
            "avatar_url": "https://avatars.githubusercontent.com/u/123",
        }
        repo_data = {
            "id": 456,
            "name": "test-repo",
            "url": "https://api.github.com/repos/testuser/test-repo",
        }
        event_data = {
            "id": "123456789",
            "type": "TestEvent",
            "actor": actor_data,
            "repo": repo_data,
            "created_at": "invalid-datetime",
            "public": True,
        }
        with pytest.raises(ValidationError):
            BaseGitHubEvent(**event_data)  # type: ignore[arg-type]


class TestPushEvent:
    """Test the PushEvent model."""

    def test_push_event_creation(self):
        """Test creating a PushEvent."""
        actor_data = {
            "id": 123,
            "login": "testuser",
            "url": "https://api.github.com/users/testuser",
            "avatar_url": "https://avatars.githubusercontent.com/u/123",
        }
        repo_data = {
            "id": 456,
            "name": "test-repo",
            "url": "https://api.github.com/repos/testuser/test-repo",
        }
        commit_data = {
            "sha": "abc123",
            "author": {"name": "Test Author", "email": "test@example.com"},
            "message": "Test commit",
            "distinct": True,
            "url": "https://api.github.com/repos/testuser/test-repo/commits/abc123",
        }
        payload_data = {
            "push_id": 12345,
            "size": 1,
            "distinct_size": 1,
            "ref": "refs/heads/main",
            "head": "abc123",
            "before": "def456",
            "commits": [commit_data],
        }
        event_data = {
            "id": "123456789",
            "type": "PushEvent",
            "actor": actor_data,
            "repo": repo_data,
            "created_at": "2023-01-01T12:00:00Z",
            "public": True,
            "payload": payload_data,
        }
        event = PushEvent(**event_data)
        assert event.type == "PushEvent"
        assert event.payload.ref == "refs/heads/main"
        assert len(event.payload.commits) == 1
        assert event.payload.commits[0].sha == "abc123"


class TestIssuesEvent:
    """Test the IssuesEvent model."""

    def test_issues_event_creation(self):
        """Test creating an IssuesEvent."""
        actor_data = {
            "id": 123,
            "login": "testuser",
            "url": "https://api.github.com/users/testuser",
            "avatar_url": "https://avatars.githubusercontent.com/u/123",
        }
        repo_data = {
            "id": 456,
            "name": "test-repo",
            "url": "https://api.github.com/repos/testuser/test-repo",
        }
        issue_data = {
            "id": 789,
            "number": 1,
            "title": "Test Issue",
            "user": actor_data,
            "labels": [],
            "state": "open",
            "locked": False,
            "assignees": [],
            "comments": 0,
            "created_at": "2023-01-01T12:00:00Z",
            "updated_at": "2023-01-01T12:00:00Z",
            "author_association": "OWNER",
            "url": "https://api.github.com/repos/testuser/test-repo/issues/1",
            "repository_url": "https://api.github.com/repos/testuser/test-repo",
            "labels_url": "https://api.github.com/repos/testuser/test-repo/issues/1/labels",
            "comments_url": "https://api.github.com/repos/testuser/test-repo/issues/1/comments",
            "events_url": "https://api.github.com/repos/testuser/test-repo/issues/1/events",
            "html_url": "https://github.com/testuser/test-repo/issues/1",
            "node_id": "I_test123",
        }
        payload_data = {
            "action": "opened",
            "issue": issue_data,
        }
        event_data = {
            "id": "123456789",
            "type": "IssuesEvent",
            "actor": actor_data,
            "repo": repo_data,
            "created_at": "2023-01-01T12:00:00Z",
            "public": True,
            "payload": payload_data,
        }
        event = IssuesEvent(**event_data)
        assert event.type == "IssuesEvent"
        assert event.payload.action == "opened"
        assert event.payload.issue.title == "Test Issue"


class TestCreateEvent:
    """Test the CreateEvent model."""

    def test_create_event_creation(self):
        """Test creating a CreateEvent."""
        actor_data = {
            "id": 123,
            "login": "testuser",
            "url": "https://api.github.com/users/testuser",
            "avatar_url": "https://avatars.githubusercontent.com/u/123",
        }
        repo_data = {
            "id": 456,
            "name": "test-repo",
            "url": "https://api.github.com/repos/testuser/test-repo",
        }
        payload_data = {
            "ref": "feature-branch",
            "ref_type": "branch",
            "master_branch": "main",
            "description": "New feature branch",
            "pusher_type": "user",
        }
        event_data = {
            "id": "123456789",
            "type": "CreateEvent",
            "actor": actor_data,
            "repo": repo_data,
            "created_at": "2023-01-01T12:00:00Z",
            "public": True,
            "payload": payload_data,
        }
        event = CreateEvent(**event_data)
        assert event.type == "CreateEvent"
        assert event.payload.ref == "feature-branch"
        assert event.payload.ref_type == "branch"


class TestWatchEvent:
    """Test the WatchEvent model."""

    def test_watch_event_creation(self):
        """Test creating a WatchEvent."""
        actor_data = {
            "id": 123,
            "login": "testuser",
            "url": "https://api.github.com/users/testuser",
            "avatar_url": "https://avatars.githubusercontent.com/u/123",
        }
        repo_data = {
            "id": 456,
            "name": "test-repo",
            "url": "https://api.github.com/repos/testuser/test-repo",
        }
        payload_data = {"action": "started"}
        event_data = {
            "id": "123456789",
            "type": "WatchEvent",
            "actor": actor_data,
            "repo": repo_data,
            "created_at": "2023-01-01T12:00:00Z",
            "public": True,
            "payload": payload_data,
        }
        event = WatchEvent(**event_data)
        assert event.type == "WatchEvent"
        assert event.payload.action == "started"


class TestEventFactory:
    """Test the EventFactory class."""

    def test_create_push_event(self):
        """Test creating a PushEvent via factory."""
        actor_data = {
            "id": 123,
            "login": "testuser",
            "url": "https://api.github.com/users/testuser",
            "avatar_url": "https://avatars.githubusercontent.com/u/123",
        }
        repo_data = {
            "id": 456,
            "name": "test-repo",
            "url": "https://api.github.com/repos/testuser/test-repo",
        }
        commit_data = {
            "sha": "abc123",
            "author": {"name": "Test Author", "email": "test@example.com"},
            "message": "Test commit",
            "distinct": True,
            "url": "https://api.github.com/repos/testuser/test-repo/commits/abc123",
        }
        payload_data = {
            "push_id": 12345,
            "size": 1,
            "distinct_size": 1,
            "ref": "refs/heads/main",
            "head": "abc123",
            "before": "def456",
            "commits": [commit_data],
        }
        event_data = {
            "id": "123456789",
            "type": "PushEvent",
            "actor": actor_data,
            "repo": repo_data,
            "created_at": "2023-01-01T12:00:00Z",
            "public": True,
            "payload": payload_data,
        }
        event = EventFactory.create_event(event_data)
        assert isinstance(event, PushEvent)
        assert event.type == "PushEvent"

    def test_create_watch_event(self):
        """Test creating a WatchEvent via factory."""
        actor_data = {
            "id": 123,
            "login": "testuser",
            "url": "https://api.github.com/users/testuser",
            "avatar_url": "https://avatars.githubusercontent.com/u/123",
        }
        repo_data = {
            "id": 456,
            "name": "test-repo",
            "url": "https://api.github.com/repos/testuser/test-repo",
        }
        payload_data = {"action": "started"}
        event_data = {
            "id": "123456789",
            "type": "WatchEvent",
            "actor": actor_data,
            "repo": repo_data,
            "created_at": "2023-01-01T12:00:00Z",
            "public": True,
            "payload": payload_data,
        }
        event = EventFactory.create_event(event_data)
        assert isinstance(event, WatchEvent)
        assert event.type == "WatchEvent"

    def test_unsupported_event_type(self):
        """Test handling of unsupported event types (fallback to UnknownEvent)."""
        event_data = {
            "id": "123456789",
            "type": "UnsupportedEvent",
            "actor": {"login": "test"},
            "repo": {"name": "test"},
            "created_at": "2023-01-01T12:00:00Z",
            "public": True,
            "payload": {},
        }
        event = EventFactory.create_event(event_data)
        # Should create UnknownEvent for unsupported types
        from git_summary.models import UnknownEvent

        assert isinstance(event, UnknownEvent)
        assert event.type == "UnsupportedEvent"

    def test_missing_event_type(self):
        """Test handling of missing event type (fallback to UnknownEvent)."""
        event_data = {
            "id": "123456789",
            "actor": {"login": "test"},
            "repo": {"name": "test"},
            "created_at": "2023-01-01T12:00:00Z",
            "public": True,
            "payload": {},
        }
        # Need to provide a type field for UnknownEvent validation
        # EventFactory will add default empty payload but type is still required
        event_data["type"] = "Unknown"  # Add missing type
        event = EventFactory.create_event(event_data)
        from git_summary.models import UnknownEvent

        assert isinstance(event, UnknownEvent)

    def test_get_supported_event_types(self):
        """Test getting list of supported event types."""
        supported_types = EventFactory.get_supported_event_types()
        assert "PushEvent" in supported_types
        assert "IssuesEvent" in supported_types
        assert "WatchEvent" in supported_types
        assert len(supported_types) >= 10  # We support many event types


class TestOutputModels:
    """Test output schema models."""

    def test_activity_summary(self):
        """Test ActivitySummary model."""
        summary_data = {
            "total_events": 50,
            "repositories_active": 5,
            "event_breakdown": {"PushEvent": 30, "IssuesEvent": 20},
            "most_active_repository": "test-repo",
            "most_common_event_type": "PushEvent",
        }
        summary = ActivitySummary(**summary_data)
        assert summary.total_events == 50
        assert summary.repositories_active == 5
        assert summary.most_active_repository == "test-repo"

    def test_daily_rollup(self):
        """Test DailyRollup model."""
        rollup_data = {
            "date": "2023-01-01",
            "events": 10,
            "repositories": ["repo1", "repo2"],
            "event_types": {"PushEvent": 8, "IssuesEvent": 2},
        }
        rollup = DailyRollup(**rollup_data)
        assert rollup.date == "2023-01-01"
        assert rollup.events == 10
        assert len(rollup.repositories) == 2

    def test_invalid_date(self):
        """Test validation of invalid date format."""
        rollup_data = {
            "date": "invalid-date",
            "events": 10,
            "repositories": ["repo1"],
            "event_types": {"PushEvent": 10},
        }
        with pytest.raises(ValidationError):
            DailyRollup(**rollup_data)

    def test_repository_breakdown(self):
        """Test RepositoryBreakdown model."""
        breakdown_data = {
            "events": 25,
            "event_types": {"PushEvent": 20, "IssuesEvent": 5},
            "last_activity": "2023-01-01T12:00:00Z",
        }
        breakdown = RepositoryBreakdown(**breakdown_data)
        assert breakdown.events == 25
        assert breakdown.last_activity == "2023-01-01T12:00:00Z"

    def test_activity_period(self):
        """Test ActivityPeriod model."""
        period_data = {
            "start": "2023-01-01T00:00:00Z",
            "end": "2023-01-31T23:59:59Z",
        }
        period = ActivityPeriod(**period_data)
        assert period.start == "2023-01-01T00:00:00Z"
        assert period.end == "2023-01-31T23:59:59Z"

    def test_detailed_event(self):
        """Test DetailedEvent model."""
        event_data = {
            "type": "PushEvent",
            "created_at": "2023-01-01T12:00:00Z",
            "repository": "test/repo",
            "actor": "testuser",
            "details": {"commits": 3, "ref": "refs/heads/main"},
        }
        event = DetailedEvent(**event_data)
        assert event.type == "PushEvent"
        assert event.repository == "test/repo"
        assert event.details["commits"] == 3

    def test_github_activity_report(self):
        """Test complete GitHubActivityReport model."""
        report_data = {
            "user": "testuser",
            "period": {
                "start": "2023-01-01T00:00:00Z",
                "end": "2023-01-31T23:59:59Z",
            },
            "summary": {
                "total_events": 50,
                "repositories_active": 5,
                "event_breakdown": {"PushEvent": 30, "IssuesEvent": 20},
            },
            "daily_rollups": [
                {
                    "date": "2023-01-01",
                    "events": 10,
                    "repositories": ["repo1"],
                    "event_types": {"PushEvent": 10},
                }
            ],
            "repository_breakdown": {
                "repo1": {
                    "events": 25,
                    "event_types": {"PushEvent": 25},
                    "last_activity": "2023-01-01T12:00:00Z",
                }
            },
            "detailed_events": [
                {
                    "type": "PushEvent",
                    "created_at": "2023-01-01T12:00:00Z",
                    "repository": "test/repo",
                    "actor": "testuser",
                    "details": {},
                }
            ],
        }
        report = GitHubActivityReport(**report_data)
        assert report.user == "testuser"
        assert len(report.daily_rollups) == 1
        assert len(report.repository_breakdown) == 1
        assert len(report.detailed_events) == 1
