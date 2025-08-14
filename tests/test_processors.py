"""Event processor tests using actual GitHub API data structures.

Following Guilfoyle's philosophy: Use real data shapes, test actual logic.
"""
from datetime import UTC, datetime

from git_summary.models import Actor, BaseGitHubEvent, Repository
from git_summary.processors import EventProcessor


def create_github_event(event_type, repo_name, created_at, **payload_data):
    """Helper to create realistic GitHub events."""
    return BaseGitHubEvent(
        id=f"test_{event_type}_{repo_name.replace('/', '_')}",
        type=event_type,
        actor=Actor(
            id=12345,
            login="testuser",
            url="https://api.github.com/users/testuser",
            avatar_url="https://avatars.githubusercontent.com/u/12345",
        ),
        repo=Repository(
            id=67890, name=repo_name, url=f"https://api.github.com/repos/{repo_name}"
        ),
        created_at=created_at,
        public=True,  # Add required field
        payload=payload_data,
    )


class TestEventProcessor:
    """Test event processing with realistic GitHub data."""

    def setup_method(self):
        self.processor = EventProcessor()

    def test_process_push_event_details(self):
        """Test extracting details from a real PushEvent structure."""
        # Create a mock event with the minimal interface the processor needs
        from types import SimpleNamespace

        # Create mock event that looks like what the processor expects
        commit1 = SimpleNamespace(message="Fix critical bug")
        commit2 = SimpleNamespace(message="Add new feature")
        payload = SimpleNamespace(commits=[commit1, commit2], ref="refs/heads/main")

        mock_event = SimpleNamespace(type="PushEvent", payload=payload)

        details = self.processor._extract_event_details(mock_event)

        assert details["commits_count"] == 2
        assert details["ref"] == "refs/heads/main"
        assert "Fix critical bug" in details["commit_messages"]
        assert "Add new feature" in details["commit_messages"]

    def test_process_issues_event_details(self):
        """Test extracting details from IssuesEvent."""
        from types import SimpleNamespace

        issue = SimpleNamespace(title="Bug report", number=42)
        payload = SimpleNamespace(action="opened", issue=issue)
        mock_event = SimpleNamespace(type="IssuesEvent", payload=payload)

        details = self.processor._extract_event_details(mock_event)

        assert details["action"] == "opened"
        assert details["issue_number"] == 42

    def test_process_pull_request_event_details(self):
        """Test extracting details from PullRequestEvent."""
        from types import SimpleNamespace

        pr = SimpleNamespace(title="Add new feature", number=15)
        payload = SimpleNamespace(action="opened", pull_request=pr)
        mock_event = SimpleNamespace(type="PullRequestEvent", payload=payload)

        details = self.processor._extract_event_details(mock_event)

        assert details["action"] == "opened"
        assert details["pr_number"] == 15

    def test_process_release_event_details(self):
        """Test extracting details from ReleaseEvent."""
        from types import SimpleNamespace

        release = SimpleNamespace(
            tag_name="v1.2.0",
            name="Version 1.2.0",
            body="This release includes bug fixes and new features.",
            prerelease=False,
        )
        payload = SimpleNamespace(action="published", release=release)
        mock_event = SimpleNamespace(type="ReleaseEvent", payload=payload)

        details = self.processor._extract_event_details(mock_event)

        assert details["action"] == "published"
        assert details["version"] == "v1.2.0"
        assert details["release_name"] == "Version 1.2.0"
        assert "bug fixes and new features" in details["release_notes"]
        assert details["prerelease"] == False

    def test_process_unknown_event_type(self):
        """Test handling of unknown event types gracefully."""
        from types import SimpleNamespace

        mock_event = SimpleNamespace(
            type="SomeNewEventType", payload=SimpleNamespace(data="something")
        )

        details = self.processor._extract_event_details(mock_event)

        # Should return empty dict for unknown types, not crash
        assert isinstance(details, dict)

    def test_generate_activity_report(self):
        """Test full report generation with multiple realistic events."""
        from types import SimpleNamespace

        # Create mock events with simple structure
        events = [
            SimpleNamespace(
                type="PushEvent",
                repo=SimpleNamespace(name="repo1"),
                actor=SimpleNamespace(login="testuser"),
                created_at="2024-01-15T10:00:00Z",
                payload=SimpleNamespace(),
            ),
            SimpleNamespace(
                type="IssuesEvent",
                repo=SimpleNamespace(name="repo2"),
                actor=SimpleNamespace(login="testuser"),
                created_at="2024-01-15T11:00:00Z",
                payload=SimpleNamespace(),
            ),
            SimpleNamespace(
                type="PushEvent",
                repo=SimpleNamespace(name="repo1"),
                actor=SimpleNamespace(login="testuser"),
                created_at="2024-01-15T12:00:00Z",
                payload=SimpleNamespace(),
            ),
        ]

        start_date = datetime(2024, 1, 15, tzinfo=UTC)
        end_date = datetime(2024, 1, 15, 23, 59, 59, tzinfo=UTC)

        report = self.processor.process(events, "testuser", start_date, end_date)

        # Test basic structure
        assert report.user == "testuser"
        assert report.summary.total_events == 3
        assert report.summary.repositories_active == 2
        assert report.summary.event_breakdown["PushEvent"] == 2
        assert report.summary.event_breakdown["IssuesEvent"] == 1

        # Test daily rollups
        assert len(report.daily_rollups) == 1
        daily = report.daily_rollups[0]
        assert daily.date == "2024-01-15"
        assert daily.events == 3
        assert len(daily.repositories) == 2
        assert "repo1" in daily.repositories
        assert "repo2" in daily.repositories

        # Test repository breakdown
        assert len(report.repository_breakdown) == 2
        assert report.repository_breakdown["repo1"].events == 2
        assert report.repository_breakdown["repo2"].events == 1

        # Test detailed events
        assert len(report.detailed_events) == 3

    def test_process_empty_events(self):
        """Test processing empty event list."""
        report = self.processor.process([], "testuser")

        assert report.user == "testuser"
        assert report.summary.total_events == 0
        assert report.summary.repositories_active == 0
        assert report.summary.event_breakdown == {}
        assert report.daily_rollups == []
        assert report.repository_breakdown == {}
        assert report.detailed_events == []

    def test_count_events_by_type(self):
        """Test event counting by type."""
        events = [
            create_github_event("PushEvent", "repo1", "2024-01-15T10:00:00Z"),
            create_github_event("PushEvent", "repo2", "2024-01-15T11:00:00Z"),
            create_github_event("IssuesEvent", "repo1", "2024-01-15T12:00:00Z"),
        ]

        counts = self.processor._count_events_by_type(events)

        assert counts["PushEvent"] == 2
        assert counts["IssuesEvent"] == 1
        assert len(counts) == 2

    def test_create_daily_rollups(self):
        """Test daily rollup creation across multiple days."""
        events = [
            create_github_event("PushEvent", "repo1", "2024-01-15T10:00:00Z"),
            create_github_event("IssuesEvent", "repo1", "2024-01-15T14:00:00Z"),
            create_github_event("PushEvent", "repo2", "2024-01-16T09:00:00Z"),
        ]

        rollups = self.processor._create_daily_rollups(events)

        assert len(rollups) == 2

        # Day 1
        day1 = rollups[0]
        assert day1.date == "2024-01-15"
        assert day1.events == 2
        assert day1.repositories == ["repo1"]
        assert day1.event_types["PushEvent"] == 1
        assert day1.event_types["IssuesEvent"] == 1

        # Day 2
        day2 = rollups[1]
        assert day2.date == "2024-01-16"
        assert day2.events == 1
        assert day2.repositories == ["repo2"]
        assert day2.event_types["PushEvent"] == 1

    def test_create_repository_breakdown(self):
        """Test repository breakdown creation."""
        events = [
            create_github_event("PushEvent", "repo1", "2024-01-15T10:00:00Z"),
            create_github_event("IssuesEvent", "repo1", "2024-01-15T14:00:00Z"),
            create_github_event("PushEvent", "repo2", "2024-01-16T09:00:00Z"),
        ]

        breakdown = self.processor._create_repository_breakdown(events)

        assert len(breakdown) == 2

        # Repo 1
        repo1_breakdown = breakdown["repo1"]
        assert repo1_breakdown.events == 2
        assert repo1_breakdown.event_types["PushEvent"] == 1
        assert repo1_breakdown.event_types["IssuesEvent"] == 1
        assert repo1_breakdown.first_activity == "2024-01-15T10:00:00Z"
        assert repo1_breakdown.last_activity == "2024-01-15T14:00:00Z"

        # Repo 2
        repo2_breakdown = breakdown["repo2"]
        assert repo2_breakdown.events == 1
        assert repo2_breakdown.event_types["PushEvent"] == 1
        assert repo2_breakdown.first_activity == "2024-01-16T09:00:00Z"
        assert repo2_breakdown.last_activity == "2024-01-16T09:00:00Z"
