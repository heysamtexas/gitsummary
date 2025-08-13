"""Tests for GitHub event processors."""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from git_summary.models import (
    Actor,
    BaseGitHubEvent,
    CreateEventPayload,
    DetailedEvent,
    GitHubActivityReport,
    Repository,
)
from git_summary.processors import EventProcessor


class TestEventProcessor:
    """Test the EventProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create an EventProcessor for testing."""
        return EventProcessor()

    @pytest.fixture
    def sample_actor(self):
        """Create a sample actor for testing."""
        return Actor(
            id=12345,
            login="testuser",
            url="https://api.github.com/users/testuser",
            avatar_url="https://avatars.githubusercontent.com/u/12345",
        )

    @pytest.fixture
    def sample_repo(self):
        """Create a sample repository for testing."""
        return Repository(
            id=67890,
            name="testuser/testrepo",
            url="https://api.github.com/repos/testuser/testrepo",
        )

    @pytest.fixture
    def sample_events(self, sample_actor, sample_repo):
        """Create sample events for testing."""
        # Create different types of events across multiple days and repositories
        events = []

        # Day 1: 2023-01-15 - testuser/repo1
        repo1 = Repository(
            id=1,
            name="testuser/repo1",
            url="https://api.github.com/repos/testuser/repo1",
        )

        # PushEvent on Day 1
        push_payload = Mock()
        push_payload.commits = [Mock(), Mock()]  # 2 commits
        push_payload.ref = "refs/heads/main"

        push_event = Mock(spec=BaseGitHubEvent)
        push_event.id = "event1"
        push_event.type = "PushEvent"
        push_event.actor = sample_actor
        push_event.repo = repo1
        push_event.created_at = "2023-01-15T10:00:00Z"
        push_event.payload = push_payload
        events.append(push_event)

        # IssuesEvent on Day 1
        issue_payload = Mock()
        issue_payload.action = "opened"
        issue_payload.issue = Mock()
        issue_payload.issue.number = 123

        issue_event = Mock(spec=BaseGitHubEvent)
        issue_event.id = "event2"
        issue_event.type = "IssuesEvent"
        issue_event.actor = sample_actor
        issue_event.repo = repo1
        issue_event.created_at = "2023-01-15T14:00:00Z"
        issue_event.payload = issue_payload
        events.append(issue_event)

        # Day 2: 2023-01-16 - testuser/repo2
        repo2 = Repository(
            id=2,
            name="testuser/repo2",
            url="https://api.github.com/repos/testuser/repo2",
        )

        # PullRequestEvent on Day 2
        pr_payload = Mock()
        pr_payload.action = "opened"
        pr_payload.pull_request = Mock()
        pr_payload.pull_request.number = 456

        pr_event = Mock(spec=BaseGitHubEvent)
        pr_event.id = "event3"
        pr_event.type = "PullRequestEvent"
        pr_event.actor = sample_actor
        pr_event.repo = repo2
        pr_event.created_at = "2023-01-16T09:00:00Z"
        pr_event.payload = pr_payload
        events.append(pr_event)

        # Another PushEvent on Day 2, same repo
        push_payload2 = Mock()
        push_payload2.commits = [Mock()]  # 1 commit
        push_payload2.ref = "refs/heads/feature"

        push_event2 = Mock(spec=BaseGitHubEvent)
        push_event2.id = "event4"
        push_event2.type = "PushEvent"
        push_event2.actor = sample_actor
        push_event2.repo = repo2
        push_event2.created_at = "2023-01-16T15:00:00Z"
        push_event2.payload = push_payload2
        events.append(push_event2)

        return events

    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = EventProcessor()
        assert processor is not None

    def test_process_with_sample_events(self, processor, sample_events):
        """Test processing with sample events."""
        start_date = datetime(2023, 1, 15, tzinfo=UTC)
        end_date = datetime(2023, 1, 16, tzinfo=UTC)

        report = processor.process(sample_events, "testuser", start_date, end_date)

        # Check basic structure
        assert isinstance(report, GitHubActivityReport)
        assert report.user == "testuser"

        # Check period
        assert report.period.start == start_date.isoformat()
        assert report.period.end == end_date.isoformat()

        # Check summary
        assert report.summary.total_events == 4
        assert report.summary.repositories_active == 2
        assert report.summary.event_breakdown == {
            "PushEvent": 2,
            "IssuesEvent": 1,
            "PullRequestEvent": 1,
        }
        assert report.summary.most_common_event_type == "PushEvent"
        assert report.summary.most_active_repository in [
            "testuser/repo1",
            "testuser/repo2",
        ]

        # Check daily rollups
        assert len(report.daily_rollups) == 2

        # Day 1 rollup
        day1_rollup = next(r for r in report.daily_rollups if r.date == "2023-01-15")
        assert day1_rollup.events == 2
        assert day1_rollup.repositories == ["testuser/repo1"]
        assert day1_rollup.event_types == {"PushEvent": 1, "IssuesEvent": 1}

        # Day 2 rollup
        day2_rollup = next(r for r in report.daily_rollups if r.date == "2023-01-16")
        assert day2_rollup.events == 2
        assert day2_rollup.repositories == ["testuser/repo2"]
        assert day2_rollup.event_types == {"PullRequestEvent": 1, "PushEvent": 1}

        # Check repository breakdown
        assert len(report.repository_breakdown) == 2

        repo1_breakdown = report.repository_breakdown["testuser/repo1"]
        assert repo1_breakdown.events == 2
        assert repo1_breakdown.event_types == {"PushEvent": 1, "IssuesEvent": 1}
        assert repo1_breakdown.first_activity == "2023-01-15T10:00:00Z"
        assert repo1_breakdown.last_activity == "2023-01-15T14:00:00Z"

        repo2_breakdown = report.repository_breakdown["testuser/repo2"]
        assert repo2_breakdown.events == 2
        assert repo2_breakdown.event_types == {"PullRequestEvent": 1, "PushEvent": 1}
        assert repo2_breakdown.first_activity == "2023-01-16T09:00:00Z"
        assert repo2_breakdown.last_activity == "2023-01-16T15:00:00Z"

        # Check detailed events
        assert len(report.detailed_events) == 4

        # Check first detailed event (PushEvent)
        push_detail = next(e for e in report.detailed_events if e.type == "PushEvent")
        assert push_detail.repository in ["testuser/repo1", "testuser/repo2"]
        assert push_detail.actor == "testuser"
        assert "commits" in push_detail.details
        assert "ref" in push_detail.details

    def test_process_empty_events(self, processor):
        """Test processing with empty events list."""
        report = processor.process([], "testuser")

        assert isinstance(report, GitHubActivityReport)
        assert report.user == "testuser"
        assert report.summary.total_events == 0
        assert report.summary.repositories_active == 0
        assert report.summary.event_breakdown == {}
        assert report.summary.most_active_repository is None
        assert report.summary.most_common_event_type is None
        assert report.daily_rollups == []
        assert report.repository_breakdown == {}
        assert report.detailed_events == []

    def test_process_single_event(self, processor, sample_actor, sample_repo):
        """Test processing with a single event."""
        # Create a single CreateEvent
        payload = CreateEventPayload(
            ref_type="branch",
            ref="feature-branch",
            master_branch="main",
            description="Test repository",
            pusher_type="user",
        )

        event = Mock(spec=BaseGitHubEvent)
        event.id = "single_event"
        event.type = "CreateEvent"
        event.actor = sample_actor
        event.repo = sample_repo
        event.created_at = "2023-01-15T12:00:00Z"
        event.payload = payload

        report = processor.process([event], "testuser")

        assert report.summary.total_events == 1
        assert report.summary.repositories_active == 1
        assert report.summary.event_breakdown == {"CreateEvent": 1}
        assert report.summary.most_common_event_type == "CreateEvent"
        assert report.summary.most_active_repository == "testuser/testrepo"

        assert len(report.daily_rollups) == 1
        assert report.daily_rollups[0].date == "2023-01-15"
        assert report.daily_rollups[0].events == 1

        assert len(report.repository_breakdown) == 1
        assert report.repository_breakdown["testuser/testrepo"].events == 1

        assert len(report.detailed_events) == 1
        assert report.detailed_events[0].type == "CreateEvent"

    def test_count_events_by_type(self, processor, sample_events):
        """Test event counting by type."""
        counts = processor._count_events_by_type(sample_events)

        assert counts == {
            "PushEvent": 2,
            "IssuesEvent": 1,
            "PullRequestEvent": 1,
        }

    def test_create_daily_rollups(self, processor, sample_events):
        """Test daily rollup creation."""
        rollups = processor._create_daily_rollups(sample_events)

        assert len(rollups) == 2

        # Check rollups are sorted by date
        assert rollups[0].date == "2023-01-15"
        assert rollups[1].date == "2023-01-16"

        # Check Day 1 rollup
        day1 = rollups[0]
        assert day1.events == 2
        assert day1.repositories == ["testuser/repo1"]
        assert day1.event_types == {"PushEvent": 1, "IssuesEvent": 1}

        # Check Day 2 rollup
        day2 = rollups[1]
        assert day2.events == 2
        assert day2.repositories == ["testuser/repo2"]
        assert day2.event_types == {"PullRequestEvent": 1, "PushEvent": 1}

    def test_create_repository_breakdown(self, processor, sample_events):
        """Test repository breakdown creation."""
        breakdown = processor._create_repository_breakdown(sample_events)

        assert len(breakdown) == 2
        assert "testuser/repo1" in breakdown
        assert "testuser/repo2" in breakdown

        # Check repo1 breakdown
        repo1 = breakdown["testuser/repo1"]
        assert repo1.events == 2
        assert repo1.event_types == {"PushEvent": 1, "IssuesEvent": 1}
        assert repo1.first_activity == "2023-01-15T10:00:00Z"
        assert repo1.last_activity == "2023-01-15T14:00:00Z"

        # Check repo2 breakdown
        repo2 = breakdown["testuser/repo2"]
        assert repo2.events == 2
        assert repo2.event_types == {"PullRequestEvent": 1, "PushEvent": 1}
        assert repo2.first_activity == "2023-01-16T09:00:00Z"
        assert repo2.last_activity == "2023-01-16T15:00:00Z"

    def test_create_detailed_events(self, processor, sample_events):
        """Test detailed events creation."""
        detailed = processor._create_detailed_events(sample_events)

        assert len(detailed) == 4

        # Check all events are DetailedEvent instances
        for event in detailed:
            assert isinstance(event, DetailedEvent)
            assert event.actor == "testuser"
            assert event.repository in ["testuser/repo1", "testuser/repo2"]
            assert isinstance(event.details, dict)

    def test_extract_event_details_push_event(self, processor):
        """Test detail extraction for PushEvent."""
        payload = Mock()
        payload.commits = [Mock(), Mock()]
        payload.ref = "refs/heads/main"

        event = Mock()
        event.type = "PushEvent"
        event.payload = payload

        details = processor._extract_event_details(event)
        assert details["commits"] == 2
        assert details["ref"] == "refs/heads/main"

    def test_extract_event_details_issues_event(self, processor):
        """Test detail extraction for IssuesEvent."""
        payload = Mock()
        payload.action = "closed"
        payload.issue = Mock()
        payload.issue.number = 789

        event = Mock()
        event.type = "IssuesEvent"
        event.payload = payload

        details = processor._extract_event_details(event)
        assert details["action"] == "closed"
        assert details["issue_number"] == 789

    def test_extract_event_details_pull_request_event(self, processor):
        """Test detail extraction for PullRequestEvent."""
        payload = Mock()
        payload.action = "merged"
        payload.pull_request = Mock()
        payload.pull_request.number = 101

        event = Mock()
        event.type = "PullRequestEvent"
        event.payload = payload

        details = processor._extract_event_details(event)
        assert details["action"] == "merged"
        assert details["pr_number"] == 101

    def test_extract_event_details_create_event(self, processor):
        """Test detail extraction for CreateEvent."""
        payload = Mock()
        payload.ref_type = "tag"
        payload.ref = "v1.0.0"

        event = Mock()
        event.type = "CreateEvent"
        event.payload = payload

        details = processor._extract_event_details(event)
        assert details["ref_type"] == "tag"
        assert details["ref"] == "v1.0.0"

    def test_extract_event_details_delete_event(self, processor):
        """Test detail extraction for DeleteEvent."""
        payload = Mock()
        payload.ref_type = "branch"
        payload.ref = "old-feature"

        event = Mock()
        event.type = "DeleteEvent"
        event.payload = payload

        details = processor._extract_event_details(event)
        assert details["ref_type"] == "branch"
        assert details["ref"] == "old-feature"

    def test_extract_event_details_fork_event(self, processor):
        """Test detail extraction for ForkEvent."""
        payload = Mock()
        payload.forkee = Mock()
        payload.forkee.full_name = "otheruser/forked-repo"

        event = Mock()
        event.type = "ForkEvent"
        event.payload = payload

        details = processor._extract_event_details(event)
        assert details["forked_to"] == "otheruser/forked-repo"

    def test_extract_event_details_watch_event(self, processor):
        """Test detail extraction for WatchEvent."""
        payload = Mock()
        payload.action = "started"

        event = Mock()
        event.type = "WatchEvent"
        event.payload = payload

        details = processor._extract_event_details(event)
        assert details["action"] == "started"

    def test_extract_event_details_unknown_event(self, processor):
        """Test detail extraction for unknown event type."""
        event = Mock()
        event.type = "UnknownEvent"
        event.payload = Mock()

        details = processor._extract_event_details(event)
        assert details == {}

    def test_calculate_summary_statistics(self, processor):
        """Test summary statistics calculation."""
        # Mock data for testing
        events = [Mock() for _ in range(10)]  # 10 events total

        event_breakdown = {
            "PushEvent": 6,
            "IssuesEvent": 3,
            "PullRequestEvent": 1,
        }

        repository_breakdown = {
            "repo1": Mock(events=7),
            "repo2": Mock(events=2),
            "repo3": Mock(events=1),
        }

        summary = processor._calculate_summary_statistics(
            events, event_breakdown, repository_breakdown
        )

        assert summary.total_events == 10
        assert summary.repositories_active == 3
        assert summary.event_breakdown == event_breakdown
        assert summary.most_common_event_type == "PushEvent"
        assert summary.most_active_repository == "repo1"

    def test_calculate_summary_statistics_empty(self, processor):
        """Test summary statistics calculation with empty data."""
        summary = processor._calculate_summary_statistics([], {}, {})

        assert summary.total_events == 0
        assert summary.repositories_active == 0
        assert summary.event_breakdown == {}
        assert summary.most_common_event_type is None
        assert summary.most_active_repository is None

    def test_process_with_explicit_dates(self, processor, sample_events):
        """Test processing with explicit start and end dates."""
        start_date = datetime(2023, 1, 10, tzinfo=UTC)
        end_date = datetime(2023, 1, 20, tzinfo=UTC)

        report = processor.process(sample_events, "testuser", start_date, end_date)

        assert report.period.start == start_date.isoformat()
        assert report.period.end == end_date.isoformat()

    def test_process_infers_dates_from_events(self, processor, sample_events):
        """Test that processing infers dates from events when not provided."""
        report = processor.process(sample_events, "testuser")

        # Should infer dates from the event timestamps
        assert report.period.start == "2023-01-15T10:00:00+00:00"
        assert report.period.end == "2023-01-16T15:00:00+00:00"

    def test_process_handles_timezone_parsing(
        self, processor, sample_actor, sample_repo
    ):
        """Test that processor correctly handles timezone parsing."""
        # Create a proper mock payload for PushEvent
        payload = Mock()
        payload.commits = []  # Empty list to avoid len() error
        payload.ref = "refs/heads/main"

        event = Mock(spec=BaseGitHubEvent)
        event.id = "tz_test"
        event.type = "PushEvent"
        event.actor = sample_actor
        event.repo = sample_repo
        event.created_at = "2023-01-15T12:00:00Z"  # UTC timezone
        event.payload = payload

        report = processor.process([event], "testuser")

        # Should parse the timezone correctly
        assert report.detailed_events[0].created_at == "2023-01-15T12:00:00Z"

    def test_process_sorts_daily_rollups_chronologically(
        self, processor, sample_actor, sample_repo
    ):
        """Test that daily rollups are sorted chronologically."""
        # Create events in reverse chronological order
        events = []
        for i, date_str in enumerate(["2023-01-17", "2023-01-15", "2023-01-16"]):
            # Create proper mock payload for PushEvent
            payload = Mock()
            payload.commits = []  # Empty list to avoid len() error
            payload.ref = "refs/heads/main"

            event = Mock(spec=BaseGitHubEvent)
            event.id = f"event_{i}"
            event.type = "PushEvent"
            event.actor = sample_actor
            event.repo = sample_repo
            event.created_at = f"{date_str}T12:00:00Z"
            event.payload = payload
            events.append(event)

        report = processor.process(events, "testuser")

        # Daily rollups should be in chronological order
        dates = [rollup.date for rollup in report.daily_rollups]
        assert dates == ["2023-01-15", "2023-01-16", "2023-01-17"]
