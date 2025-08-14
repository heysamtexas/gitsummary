"""Unit tests for AutomationDetector.

Following Guilfoyle's guidance: test algorithm correctness with synthetic data,
focus on threshold logic and business rules, not API simulation.
"""

from unittest.mock import AsyncMock

import pytest

from git_summary.models import Actor, BaseGitHubEvent, Repository
from git_summary.user_profiling import AutomationDetector, UserProfile
from tests.fixtures.github_responses import (
    create_automation_user_events,
    create_normal_user_events,
)


@pytest.fixture
def mock_github_client():
    """Create a mock GitHub client for testing."""
    client = AsyncMock()
    client.get_user_events_paginated = AsyncMock()
    return client


@pytest.fixture
def automation_detector(mock_github_client):
    """Create AutomationDetector instance for testing."""
    return AutomationDetector(mock_github_client)


class TestAutomationDetector:
    """Test the automation detection algorithm logic."""

    @pytest.mark.parametrize("issue_ratio,expected", [
        (0.1, False),  # Low issue ratio - normal user
        (0.75, True),  # High issue ratio - automation
        (0.70, False), # Exactly at threshold - just under
        (0.71, True),  # Just over threshold - automation
    ])
    def test_issue_ratio_threshold(self, automation_detector, issue_ratio, expected):
        """Test the issue event ratio threshold detection."""
        total_events = 100
        issue_events = int(total_events * issue_ratio)
        other_events = total_events - issue_events

        # Create synthetic events focusing on the ratio we want to test
        events = []

        # Add issue events
        for i in range(issue_events):
            events.append(self._create_test_event("IssuesEvent", f"event_issue_{i}"))

        # Add other event types
        for i in range(other_events):
            events.append(self._create_test_event("PushEvent", f"event_push_{i}"))

        # Test the actual implementation
        indicators = automation_detector._calculate_automation_indicators(events, days=7)
        actual_ratio = indicators["issue_event_ratio"]
        is_automation = actual_ratio > automation_detector.ISSUE_RATIO_THRESHOLD
        assert is_automation == expected

        # Also verify the ratio calculation is correct
        assert abs(actual_ratio - issue_ratio) < 0.01  # Allow small floating point differences

    @pytest.mark.parametrize("repo_dominance,expected", [
        (0.5, False),   # Balanced across repos - normal
        (0.85, True),   # Single repo dominance - automation
        (0.80, False),  # Exactly at threshold - just under
        (0.81, True),   # Just over threshold - automation
    ])
    def test_single_repo_dominance(self, automation_detector, repo_dominance, expected):
        """Test single repository dominance detection."""
        total_events = 100
        dominant_repo_events = int(total_events * repo_dominance)
        other_repo_events = total_events - dominant_repo_events

        events = []

        # Events from dominant repository
        for i in range(dominant_repo_events):
            events.append(self._create_test_event(
                "PushEvent",
                f"dominant_{i}",
                repo_name="dominant-repo"
            ))

        # Events from other repositories
        repos = ["repo1", "repo2", "repo3", "repo4"]
        for i in range(other_repo_events):
            repo_name = repos[i % len(repos)]
            events.append(self._create_test_event(
                "PushEvent",
                f"other_{i}",
                repo_name=repo_name
            ))

        # Test the actual implementation
        indicators = automation_detector._calculate_automation_indicators(events, days=7)
        actual_dominance = indicators["single_repo_dominance"]
        is_automation = actual_dominance > automation_detector.SINGLE_REPO_DOMINANCE_THRESHOLD
        assert is_automation == expected

        # Verify the dominance calculation is correct
        assert abs(actual_dominance - repo_dominance) < 0.01

    @pytest.mark.parametrize("events_per_hour,expected", [
        (2.0, False),  # Normal frequency
        (6.0, True),   # High frequency - automation
        (5.0, False),  # Exactly at threshold - just under
        (5.1, True),   # Just over threshold - automation
    ])
    def test_high_frequency_detection(self, automation_detector, events_per_hour, expected):
        """Test high frequency event detection."""
        # Create events at the specified frequency over 24 hours
        hours = 24
        total_events = int(events_per_hour * hours)

        events = []
        for i in range(total_events):
            # Distribute events evenly across 24 hours
            hour = i % hours
            minute = (i * 60 // total_events) % 60
            timestamp = f"2024-01-01T{hour:02d}:{minute:02d}:00Z"

            events.append(self._create_test_event(
                "PushEvent",
                f"freq_event_{i}",
                created_at=timestamp
            ))

        # Test the actual implementation
        indicators = automation_detector._calculate_automation_indicators(events, days=7)
        actual_frequency = indicators["events_per_hour"]
        is_automation = indicators["high_frequency_score"] > 1.0  # High frequency score threshold
        assert is_automation == expected

        # Verify frequency calculation is roughly correct (allow some variance for time calculations)
        assert abs(actual_frequency - events_per_hour) < 1.0

    def test_timing_pattern_detection(self, automation_detector):
        """Test detection of regular timing patterns (automation indicator)."""
        # Create events at exact hourly intervals - strong automation signal
        regular_events = []
        for hour in range(24):
            regular_events.append(self._create_test_event(
                "PushEvent",
                f"regular_{hour}",
                created_at=f"2024-01-01T{hour:02d}:00:00Z"
            ))

        # Create events at random times - normal user pattern
        random_events = []
        random_times = ["09:15", "14:33", "16:07", "11:42", "15:58"]
        for i, time in enumerate(random_times):
            random_events.append(self._create_test_event(
                "PushEvent",
                f"random_{i}",
                created_at=f"2024-01-01T{time}:00Z"
            ))

        # Test the actual implementation for timing patterns
        regular_indicators = automation_detector._calculate_automation_indicators(regular_events, days=7)
        random_indicators = automation_detector._calculate_automation_indicators(random_events, days=7)

        # Regular patterns should have higher frequency scores
        regular_freq = regular_indicators["events_per_hour"]
        random_freq = random_indicators["events_per_hour"]

        # Regular events should have consistently higher frequency
        assert regular_freq > random_freq

    @pytest.mark.asyncio
    async def test_classify_user_automation_bot(self, automation_detector, mock_github_client):
        """Test classification of clear automation bot."""
        # Setup mock to return automation-like events
        automation_events = self._convert_to_github_events(create_automation_user_events()[:50])
        mock_github_client.get_user_events_paginated.return_value = [automation_events]

        profile = await automation_detector.classify_user("automation-bot", days=7)

        assert isinstance(profile, UserProfile)
        assert profile.is_automation is True
        assert profile.confidence_score > 0.8
        assert "high issue ratio" in profile.indicators or "repo dominance" in profile.indicators

    @pytest.mark.asyncio
    async def test_classify_user_normal_developer(self, automation_detector, mock_github_client):
        """Test classification of normal developer."""
        # Setup mock to return normal developer events
        normal_events = self._convert_to_github_events(create_normal_user_events())
        mock_github_client.get_user_events_paginated.return_value = [normal_events]

        profile = await automation_detector.classify_user("normal-dev", days=7)

        assert isinstance(profile, UserProfile)
        assert profile.is_automation is False
        assert profile.confidence_score > 0.6
        assert len(profile.indicators) >= 1

    @pytest.mark.asyncio
    async def test_classify_user_no_activity(self, automation_detector, mock_github_client):
        """Test classification when user has no activity."""
        # Setup mock to return empty events
        mock_github_client.get_user_events_paginated.return_value = [[]]

        profile = await automation_detector.classify_user("inactive-user", days=7)

        assert isinstance(profile, UserProfile)
        assert profile.is_automation is False
        assert profile.confidence_score == 0.0
        assert "insufficient activity" in profile.indicators

    def test_edge_case_single_event(self, automation_detector):
        """Test behavior with only one event."""
        events = [self._create_test_event("PushEvent", "single_event")]

        # Should not crash and should handle gracefully
        indicators = automation_detector._calculate_automation_indicators(events, days=7)

        assert 0.0 <= indicators["issue_event_ratio"] <= 1.0
        assert 0.0 <= indicators["single_repo_dominance"] <= 1.0
        assert indicators["events_per_hour"] >= 0.0

    def test_edge_case_all_same_timestamp(self, automation_detector):
        """Test behavior when all events have identical timestamps."""
        same_time = "2024-01-01T12:00:00Z"
        events = [
            self._create_test_event("PushEvent", f"event_{i}", created_at=same_time)
            for i in range(10)
        ]

        # Should handle without crashing
        indicators = automation_detector._calculate_automation_indicators(events, days=7)

        assert indicators["events_per_hour"] >= 0.0
        assert 0.0 <= indicators["high_frequency_score"] <= 10.0  # Should be bounded

    def _create_test_event(
        self,
        event_type: str,
        event_id: str,
        repo_name: str = "test-repo",
        created_at: str = "2024-01-01T12:00:00Z"
    ) -> BaseGitHubEvent:
        """Create a test GitHub event for testing."""
        return BaseGitHubEvent(
            id=event_id,
            type=event_type,
            actor=Actor(
                id=123,
                login="test-user",
                display_login="test-user",
                gravatar_id="",
                url="https://api.github.com/users/test-user",
                avatar_url="https://avatars.githubusercontent.com/u/123"
            ),
            repo=Repository(
                id=456,
                name=repo_name,
                url=f"https://api.github.com/repos/test-owner/{repo_name}",
                full_name=f"test-owner/{repo_name}"
            ),
            payload={},
            public=True,
            created_at=created_at
        )

    def _convert_to_github_events(self, event_dicts: list[dict]) -> list[BaseGitHubEvent]:
        """Convert dictionary events to BaseGitHubEvent objects."""
        events = []
        for event_dict in event_dicts:
            repo_info = event_dict.get("repo", {})
            actor_info = event_dict.get("actor", {})

            event = BaseGitHubEvent(
                id=event_dict["id"],
                type=event_dict["type"],
                actor=Actor(
                    id=123,
                    login=actor_info.get("login", "test-user"),
                    display_login=actor_info.get("login", "test-user"),
                    gravatar_id="",
                    url="https://api.github.com/users/test-user",
                    avatar_url="https://avatars.githubusercontent.com/u/123"
                ),
                repo=Repository(
                    id=456,
                    name=repo_info.get("name", "test-repo"),
                    url=f"https://api.github.com/repos/test-owner/{repo_info.get('name', 'test-repo')}",
                    full_name=repo_info.get("full_name", f"test-owner/{repo_info.get('name', 'test-repo')}")
                ),
                payload=event_dict.get("payload", {}),
                public=True,
                created_at=event_dict["created_at"]
            )
            events.append(event)

        return events
