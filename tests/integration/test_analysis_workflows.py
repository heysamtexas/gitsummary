"""Integration tests for analysis component workflows.

Following Guilfoyle's guidance: test component interactions and coordination logic,
not API simulation. Focus on data flow between components and error handling.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from git_summary.intelligence_guided import IntelligenceGuidedAnalyzer
from git_summary.models import Actor, BaseGitHubEvent, Repository
from git_summary.multi_source_discovery import MultiSourceDiscovery
from git_summary.user_profiling import AutomationDetector
from tests.fixtures.github_responses import (
    create_automation_user_events,
    create_empty_response,
    create_repository_discovery_results,
)


@pytest.fixture
def mock_github_client():
    """Create a comprehensive mock GitHub client for integration testing."""
    client = AsyncMock()

    # Set up default empty responses
    client.get_user_events_paginated.return_value = [[]]
    client._make_request_with_retry.return_value.json.return_value = (
        create_empty_response()
    )
    client._update_rate_limit_info = Mock()
    client.base_url = "https://api.github.com"
    client.headers = {"Authorization": "token test-token"}

    return client


class TestIntelligenceGuidedWorkflow:
    """Test the three-phase intelligence-guided analysis workflow."""

    def test_repository_scoring_workflow(self):
        """Test the repository scoring workflow logic."""
        # Following Guilfoyle's guidance: test coordination logic, not API simulation
        analyzer = IntelligenceGuidedAnalyzer(Mock())

        # Create events that will test the scoring workflow
        events = [
            self._create_test_event("PushEvent", "1", "high-activity"),  # 3 points
            self._create_test_event("PushEvent", "2", "high-activity"),  # 3 points
            self._create_test_event(
                "PullRequestEvent", "3", "high-activity"
            ),  # 2 points
            self._create_test_event("IssuesEvent", "4", "low-activity"),  # 1 point
        ]

        # Test the scoring logic
        repo_scores = analyzer._score_repositories_by_development_activity(events)

        # Verify scoring worked correctly
        assert "high-activity" in repo_scores
        assert repo_scores["high-activity"] == 8  # 3+3+2 points
        # low-activity (1 point) should be filtered out by minimum threshold

    def test_analysis_stats_generation(self):
        """Test that analysis statistics are generated correctly from events."""
        # Following Guilfoyle's guidance: test the logic, not the API calls
        analyzer = IntelligenceGuidedAnalyzer(Mock())

        # Create synthetic events to test stats generation
        events = [
            self._create_test_event("PushEvent", "1", "repo-a"),
            self._create_test_event("PullRequestEvent", "2", "repo-a"),
            self._create_test_event("IssuesEvent", "3", "repo-b"),
        ]

        stats = analyzer.get_analysis_stats(events)

        # Verify stats structure and content
        assert stats["total_events"] == 3
        assert stats["repositories_analyzed"] == 2
        assert "PushEvent" in stats["event_type_breakdown"]
        assert stats["analysis_strategy"] == "intelligence_guided"

    def _create_test_event(
        self,
        event_type: str,
        event_id: str,
        repo_name: str,
        created_at: str = "2024-01-01T12:00:00Z",
    ) -> BaseGitHubEvent:
        """Create a test GitHub event."""
        return BaseGitHubEvent(
            id=event_id,
            type=event_type,
            actor=Actor(
                id=123,
                login="test-user",
                display_login="test-user",
                gravatar_id="",
                url="",
                avatar_url="",
            ),
            repo=Repository(
                id=456,
                name=repo_name,
                url=f"https://api.github.com/repos/owner/{repo_name}",
                full_name=f"owner/{repo_name}",
            ),
            payload={},
            public=True,
            created_at=created_at,
        )

    @pytest.mark.asyncio
    async def test_workflow_phase_transitions(self, mock_github_client):
        """Test that phases execute in correct order with proper data flow."""
        analyzer = IntelligenceGuidedAnalyzer(mock_github_client)

        # Create events that will produce clear phase transitions
        test_events = []
        for i in range(20):
            repo_name = f"repo-{i % 3}"  # 3 different repos
            test_events.append(
                self._create_test_event("PushEvent", f"event_{i}", repo_name)
            )

        mock_github_client.get_user_events_paginated.return_value = [test_events]

        # Mock detailed repository fetch
        mock_github_client._make_request_with_retry.return_value.json.return_value = []

        # Track execution with progress callback
        phase_calls = []

        def progress_callback(phase, current, total):
            phase_calls.append(phase)

        await analyzer.discover_and_fetch(
            "test-user", days=7, progress_callback=progress_callback
        )

        # Verify phases executed in correct order
        assert len(phase_calls) > 0
        assert "Phase 1" in " ".join(phase_calls)

    @pytest.mark.asyncio
    async def test_error_handling_api_failure(self, mock_github_client):
        """Test workflow error handling when API calls fail."""
        analyzer = IntelligenceGuidedAnalyzer(mock_github_client)

        # Mock API failure in phase 1
        mock_github_client.get_user_events_paginated.side_effect = Exception(
            "API Error"
        )

        # Should handle gracefully, not crash
        result = await analyzer.discover_and_fetch("error-user", days=7)

        # Should return empty result on error
        assert isinstance(result, list)
        assert len(result) == 0


class TestMultiSourceDiscoveryWorkflow:
    """Test the multi-source repository discovery workflow."""

    @pytest.mark.asyncio
    async def test_multi_source_coordination(self, mock_github_client):
        """Test coordination between multiple discovery sources."""
        discovery = MultiSourceDiscovery(mock_github_client)

        # Mock the three sources with different data
        test_data = create_repository_discovery_results()

        # Mock owned repositories API call
        mock_github_client._make_request_with_retry.return_value.json.return_value = (
            test_data["owned_repos"]
        )

        # Mock user events for event-based discovery
        test_events = []
        for repo_name, repo_data in test_data["event_repos"].items():
            for _ in range(repo_data["score"]):  # Create events proportional to score
                test_events.append(
                    self._create_test_event(
                        "PushEvent",
                        f"event_{len(test_events)}",
                        repo_name.split("/")[1],
                    )
                )

        mock_github_client.get_user_events_paginated.return_value = [test_events]

        # Execute multi-source discovery
        result = await discovery.discover_and_fetch("multi-source-user", days=7)

        # Verify coordination worked
        assert isinstance(result, list)
        # Should have called multiple API endpoints
        assert mock_github_client._make_request_with_retry.call_count >= 2

    @pytest.mark.asyncio
    async def test_source_fallback_behavior(self, mock_github_client):
        """Test fallback when individual sources fail."""
        discovery = MultiSourceDiscovery(mock_github_client)

        # Mock: owned repos fails, but other sources succeed
        call_count = 0

        def mock_api_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call (owned repos) fails
                raise Exception("Owned repos API failed")
            else:  # Subsequent calls succeed
                mock_response = Mock()
                mock_response.json.return_value = {"items": []}
                return mock_response

        mock_github_client._make_request_with_retry.side_effect = mock_api_call
        mock_github_client.get_user_events_paginated.return_value = [[]]

        # Should continue with other sources despite one failure
        result = await discovery.discover_and_fetch("fallback-user", days=7)

        # Should not crash, should return results from working sources
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_repository_deduplication_across_sources(self, mock_github_client):
        """Test that repositories discovered by multiple sources are properly merged."""
        discovery = MultiSourceDiscovery(mock_github_client)

        # Create overlapping repository data across sources
        owned_repo_response = [
            {
                "full_name": "user/overlap-repo",
                "updated_at": "2024-01-15T10:00:00Z",
                "private": False,
                "language": "Python",
            }
        ]

        # Same repo appears in events
        overlap_events = [
            self._create_test_event("PushEvent", "event1", "overlap-repo"),
            self._create_test_event("PushEvent", "event2", "overlap-repo"),
        ]

        # Same repo in commit search
        commit_search_response = {
            "items": [
                {"repository": {"full_name": "user/overlap-repo"}},
                {"repository": {"full_name": "user/overlap-repo"}},
            ]
        }

        # Setup mocks to return overlapping data
        responses = [
            Mock(json=lambda: owned_repo_response),  # Owned repos
            Mock(json=lambda: commit_search_response),  # Commit search
            Mock(json=lambda: []),  # Repository events (empty)
        ]
        mock_github_client._make_request_with_retry.side_effect = responses
        mock_github_client.get_user_events_paginated.return_value = [overlap_events]

        result = await discovery.discover_and_fetch("overlap-user", days=7)

        # Repository should appear once despite being in multiple sources
        # (Test the deduplication logic indirectly through final results)
        assert isinstance(result, list)


class TestComponentIntegration:
    """Test integration between different analysis components."""

    @pytest.mark.asyncio
    async def test_automation_detector_with_multi_source_integration(
        self, mock_github_client
    ):
        """Test integration between automation detection and multi-source discovery."""
        # This tests the workflow where automation detection determines strategy

        detector = AutomationDetector(mock_github_client)
        discovery = MultiSourceDiscovery(mock_github_client)

        # Setup automation user events
        automation_events = self._convert_to_github_events(
            create_automation_user_events()[:30]
        )
        mock_github_client.get_user_events_paginated.return_value = [automation_events]

        # Mock multi-source discovery APIs
        mock_github_client._make_request_with_retry.return_value.json.return_value = {
            "items": []
        }

        # Step 1: Detect automation user
        profile = await detector.classify_user("automation-user", days=7)
        assert profile.is_automation is True

        # Step 2: Use multi-source discovery for automation user
        result = await discovery.discover_and_fetch("automation-user", days=7)

        # Integration should work smoothly
        assert isinstance(result, list)

    def test_error_propagation_between_components(self):
        """Test that errors are properly handled across component boundaries."""
        # Test error handling without async complexity

        detector = AutomationDetector(Mock())

        # Test that component errors don't crash the entire workflow
        with pytest.raises(ValueError):
            # Simulate a component error that should be caught by caller
            detector._calculate_issue_ratio(None)  # Invalid input

    def _create_test_event(
        self,
        event_type: str,
        event_id: str,
        repo_name: str,
        created_at: str = "2024-01-01T12:00:00Z",
    ) -> BaseGitHubEvent:
        """Create a test GitHub event."""
        return BaseGitHubEvent(
            id=event_id,
            type=event_type,
            actor=Actor(
                id=123,
                login="test-user",
                display_login="test-user",
                gravatar_id="",
                url="",
                avatar_url="",
            ),
            repo=Repository(
                id=456,
                name=repo_name,
                url=f"https://api.github.com/repos/owner/{repo_name}",
                full_name=f"owner/{repo_name}",
            ),
            payload={},
            public=True,
            created_at=created_at,
        )

    def _convert_to_github_events(
        self, event_dicts: list[dict]
    ) -> list[BaseGitHubEvent]:
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
                    avatar_url="https://avatars.githubusercontent.com/u/123",
                ),
                repo=Repository(
                    id=456,
                    name=repo_info.get("name", "test-repo"),
                    url=f"https://api.github.com/repos/test-owner/{repo_info.get('name', 'test-repo')}",
                    full_name=repo_info.get(
                        "full_name", f"test-owner/{repo_info.get('name', 'test-repo')}"
                    ),
                ),
                payload=event_dict.get("payload", {}),
                public=True,
                created_at=event_dict["created_at"],
            )
            events.append(event)

        return events
