"""Unit tests for repository scoring algorithms.

Tests the mathematical logic of repository scoring and ranking
without API dependencies. Focuses on edge cases and algorithm correctness.
"""

from collections import Counter
from unittest.mock import Mock

import pytest

from git_summary.intelligence_guided import IntelligenceGuidedAnalyzer
from git_summary.models import Actor, BaseGitHubEvent, Repository
from git_summary.multi_source_discovery import MultiSourceDiscovery


@pytest.fixture
def mock_github_client():
    """Create a mock GitHub client."""
    return Mock()


class TestRepositoryScoringAlgorithms:
    """Test repository scoring and ranking logic."""

    def test_event_weight_calculation(self):
        """Test that different event types receive correct weights."""
        analyzer = IntelligenceGuidedAnalyzer(Mock())

        # Test known event weights
        assert analyzer._get_event_weight("PushEvent") == 3
        assert analyzer._get_event_weight("PullRequestEvent") == 2
        assert analyzer._get_event_weight("ReleaseEvent") == 2
        assert analyzer._get_event_weight("IssuesEvent") == 1
        assert analyzer._get_event_weight("UnknownEvent") == 1

    def test_repository_scoring_single_pass(self):
        """Test that repository scoring uses single-pass algorithm (Guilfoyle's fix)."""
        analyzer = IntelligenceGuidedAnalyzer(Mock())

        # Create events that would expose O(n²) behavior if present
        events = []
        repo_names = [f"repo-{i}" for i in range(100)]

        for i, repo_name in enumerate(repo_names):
            # Each repo gets one high-value event
            events.append(self._create_test_event("PushEvent", f"event_{i}", repo_name))

        # This should complete quickly with single-pass algorithm
        scores = analyzer._score_repositories_by_development_activity(events)

        # All repos should have same score (one PushEvent each)
        expected_score = analyzer._get_event_weight("PushEvent")
        for repo_name in repo_names:
            assert scores[repo_name] == expected_score

    def test_repository_scoring_aggregation(self):
        """Test that repository scores aggregate correctly."""
        analyzer = IntelligenceGuidedAnalyzer(Mock())

        events = [
            self._create_test_event("PushEvent", "1", "repo-a"),      # Weight 3
            self._create_test_event("PushEvent", "2", "repo-a"),      # Weight 3
            self._create_test_event("PullRequestEvent", "3", "repo-a"), # Weight 2
            self._create_test_event("IssuesEvent", "4", "repo-b"),    # Weight 1
            self._create_test_event("ReleaseEvent", "5", "repo-b"),   # Weight 2
        ]

        scores = analyzer._score_repositories_by_development_activity(events)

        assert scores["repo-a"] == 3 + 3 + 2  # 8 total
        assert scores["repo-b"] == 1 + 2      # 3 total

    def test_repository_scoring_with_threshold(self):
        """Test that repository scoring applies minimum threshold correctly."""
        analyzer = IntelligenceGuidedAnalyzer(Mock())

        # Create events with varying activity levels
        events = [
            self._create_test_event("PushEvent", "1", "high-activity"),    # 3 points
            self._create_test_event("PushEvent", "2", "high-activity"),    # 3 points
            self._create_test_event("IssuesEvent", "3", "low-activity"),   # 1 point - below threshold
        ]

        # Test scoring with threshold
        repo_scores = analyzer._score_repositories_by_development_activity(events)

        # High activity should be included, low activity might be filtered by threshold
        assert "high-activity" in repo_scores
        assert repo_scores["high-activity"] >= analyzer.min_repo_score

    def test_repository_scoring_empty_results(self):
        """Test behavior when no repositories meet threshold."""
        analyzer = IntelligenceGuidedAnalyzer(Mock())

        # Create very low activity events
        events = [
            self._create_test_event("WatchEvent", "1", "minimal-activity"),  # Low weight event
        ]

        repo_scores = analyzer._score_repositories_by_development_activity(events)

        # Should handle gracefully, might return empty dict if below threshold
        assert isinstance(repo_scores, dict)

    def test_multi_source_scoring_algorithm(self, mock_github_client):
        """Test the corrected multi-source scoring algorithm."""
        discovery = MultiSourceDiscovery(mock_github_client)

        # Test data that would expose the previous scoring bug
        owned_repos = {
            "user/owned-repo": {"source": "owned", "score": 3}
        }

        event_repos = {
            "user/owned-repo": {"source": "events", "score": 5},
            "external/contributed": {"source": "events", "score": 8}
        }

        commit_repos = {
            "user/owned-repo": {"source": "commits", "score": 2},
            "external/contributed": {"source": "commits", "score": 1}
        }

        final_repos = discovery._merge_repository_sources(
            owned_repos, event_repos, commit_repos
        )

        # user/owned-repo should be ranked higher due to owned repo priority boost
        # Score calculation: (3 + 5 + 2) * 1.5 (multi-source) * 2.0 (owned) = 30
        # external/contributed: (8 + 1) * 1.5 (multi-source) = 13.5

        assert final_repos[0] == "user/owned-repo"
        assert final_repos[1] == "external/contributed"

    def test_multi_source_priority_calculation(self, mock_github_client):
        """Test the priority calculation logic in multi-source discovery."""
        discovery = MultiSourceDiscovery(mock_github_client)

        # Create test scenario with different combinations
        owned_repos = {"user/main": {"source": "owned", "score": 3}}
        event_repos = {
            "user/main": {"source": "events", "score": 4},
            "org/work": {"source": "events", "score": 10},  # Higher base score
            "user/side": {"source": "events", "score": 2}
        }
        commit_repos = {"user/main": {"source": "commits", "score": 1}}

        final_repos = discovery._merge_repository_sources(
            owned_repos, event_repos, commit_repos
        )

        # user/main should be first despite lower base activity due to:
        # - Multi-source bonus (3 sources)
        # - Owned repository 2x boost
        # Final score: (3 + 4 + 1) * 1.5 * 2.0 = 24
        # org/work score: 10 (single source, no boosts)

        assert final_repos[0] == "user/main"
        assert "org/work" in final_repos

    def test_scoring_edge_case_zero_events(self):
        """Test scoring behavior with zero events."""
        analyzer = IntelligenceGuidedAnalyzer(Mock())

        scores = analyzer._score_repositories_by_activity([])

        assert isinstance(scores, Counter)
        assert len(scores) == 0

    def test_scoring_edge_case_missing_repo_info(self):
        """Test scoring with events missing repository information."""
        analyzer = IntelligenceGuidedAnalyzer(Mock())

        # Create event without repo info
        event = BaseGitHubEvent(
            id="test",
            type="PushEvent",
            actor=Actor(
                id=123, login="user", display_login="user",
                gravatar_id="", url="", avatar_url=""
            ),
            repo=None,  # Missing repo info
            payload={},
            public=True,
            created_at="2024-01-01T12:00:00Z"
        )

        # Should not crash, should skip events without repo info
        scores = analyzer._score_repositories_by_activity([event])
        assert len(scores) == 0

    def test_event_deduplication_algorithm(self, mock_github_client):
        """Test event deduplication logic."""
        discovery = MultiSourceDiscovery(mock_github_client)

        # Create duplicate events (same ID)
        event1 = self._create_test_event("PushEvent", "duplicate_id", "repo1")
        event2 = self._create_test_event("PullRequestEvent", "duplicate_id", "repo2")  # Same ID
        event3 = self._create_test_event("IssuesEvent", "unique_id", "repo1")

        events = [event1, event2, event3]
        deduplicated = discovery._deduplicate_events(events)

        # Should keep first occurrence of duplicate ID
        assert len(deduplicated) == 2
        assert deduplicated[0].id == "duplicate_id"
        assert deduplicated[0].type == "PushEvent"  # First occurrence
        assert deduplicated[1].id == "unique_id"

    def test_source_count_priority_algorithm(self, mock_github_client):
        """Test that multi-source repositories are prioritized correctly."""
        discovery = MultiSourceDiscovery(mock_github_client)

        owned_repos = {"user/multi": {"score": 1}}
        event_repos = {"user/multi": {"score": 1}, "user/single": {"score": 10}}
        commit_repos = {"user/multi": {"score": 1}}

        final_repos = discovery._merge_repository_sources(
            owned_repos, event_repos, commit_repos
        )

        # user/multi should rank higher despite lower base score because:
        # - 3 sources vs 1 source (primary sort key)
        # - Multi-source bonus and owned repo boost
        assert final_repos[0] == "user/multi"

    def _create_test_event(
        self,
        event_type: str,
        event_id: str,
        repo_name: str,
        created_at: str = "2024-01-01T12:00:00Z"
    ) -> BaseGitHubEvent:
        """Create a test GitHub event."""
        return BaseGitHubEvent(
            id=event_id,
            type=event_type,
            actor=Actor(
                id=123, login="test-user", display_login="test-user",
                gravatar_id="", url="", avatar_url=""
            ),
            repo=Repository(
                id=456, name=repo_name,
                url=f"https://api.github.com/repos/owner/{repo_name}",
                full_name=f"owner/{repo_name}"
            ),
            payload={},
            public=True,
            created_at=created_at
        )
