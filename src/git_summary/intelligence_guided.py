"""Intelligence-guided repository discovery for normal users (98% of cases)."""

import logging
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from git_summary.fetchers import RepositoryEventsFetcher, SimpleRateLimiter
from git_summary.github_client import GitHubClient
from git_summary.models import BaseGitHubEvent

logger = logging.getLogger(__name__)


class IntelligenceGuidedAnalyzer:
    """Intelligence-guided repository discovery for normal users.

    Uses the 300-event stream as a repository filter (not primary data source),
    then performs sequential deep-dive fetching with rate limiting.
    """

    # Configuration constants
    DEFAULT_MAX_INITIAL_EVENTS = 300
    DEFAULT_MAX_REPOS = 15
    DEFAULT_MIN_REPO_SCORE = 1
    DEFAULT_MAX_PAGES_PER_REPO = 5
    DEFAULT_MAX_EVENTS_PER_REPO = 100

    def __init__(self, github_client: GitHubClient) -> None:
        """Initialize the analyzer.

        Args:
            github_client: GitHub API client for fetching events
        """
        self.github_client = github_client
        self.rate_limiter = SimpleRateLimiter(
            requests_per_minute=180
        )  # 0.33s between requests - well within 5000/hour budget

        # Development event types that indicate actual development work
        self.DEVELOPMENT_EVENTS = {
            "PushEvent",
            "PullRequestEvent",
            "ReleaseEvent",
            "CreateEvent",  # Branch/tag creation
            "DeleteEvent",  # Branch/tag deletion
            "ForkEvent",
            "PublicEvent",  # Repository made public
        }

        # Configuration (can be overridden)
        self.min_repo_score = self.DEFAULT_MIN_REPO_SCORE
        self.max_repos_to_analyze = self.DEFAULT_MAX_REPOS
        self.max_initial_events = self.DEFAULT_MAX_INITIAL_EVENTS
        self.max_pages_per_repo = self.DEFAULT_MAX_PAGES_PER_REPO
        self.max_events_per_repo = self.DEFAULT_MAX_EVENTS_PER_REPO

    async def discover_and_fetch(
        self,
        username: str,
        days: int,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Discover repositories via intelligence and fetch detailed events.

        Args:
            username: GitHub username to analyze
            days: Number of days to analyze
            progress_callback: Optional callback for progress updates (phase, current, total)

        Returns:
            List of detailed GitHub events from discovered repositories
        """
        logger.info(
            f"Starting intelligence-guided analysis for {username} over {days} days"
        )

        if progress_callback:
            progress_callback("Analyzing user activity", 0, 100)

        # Step 1: Get 300 recent events to identify development repositories
        cutoff_date = datetime.now(tz=UTC) - timedelta(days=days)
        user_events = await self._fetch_user_events_for_discovery(username, cutoff_date)

        if progress_callback:
            progress_callback("Identifying development repositories", 20, 100)

        # Step 2: Score repositories by development activity
        repo_scores = self._score_repositories_by_development_activity(user_events)

        if not repo_scores:
            logger.info(f"No development repositories found for {username}")
            return []

        # Step 3: Select top repositories for deep-dive analysis
        selected_repos = self._select_repositories_for_analysis(repo_scores)

        if progress_callback:
            progress_callback("Deep-diving into repositories", 40, 100)

        # Step 4: Perform sequential deep-dive fetching
        all_events = await self._fetch_detailed_repository_events(
            username, selected_repos, cutoff_date, progress_callback
        )

        if progress_callback:
            progress_callback("Analysis complete", 100, 100)

        logger.info(
            f"Intelligence-guided analysis complete: {len(all_events)} events from "
            f"{len(selected_repos)} repositories for {username}"
        )

        return all_events

    async def _fetch_user_events_for_discovery(
        self, username: str, since: datetime
    ) -> list[BaseGitHubEvent]:
        """Fetch user events for repository discovery (up to 300 events).

        Args:
            username: GitHub username
            since: Only events after this datetime

        Returns:
            List of recent user events for analysis
        """
        events = []
        event_count = 0

        logger.debug(
            f"Fetching user events for repository discovery (max {self.max_initial_events})"
        )

        max_pages = (self.max_initial_events + 99) // 100  # Calculate pages needed
        async for event_batch in self.github_client.get_user_events_paginated(
            username,
            per_page=100,
            max_pages=max_pages,
        ):
            for event in event_batch:
                # Filter by date
                try:
                    event_time = datetime.fromisoformat(
                        event.created_at.replace("Z", "+00:00")
                    )
                    if event_time < since:
                        continue
                except (ValueError, AttributeError):
                    continue

                events.append(event)
                event_count += 1

                if event_count >= self.max_initial_events:
                    break

            if event_count >= self.max_initial_events:
                break

        logger.info(f"Retrieved {len(events)} user events for repository discovery")
        return events

    def _score_repositories_by_development_activity(
        self, events: list[BaseGitHubEvent]
    ) -> dict[str, int]:
        """Score repositories by development event frequency in single pass.

        Args:
            events: List of user events

        Returns:
            Dictionary mapping repository names to development scores
        """
        repo_scores: Counter[str] = Counter()

        # Single pass through events (O(n) instead of O(n²))
        for event in events:
            if event.type in self.DEVELOPMENT_EVENTS and event.repo and event.repo.name:
                weight = self._get_event_weight(event.type)
                repo_scores[event.repo.name] += weight

        # Filter out repositories with insufficient activity
        filtered_scores = {
            repo: score
            for repo, score in repo_scores.items()
            if score >= self.min_repo_score
        }

        # Defensive check for empty results
        if not filtered_scores:
            logger.warning(
                f"No repositories meet minimum score threshold ({self.min_repo_score}). "
                f"Found {len(repo_scores)} repositories with scores: {dict(repo_scores.most_common(5))}"
            )
            return {}

        logger.info(
            f"Scored {len(repo_scores)} repositories, {len(filtered_scores)} above threshold. "
            f"Top repositories: {dict(repo_scores.most_common(5))}"
        )

        return filtered_scores

    def _get_event_weight(self, event_type: str) -> int:
        """Get weight for different event types in scoring.

        Args:
            event_type: GitHub event type

        Returns:
            Weight for this event type (higher = more important)
        """
        weights = {
            "PushEvent": 3,  # High weight - actual code commits
            "PullRequestEvent": 2,  # Medium-high weight - code review
            "ReleaseEvent": 2,  # Medium-high weight - releases
            "CreateEvent": 1,  # Medium weight - branch/tag creation
            "DeleteEvent": 1,  # Medium weight - cleanup
            "ForkEvent": 1,  # Medium weight - forking
            "PublicEvent": 1,  # Medium weight - publishing
        }
        return weights.get(event_type, 1)

    def _select_repositories_for_analysis(
        self, repo_scores: dict[str, int]
    ) -> list[str]:
        """Select top repositories for detailed analysis.

        Args:
            repo_scores: Repository names mapped to development scores

        Returns:
            List of repository names to analyze in detail
        """
        # Sort by score descending and take top repositories
        sorted_repos = sorted(repo_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [repo for repo, score in sorted_repos[: self.max_repos_to_analyze]]

        logger.info(
            f"Selected {len(selected)} repositories for deep-dive analysis: {selected}"
        )
        return selected

    async def _fetch_detailed_repository_events(
        self,
        username: str,
        repositories: list[str],
        since: datetime,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Fetch detailed events from repositories with sequential rate limiting.

        Args:
            username: GitHub username to filter events for
            repositories: List of repository names to analyze
            since: Only events after this datetime
            progress_callback: Optional progress callback

        Returns:
            List of detailed events from all repositories
        """
        all_events = []
        repo_fetcher = RepositoryEventsFetcher(self.github_client, self.rate_limiter)

        logger.info(
            f"Starting sequential deep-dive into {len(repositories)} repositories"
        )

        for i, repo_name in enumerate(repositories):
            try:
                if progress_callback:
                    progress_percentage = 40 + int((i / len(repositories)) * 50)
                    progress_callback(
                        f"Analyzing {repo_name}", progress_percentage, 100
                    )

                logger.debug(
                    f"Deep-diving into repository {i + 1}/{len(repositories)}: {repo_name}"
                )

                # Apply rate limiting before each repository request
                await self.rate_limiter.wait_if_needed()

                # Fetch events for this repository, filtered to the user
                repo_events = await repo_fetcher.fetch_events_list(
                    repository=repo_name,
                    username=username,
                    since=since,
                    max_pages=self.max_pages_per_repo,
                    max_events=self.max_events_per_repo,
                )

                if repo_events:
                    all_events.extend(repo_events)
                    logger.debug(
                        f"Retrieved {len(repo_events)} events from {repo_name}"
                    )
                else:
                    logger.debug(f"No events found for user {username} in {repo_name}")
                    # Early termination optimization - if no user activity found, skip similar repos
                    # This could be enhanced with similarity detection

            except Exception as e:
                logger.warning(f"Failed to fetch events from {repo_name}: {e}")
                # Continue with next repository instead of failing entire analysis
                continue

        # Deduplicate events by ID (user events + repo events may overlap)
        deduplicated_events = self._deduplicate_events(all_events)

        # Sort chronologically
        deduplicated_events.sort(key=lambda e: e.created_at)

        logger.info(
            f"Deep-dive complete: {len(all_events)} total events, {len(deduplicated_events)} after deduplication"
        )
        return deduplicated_events

    def _deduplicate_events(
        self, events: list[BaseGitHubEvent]
    ) -> list[BaseGitHubEvent]:
        """Deduplicate events by GitHub event ID.

        Args:
            events: List of events that may contain duplicates

        Returns:
            List of unique events
        """
        seen_ids = set()
        unique_events = []

        for event in events:
            if event.id not in seen_ids:
                seen_ids.add(event.id)
                unique_events.append(event)

        return unique_events

    def get_analysis_stats(self, events: list[BaseGitHubEvent]) -> dict[str, Any]:
        """Generate analysis statistics for the discovered events.

        Args:
            events: List of events from the analysis

        Returns:
            Dictionary with analysis statistics
        """
        if not events:
            return {
                "total_events": 0,
                "repositories_analyzed": 0,
                "event_type_breakdown": {},
                "date_range": None,
                "analysis_strategy": "intelligence_guided",
            }

        # Count event types
        event_types = Counter(event.type for event in events)

        # Count unique repositories
        repositories = {
            event.repo.name for event in events if event.repo and event.repo.name
        }

        # Determine date range
        event_times = []
        for event in events:
            try:
                event_time = datetime.fromisoformat(
                    event.created_at.replace("Z", "+00:00")
                )
                event_times.append(event_time)
            except (ValueError, AttributeError):
                continue

        date_range = None
        if event_times:
            event_times.sort()
            date_range = {
                "start": event_times[0].isoformat(),
                "end": event_times[-1].isoformat(),
                "span_days": (event_times[-1] - event_times[0]).days,
            }

        return {
            "total_events": len(events),
            "repositories_analyzed": len(repositories),
            "event_type_breakdown": dict(event_types),
            "date_range": date_range,
            "analysis_strategy": "intelligence_guided",
            "repository_list": sorted(repositories),
        }
