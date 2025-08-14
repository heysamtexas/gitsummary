"""Multi-source repository discovery for heavy automation users (outliers)."""

import logging
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from git_summary.fetchers import RepositoryEventsFetcher, SimpleRateLimiter
from git_summary.github_client import GitHubClient
from git_summary.models import BaseGitHubEvent

logger = logging.getLogger(__name__)


class MultiSourceDiscovery:
    """Multi-source repository discovery for heavy automation users.

    Implements comprehensive discovery for users with heavy automation pollution
    by combining multiple GitHub API sources to find actual development work.
    """

    # Configuration constants
    DEFAULT_MAX_OWNED_REPOS = 50
    DEFAULT_MAX_COMMIT_SEARCH_RESULTS = 100
    DEFAULT_MAX_EVENTS_PER_REPO = 50
    DEFAULT_MAX_TOTAL_REPOS = 25  # Final limit after merging sources

    def __init__(self, github_client: GitHubClient) -> None:
        """Initialize the multi-source discovery analyzer.

        Args:
            github_client: GitHub API client for making requests
        """
        self.github_client = github_client
        # Use more conservative rate limiting for search API
        self.rate_limiter = SimpleRateLimiter(
            requests_per_minute=60  # 30 searches/min + buffer for rest API calls
        )

        # Configuration (can be overridden)
        self.max_owned_repos = self.DEFAULT_MAX_OWNED_REPOS
        self.max_commit_search_results = self.DEFAULT_MAX_COMMIT_SEARCH_RESULTS
        self.max_events_per_repo = self.DEFAULT_MAX_EVENTS_PER_REPO
        self.max_total_repos = self.DEFAULT_MAX_TOTAL_REPOS

        # Development event types (filters out automation noise)
        self.DEVELOPMENT_EVENTS = {
            "PushEvent",
            "PullRequestEvent",
            "ReleaseEvent",
            "CreateEvent",  # Branch/tag creation
            "DeleteEvent",  # Branch/tag deletion
        }

    async def discover_and_fetch(
        self,
        username: str,
        days: int,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Discover repositories via multiple sources and fetch detailed events.

        Args:
            username: GitHub username to analyze
            days: Number of days to analyze
            progress_callback: Optional callback for progress updates (phase, current, total)

        Returns:
            List of detailed GitHub events from discovered repositories
        """
        logger.info(f"Starting multi-source discovery for {username} over {days} days")

        cutoff_date = datetime.now(tz=UTC) - timedelta(days=days)

        if progress_callback:
            progress_callback("Discovering repositories from multiple sources", 0, 100)

        # Source 1: Recently updated owned repositories (critical path)
        if progress_callback:
            progress_callback("Fetching owned repositories", 10, 100)
        try:
            owned_repos = await self._fetch_owned_repositories(username, cutoff_date)
        except Exception as e:
            logger.error(f"Failed to fetch owned repositories: {e}")
            logger.warning(
                "Owned repository discovery failed - results may be incomplete"
            )
            owned_repos = {}

        # Source 2: Development events from user stream (filtered)
        if progress_callback:
            progress_callback("Analyzing user event stream", 30, 100)
        event_repos = await self._extract_repos_from_user_events(username, cutoff_date)

        # Source 3: Commit search API for contribution detection
        if progress_callback:
            progress_callback("Searching commit history", 50, 100)
        commit_repos = await self._search_commit_contributions(username, cutoff_date)

        # Merge sources with priority scoring
        if progress_callback:
            progress_callback("Merging repository sources", 70, 100)
        final_repos = self._merge_repository_sources(
            owned_repos, event_repos, commit_repos
        )

        if not final_repos:
            logger.info(f"No repositories found for {username}")
            return []

        # Fetch detailed events from discovered repositories
        if progress_callback:
            progress_callback("Fetching detailed repository events", 80, 100)

        all_events = await self._fetch_detailed_repository_events(
            username, final_repos, cutoff_date, progress_callback
        )

        if progress_callback:
            progress_callback("Multi-source discovery complete", 100, 100)

        logger.info(
            f"Multi-source discovery complete: {len(all_events)} events from "
            f"{len(final_repos)} repositories for {username}"
        )

        return all_events

    async def _fetch_owned_repositories(
        self, username: str, since: datetime
    ) -> dict[str, dict[str, Any]]:
        """Fetch recently updated owned repositories.

        Args:
            username: GitHub username
            since: Only repos updated after this datetime

        Returns:
            Dictionary mapping repository names to metadata
        """
        repos_data = {}

        try:
            # Use /users/{username}/repos endpoint with updated sort
            url = f"{self.github_client.base_url}/users/{username}/repos"
            params = {
                "sort": "updated",
                "direction": "desc",
                "per_page": 100,
                "page": 1,
            }

            logger.debug(f"Fetching owned repositories for {username}")

            # Apply rate limiting
            await self.rate_limiter.wait_if_needed()

            response = await self.github_client._make_request_with_retry(
                "GET", url, headers=self.github_client.headers, params=params
            )
            self.github_client._update_rate_limit_info(response)

            repos = response.json()

            for repo in repos[: self.max_owned_repos]:
                try:
                    updated_at = datetime.fromisoformat(
                        repo["updated_at"].replace("Z", "+00:00")
                    )

                    # Filter by update date
                    if updated_at >= since:
                        repos_data[repo["full_name"]] = {
                            "source": "owned",
                            "updated_at": updated_at.isoformat(),
                            "private": repo.get("private", False),
                            "language": repo.get("language"),
                            "score": 3,  # High priority for owned repos
                        }

                except (ValueError, KeyError) as e:
                    logger.debug(f"Failed to parse repo data: {e}")
                    continue

            logger.info(f"Found {len(repos_data)} recently updated owned repositories")

        except Exception as e:
            logger.warning(f"Failed to fetch owned repositories: {e}")

        return repos_data

    async def _extract_repos_from_user_events(
        self, username: str, since: datetime
    ) -> dict[str, dict[str, Any]]:
        """Extract repositories from user event stream with development focus.

        Args:
            username: GitHub username
            since: Only events after this datetime

        Returns:
            Dictionary mapping repository names to metadata
        """
        repos_data = {}

        try:
            # Fetch user events for development activity discovery
            events = []
            event_count = 0
            max_events = 300

            async for event_batch in self.github_client.get_user_events_paginated(
                username, per_page=100, max_pages=3
            ):
                for event in event_batch:
                    # Filter by date and development events only
                    try:
                        event_time = datetime.fromisoformat(
                            event.created_at.replace("Z", "+00:00")
                        )
                        if event_time < since:
                            continue

                        # Only count development events (filter automation noise)
                        if event.type in self.DEVELOPMENT_EVENTS:
                            events.append(event)
                            event_count += 1

                        if event_count >= max_events:
                            break

                    except (ValueError, AttributeError):
                        continue

                if event_count >= max_events:
                    break

            # Score repositories by development activity
            repo_scores = Counter()
            for event in events:
                if event.repo and event.repo.name:
                    # Weight by event type
                    weight = self._get_event_weight(event.type)
                    repo_scores[event.repo.name] += weight

            # Convert to metadata format
            for repo_name, score in repo_scores.items():
                repos_data[repo_name] = {
                    "source": "events",
                    "score": score,
                    "event_count": repo_scores[repo_name],
                }

            logger.info(f"Extracted {len(repos_data)} repositories from user events")

        except Exception as e:
            logger.warning(f"Failed to extract repositories from user events: {e}")

        return repos_data

    async def _search_commit_contributions(
        self, username: str, since: datetime
    ) -> dict[str, dict[str, Any]]:
        """Search for commit contributions using GitHub Search API.

        Args:
            username: GitHub username
            since: Only commits after this datetime

        Returns:
            Dictionary mapping repository names to metadata
        """
        repos_data = {}

        try:
            # Format date for GitHub search API (YYYY-MM-DD)
            since_date = since.strftime("%Y-%m-%d")

            # Search for commits by this author
            search_query = f"author:{username} committer-date:>{since_date}"
            url = f"{self.github_client.base_url}/search/commits"
            params = {
                "q": search_query,
                "sort": "committer-date",
                "order": "desc",
                "per_page": min(100, self.max_commit_search_results),
            }

            logger.debug(f"Searching commits for {username} since {since_date}")

            # Apply rate limiting (search API has stricter limits)
            await self.rate_limiter.wait_if_needed()

            response = await self.github_client._make_request_with_retry(
                "GET",
                url,
                headers={
                    **self.github_client.headers,
                    "Accept": "application/vnd.github.cloak-preview+json",  # Required for commit search
                },
                params=params,
            )
            self.github_client._update_rate_limit_info(response)

            data = response.json()
            commits = data.get("items", [])

            # Extract repositories from commit results
            repo_scores = Counter()
            for commit in commits:
                if "repository" in commit and commit["repository"]:
                    repo_name = commit["repository"]["full_name"]
                    repo_scores[repo_name] += 1

            # Convert to metadata format
            for repo_name, commit_count in repo_scores.items():
                repos_data[repo_name] = {
                    "source": "commits",
                    "score": min(commit_count, 5),  # Cap score to prevent skew
                    "commit_count": commit_count,
                }

            logger.info(f"Found {len(repos_data)} repositories from commit search")

        except Exception as e:
            logger.warning(f"Failed to search commit contributions: {e}")
            # Graceful degradation - commit search is nice-to-have

        return repos_data

    def _get_event_weight(self, event_type: str) -> int:
        """Get weight for different event types in scoring.

        Args:
            event_type: GitHub event type

        Returns:
            Weight for this event type
        """
        weights = {
            "PushEvent": 3,  # High weight - actual code commits
            "PullRequestEvent": 2,  # Medium-high weight - code review
            "ReleaseEvent": 2,  # Medium-high weight - releases
            "CreateEvent": 1,  # Medium weight - branch/tag creation
            "DeleteEvent": 1,  # Medium weight - cleanup
        }
        return weights.get(event_type, 1)

    def _merge_repository_sources(
        self,
        owned_repos: dict[str, dict[str, Any]],
        event_repos: dict[str, dict[str, Any]],
        commit_repos: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Merge repository sources with corrected priority scoring.

        Args:
            owned_repos: Repositories from ownership
            event_repos: Repositories from event stream
            commit_repos: Repositories from commit search

        Returns:
            List of repository names ordered by priority
        """
        # Get all unique repository names
        all_repo_names = set()
        all_repo_names.update(owned_repos.keys())
        all_repo_names.update(event_repos.keys())
        all_repo_names.update(commit_repos.keys())

        # Calculate priority scores correctly
        all_repos = {}
        owned_repo_names = set(owned_repos.keys())

        for repo_name in all_repo_names:
            # Start with source contributions
            score = 0.0
            source_count = 0
            sources = []
            data = {}

            # Add weighted contributions from each source
            if repo_name in owned_repos:
                score += 3.0  # Base weight for owned repos
                source_count += 1
                sources.append("owned")
                data["owned"] = owned_repos[repo_name]

            if repo_name in event_repos:
                score += event_repos[repo_name]["score"]
                source_count += 1
                sources.append("events")
                data["events"] = event_repos[repo_name]

            if repo_name in commit_repos:
                score += commit_repos[repo_name]["score"]
                source_count += 1
                sources.append("commits")
                data["commits"] = commit_repos[repo_name]

            # Multi-source bonus (after base scoring)
            if source_count > 1:
                score *= 1.5

            # Owned repo priority boost (final multiplier)
            if repo_name in owned_repo_names:
                score *= 2.0

            all_repos[repo_name] = {
                "total_score": score,
                "sources": sources,
                "source_count": source_count,
                "data": data,
                "is_owned": repo_name in owned_repo_names,
            }

        # Sort by source count first (multi-source preferred), then by score
        sorted_repos = sorted(
            all_repos.items(),
            key=lambda x: (x[1]["source_count"], x[1]["total_score"]),
            reverse=True,
        )

        final_repos = [repo for repo, _ in sorted_repos[: self.max_total_repos]]

        # Validation and logging
        self._validate_discovery_results(final_repos, all_repos)

        logger.info(
            f"Merged sources: {len(owned_repos)} owned + {len(event_repos)} events + "
            f"{len(commit_repos)} commits → {len(final_repos)} final repositories"
        )

        return final_repos

    def _validate_discovery_results(
        self, final_repos: list[str], all_repos: dict[str, dict[str, Any]]
    ) -> None:
        """Validate discovery results and log warnings for potential issues.

        Args:
            final_repos: List of selected repository names
            all_repos: Dictionary of all discovered repositories with metadata
        """
        if len(final_repos) < 5:
            logger.warning(
                f"Low repository count: {len(final_repos)}. "
                "Consider adjusting discovery parameters for better coverage."
            )

        owned_count = sum(1 for repo in final_repos if all_repos[repo]["is_owned"])
        if owned_count == 0:
            logger.warning(
                "No owned repositories found in final results. "
                "Results may be incomplete for this user."
            )

        # Check source diversity
        multi_source_count = sum(
            1 for repo in final_repos if all_repos[repo]["source_count"] > 1
        )
        if multi_source_count == 0:
            logger.info("No multi-source repositories found - sources are disjoint")

        # Log debug breakdown if enabled
        if logger.isEnabledFor(logging.DEBUG):
            self._log_discovery_breakdown(final_repos, all_repos)

    def _log_discovery_breakdown(
        self, final_repos: list[str], all_repos: dict[str, dict[str, Any]]
    ) -> None:
        """Log detailed breakdown of discovery results for debugging.

        Args:
            final_repos: List of selected repository names
            all_repos: Dictionary of all discovered repositories with metadata
        """
        logger.debug("=== Repository Discovery Breakdown ===")
        for i, repo_name in enumerate(final_repos[:10]):  # Top 10 only
            repo_data = all_repos[repo_name]
            logger.debug(
                f"{i + 1:2d}. {repo_name} "
                f"(score: {repo_data['total_score']:.1f}, "
                f"sources: {repo_data['sources']}, "
                f"owned: {repo_data['is_owned']})"
            )

    async def _fetch_detailed_repository_events(
        self,
        username: str,
        repositories: list[str],
        since: datetime,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Fetch detailed events from repositories with rate limiting.

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

        logger.info(f"Fetching detailed events from {len(repositories)} repositories")

        for i, repo_name in enumerate(repositories):
            try:
                if progress_callback:
                    progress_percentage = 80 + int((i / len(repositories)) * 15)
                    progress_callback(
                        f"Analyzing {repo_name}", progress_percentage, 100
                    )

                logger.debug(
                    f"Fetching events from repository {i + 1}/{len(repositories)}: {repo_name}"
                )

                # Apply rate limiting
                await self.rate_limiter.wait_if_needed()

                # Fetch events for this repository, filtered to the user
                repo_events = await repo_fetcher.fetch_events_list(
                    repository=repo_name,
                    username=username,
                    since=since,
                    max_pages=3,  # Limit per repo to manage API usage
                    max_events=self.max_events_per_repo,
                )

                if repo_events:
                    all_events.extend(repo_events)
                    logger.debug(
                        f"Retrieved {len(repo_events)} events from {repo_name}"
                    )

            except Exception as e:
                logger.warning(f"Failed to fetch events from {repo_name}: {e}")
                # Continue with next repository
                continue

        # Deduplicate and sort events
        deduplicated_events = self._deduplicate_events(all_events)
        deduplicated_events.sort(key=lambda e: e.created_at)

        logger.info(
            f"Multi-source fetch complete: {len(all_events)} total events, "
            f"{len(deduplicated_events)} after deduplication"
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
                "analysis_strategy": "multi_source_discovery",
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
            "analysis_strategy": "multi_source_discovery",
            "repository_list": sorted(repositories),
        }
