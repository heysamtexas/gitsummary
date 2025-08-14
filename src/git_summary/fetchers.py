"""Event fetchers for GitHub API data collection."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from typing import Any

from git_summary.github_client import GitHubClient
from git_summary.models import BaseGitHubEvent, EventFactory

logger = logging.getLogger(__name__)


class FetcherError(Exception):
    """Base exception for fetcher errors."""


class AuthenticationError(FetcherError):
    """Raised when GitHub authentication fails."""


class SimpleRateLimiter:
    """Simple rate limiter to prevent GitHub API abuse."""

    def __init__(self, requests_per_minute: int = 60) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute  # seconds between requests
        self.last_request_time = 0.0

    async def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)

        self.last_request_time = asyncio.get_event_loop().time()


class UserEventsFetcher:
    """Fetcher for authenticated user's events via /users/{username}/events endpoint."""

    def __init__(self, client: GitHubClient, username: str) -> None:
        """Initialize the fetcher with a GitHub client and username.

        Args:
            client: GitHub API client for making requests
            username: GitHub username of the authenticated user
        """
        self.client = client
        self.username = username

    async def fetch_events(
        self,
        since: datetime | None = None,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> AsyncGenerator[BaseGitHubEvent, None]:
        """Fetch events for the authenticated user.

        Args:
            since: Only events after this datetime (GitHub API 'since' parameter)
            max_pages: Maximum number of pages to fetch (None for unlimited)
            max_events: Maximum number of events to yield (None for unlimited)
            progress_callback: Optional callback for progress updates

        Yields:
            BaseGitHubEvent: Individual GitHub events

        Raises:
            AuthenticationError: If authentication fails (401/403)
            FetcherError: For other API errors
        """
        try:
            # Build URL with optional since parameter - use /users/{username}/events for authenticated access
            url = f"{self.client.base_url}/users/{self.username}/events"
            params: dict[str, Any] = {"per_page": 100}

            if since:
                # GitHub API expects ISO 8601 format
                params["since"] = since.isoformat()

            logger.info(
                f"Fetching authenticated user events for {self.username} since {since}, max_pages={max_pages}, max_events={max_events}"
            )

            events_yielded = 0

            # Use the client's pagination to fetch all pages
            async for response in self.client.paginate(
                url,
                max_pages=max_pages,
                progress_callback=progress_callback,
                headers=self.client.headers,
                params=params,
            ):
                data = response.json()

                # Parse each event using EventFactory
                for event_data in data:
                    # Check max_events limit before processing
                    if max_events is not None and events_yielded >= max_events:
                        logger.info(
                            f"Reached max_events limit of {max_events}, stopping"
                        )
                        return

                    try:
                        event = EventFactory.create_event(event_data)
                        yield event
                        events_yielded += 1
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse event {event_data.get('id', 'unknown')}: {e}"
                        )
                        continue

        except Exception as e:
            # Handle authentication errors specifically
            if (
                hasattr(e, "response")
                and hasattr(e.response, "status_code")
                and e.response.status_code in (401, 403)
            ):
                logger.error(
                    f"Authentication failed for /users/{self.username}/events: HTTP {e.response.status_code}"
                )
                raise AuthenticationError(
                    f"GitHub authentication failed (HTTP {e.response.status_code}) when fetching user events. "
                    "Please check your token permissions and ensure it has the required scopes."
                ) from e

            # Log the error for debugging
            logger.error(f"Failed to fetch user events with params {params}: {e}")

            # Re-raise other errors as FetcherError with more context
            raise FetcherError(
                f"Failed to fetch authenticated user events for {self.username} from GitHub API. "
                f"Parameters: since={since}, max_pages={max_pages}. "
                f"Error: {e}"
            ) from e

    async def fetch_events_list(
        self,
        since: datetime | None = None,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Fetch events and return as a list.

        Args:
            since: Only events after this datetime
            max_pages: Maximum number of pages to fetch
            max_events: Maximum number of events to return
            progress_callback: Optional callback for progress updates

        Returns:
            List of GitHub events

        Raises:
            AuthenticationError: If authentication fails (401/403)
            FetcherError: For other API errors
        """
        events = []
        async for event in self.fetch_events(
            since, max_pages, max_events, progress_callback
        ):
            events.append(event)

        logger.info(f"Fetched {len(events)} user events")
        return events

    async def fetch_events_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Fetch events within a specific date range.

        Note: GitHub API only supports 'since' parameter, so we fetch from start_date
        and filter client-side for end_date.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            max_pages: Maximum number of pages to fetch
            max_events: Maximum number of events to return
            progress_callback: Optional callback for progress updates

        Returns:
            List of GitHub events within the date range

        Raises:
            AuthenticationError: If authentication fails (401/403)
            FetcherError: For other API errors
        """
        logger.info(
            f"Fetching events from {start_date} to {end_date}, max_events={max_events}"
        )

        events = []
        async for event in self.fetch_events(
            start_date, max_pages, max_events, progress_callback
        ):
            # Parse event timestamp and filter by end_date
            event_time = datetime.fromisoformat(event.created_at.replace("Z", "+00:00"))

            if event_time > end_date:
                continue  # Skip events after end_date

            events.append(event)

        logger.info(f"Fetched {len(events)} events in date range")
        return events


class PublicEventsFetcher:
    """Fetcher for public user events via /users/{username}/events/public endpoint."""

    def __init__(self, client: GitHubClient, username: str) -> None:
        """Initialize the fetcher with a GitHub client and username.

        Args:
            client: GitHub API client for making requests
            username: GitHub username to fetch public events for
        """
        self.client = client
        self.username = username

    async def fetch_events(
        self,
        since: datetime | None = None,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> AsyncGenerator[BaseGitHubEvent, None]:
        """Fetch public events for the specified user.

        Args:
            since: Only events after this datetime (GitHub API 'since' parameter)
            max_pages: Maximum number of pages to fetch (None for unlimited)
            max_events: Maximum number of events to yield (None for unlimited)
            progress_callback: Optional callback for progress updates

        Yields:
            BaseGitHubEvent: Individual GitHub events

        Raises:
            FetcherError: If user not found (404) or other API errors
        """
        try:
            # Build URL with username
            url = f"{self.client.base_url}/users/{self.username}/events/public"
            params: dict[str, Any] = {"per_page": 100}

            if since:
                # GitHub API expects ISO 8601 format
                params["since"] = since.isoformat()

            logger.info(
                f"Fetching public events for user '{self.username}' since {since}, max_pages={max_pages}, max_events={max_events}"
            )

            events_yielded = 0

            # Use the client's pagination to fetch all pages
            async for response in self.client.paginate(
                url,
                max_pages=max_pages,
                progress_callback=progress_callback,
                headers=self.client.headers,
                params=params,
            ):
                data = response.json()

                # Parse each event using EventFactory
                for event_data in data:
                    # Check max_events limit before processing
                    if max_events is not None and events_yielded >= max_events:
                        logger.info(
                            f"Reached max_events limit of {max_events}, stopping"
                        )
                        return

                    try:
                        event = EventFactory.create_event(event_data)
                        yield event
                        events_yielded += 1
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse event {event_data.get('id', 'unknown')}: {e}"
                        )
                        continue

        except Exception as e:
            # Handle user not found errors specifically
            if (
                hasattr(e, "response")
                and hasattr(e.response, "status_code")
                and e.response.status_code == 404
            ):
                logger.error(
                    f"User '{self.username}' not found: HTTP {e.response.status_code}"
                )
                raise FetcherError(
                    f"GitHub user '{self.username}' not found. "
                    "Please check the username and try again."
                ) from e

            # Log the error for debugging
            logger.error(
                f"Failed to fetch public events for user '{self.username}' with params {params}: {e}"
            )

            # Re-raise other errors as FetcherError with more context
            raise FetcherError(
                f"Failed to fetch public events for user '{self.username}' from GitHub API. "
                f"Parameters: since={since}, max_pages={max_pages}. "
                f"Error: {e}"
            ) from e

    async def fetch_events_list(
        self,
        since: datetime | None = None,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Fetch public events and return as a list.

        Args:
            since: Only events after this datetime
            max_pages: Maximum number of pages to fetch
            max_events: Maximum number of events to return
            progress_callback: Optional callback for progress updates

        Returns:
            List of GitHub events

        Raises:
            FetcherError: If user not found (404) or other API errors
        """
        events = []
        async for event in self.fetch_events(
            since, max_pages, max_events, progress_callback
        ):
            events.append(event)

        logger.info(f"Fetched {len(events)} public events for user '{self.username}'")
        return events

    async def fetch_events_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Fetch public events within a specific date range.

        Note: GitHub API only supports 'since' parameter, so we fetch from start_date
        and filter client-side for end_date.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            max_pages: Maximum number of pages to fetch
            max_events: Maximum number of events to return
            progress_callback: Optional callback for progress updates

        Returns:
            List of GitHub events within the date range

        Raises:
            FetcherError: If user not found (404) or other API errors
        """
        logger.info(
            f"Fetching public events for user '{self.username}' from {start_date} to {end_date}, max_events={max_events}"
        )

        events = []
        async for event in self.fetch_events(
            start_date, max_pages, max_events, progress_callback
        ):
            # Parse event timestamp and filter by end_date
            event_time = datetime.fromisoformat(event.created_at.replace("Z", "+00:00"))

            if event_time > end_date:
                continue  # Skip events after end_date

            events.append(event)

        logger.info(
            f"Fetched {len(events)} public events for user '{self.username}' in date range"
        )
        return events


class EventCoordinatorLegacy:
    """Legacy coordinator that selects optimal fetching strategy based on authentication and user context."""

    def __init__(self, client: GitHubClient) -> None:
        """Initialize the coordinator with a GitHub client.

        Args:
            client: GitHub API client for making requests
        """
        self.client = client
        self._authenticated_user: str | None = None

    async def get_authenticated_user(self) -> str | None:
        """Get the authenticated user's login name.

        Returns:
            Username of the authenticated user, or None if not authenticated

        Raises:
            FetcherError: If API request fails
        """
        if self._authenticated_user is not None:
            return self._authenticated_user

        try:
            url = f"{self.client.base_url}/user"

            async for response in self.client.paginate(
                url, max_pages=1, headers=self.client.headers
            ):
                data = response.json()
                self._authenticated_user = data.get("login")
                logger.info(f"Detected authenticated user: {self._authenticated_user}")
                return self._authenticated_user

        except Exception as e:
            if (
                hasattr(e, "response")
                and hasattr(e.response, "status_code")
                and e.response.status_code in (401, 403)
            ):
                logger.info("No valid authentication detected")
                return None

            logger.error(f"Failed to get authenticated user: {e}")
            raise FetcherError(f"Failed to detect authenticated user: {e}") from e

        return None

    async def fetch_events(
        self,
        username: str,
        since: datetime | None = None,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> AsyncGenerator[BaseGitHubEvent, None]:
        """Fetch events using optimal strategy based on authentication context.

        Args:
            username: GitHub username to fetch events for
            since: Only events after this datetime
            max_pages: Maximum number of pages to fetch (None for unlimited)
            max_events: Maximum number of events to yield (None for unlimited)
            progress_callback: Optional callback for progress updates

        Yields:
            BaseGitHubEvent: Individual GitHub events

        Raises:
            AuthenticationError: If authentication is required but fails
            FetcherError: For other API errors
        """
        authenticated_user = await self.get_authenticated_user()

        # Use UserEventsFetcher if requesting own events
        if authenticated_user and username.lower() == authenticated_user.lower():
            logger.info(f"Using UserEventsFetcher for authenticated user {username}")
            try:
                user_fetcher = UserEventsFetcher(self.client, authenticated_user)
                async for event in user_fetcher.fetch_events(
                    since, max_pages, max_events, progress_callback
                ):
                    yield event
                return
            except Exception as e:
                logger.warning(
                    f"UserEventsFetcher failed: {e}, falling back to public fetcher"
                )
                # Fall through to public fetcher

        # Use PublicEventsFetcher for other users or as fallback
        logger.info(f"Using PublicEventsFetcher for user {username}")
        try:
            public_fetcher = PublicEventsFetcher(self.client, username)
            async for event in public_fetcher.fetch_events(
                since, max_pages, max_events, progress_callback
            ):
                yield event
        except Exception as e:
            # If authenticated and public fetcher fails, try user fetcher as last resort
            if authenticated_user and username.lower() == authenticated_user.lower():
                logger.warning(
                    f"Public fetcher failed, retrying with UserEventsFetcher: {e}"
                )
                user_fetcher = UserEventsFetcher(self.client, authenticated_user)
                async for event in user_fetcher.fetch_events(
                    since, max_pages, max_events, progress_callback
                ):
                    yield event
            else:
                raise

    async def fetch_events_list(
        self,
        username: str,
        since: datetime | None = None,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Fetch events using optimal strategy and return as a list.

        Args:
            username: GitHub username to fetch events for
            since: Only events after this datetime
            max_pages: Maximum number of pages to fetch
            max_events: Maximum number of events to return
            progress_callback: Optional callback for progress updates

        Returns:
            List of GitHub events

        Raises:
            AuthenticationError: If authentication is required but fails
            FetcherError: For other API errors
        """
        events = []
        async for event in self.fetch_events(
            username, since, max_pages, max_events, progress_callback
        ):
            events.append(event)

        logger.info(
            f"Fetched {len(events)} events for {username} using smart coordination"
        )
        return events

    async def fetch_events_by_date_range(
        self,
        username: str,
        start_date: datetime,
        end_date: datetime,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Fetch events within a specific date range using optimal strategy.

        Args:
            username: GitHub username to fetch events for
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            max_pages: Maximum number of pages to fetch
            max_events: Maximum number of events to return
            progress_callback: Optional callback for progress updates

        Returns:
            List of GitHub events within the date range

        Raises:
            AuthenticationError: If authentication is required but fails
            FetcherError: For other API errors
        """
        logger.info(
            f"Fetching events for {username} from {start_date} to {end_date}, max_events={max_events}"
        )

        events = []
        async for event in self.fetch_events(
            username, start_date, max_pages, max_events, progress_callback
        ):
            # Parse event timestamp and filter by end_date
            event_time = datetime.fromisoformat(event.created_at.replace("Z", "+00:00"))

            if event_time > end_date:
                continue  # Skip events after end_date

            events.append(event)

        logger.info(
            f"Fetched {len(events)} events for {username} in date range using smart coordination"
        )
        return events


class RepositoryEventsFetcher:
    """Fetcher for repository events via /repos/{owner}/{repo}/events endpoint."""

    MAX_REPO_FETCHES = 10  # Maximum repositories to fetch before falling back

    def __init__(
        self, client: GitHubClient, rate_limiter: SimpleRateLimiter | None = None
    ) -> None:
        """Initialize the fetcher with a GitHub client.

        Args:
            client: GitHub API client for making requests
            rate_limiter: Optional rate limiter (creates default if None)
        """
        self.client = client
        self.rate_limiter = rate_limiter or SimpleRateLimiter()

    def _parse_repo_string(self, repo: str) -> tuple[str, str]:
        """Parse 'owner/repo' format with validation.

        Args:
            repo: Repository string in 'owner/repo' format

        Returns:
            Tuple of (owner, repo_name)

        Raises:
            ValueError: If repository string format is invalid
        """
        if "/" not in repo:
            raise ValueError(f"Repository must be in 'owner/repo' format, got: {repo}")

        parts = repo.split("/")
        if len(parts) != 2:
            raise ValueError(f"Repository must be in 'owner/repo' format, got: {repo}")

        owner, repo_name = parts
        if not owner or not repo_name:
            raise ValueError(f"Both owner and repo name required, got: {repo}")

        return owner.strip(), repo_name.strip()

    async def fetch_events(
        self,
        repository: str,
        username: str | None = None,
        since: datetime | None = None,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> AsyncGenerator[BaseGitHubEvent, None]:
        """Fetch events for a specific repository.

        Args:
            repository: Repository in 'owner/repo' format
            username: Filter events to this user (None for all users)
            since: Only events after this datetime
            max_pages: Maximum number of pages to fetch (None for unlimited)
            max_events: Maximum number of events to yield (None for unlimited)
            progress_callback: Optional callback for progress updates

        Yields:
            BaseGitHubEvent: Individual GitHub events

        Raises:
            ValueError: If repository format is invalid
            FetcherError: For API errors (including private/missing repos)
        """
        try:
            owner, repo_name = self._parse_repo_string(repository)

            # Build URL for repository events
            url = f"{self.client.base_url}/repos/{owner}/{repo_name}/events"
            params: dict[str, Any] = {"per_page": 100}

            if since:
                # GitHub API expects ISO 8601 format
                params["since"] = since.isoformat()

            logger.info(
                f"Fetching repository events for {repository}"
                + (f" filtered to user {username}" if username else "")
                + f" since {since}, max_pages={max_pages}, max_events={max_events}"
            )

            events_yielded = 0

            # Use the client's pagination to fetch all pages
            async for response in self.client.paginate(
                url,
                max_pages=max_pages,
                progress_callback=progress_callback,
                headers=self.client.headers,
                params=params,
            ):
                data = response.json()

                # Parse each event using EventFactory
                for event_data in data:
                    # Check max_events limit before processing
                    if max_events is not None and events_yielded >= max_events:
                        logger.info(
                            f"Reached max_events limit of {max_events}, stopping"
                        )
                        return

                    # Filter by username if specified
                    if (
                        username
                        and event_data.get("actor", {}).get("login") != username
                    ):
                        continue

                    try:
                        event = EventFactory.create_event(event_data)
                        yield event
                        events_yielded += 1
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse event {event_data.get('id', 'unknown')}: {e}"
                        )
                        continue

        except ValueError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Handle specific API errors
            if hasattr(e, "response") and hasattr(e.response, "status_code"):
                status_code = e.response.status_code
                if status_code == 404:
                    raise FetcherError(
                        f"Repository '{repository}' not found or not accessible. "
                        "Please check the repository name and your access permissions."
                    ) from e
                elif status_code in (401, 403):
                    raise FetcherError(
                        f"Access denied to repository '{repository}' (HTTP {status_code}). "
                        "Please check your token permissions."
                    ) from e

            # Log the error for debugging
            logger.error(
                f"Failed to fetch repository events for {repository} with params {params}: {e}"
            )

            # Re-raise other errors as FetcherError with more context
            raise FetcherError(
                f"Failed to fetch events for repository '{repository}' from GitHub API. "
                f"Parameters: since={since}, max_pages={max_pages}. "
                f"Error: {e}"
            ) from e

    async def fetch_events_list(
        self,
        repository: str,
        username: str | None = None,
        since: datetime | None = None,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Fetch repository events and return as a list.

        Args:
            repository: Repository in 'owner/repo' format
            username: Filter events to this user (None for all users)
            since: Only events after this datetime
            max_pages: Maximum number of pages to fetch
            max_events: Maximum number of events to return
            progress_callback: Optional callback for progress updates

        Returns:
            List of GitHub events

        Raises:
            ValueError: If repository format is invalid
            FetcherError: For API errors
        """
        events = []
        async for event in self.fetch_events(
            repository, username, since, max_pages, max_events, progress_callback
        ):
            events.append(event)

        logger.info(f"Fetched {len(events)} events for repository '{repository}'")
        return events

    async def fetch_multiple_repositories(
        self,
        repositories: list[str],
        username: str | None = None,
        since: datetime | None = None,
        max_pages: int | None = None,
        max_events_per_repo: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Fetch events from multiple repositories and merge results.

        Args:
            repositories: List of repositories in 'owner/repo' format
            username: Filter events to this user (None for all users)
            since: Only events after this datetime
            max_pages: Maximum number of pages to fetch per repository
            max_events_per_repo: Maximum number of events per repository
            progress_callback: Optional callback for progress updates

        Returns:
            List of GitHub events from all repositories, deduplicated and sorted

        Raises:
            ValueError: If any repository format is invalid
            FetcherError: For API errors
        """
        all_events = []
        total_repos = len(repositories)

        logger.info(
            f"Fetching events from {total_repos} repositories"
            + (f" for user {username}" if username else "")
        )

        for i, repository in enumerate(repositories):
            logger.debug(f"Processing repository {i+1}/{total_repos}: {repository}")

            # Rate limit before each repository request
            await self.rate_limiter.wait_if_needed()

            try:
                repo_events = await self.fetch_events_list(
                    repository=repository,
                    username=username,
                    since=since,
                    max_pages=max_pages,
                    max_events=max_events_per_repo,
                    progress_callback=progress_callback,
                )
                all_events.extend(repo_events)

            except FetcherError as e:
                logger.warning(f"Failed to fetch events from {repository}: {e}")
                # Continue with other repositories instead of failing completely
                continue

        # Deduplicate and sort events
        deduplicated_events = self._deduplicate_events(all_events)

        logger.info(
            f"Fetched {len(all_events)} total events from {total_repos} repositories, "
            f"{len(deduplicated_events)} after deduplication"
        )

        return deduplicated_events

    def _deduplicate_events(
        self, events: list[BaseGitHubEvent]
    ) -> list[BaseGitHubEvent]:
        """Deduplicate events by event ID and created_at timestamp.

        Args:
            events: List of events that may contain duplicates

        Returns:
            List of unique events sorted by creation time
        """
        seen = set()
        unique_events = []

        for event in events:
            # GitHub event IDs are unique, but created_at provides backup
            key = (event.id, event.created_at)
            if key not in seen:
                seen.add(key)
                unique_events.append(event)

        # Sort events chronologically
        return sorted(unique_events, key=lambda e: e.created_at)


class FetchStrategy(ABC):
    """Abstract base class for event fetching strategies."""

    @abstractmethod
    async def fetch_events(
        self,
        username: str,
        since: datetime | None = None,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Fetch events using this strategy.

        Args:
            username: GitHub username to fetch events for
            since: Only events after this datetime
            max_pages: Maximum number of pages to fetch
            max_events: Maximum number of events to return
            progress_callback: Optional callback for progress updates

        Returns:
            List of GitHub events

        Raises:
            FetcherError: For API errors
        """
        pass


class UserEventStrategy(FetchStrategy):
    """Strategy for fetching events from user event streams."""

    def __init__(
        self, client: GitHubClient, client_filter_repos: list[str] | None = None
    ) -> None:
        """Initialize the strategy.

        Args:
            client: GitHub API client
            client_filter_repos: Optional list of repos to filter client-side
        """
        self.client = client
        self.client_filter_repos = client_filter_repos

    async def fetch_events(
        self,
        username: str,
        since: datetime | None = None,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Fetch events from user event streams."""
        coordinator = EventCoordinatorLegacy(self.client)
        events = await coordinator.fetch_events_by_date_range(
            username=username,
            start_date=since or datetime.min.replace(tzinfo=UTC),
            end_date=datetime.now(UTC),
            max_pages=max_pages,
            max_events=max_events,
            progress_callback=progress_callback,
        )

        # Apply client-side repository filtering if specified
        if self.client_filter_repos:
            repo_names = [r.lower() for r in self.client_filter_repos]
            events = [
                event
                for event in events
                if event.repo
                and (
                    event.repo.name.lower() in repo_names
                    or (
                        event.repo.full_name
                        and event.repo.full_name.lower() in repo_names
                    )
                )
            ]
            logger.info(
                f"Client-side filtered to {len(events)} events from {len(self.client_filter_repos)} repositories"
            )

        return events


class RepositoryEventStrategy(FetchStrategy):
    """Strategy for fetching events directly from repository event streams."""

    def __init__(self, repositories: list[str], client: GitHubClient) -> None:
        """Initialize the strategy.

        Args:
            repositories: List of repositories in 'owner/repo' format
            client: GitHub API client
        """
        self.repositories = repositories
        self.client = client

    async def fetch_events(
        self,
        username: str,
        since: datetime | None = None,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Fetch events from repository event streams."""
        fetcher = RepositoryEventsFetcher(self.client)

        # Calculate max events per repo to distribute fairly
        max_events_per_repo = None
        if max_events:
            max_events_per_repo = max_events // len(self.repositories)

        return await fetcher.fetch_multiple_repositories(
            repositories=self.repositories,
            username=username,
            since=since,
            max_pages=max_pages,
            max_events_per_repo=max_events_per_repo,
            progress_callback=progress_callback,
        )


class EventCoordinator:
    """Smart coordinator that selects optimal fetching strategy based on repository filters."""

    def __init__(self, client: GitHubClient) -> None:
        """Initialize the coordinator with a GitHub client.

        Args:
            client: GitHub API client for making requests
        """
        self.client = client

    def _determine_strategy(
        self, username: str, repo_filters: list[str] | None = None
    ) -> FetchStrategy:
        """Determine the best fetching strategy based on parameters.

        Args:
            username: GitHub username to fetch events for
            repo_filters: Optional list of repository filters

        Returns:
            FetchStrategy: The optimal strategy for this request
        """
        if not repo_filters:
            # No repository filters - use user events
            logger.info("Using UserEventStrategy (no repository filters)")
            return UserEventStrategy(self.client)

        if len(repo_filters) > RepositoryEventsFetcher.MAX_REPO_FETCHES:
            # Too many repositories - fall back to user events with client-side filtering
            logger.info(
                f"Using UserEventStrategy with client-side filtering "
                f"({len(repo_filters)} repositories > {RepositoryEventsFetcher.MAX_REPO_FETCHES} max)"
            )
            return UserEventStrategy(self.client, client_filter_repos=repo_filters)

        # Optimal case - use repository-specific fetching
        logger.info(
            f"Using RepositoryEventStrategy for {len(repo_filters)} repositories"
        )
        return RepositoryEventStrategy(repo_filters, self.client)

    async def fetch_events_by_date_range(
        self,
        username: str,
        start_date: datetime,
        end_date: datetime,
        repo_filters: list[str] | None = None,
        max_pages: int | None = None,
        max_events: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> list[BaseGitHubEvent]:
        """Fetch events within a specific date range using optimal strategy.

        Args:
            username: GitHub username to fetch events for
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            repo_filters: Optional list of repositories to filter to
            max_pages: Maximum number of pages to fetch
            max_events: Maximum number of events to return
            progress_callback: Optional callback for progress updates

        Returns:
            List of GitHub events within the date range

        Raises:
            FetcherError: For API errors
        """
        logger.info(
            f"Fetching events for {username} from {start_date} to {end_date}"
            + (f" for {len(repo_filters)} repositories" if repo_filters else "")
            + (f", max_events={max_events}" if max_events else "")
        )

        # Select strategy based on parameters
        strategy = self._determine_strategy(username, repo_filters)

        # Fetch events using the selected strategy
        events = await strategy.fetch_events(
            username=username,
            since=start_date,
            max_pages=max_pages,
            max_events=max_events,
            progress_callback=progress_callback,
        )

        # Filter by end_date (since GitHub API only supports 'since' parameter)
        filtered_events = []
        for event in events:
            event_time = datetime.fromisoformat(event.created_at.replace("Z", "+00:00"))
            if event_time <= end_date:
                filtered_events.append(event)

        logger.info(
            f"Fetched {len(events)} events, {len(filtered_events)} within date range"
        )
        return filtered_events
