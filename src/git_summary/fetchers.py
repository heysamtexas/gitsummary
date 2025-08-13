"""Event fetchers for GitHub API data collection."""

import logging
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from typing import Any

from git_summary.github_client import GitHubClient
from git_summary.models import BaseGitHubEvent, EventFactory

logger = logging.getLogger(__name__)


class FetcherError(Exception):
    """Base exception for fetcher errors."""


class AuthenticationError(FetcherError):
    """Raised when GitHub authentication fails."""


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


class EventCoordinator:
    """Smart coordinator that selects optimal fetching strategy based on authentication and user context."""

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
