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
    """Fetcher for authenticated user's events via /user/events endpoint."""

    def __init__(self, client: GitHubClient) -> None:
        """Initialize the fetcher with a GitHub client.

        Args:
            client: GitHub API client for making requests
        """
        self.client = client

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
            # Build URL with optional since parameter
            url = f"{self.client.base_url}/user/events"
            params: dict[str, Any] = {"per_page": 100}

            if since:
                # GitHub API expects ISO 8601 format
                params["since"] = since.isoformat()

            logger.info(
                f"Fetching user events since {since}, max_pages={max_pages}, max_events={max_events}"
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
                    f"Authentication failed for /user/events: HTTP {e.response.status_code}"
                )
                raise AuthenticationError(
                    f"GitHub authentication failed (HTTP {e.response.status_code}) when fetching user events. "
                    "Please check your token permissions and ensure it has the required scopes."
                ) from e

            # Log the error for debugging
            logger.error(f"Failed to fetch user events with params {params}: {e}")

            # Re-raise other errors as FetcherError with more context
            raise FetcherError(
                f"Failed to fetch user events from GitHub API. "
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
