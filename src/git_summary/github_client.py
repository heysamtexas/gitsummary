"""GitHub API client for fetching user activity data."""

import asyncio
import contextlib
import logging
import random
import threading
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit breaker is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back up


class RateLimitError(Exception):
    """Raised when GitHub API rate limit is exceeded."""

    def __init__(self, message: str, remaining: int, reset_time: datetime) -> None:
        super().__init__(message)
        self.remaining = remaining
        self.reset_time = reset_time


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class GitHubEvent(BaseModel):
    """Model for a GitHub event."""

    id: str
    type: str
    actor: dict[str, Any]
    repo: dict[str, Any]
    payload: dict[str, Any]
    created_at: str
    public: bool


class GitHubClient:
    """Client for interacting with the GitHub API."""

    def __init__(self, token: str, base_url: str = "https://api.github.com") -> None:
        """Initialize the GitHub client.

        Args:
            token: GitHub Personal Access Token
            base_url: GitHub API base URL
        """
        self.token = token
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        # Rate limiting tracking
        self.rate_limit_remaining: int | None = None
        self.rate_limit_limit: int | None = None
        self.rate_limit_reset: datetime | None = None

        # Circuit breaker state
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.circuit_breaker_timeout = 60  # seconds
        self._circuit_breaker_lock = threading.Lock()

        # Persistent HTTP client for better connection reuse
        self._http_client: httpx.AsyncClient | None = None

    def _update_rate_limit_info(self, response: httpx.Response) -> None:
        """Update rate limit information from GitHub API response headers."""
        if "x-ratelimit-remaining" in response.headers:
            self.rate_limit_remaining = int(response.headers["x-ratelimit-remaining"])
        if "x-ratelimit-limit" in response.headers:
            self.rate_limit_limit = int(response.headers["x-ratelimit-limit"])
        if "x-ratelimit-reset" in response.headers:
            reset_timestamp = int(response.headers["x-ratelimit-reset"])
            self.rate_limit_reset = datetime.fromtimestamp(reset_timestamp, tz=UTC)

    async def _check_rate_limit(self) -> None:
        """Check rate limit and apply backoff if necessary."""
        if self.rate_limit_remaining is not None and self.rate_limit_remaining <= 0:
            if self.rate_limit_reset:
                raise RateLimitError(
                    f"GitHub API rate limit exceeded. Rate limit resets at {self.rate_limit_reset}",
                    remaining=self.rate_limit_remaining,
                    reset_time=self.rate_limit_reset,
                )
            else:
                raise RateLimitError(
                    "GitHub API rate limit exceeded",
                    remaining=self.rate_limit_remaining,
                    reset_time=datetime.now(tz=UTC),
                )

        # Implement exponential backoff when approaching rate limit
        if self.rate_limit_remaining is not None and self.rate_limit_remaining < 100:
            # Calculate delay based on remaining requests
            if self.rate_limit_remaining < 10:
                delay = 5.0  # 5 seconds for very low remaining requests
            elif self.rate_limit_remaining < 50:
                delay = 2.0  # 2 seconds for low remaining requests
            else:
                delay = 1.0  # 1 second for moderate remaining requests

            await asyncio.sleep(delay)

    def _calculate_retry_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate exponential backoff delay with jitter to prevent thundering herd."""
        # Exponential backoff: base_delay * 2^attempt, capped at 60 seconds
        exponential_delay = min(base_delay * (2**attempt), 60.0)

        # Add jitter: ±10% randomness to prevent thundering herd
        jitter_factor = 0.1
        jitter_range = exponential_delay * jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)

        # Ensure minimum delay of 0.1 seconds
        final_delay: float = max(0.1, exponential_delay + jitter)

        logger.debug(
            f"Calculated retry delay: {final_delay:.2f}s (attempt {attempt}, base {base_delay}s)"
        )
        return final_delay

    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry."""
        if isinstance(error, httpx.HTTPStatusError):
            # Don't retry authentication/authorization errors
            if error.response.status_code in (401, 403):
                logger.info(
                    f"Not retrying authentication error: {error.response.status_code}"
                )
                return False
            # Don't retry client errors (4xx) except rate limiting
            if (
                400 <= error.response.status_code < 500
                and error.response.status_code != 429
            ):
                logger.info(f"Not retrying client error: {error.response.status_code}")
                return False

        # Retry network errors and server errors
        if isinstance(error, httpx.NetworkError | httpx.TimeoutException):
            logger.info(f"Will retry network error: {type(error).__name__}")
            return True

        # Retry server errors (5xx) and rate limit errors (429)
        if isinstance(error, httpx.HTTPStatusError) and (
            error.response.status_code >= 500 or error.response.status_code == 429
        ):
            logger.info(f"Will retry server error: {error.response.status_code}")
            return True

        return False

    def _check_circuit_breaker(self) -> None:
        """Check circuit breaker state and potentially fail fast."""
        with self._circuit_breaker_lock:
            if self.circuit_breaker_state == CircuitBreakerState.OPEN:
                # Check if we should transition to half-open
                if (
                    self.last_failure_time
                    and (datetime.now(tz=UTC) - self.last_failure_time).total_seconds()
                    > self.circuit_breaker_timeout
                ):
                    old_state = self.circuit_breaker_state
                    self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                    logger.info(
                        f"Circuit breaker transitioning: {old_state.value} -> {self.circuit_breaker_state.value}"
                    )
                else:
                    logger.warning("Circuit breaker is OPEN, failing fast")
                    raise CircuitBreakerError(
                        "Circuit breaker is open, requests are being rejected"
                    )

    def _record_success(self) -> None:
        """Record a successful request, potentially closing the circuit breaker."""
        with self._circuit_breaker_lock:
            old_state = self.circuit_breaker_state
            if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                self.circuit_breaker_state = CircuitBreakerState.CLOSED
                if old_state != self.circuit_breaker_state:
                    logger.info("Circuit breaker closing after successful request")

            self.failure_count = 0
            self.last_failure_time = None

    def _record_failure(self) -> None:
        """Record a failed request, potentially opening the circuit breaker."""
        with self._circuit_breaker_lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(tz=UTC)

            logger.warning(f"Request failure recorded. Count: {self.failure_count}")

            old_state = self.circuit_breaker_state

            # Open circuit breaker after 5 consecutive failures
            if (
                self.failure_count >= 5
                and self.circuit_breaker_state != CircuitBreakerState.OPEN
            ):
                self.circuit_breaker_state = CircuitBreakerState.OPEN

            # Log state transitions
            if old_state != self.circuit_breaker_state:
                logger.error(
                    f"Circuit breaker state changed: {old_state.value} -> {self.circuit_breaker_state.value}"
                )

    async def _make_request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make HTTP request with retry logic and circuit breaker."""
        self._check_circuit_breaker()

        max_retries = 3
        base_delay = 1.0  # Start with 1 second

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                if self._http_client is None:
                    self._http_client = httpx.AsyncClient(
                        timeout=30.0,
                        limits=httpx.Limits(
                            max_keepalive_connections=10, max_connections=20
                        ),
                    )

                response = await self._http_client.request(method, url, **kwargs)
                response.raise_for_status()

                # Success - record it and return
                self._record_success()
                return response

            except Exception as error:
                last_exception = error
                self._record_failure()

                # If this is the last attempt, don't sleep
                if attempt == max_retries:
                    break

                # Check if we should retry
                if not self._should_retry(error):
                    logger.info(f"Not retrying error on attempt {attempt + 1}: {error}")
                    break

                # Calculate exponential backoff delay with jitter
                delay = self._calculate_retry_delay(attempt, base_delay)
                logger.info(
                    f"Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)

        # All retries exhausted, raise the last exception
        logger.error(f"All retries exhausted. Final error: {last_exception}")
        if last_exception:
            raise last_exception
        else:
            raise Exception("Request failed with no recorded exception")

    def _parse_link_header(self, link_header: str) -> dict[str, str]:
        """Parse GitHub's Link header for pagination URLs.

        Args:
            link_header: The Link header value from GitHub API response

        Returns:
            Dictionary with 'next', 'prev', 'first', 'last' URLs if present
        """
        links: dict[str, str] = {}

        if not link_header:
            return links

        # Split by comma and process each link
        for link in link_header.split(","):
            link = link.strip()
            if ";" not in link:
                continue

            url_part, rel_part = link.split(";", 1)
            url = url_part.strip(" <>")

            # Extract rel value
            for param in rel_part.split(";"):
                param = param.strip()
                if param.startswith("rel="):
                    rel = param[4:].strip("\"'")
                    links[rel] = url
                    break

        return links

    async def paginate(
        self,
        initial_url: str,
        max_pages: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[httpx.Response, None]:
        """Paginate through GitHub API responses using Link headers.

        Args:
            initial_url: The first URL to fetch
            max_pages: Maximum number of pages to fetch (None for unlimited)
            progress_callback: Optional callback for progress updates (current_page, total_pages)
            **kwargs: Additional arguments to pass to the HTTP request

        Yields:
            httpx.Response objects for each page
        """
        current_url: str | None = initial_url
        page_count = 0
        total_pages = None

        while current_url and (max_pages is None or page_count < max_pages):
            page_count += 1

            # Make request for current page
            response = await self._make_request_with_retry("GET", current_url, **kwargs)

            # Update rate limit info
            self._update_rate_limit_info(response)

            # Try to determine total pages from Link header (GitHub specific)
            if total_pages is None and "link" in response.headers:
                links = self._parse_link_header(response.headers["link"])
                if "last" in links:
                    # Extract page number from last URL
                    parsed = urlparse(links["last"])
                    query_params = parse_qs(parsed.query)
                    if "page" in query_params:
                        with contextlib.suppress(ValueError, IndexError):
                            total_pages = int(query_params["page"][0])

            # Call progress callback if provided
            if progress_callback:
                progress_callback(page_count, total_pages)

            # Yield the response
            yield response

            # Check if there's a next page
            if "link" not in response.headers:
                break

            links = self._parse_link_header(response.headers["link"])
            current_url = links.get("next")

            # If no next link, we're done
            if not current_url:
                break

    def get_rate_limit_status(self) -> dict[str, Any]:
        """Return current rate limit status.

        Returns:
            Dictionary containing rate limit information
        """
        return {
            "remaining": self.rate_limit_remaining,
            "limit": self.rate_limit_limit,
            "reset_time": self.rate_limit_reset,
            "reset_in_seconds": (
                int((self.rate_limit_reset - datetime.now(tz=UTC)).total_seconds())
                if self.rate_limit_reset
                else None
            ),
        }

    async def get_user_events(
        self,
        username: str,
        per_page: int = 100,
        page: int = 1,
    ) -> list[GitHubEvent]:
        """Fetch events for a specific user.

        Args:
            username: GitHub username
            per_page: Number of events per page (max 100)
            page: Page number to fetch

        Returns:
            List of GitHub events

        Raises:
            httpx.HTTPStatusError: If the API request fails
        """
        await self._check_rate_limit()

        response = await self._make_request_with_retry(
            "GET",
            f"{self.base_url}/users/{username}/events",
            headers=self.headers,
            params={"per_page": per_page, "page": page},
        )
        self._update_rate_limit_info(response)
        data = response.json()

        return [GitHubEvent(**event) for event in data]

    async def get_user_events_paginated(
        self,
        username: str,
        per_page: int = 100,
        max_pages: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> AsyncGenerator[list[GitHubEvent], None]:
        """Fetch events for a specific user with automatic pagination.

        Args:
            username: GitHub username
            per_page: Number of events per page (max 100)
            max_pages: Maximum number of pages to fetch (None for unlimited)
            progress_callback: Optional callback for progress updates

        Yields:
            Lists of GitHubEvent objects for each page
        """
        await self._check_rate_limit()

        initial_url = f"{self.base_url}/users/{username}/events"

        async for response in self.paginate(
            initial_url,
            max_pages=max_pages,
            progress_callback=progress_callback,
            headers=self.headers,
            params={"per_page": per_page},
        ):
            data = response.json()
            yield [GitHubEvent(**event) for event in data]

    async def get_user_received_events(
        self,
        username: str,
        per_page: int = 100,
        page: int = 1,
    ) -> list[GitHubEvent]:
        """Fetch events received by a specific user.

        Args:
            username: GitHub username
            per_page: Number of events per page (max 100)
            page: Page number to fetch

        Returns:
            List of GitHub events received by the user

        Raises:
            httpx.HTTPStatusError: If the API request fails
        """
        await self._check_rate_limit()

        response = await self._make_request_with_retry(
            "GET",
            f"{self.base_url}/users/{username}/received_events",
            headers=self.headers,
            params={"per_page": per_page, "page": page},
        )
        self._update_rate_limit_info(response)
        data = response.json()

        return [GitHubEvent(**event) for event in data]

    async def get_user_received_events_paginated(
        self,
        username: str,
        per_page: int = 100,
        max_pages: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> AsyncGenerator[list[GitHubEvent], None]:
        """Fetch events received by a specific user with automatic pagination.

        Args:
            username: GitHub username
            per_page: Number of events per page (max 100)
            max_pages: Maximum number of pages to fetch (None for unlimited)
            progress_callback: Optional callback for progress updates

        Yields:
            Lists of GitHubEvent objects received by the user for each page
        """
        await self._check_rate_limit()

        initial_url = f"{self.base_url}/users/{username}/received_events"

        async for response in self.paginate(
            initial_url,
            max_pages=max_pages,
            progress_callback=progress_callback,
            headers=self.headers,
            params={"per_page": per_page},
        ):
            data = response.json()
            yield [GitHubEvent(**event) for event in data]

    async def validate_token(self) -> dict[str, Any]:
        """Validate the GitHub token and return user information.

        Returns:
            User information from GitHub API

        Raises:
            httpx.HTTPStatusError: If the token is invalid
        """
        await self._check_rate_limit()

        response = await self._make_request_with_retry(
            "GET",
            f"{self.base_url}/user",
            headers=self.headers,
        )
        self._update_rate_limit_info(response)
        return response.json()  # type: ignore[no-any-return]

    async def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self) -> "GitHubClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit with cleanup."""
        await self.close()
