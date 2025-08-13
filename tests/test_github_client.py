"""Tests for the GitHub API client."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from git_summary.github_client import (
    CircuitBreakerError,
    CircuitBreakerState,
    GitHubClient,
    GitHubEvent,
    RateLimitError,
)


class TestRateLimitError:
    """Test the RateLimitError exception."""

    def test_rate_limit_error_creation(self):
        """Test creating a RateLimitError with all attributes."""
        reset_time = datetime.now(tz=UTC)
        error = RateLimitError(
            "Rate limit exceeded", remaining=0, reset_time=reset_time
        )

        assert str(error) == "Rate limit exceeded"
        assert error.remaining == 0
        assert error.reset_time == reset_time


class TestGitHubClient:
    """Test the GitHubClient class."""

    @pytest.fixture
    def client(self):
        """Create a GitHub client for testing."""
        return GitHubClient(token="test_token")

    def test_client_initialization(self, client):
        """Test client initialization with default values."""
        assert client.token == "test_token"
        assert client.base_url == "https://api.github.com"
        assert client.headers["Authorization"] == "Bearer test_token"
        assert client.rate_limit_remaining is None
        assert client.rate_limit_limit is None
        assert client.rate_limit_reset is None

    def test_client_initialization_with_custom_base_url(self):
        """Test client initialization with custom base URL."""
        client = GitHubClient(
            token="test_token", base_url="https://api.github.enterprise.com"
        )
        assert client.base_url == "https://api.github.enterprise.com"

    def test_update_rate_limit_info_all_headers(self, client):
        """Test updating rate limit info with all headers present."""
        mock_response = Mock()
        mock_response.headers = {
            "x-ratelimit-remaining": "4999",
            "x-ratelimit-limit": "5000",
            "x-ratelimit-reset": "1672531200",  # 2023-01-01 00:00:00 UTC
        }

        client._update_rate_limit_info(mock_response)

        assert client.rate_limit_remaining == 4999
        assert client.rate_limit_limit == 5000
        assert client.rate_limit_reset == datetime.fromtimestamp(1672531200, tz=UTC)

    def test_update_rate_limit_info_partial_headers(self, client):
        """Test updating rate limit info with only some headers present."""
        mock_response = Mock()
        mock_response.headers = {
            "x-ratelimit-remaining": "100",
        }

        client._update_rate_limit_info(mock_response)

        assert client.rate_limit_remaining == 100
        assert client.rate_limit_limit is None
        assert client.rate_limit_reset is None

    def test_update_rate_limit_info_no_headers(self, client):
        """Test updating rate limit info with no headers present."""
        mock_response = Mock()
        mock_response.headers = {}

        # Should not raise an error
        client._update_rate_limit_info(mock_response)

        assert client.rate_limit_remaining is None
        assert client.rate_limit_limit is None
        assert client.rate_limit_reset is None

    @pytest.mark.asyncio
    async def test_check_rate_limit_no_limit_info(self, client):
        """Test rate limit check when no limit info is available."""
        # Should not raise an error or delay
        await client._check_rate_limit()

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded_with_reset_time(self, client):
        """Test rate limit check when limit is exceeded with reset time."""
        reset_time = datetime.now(tz=UTC)
        client.rate_limit_remaining = 0
        client.rate_limit_reset = reset_time

        with pytest.raises(RateLimitError) as exc_info:
            await client._check_rate_limit()

        assert exc_info.value.remaining == 0
        assert exc_info.value.reset_time == reset_time
        assert "Rate limit resets at" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded_without_reset_time(self, client):
        """Test rate limit check when limit is exceeded without reset time."""
        client.rate_limit_remaining = 0
        client.rate_limit_reset = None

        with pytest.raises(RateLimitError) as exc_info:
            await client._check_rate_limit()

        assert exc_info.value.remaining == 0
        assert "GitHub API rate limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_check_rate_limit_backoff_very_low(self, client):
        """Test backoff when remaining requests are very low."""
        client.rate_limit_remaining = 5

        start_time = asyncio.get_event_loop().time()
        await client._check_rate_limit()
        end_time = asyncio.get_event_loop().time()

        # Should delay for approximately 5 seconds
        assert end_time - start_time >= 4.5  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_check_rate_limit_backoff_low(self, client):
        """Test backoff when remaining requests are low."""
        client.rate_limit_remaining = 25

        start_time = asyncio.get_event_loop().time()
        await client._check_rate_limit()
        end_time = asyncio.get_event_loop().time()

        # Should delay for approximately 2 seconds
        assert end_time - start_time >= 1.5  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_check_rate_limit_backoff_moderate(self, client):
        """Test backoff when remaining requests are moderate."""
        client.rate_limit_remaining = 75

        start_time = asyncio.get_event_loop().time()
        await client._check_rate_limit()
        end_time = asyncio.get_event_loop().time()

        # Should delay for approximately 1 second
        assert end_time - start_time >= 0.5  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_check_rate_limit_no_backoff(self, client):
        """Test no backoff when remaining requests are sufficient."""
        client.rate_limit_remaining = 150

        start_time = asyncio.get_event_loop().time()
        await client._check_rate_limit()
        end_time = asyncio.get_event_loop().time()

        # Should not delay
        assert end_time - start_time < 0.1

    def test_get_rate_limit_status_all_info(self, client):
        """Test getting rate limit status with all information available."""
        reset_time = datetime.now(tz=UTC).replace(second=0, microsecond=0)
        client.rate_limit_remaining = 4500
        client.rate_limit_limit = 5000
        client.rate_limit_reset = reset_time

        status = client.get_rate_limit_status()

        assert status["remaining"] == 4500
        assert status["limit"] == 5000
        assert status["reset_time"] == reset_time
        assert isinstance(status["reset_in_seconds"], int)

    def test_get_rate_limit_status_no_info(self, client):
        """Test getting rate limit status with no information available."""
        status = client.get_rate_limit_status()

        assert status["remaining"] is None
        assert status["limit"] is None
        assert status["reset_time"] is None
        assert status["reset_in_seconds"] is None

    @pytest.mark.asyncio
    async def test_get_user_events_success(self, client):
        """Test successful user events retrieval with rate limiting."""
        mock_response_data = [
            {
                "id": "123",
                "type": "PushEvent",
                "actor": {"login": "testuser"},
                "repo": {"name": "test/repo"},
                "payload": {"commits": []},
                "created_at": "2023-01-01T00:00:00Z",
                "public": True,
            }
        ]

        # Mock the retry method to avoid complexity
        with patch.object(client, "_make_request_with_retry") as mock_retry:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.headers = {"x-ratelimit-remaining": "4999"}
            mock_retry.return_value = mock_response

            events = await client.get_user_events("testuser")

            assert len(events) == 1
            assert isinstance(events[0], GitHubEvent)
            assert events[0].id == "123"
            assert events[0].type == "PushEvent"
            assert client.rate_limit_remaining == 4999

    @pytest.mark.asyncio
    async def test_get_user_received_events_success(self, client):
        """Test successful user received events retrieval with rate limiting."""
        mock_response_data = [
            {
                "id": "456",
                "type": "IssuesEvent",
                "actor": {"login": "otheruser"},
                "repo": {"name": "test/repo"},
                "payload": {"action": "opened"},
                "created_at": "2023-01-01T01:00:00Z",
                "public": True,
            }
        ]

        # Mock the retry method to avoid complexity
        with patch.object(client, "_make_request_with_retry") as mock_retry:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.headers = {"x-ratelimit-remaining": "4998"}
            mock_retry.return_value = mock_response

            events = await client.get_user_received_events("testuser")

            assert len(events) == 1
            assert isinstance(events[0], GitHubEvent)
            assert events[0].id == "456"
            assert events[0].type == "IssuesEvent"
            assert client.rate_limit_remaining == 4998

    @pytest.mark.asyncio
    async def test_validate_token_success(self, client):
        """Test successful token validation with rate limiting."""
        mock_user_data = {
            "login": "testuser",
            "id": 12345,
            "name": "Test User",
        }

        # Mock the retry method to avoid complexity
        with patch.object(client, "_make_request_with_retry") as mock_retry:
            mock_response = Mock()
            mock_response.json.return_value = mock_user_data
            mock_response.headers = {"x-ratelimit-remaining": "4997"}
            mock_retry.return_value = mock_response

            user_data = await client.validate_token()

            assert user_data["login"] == "testuser"
            assert user_data["id"] == 12345
            assert client.rate_limit_remaining == 4997

    @pytest.mark.asyncio
    async def test_api_method_with_rate_limit_exceeded(self, client):
        """Test API method behavior when rate limit is exceeded."""
        client.rate_limit_remaining = 0
        client.rate_limit_reset = datetime.now(tz=UTC)

        with pytest.raises(RateLimitError):
            await client.get_user_events("testuser")

    @pytest.mark.asyncio
    async def test_api_method_with_http_error(self, client):
        """Test API method behavior with HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 401
        http_error = httpx.HTTPStatusError(
            "401 Unauthorized", request=Mock(), response=mock_response
        )

        # Mock the retry method to raise the error
        with patch.object(client, "_make_request_with_retry") as mock_retry:
            mock_retry.side_effect = http_error

            with pytest.raises(httpx.HTTPStatusError):
                await client.get_user_events("testuser")


class TestGitHubEvent:
    """Test the GitHubEvent model."""

    def test_github_event_creation(self):
        """Test creating a GitHubEvent instance."""
        event_data = {
            "id": "123456789",
            "type": "PushEvent",
            "actor": {"login": "testuser", "id": 12345},
            "repo": {"name": "test/repo", "id": 67890},
            "payload": {"commits": [{"message": "Test commit"}]},
            "created_at": "2023-01-01T00:00:00Z",
            "public": True,
        }

        event = GitHubEvent(**event_data)  # type: ignore[arg-type]

        assert event.id == "123456789"
        assert event.type == "PushEvent"
        assert event.actor["login"] == "testuser"
        assert event.repo["name"] == "test/repo"
        assert event.payload["commits"][0]["message"] == "Test commit"
        assert event.created_at == "2023-01-01T00:00:00Z"
        assert event.public is True


class TestCircuitBreakerError:
    """Test the CircuitBreakerError exception."""

    def test_circuit_breaker_error_creation(self):
        """Test creating a CircuitBreakerError."""
        error = CircuitBreakerError("Circuit breaker is open")

        assert str(error) == "Circuit breaker is open"


class TestRetryLogic:
    """Test the retry logic and circuit breaker functionality."""

    @pytest.fixture
    def client(self):
        """Create a GitHub client for testing."""
        return GitHubClient(token="test_token")

    def test_should_retry_network_errors(self, client):
        """Test that network errors should be retried."""
        network_error = httpx.NetworkError("Connection failed")
        timeout_error = httpx.TimeoutException("Request timed out")

        assert client._should_retry(network_error) is True
        assert client._should_retry(timeout_error) is True

    def test_should_retry_server_errors(self, client):
        """Test that server errors should be retried."""
        mock_response = Mock()
        mock_response.status_code = 500
        server_error = httpx.HTTPStatusError(
            "Server error", request=Mock(), response=mock_response
        )

        assert client._should_retry(server_error) is True

    def test_should_retry_rate_limit_errors(self, client):
        """Test that rate limit errors should be retried."""
        mock_response = Mock()
        mock_response.status_code = 429
        rate_limit_error = httpx.HTTPStatusError(
            "Rate limit", request=Mock(), response=mock_response
        )

        assert client._should_retry(rate_limit_error) is True

    def test_should_not_retry_auth_errors(self, client):
        """Test that authentication errors should not be retried."""
        mock_response_401 = Mock()
        mock_response_401.status_code = 401
        auth_error = httpx.HTTPStatusError(
            "Unauthorized", request=Mock(), response=mock_response_401
        )

        mock_response_403 = Mock()
        mock_response_403.status_code = 403
        forbidden_error = httpx.HTTPStatusError(
            "Forbidden", request=Mock(), response=mock_response_403
        )

        assert client._should_retry(auth_error) is False
        assert client._should_retry(forbidden_error) is False

    def test_should_not_retry_client_errors(self, client):
        """Test that client errors (4xx except 429) should not be retried."""
        mock_response = Mock()
        mock_response.status_code = 404
        not_found_error = httpx.HTTPStatusError(
            "Not found", request=Mock(), response=mock_response
        )

        assert client._should_retry(not_found_error) is False

    def test_circuit_breaker_initial_state(self, client):
        """Test that circuit breaker starts in CLOSED state."""
        assert client.circuit_breaker_state == CircuitBreakerState.CLOSED
        assert client.failure_count == 0
        assert client.last_failure_time is None

    def test_record_success_resets_failure_count(self, client):
        """Test that recording success resets the failure count."""
        client.failure_count = 3
        client.last_failure_time = datetime.now(tz=UTC)

        client._record_success()

        assert client.failure_count == 0
        assert client.last_failure_time is None

    def test_record_success_closes_half_open_circuit(self, client):
        """Test that success closes a half-open circuit breaker."""
        client.circuit_breaker_state = CircuitBreakerState.HALF_OPEN

        client._record_success()

        assert client.circuit_breaker_state == CircuitBreakerState.CLOSED

    def test_record_failure_increments_count(self, client):
        """Test that recording failure increments the count."""
        initial_count = client.failure_count

        client._record_failure()

        assert client.failure_count == initial_count + 1
        assert client.last_failure_time is not None

    def test_circuit_breaker_opens_after_five_failures(self, client):
        """Test that circuit breaker opens after 5 consecutive failures."""
        for _ in range(4):
            client._record_failure()
            assert client.circuit_breaker_state == CircuitBreakerState.CLOSED

        # Fifth failure should open the circuit
        client._record_failure()
        assert client.circuit_breaker_state == CircuitBreakerState.OPEN
        assert client.failure_count == 5

    def test_circuit_breaker_fails_fast_when_open(self, client):
        """Test that circuit breaker fails fast when open."""
        client.circuit_breaker_state = CircuitBreakerState.OPEN
        client.last_failure_time = datetime.now(tz=UTC)

        with pytest.raises(CircuitBreakerError) as exc_info:
            client._check_circuit_breaker()

        assert "Circuit breaker is open" in str(exc_info.value)

    def test_circuit_breaker_transitions_to_half_open(self, client):
        """Test that circuit breaker transitions to half-open after timeout."""
        # Set circuit breaker to open with an old failure time
        from datetime import timedelta

        client.circuit_breaker_state = CircuitBreakerState.OPEN
        client.last_failure_time = datetime.now(tz=UTC) - timedelta(
            seconds=120
        )  # 2 minutes ago

        # This should transition to half-open without raising an exception
        client._check_circuit_breaker()

        assert client.circuit_breaker_state == CircuitBreakerState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_make_request_with_retry_success(self, client):
        """Test successful request without retries."""
        mock_response = Mock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        mock_http_client = AsyncMock()
        mock_http_client.request.return_value = mock_response
        client._http_client = mock_http_client

        response = await client._make_request_with_retry(
            "GET", "https://api.github.com/test", headers={"test": "header"}
        )

        assert response == mock_response
        assert client.failure_count == 0
        mock_http_client.request.assert_called_once_with(
            "GET", "https://api.github.com/test", headers={"test": "header"}
        )

    @pytest.mark.asyncio
    async def test_make_request_with_retry_network_error(self, client):
        """Test retry behavior with network errors."""
        # First 3 calls fail with network error, 4th succeeds
        network_error = httpx.NetworkError("Connection failed")
        mock_response = Mock()
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        mock_http_client = AsyncMock()
        mock_http_client.request.side_effect = [
            network_error,
            network_error,
            network_error,
            mock_response,
        ]
        client._http_client = mock_http_client

        response = await client._make_request_with_retry(
            "GET", "https://api.github.com/test"
        )

        assert response == mock_response
        assert mock_http_client.request.call_count == 4
        # Success should reset failure count
        assert client.failure_count == 0

    @pytest.mark.asyncio
    async def test_make_request_with_retry_exhausted(self, client):
        """Test that all retries are exhausted and final error is raised."""
        # All calls fail with network error
        network_error = httpx.NetworkError("Connection failed")

        mock_http_client = AsyncMock()
        mock_http_client.request.side_effect = network_error
        client._http_client = mock_http_client

        with pytest.raises(httpx.NetworkError):
            await client._make_request_with_retry("GET", "https://api.github.com/test")

        # Should have tried 4 times (initial + 3 retries)
        assert mock_http_client.request.call_count == 4
        assert client.failure_count == 4

    @pytest.mark.asyncio
    async def test_make_request_no_retry_on_auth_error(self, client):
        """Test that authentication errors are not retried."""
        # Auth error should not be retried
        mock_response = Mock()
        mock_response.status_code = 401
        auth_error = httpx.HTTPStatusError(
            "Unauthorized", request=Mock(), response=mock_response
        )

        mock_http_client = AsyncMock()
        mock_http_client.request.side_effect = auth_error
        client._http_client = mock_http_client

        with pytest.raises(httpx.HTTPStatusError):
            await client._make_request_with_retry("GET", "https://api.github.com/test")

        # Should only try once
        assert mock_http_client.request.call_count == 1
        assert client.failure_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_requests(self, client):
        """Test that open circuit breaker prevents requests."""
        client.circuit_breaker_state = CircuitBreakerState.OPEN
        client.last_failure_time = datetime.now(tz=UTC)

        with pytest.raises(CircuitBreakerError):
            await client._make_request_with_retry("GET", "https://api.github.com/test")

    @pytest.mark.asyncio
    async def test_api_methods_use_retry_logic(self, client):
        """Test that API methods use the retry logic."""
        mock_response_data = [
            {
                "id": "123",
                "type": "PushEvent",
                "actor": {"login": "testuser"},
                "repo": {"name": "test/repo"},
                "payload": {"commits": []},
                "created_at": "2023-01-01T00:00:00Z",
                "public": True,
            }
        ]

        # Mock the retry method to verify it's called
        with patch.object(client, "_make_request_with_retry") as mock_retry:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.headers = {"x-ratelimit-remaining": "4999"}
            mock_retry.return_value = mock_response

            events = await client.get_user_events("testuser")

            assert len(events) == 1
            assert events[0].id == "123"
            mock_retry.assert_called_once()

            # Verify the call arguments
            call_args = mock_retry.call_args
            assert call_args[0][0] == "GET"
            assert "users/testuser/events" in call_args[0][1]


class TestPagination:
    """Test the pagination functionality."""

    @pytest.fixture
    def client(self):
        """Create a GitHub client for testing."""
        return GitHubClient(token="test_token")

    def test_parse_link_header_empty(self, client):
        """Test parsing empty link header."""
        links = client._parse_link_header("")
        assert links == {}

    def test_parse_link_header_single_link(self, client):
        """Test parsing link header with single next link."""
        link_header = '<https://api.github.com/user/events?page=2>; rel="next"'
        links = client._parse_link_header(link_header)

        assert "next" in links
        assert links["next"] == "https://api.github.com/user/events?page=2"

    def test_parse_link_header_multiple_links(self, client):
        """Test parsing link header with multiple pagination links."""
        link_header = (
            '<https://api.github.com/user/events?page=2>; rel="next", '
            '<https://api.github.com/user/events?page=5>; rel="last", '
            '<https://api.github.com/user/events?page=1>; rel="first", '
            '<https://api.github.com/user/events?page=1>; rel="prev"'
        )
        links = client._parse_link_header(link_header)

        assert len(links) == 4
        assert links["next"] == "https://api.github.com/user/events?page=2"
        assert links["last"] == "https://api.github.com/user/events?page=5"
        assert links["first"] == "https://api.github.com/user/events?page=1"
        assert links["prev"] == "https://api.github.com/user/events?page=1"

    def test_parse_link_header_malformed(self, client):
        """Test parsing malformed link header."""
        link_header = "malformed header without proper format"
        links = client._parse_link_header(link_header)

        assert links == {}

    @pytest.mark.asyncio
    async def test_paginate_single_page(self, client):
        """Test pagination with single page response."""
        mock_response = Mock()
        mock_response.json.return_value = [{"id": "1", "type": "PushEvent"}]
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        pages_yielded = []

        with patch.object(client, "_make_request_with_retry") as mock_retry:
            mock_retry.return_value = mock_response

            async for response in client.paginate("https://api.github.com/test"):
                pages_yielded.append(response)

        assert len(pages_yielded) == 1
        assert pages_yielded[0] == mock_response
        mock_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_paginate_multiple_pages(self, client):
        """Test pagination with multiple pages."""
        # Mock responses for 3 pages
        mock_response_1 = Mock()
        mock_response_1.json.return_value = [{"id": "1", "type": "PushEvent"}]
        mock_response_1.headers = {
            "x-ratelimit-remaining": "5000",
            "link": '<https://api.github.com/test?page=2>; rel="next", <https://api.github.com/test?page=3>; rel="last"',
        }

        mock_response_2 = Mock()
        mock_response_2.json.return_value = [{"id": "2", "type": "IssuesEvent"}]
        mock_response_2.headers = {
            "x-ratelimit-remaining": "4999",
            "link": (
                '<https://api.github.com/test?page=1>; rel="first", '
                '<https://api.github.com/test?page=1>; rel="prev", '
                '<https://api.github.com/test?page=3>; rel="next", '
                '<https://api.github.com/test?page=3>; rel="last"'
            ),
        }

        mock_response_3 = Mock()
        mock_response_3.json.return_value = [{"id": "3", "type": "PullRequestEvent"}]
        mock_response_3.headers = {
            "x-ratelimit-remaining": "4998",
            "link": (
                '<https://api.github.com/test?page=1>; rel="first", '
                '<https://api.github.com/test?page=2>; rel="prev"'
            ),
        }

        pages_yielded = []

        with patch.object(client, "_make_request_with_retry") as mock_retry:
            mock_retry.side_effect = [mock_response_1, mock_response_2, mock_response_3]

            async for response in client.paginate("https://api.github.com/test"):
                pages_yielded.append(response)

        assert len(pages_yielded) == 3
        assert mock_retry.call_count == 3

        # Verify the URLs were called in order
        call_urls = [call[0][1] for call in mock_retry.call_args_list]
        assert call_urls[0] == "https://api.github.com/test"
        assert call_urls[1] == "https://api.github.com/test?page=2"
        assert call_urls[2] == "https://api.github.com/test?page=3"

    @pytest.mark.asyncio
    async def test_paginate_max_pages_limit(self, client):
        """Test pagination with max_pages limit."""
        mock_response = Mock()
        mock_response.json.return_value = [{"id": "1", "type": "PushEvent"}]
        mock_response.headers = {
            "x-ratelimit-remaining": "5000",
            "link": '<https://api.github.com/test?page=2>; rel="next"',
        }

        pages_yielded = []

        with patch.object(client, "_make_request_with_retry") as mock_retry:
            mock_retry.return_value = mock_response

            # Limit to 2 pages even though there are more
            async for response in client.paginate(
                "https://api.github.com/test", max_pages=2
            ):
                pages_yielded.append(response)

        assert len(pages_yielded) == 2
        assert mock_retry.call_count == 2

    @pytest.mark.asyncio
    async def test_paginate_progress_callback(self, client):
        """Test pagination with progress callback."""
        mock_response = Mock()
        mock_response.json.return_value = [{"id": "1", "type": "PushEvent"}]
        mock_response.headers = {
            "x-ratelimit-remaining": "5000",
            "link": '<https://api.github.com/test?page=2>; rel="next", <https://api.github.com/test?page=3>; rel="last"',
        }

        progress_calls = []

        def progress_callback(current_page: int, total_pages: int | None) -> None:
            progress_calls.append((current_page, total_pages))

        with patch.object(client, "_make_request_with_retry") as mock_retry:
            # First response has total pages info in link header
            mock_response_with_last = Mock()
            mock_response_with_last.json.return_value = [
                {"id": "1", "type": "PushEvent"}
            ]
            mock_response_with_last.headers = {
                "x-ratelimit-remaining": "5000",
                "link": '<https://api.github.com/test?page=2>; rel="next", <https://api.github.com/test?page=3>; rel="last"',
            }

            # Second response has no next link
            mock_response_final = Mock()
            mock_response_final.json.return_value = [{"id": "2", "type": "PushEvent"}]
            mock_response_final.headers = {"x-ratelimit-remaining": "4999"}

            mock_retry.side_effect = [mock_response_with_last, mock_response_final]

            pages = []
            async for response in client.paginate(
                "https://api.github.com/test", progress_callback=progress_callback
            ):
                pages.append(response)

        assert len(pages) == 2
        assert len(progress_calls) == 2
        assert progress_calls[0] == (1, 3)  # First page, detected 3 total pages
        assert progress_calls[1] == (2, 3)  # Second page, still 3 total pages

    @pytest.mark.asyncio
    async def test_get_user_events_paginated(self, client):
        """Test paginated user events retrieval."""
        mock_response_1 = Mock()
        mock_response_1.json.return_value = [
            {
                "id": "1",
                "type": "PushEvent",
                "actor": {"login": "testuser"},
                "repo": {"name": "test/repo"},
                "payload": {"commits": []},
                "created_at": "2023-01-01T00:00:00Z",
                "public": True,
            }
        ]
        mock_response_1.headers = {
            "x-ratelimit-remaining": "5000",
            "link": '<https://api.github.com/users/testuser/events?page=2>; rel="next"',
        }

        mock_response_2 = Mock()
        mock_response_2.json.return_value = [
            {
                "id": "2",
                "type": "IssuesEvent",
                "actor": {"login": "testuser"},
                "repo": {"name": "test/repo2"},
                "payload": {"action": "opened"},
                "created_at": "2023-01-01T01:00:00Z",
                "public": True,
            }
        ]
        mock_response_2.headers = {"x-ratelimit-remaining": "4999"}

        with patch.object(client, "_make_request_with_retry") as mock_retry:
            mock_retry.side_effect = [mock_response_1, mock_response_2]

            events_pages = []
            async for events in client.get_user_events_paginated(
                "testuser", max_pages=2
            ):
                events_pages.append(events)

        assert len(events_pages) == 2
        assert len(events_pages[0]) == 1
        assert len(events_pages[1]) == 1
        assert events_pages[0][0].id == "1"
        assert events_pages[1][0].id == "2"
        assert isinstance(events_pages[0][0], GitHubEvent)
        assert isinstance(events_pages[1][0], GitHubEvent)

    @pytest.mark.asyncio
    async def test_get_user_received_events_paginated(self, client):
        """Test paginated user received events retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "id": "456",
                "type": "IssuesEvent",
                "actor": {"login": "otheruser"},
                "repo": {"name": "test/repo"},
                "payload": {"action": "opened"},
                "created_at": "2023-01-01T01:00:00Z",
                "public": True,
            }
        ]
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        with patch.object(client, "_make_request_with_retry") as mock_retry:
            mock_retry.return_value = mock_response

            events_pages = []
            async for events in client.get_user_received_events_paginated(
                "testuser", max_pages=1
            ):
                events_pages.append(events)

        assert len(events_pages) == 1
        assert len(events_pages[0]) == 1
        assert events_pages[0][0].id == "456"
        assert isinstance(events_pages[0][0], GitHubEvent)
