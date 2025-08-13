"""Tests for GitHub event fetchers."""

import os
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import httpx
import pytest

from git_summary.fetchers import AuthenticationError, FetcherError, UserEventsFetcher
from git_summary.github_client import GitHubClient
from git_summary.models import BaseGitHubEvent


class TestUserEventsFetcher:
    """Test the UserEventsFetcher class."""

    @pytest.fixture
    def client(self):
        """Create a GitHub client for testing."""
        return GitHubClient(token="test_token")

    @pytest.fixture
    def fetcher(self, client):
        """Create a UserEventsFetcher for testing."""
        return UserEventsFetcher(client)

    @pytest.fixture
    def mock_event_data(self):
        """Mock GitHub event data."""
        return {
            "id": "123456789",
            "type": "PushEvent",
            "actor": {
                "id": 12345,
                "login": "testuser",
                "url": "https://api.github.com/users/testuser",
                "avatar_url": "https://avatars.githubusercontent.com/u/12345",
            },
            "repo": {
                "id": 67890,
                "name": "test/repo",
                "url": "https://api.github.com/repos/test/repo",
            },
            "payload": {
                "push_id": 12345,
                "size": 1,
                "distinct_size": 1,
                "ref": "refs/heads/main",
                "head": "abc123",
                "before": "def456",
                "commits": [
                    {
                        "sha": "abc123",
                        "author": {"name": "Test Author", "email": "test@example.com"},
                        "message": "Test commit",
                        "distinct": True,
                        "url": "https://api.github.com/repos/test/repo/commits/abc123",
                    }
                ],
            },
            "created_at": "2023-01-01T12:00:00Z",
            "public": True,
        }

    def test_fetcher_initialization(self, client):
        """Test fetcher initialization."""
        fetcher = UserEventsFetcher(client)
        assert fetcher.client == client

    @pytest.mark.asyncio
    async def test_fetch_events_success(self, fetcher, mock_event_data):
        """Test successful event fetching."""
        # Mock pagination response
        mock_response = Mock()
        mock_response.json.return_value = [mock_event_data]
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        # Mock the paginate method to yield our response
        async def mock_paginate(*args, **kwargs):
            yield mock_response

        with patch.object(fetcher.client, "paginate", mock_paginate):
            events = []
            async for event in fetcher.fetch_events():
                events.append(event)

            assert len(events) == 1
            assert isinstance(events[0], BaseGitHubEvent)
            assert events[0].id == "123456789"
            assert events[0].type == "PushEvent"

    @pytest.mark.asyncio
    async def test_fetch_events_with_since_parameter(self, fetcher, mock_event_data):
        """Test fetching events with since parameter."""
        since_date = datetime(2023, 1, 1, tzinfo=UTC)

        mock_response = Mock()
        mock_response.json.return_value = [mock_event_data]
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        paginate_args = None

        async def mock_paginate(*args, **kwargs):
            nonlocal paginate_args
            paginate_args = (args, kwargs)
            yield mock_response

        with patch.object(fetcher.client, "paginate", mock_paginate):
            events = []
            async for event in fetcher.fetch_events(since=since_date):
                events.append(event)

            # Verify the since parameter was passed correctly
            assert paginate_args is not None
            params = paginate_args[1].get("params", {})
            assert "since" in params
            assert params["since"] == since_date.isoformat()

    @pytest.mark.asyncio
    async def test_fetch_events_with_max_pages(self, fetcher, mock_event_data):
        """Test fetching events with max_pages limit."""
        mock_response = Mock()
        mock_response.json.return_value = [mock_event_data]
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        paginate_args = None

        async def mock_paginate(*args, **kwargs):
            nonlocal paginate_args
            paginate_args = (args, kwargs)
            yield mock_response

        with patch.object(fetcher.client, "paginate", mock_paginate):
            events = []
            async for event in fetcher.fetch_events(max_pages=5):
                events.append(event)

            # Verify max_pages was passed
            assert paginate_args is not None
            assert paginate_args[1].get("max_pages") == 5

    @pytest.mark.asyncio
    async def test_fetch_events_with_progress_callback(self, fetcher, mock_event_data):
        """Test fetching events with progress callback."""
        mock_response = Mock()
        mock_response.json.return_value = [mock_event_data]
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        paginate_args = None

        async def mock_paginate(*args, **kwargs):
            nonlocal paginate_args
            paginate_args = (args, kwargs)
            yield mock_response

        with patch.object(fetcher.client, "paginate", mock_paginate):
            events = []
            async for event in fetcher.fetch_events(
                progress_callback=progress_callback
            ):
                events.append(event)

            # Verify progress_callback was passed
            assert paginate_args is not None
            assert paginate_args[1].get("progress_callback") == progress_callback

    @pytest.mark.asyncio
    async def test_fetch_events_authentication_error_401(self, fetcher):
        """Test handling of 401 authentication error."""
        mock_response = Mock()
        mock_response.status_code = 401
        auth_error = httpx.HTTPStatusError(
            "Unauthorized", request=Mock(), response=mock_response
        )

        async def mock_paginate(*args, **kwargs):
            if False:  # Never executed, just for type checking
                yield  # type: ignore[unreachable]
            raise auth_error

        with patch.object(fetcher.client, "paginate", mock_paginate):
            with pytest.raises(AuthenticationError) as exc_info:
                events = []
                async for event in fetcher.fetch_events():
                    events.append(event)

            assert "GitHub authentication failed" in str(exc_info.value)
            assert "HTTP 401" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_events_authentication_error_403(self, fetcher):
        """Test handling of 403 authentication error."""
        mock_response = Mock()
        mock_response.status_code = 403
        forbidden_error = httpx.HTTPStatusError(
            "Forbidden", request=Mock(), response=mock_response
        )

        async def mock_paginate(*args, **kwargs):
            if False:  # Never executed, just for type checking
                yield  # type: ignore[unreachable]
            raise forbidden_error

        with patch.object(fetcher.client, "paginate", mock_paginate):
            with pytest.raises(AuthenticationError) as exc_info:
                events = []
                async for event in fetcher.fetch_events():
                    events.append(event)

            assert "GitHub authentication failed" in str(exc_info.value)
            assert "HTTP 403" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_events_other_error(self, fetcher):
        """Test handling of non-authentication errors."""
        network_error = httpx.NetworkError("Connection failed")

        async def mock_paginate(*args, **kwargs):
            if False:  # Never executed, just for type checking
                yield  # type: ignore[unreachable]
            raise network_error

        with patch.object(fetcher.client, "paginate", mock_paginate):
            with pytest.raises(FetcherError) as exc_info:
                events = []
                async for event in fetcher.fetch_events():
                    events.append(event)

            assert "Failed to fetch user events" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_events_invalid_event_data(self, fetcher):
        """Test handling of invalid event data (should be skipped with warning)."""
        # Mix of valid and invalid event data
        valid_event = {
            "id": "123",
            "type": "PushEvent",
            "actor": {"login": "user"},
            "repo": {"name": "test"},
            "created_at": "2023-01-01T12:00:00Z",
            "public": True,
            "payload": {
                "commits": [],
                "push_id": 1,
                "size": 0,
                "distinct_size": 0,
                "ref": "main",
                "head": "abc",
                "before": "def",
            },
        }
        invalid_event = {"id": "456"}  # Missing required fields

        mock_response = Mock()
        mock_response.json.return_value = [valid_event, invalid_event]
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        async def mock_paginate(*args, **kwargs):
            yield mock_response

        with patch.object(fetcher.client, "paginate", mock_paginate):
            events = []
            async for event in fetcher.fetch_events():
                events.append(event)

            # Should only get the valid event
            assert len(events) == 1
            assert events[0].id == "123"

    @pytest.mark.asyncio
    async def test_fetch_events_list(self, fetcher, mock_event_data):
        """Test fetch_events_list method."""
        mock_response = Mock()
        mock_response.json.return_value = [mock_event_data, mock_event_data]
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        async def mock_paginate(*args, **kwargs):
            yield mock_response

        with patch.object(fetcher.client, "paginate", mock_paginate):
            events = await fetcher.fetch_events_list()

            assert isinstance(events, list)
            assert len(events) == 2
            assert all(isinstance(event, BaseGitHubEvent) for event in events)

    @pytest.mark.asyncio
    async def test_fetch_events_by_date_range(self, fetcher):
        """Test fetch_events_by_date_range method."""
        start_date = datetime(2023, 1, 1, tzinfo=UTC)
        end_date = datetime(2023, 1, 31, tzinfo=UTC)

        # Create events with different timestamps
        event_in_range = {
            "id": "123",
            "type": "PushEvent",
            "actor": {"login": "user"},
            "repo": {"name": "test"},
            "created_at": "2023-01-15T12:00:00Z",  # Within range
            "public": True,
            "payload": {
                "commits": [],
                "push_id": 1,
                "size": 0,
                "distinct_size": 0,
                "ref": "main",
                "head": "abc",
                "before": "def",
            },
        }
        event_out_of_range = {
            "id": "456",
            "type": "PushEvent",
            "actor": {"login": "user"},
            "repo": {"name": "test"},
            "created_at": "2023-02-15T12:00:00Z",  # Outside range
            "public": True,
            "payload": {
                "commits": [],
                "push_id": 2,
                "size": 0,
                "distinct_size": 0,
                "ref": "main",
                "head": "def",
                "before": "ghi",
            },
        }

        mock_response = Mock()
        mock_response.json.return_value = [event_in_range, event_out_of_range]
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        paginate_args = None

        async def mock_paginate(*args, **kwargs):
            nonlocal paginate_args
            paginate_args = (args, kwargs)
            yield mock_response

        with patch.object(fetcher.client, "paginate", mock_paginate):
            events = await fetcher.fetch_events_by_date_range(start_date, end_date)

            # Should only get events within the date range
            assert len(events) == 1
            assert events[0].id == "123"

            # Verify start_date was passed as since parameter
            assert paginate_args is not None
            params = paginate_args[1].get("params", {})
            assert "since" in params
            assert params["since"] == start_date.isoformat()

    @pytest.mark.asyncio
    async def test_empty_response(self, fetcher):
        """Test handling of empty API response."""
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        async def mock_paginate(*args, **kwargs):
            yield mock_response

        with patch.object(fetcher.client, "paginate", mock_paginate):
            events = []
            async for event in fetcher.fetch_events():
                events.append(event)

            assert len(events) == 0

    @pytest.mark.asyncio
    async def test_multiple_pages(self, fetcher, mock_event_data):
        """Test handling of multiple pages of results."""
        # Create two different events for two pages
        event1 = mock_event_data.copy()
        event1["id"] = "111"

        event2 = mock_event_data.copy()
        event2["id"] = "222"

        mock_response1 = Mock()
        mock_response1.json.return_value = [event1]
        mock_response1.headers = {"x-ratelimit-remaining": "5000"}

        mock_response2 = Mock()
        mock_response2.json.return_value = [event2]
        mock_response2.headers = {"x-ratelimit-remaining": "4999"}

        async def mock_paginate(*args, **kwargs):
            yield mock_response1
            yield mock_response2

        with patch.object(fetcher.client, "paginate", mock_paginate):
            events = []
            async for event in fetcher.fetch_events():
                events.append(event)

            assert len(events) == 2
            assert events[0].id == "111"
            assert events[1].id == "222"

    @pytest.mark.asyncio
    async def test_fetch_events_with_max_events(self, fetcher, mock_event_data):
        """Test fetching events with max_events limit."""
        # Create multiple events
        events_data = []
        for i in range(5):
            event = mock_event_data.copy()
            event["id"] = f"event_{i}"
            events_data.append(event)

        mock_response = Mock()
        mock_response.json.return_value = events_data
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        async def mock_paginate(*args, **kwargs):
            yield mock_response

        with patch.object(fetcher.client, "paginate", mock_paginate):
            events = []
            async for event in fetcher.fetch_events(max_events=3):
                events.append(event)

            # Should only get 3 events due to max_events limit
            assert len(events) == 3
            assert events[0].id == "event_0"
            assert events[1].id == "event_1"
            assert events[2].id == "event_2"

    @pytest.mark.asyncio
    async def test_network_failure_during_pagination(self, fetcher):
        """Test handling of network failures during pagination."""
        network_error = httpx.NetworkError("Connection lost during pagination")

        # First response succeeds, second fails
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        async def mock_paginate(*args, **kwargs):
            yield mock_response
            raise network_error

        with patch.object(fetcher.client, "paginate", mock_paginate):
            with pytest.raises(FetcherError) as exc_info:
                events = []
                async for event in fetcher.fetch_events():
                    events.append(event)

            assert "Failed to fetch user events from GitHub API" in str(exc_info.value)
            assert "Connection lost during pagination" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_consecutive_api_failures(self, fetcher):
        """Test handling of multiple consecutive API failures."""
        server_error = httpx.HTTPStatusError(
            "Server error", request=Mock(), response=Mock()
        )
        server_error.response.status_code = 500

        async def mock_paginate(*args, **kwargs):
            if False:  # Never executed, just for type checking
                yield  # type: ignore[unreachable]
            raise server_error

        with patch.object(fetcher.client, "paginate", mock_paginate):
            with pytest.raises(FetcherError) as exc_info:
                events = []
                async for event in fetcher.fetch_events():
                    events.append(event)

            assert "Failed to fetch user events from GitHub API" in str(exc_info.value)
            assert "Parameters: since=None, max_pages=None" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_malformed_response_data(self, fetcher):
        """Test handling of malformed JSON response data."""
        # Response with invalid JSON structure
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.headers = {"x-ratelimit-remaining": "5000"}

        async def mock_paginate(*args, **kwargs):
            yield mock_response

        with patch.object(fetcher.client, "paginate", mock_paginate):
            with pytest.raises(FetcherError) as exc_info:
                events = []
                async for event in fetcher.fetch_events():
                    events.append(event)

            assert "Failed to fetch user events from GitHub API" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rate_limit_during_event_processing(self, fetcher):
        """Test handling when rate limit is hit during event processing."""
        from datetime import datetime

        from git_summary.github_client import RateLimitError

        rate_limit_error = RateLimitError(
            "Rate limit exceeded", remaining=0, reset_time=datetime.now(UTC)
        )

        async def mock_paginate(*args, **kwargs):
            if False:  # Never executed, just for type checking
                yield  # type: ignore[unreachable]
            raise rate_limit_error

        with patch.object(fetcher.client, "paginate", mock_paginate):
            with pytest.raises(FetcherError) as exc_info:
                events = []
                async for event in fetcher.fetch_events():
                    events.append(event)

            assert "Failed to fetch user events from GitHub API" in str(exc_info.value)


@pytest.mark.integration
class TestUserEventsFetcherIntegration:
    """Integration tests for UserEventsFetcher with real GitHub API.

    These tests require a valid GitHub token in the GITHUB_TOKEN environment variable.
    They will be skipped if no token is provided.
    """

    @pytest.fixture
    def github_token(self):
        """Get GitHub token from environment or skip test."""
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            pytest.skip("GITHUB_TOKEN environment variable not set")
        return token

    @pytest.fixture
    def integration_client(self, github_token):
        """Create a real GitHub client for integration tests."""
        return GitHubClient(token=github_token)

    @pytest.fixture
    def integration_fetcher(self, integration_client):
        """Create a real UserEventsFetcher for integration tests."""
        return UserEventsFetcher(integration_client)

    @pytest.mark.asyncio
    async def test_fetch_user_events_real_api(self, integration_fetcher):
        """Test fetching user events from real GitHub API."""
        # Fetch a small number of events to avoid hitting rate limits
        events = await integration_fetcher.fetch_events_list(max_pages=1, max_events=10)

        # Verify we got real events
        assert isinstance(events, list)

        if events:  # Only test if we got events (account might have no recent activity)
            for event in events:
                assert isinstance(event, BaseGitHubEvent)
                assert event.id
                assert event.type
                assert event.actor
                assert event.repo
                assert event.created_at

    @pytest.mark.asyncio
    async def test_fetch_events_with_date_filter_real_api(self, integration_fetcher):
        """Test fetching events with date filtering from real API."""
        # Test fetching events from last 7 days
        since_date = datetime.now(UTC) - timedelta(days=7)

        events = await integration_fetcher.fetch_events_list(
            since=since_date,
            max_pages=1,  # Limit to avoid rate limits
        )

        # Verify all events are after the since date
        for event in events:
            event_time = datetime.fromisoformat(event.created_at.replace("Z", "+00:00"))
            assert event_time >= since_date

    @pytest.mark.asyncio
    async def test_fetch_events_with_progress_callback_real_api(
        self, integration_fetcher
    ):
        """Test fetching events with progress callback from real API."""
        progress_updates = []

        def progress_callback(current_page: int, total_pages: int | None) -> None:
            progress_updates.append((current_page, total_pages))

        events = await integration_fetcher.fetch_events_list(
            max_pages=2, progress_callback=progress_callback
        )

        # Should have received progress updates
        if events:  # Only check if we got events
            assert len(progress_updates) > 0
            # First update should be for page 1
            assert progress_updates[0][0] == 1

    @pytest.mark.asyncio
    async def test_fetch_events_rate_limit_info(self, integration_fetcher):
        """Test that rate limit info is properly updated during fetching."""
        # Fetch a few events to trigger rate limit header updates
        await integration_fetcher.fetch_events_list(max_pages=1)

        # Check that rate limit info was updated
        rate_status = integration_fetcher.client.get_rate_limit_status()

        # Should have rate limit information after making API calls
        assert rate_status["remaining"] is not None
        assert rate_status["limit"] is not None

        if rate_status["remaining"] is not None:
            assert rate_status["remaining"] >= 0

    @pytest.mark.asyncio
    async def test_invalid_token_authentication_error(self):
        """Test that invalid token raises AuthenticationError."""
        invalid_client = GitHubClient(token="invalid_token")
        invalid_fetcher = UserEventsFetcher(invalid_client)

        with pytest.raises(AuthenticationError) as exc_info:
            events = []
            async for event in invalid_fetcher.fetch_events(max_pages=1):
                events.append(event)
                break  # Stop after first event to fail fast

        assert "GitHub authentication failed" in str(exc_info.value)
