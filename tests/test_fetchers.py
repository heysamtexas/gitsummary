"""Fetcher tests that focus on contracts, not HTTP implementation details.

Following Guilfoyle's philosophy: Mock at boundaries, test contracts.
"""
from unittest.mock import Mock, patch

import httpx
import pytest

from git_summary.fetchers import (
    EventCoordinator,
    PublicEventsFetcher,
    UserEventsFetcher,
)
from git_summary.github_client import GitHubClient
from git_summary.models import BaseGitHubEvent


class TestUserEventsFetcher:
    """Test UserEventsFetcher behavior without over-mocking."""

    def setup_method(self):
        self.client = GitHubClient(token="fake_token")
        self.fetcher = UserEventsFetcher(self.client, "testuser")

    @patch.object(GitHubClient, "paginate")
    @pytest.mark.asyncio
    async def test_fetch_events_returns_list(self, mock_paginate):
        """Test that fetch_events returns a list of GitHub events."""
        # Mock just the HTTP layer response
        mock_event_data = {
            "id": "123",
            "type": "PushEvent",
            "created_at": "2024-01-15T10:00:00Z",
            "actor": {
                "id": 1,
                "login": "testuser",
                "url": "test",
                "avatar_url": "test",
            },
            "repo": {"id": 1, "name": "test/repo", "url": "test"},
            "payload": {"commits": [{"sha": "abc"}]},
            "public": True,
        }

        # Mock the response object
        mock_response = Mock()
        mock_response.json.return_value = [mock_event_data]

        # Mock the async generator
        async def mock_async_gen():
            yield mock_response

        mock_paginate.return_value = mock_async_gen()

        events = []
        async for event in self.fetcher.fetch_events():
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], BaseGitHubEvent)
        assert events[0].type == "PushEvent"

    @patch.object(GitHubClient, "paginate")
    @pytest.mark.asyncio
    async def test_fetch_events_handles_empty_response(self, mock_paginate):
        """Test handling of empty API responses."""
        mock_response = Mock()
        mock_response.json.return_value = []

        async def mock_async_gen():
            yield mock_response

        mock_paginate.return_value = mock_async_gen()

        events = []
        async for event in self.fetcher.fetch_events():
            events.append(event)

        assert events == []

    @patch.object(GitHubClient, "paginate")
    @pytest.mark.asyncio
    async def test_fetch_events_handles_http_error(self, mock_paginate):
        """Test error handling for HTTP failures."""
        mock_paginate.side_effect = httpx.HTTPError("API Error")

        with pytest.raises(Exception):  # Should propagate the error appropriately
            async for event in self.fetcher.fetch_events():
                pass


class TestPublicEventsFetcher:
    """Test PublicEventsFetcher with minimal mocking."""

    def setup_method(self):
        self.client = GitHubClient(token="fake_token")
        self.fetcher = PublicEventsFetcher(self.client, "testuser")

    @patch.object(GitHubClient, "paginate")
    @pytest.mark.asyncio
    async def test_fetch_events_returns_list(self, mock_paginate):
        """Test basic functionality of public events fetching."""
        mock_event_data = {
            "id": "456",
            "type": "IssuesEvent",
            "created_at": "2024-01-15T10:00:00Z",
            "actor": {
                "id": 1,
                "login": "testuser",
                "url": "test",
                "avatar_url": "test",
            },
            "repo": {"id": 1, "name": "test/repo", "url": "test"},
            "payload": {"action": "opened"},
            "public": True,
        }

        mock_response = Mock()
        mock_response.json.return_value = [mock_event_data]

        async def mock_async_gen():
            yield mock_response

        mock_paginate.return_value = mock_async_gen()

        events = []
        async for event in self.fetcher.fetch_events():
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], BaseGitHubEvent)
        assert events[0].type == "IssuesEvent"

    @patch.object(GitHubClient, "paginate")
    @pytest.mark.asyncio
    async def test_fetch_events_handles_404_user_not_found(self, mock_paginate):
        """Test handling of 404 for non-existent users."""
        mock_paginate.side_effect = httpx.HTTPStatusError(
            "Not Found", request=Mock(), response=Mock(status_code=404)
        )

        # Should raise FetcherError for non-existent users
        with pytest.raises(Exception):  # FetcherError or similar
            async for event in self.fetcher.fetch_events():
                pass


class TestEventCoordinator:
    """Test the EventCoordinator contract."""

    def setup_method(self):
        self.client = GitHubClient(token="fake_token")
        self.coordinator = EventCoordinator(self.client)

    @pytest.mark.asyncio
    async def test_fetch_events_by_date_range_basic_functionality(self):
        """Test basic coordinator functionality."""
        from datetime import UTC, datetime

        # Just test that coordinator doesn't crash with valid inputs
        start_date = datetime(2024, 1, 1, tzinfo=UTC)
        end_date = datetime(2024, 1, 2, tzinfo=UTC)

        # Mock both fetchers to avoid real API calls
        with patch.object(UserEventsFetcher, "fetch_events") as mock_user:
            with patch.object(PublicEventsFetcher, "fetch_events") as mock_public:
                # Return async generators
                async def empty_gen():
                    if False:
                        yield

                mock_user.return_value = empty_gen()
                mock_public.return_value = empty_gen()

                events = await self.coordinator.fetch_events_by_date_range(
                    username="testuser", start_date=start_date, end_date=end_date
                )

                assert isinstance(events, list)


class TestGitHubClient:
    """Test the GitHub client contract."""

    def test_client_accepts_valid_token(self):
        """Test that client accepts valid tokens."""
        client = GitHubClient("valid_token")
        assert client.token == "valid_token"
        assert client.headers.get("Authorization") == "Bearer valid_token"
        assert client.headers.get("Accept") == "application/vnd.github.v3+json"

    def test_client_has_required_headers(self):
        """Test that client sets required headers."""
        client = GitHubClient("test_token")

        # Verify essential headers are present
        assert "Authorization" in client.headers
        assert "Accept" in client.headers
        assert "X-GitHub-Api-Version" in client.headers
