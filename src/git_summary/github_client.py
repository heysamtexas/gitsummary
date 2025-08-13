"""GitHub API client for fetching user activity data."""

from typing import Any

import httpx
from pydantic import BaseModel


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
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/users/{username}/events",
                headers=self.headers,
                params={"per_page": per_page, "page": page},
            )
            response.raise_for_status()
            data = response.json()

            return [GitHubEvent(**event) for event in data]

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
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/users/{username}/received_events",
                headers=self.headers,
                params={"per_page": per_page, "page": page},
            )
            response.raise_for_status()
            data = response.json()

            return [GitHubEvent(**event) for event in data]

    async def validate_token(self) -> dict[str, Any]:
        """Validate the GitHub token and return user information.

        Returns:
            User information from GitHub API

        Raises:
            httpx.HTTPStatusError: If the token is invalid
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/user",
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
