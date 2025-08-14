"""Focused GitHub API response fixtures for testing.

Following Guilfoyle's guidance: test behavior, not perfect API simulation.
Each fixture focuses on specific data patterns that affect our analysis logic.
"""

from pathlib import Path
from typing import Any

import json

FIXTURES_DIR = Path(__file__).parent / "github_responses"


def load_fixture(fixture_name: str) -> dict[str, Any]:
    """Load a GitHub API response fixture."""
    fixture_path = FIXTURES_DIR / f"{fixture_name}.json"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
    
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Fixture catalog - maps test scenarios to fixture files
FIXTURES = {
    # User event patterns for automation detection
    "user_events": {
        "automation_bot": "automation_heavy_user_events.json",
        "normal_developer": "normal_user_mixed_activity.json", 
        "inactive_user": "empty_user_events.json",
        "high_frequency_user": "high_frequency_push_events.json",
        "single_repo_dominant": "single_repo_dominance.json",
    },
    
    # Repository search results for discovery testing
    "repository_search": {
        "prolific_contributor": "many_repos_varied_activity.json",
        "single_project_user": "one_main_repository.json",
        "no_repositories": "empty_repository_search.json",
        "private_repos_only": "private_repositories_result.json",
    },
    
    # Owned repositories for multi-source discovery
    "owned_repos": {
        "active_maintainer": "owned_repos_recently_updated.json",
        "inactive_maintainer": "owned_repos_stale.json",
        "no_owned_repos": "empty_owned_repos.json",
    },
    
    # Commit search results
    "commit_search": {
        "frequent_committer": "commit_search_many_results.json",
        "occasional_contributor": "commit_search_few_results.json",
        "no_commits": "empty_commit_search.json",
    },
    
    # Error responses
    "errors": {
        "rate_limited": "rate_limit_response.json",
        "user_not_found": "user_404_response.json",
        "repo_access_denied": "repo_403_response.json",
        "malformed_search": "incomplete_search_results.json",
    }
}


def create_automation_user_events() -> list[dict[str, Any]]:
    """Create synthetic events that should trigger automation detection."""
    return [
        {
            "id": f"event_{i}",
            "type": "PushEvent",
            "created_at": f"2024-01-{i:02d}T10:00:00Z",
            "repo": {"name": "automation-repo", "full_name": "bot/automation-repo"},
            "actor": {"login": "automation-bot"},
            "payload": {"commits": [{"message": f"Automated commit {i}"}]}
        }
        for i in range(1, 101)  # 100 push events - clearly automation
    ]


def create_normal_user_events() -> list[dict[str, Any]]:
    """Create synthetic events for a normal developer."""
    event_types = ["PushEvent", "PullRequestEvent", "IssuesEvent", "CreateEvent"]
    repos = ["user/project1", "org/shared-project", "user/side-project"]
    
    events = []
    for i in range(1, 21):  # 20 mixed events
        events.append({
            "id": f"event_{i}",
            "type": event_types[i % len(event_types)],
            "created_at": f"2024-01-{i:02d}T{10 + i % 14}:00:00Z",
            "repo": {"name": repos[i % len(repos)].split("/")[1], "full_name": repos[i % len(repos)]},
            "actor": {"login": "normal-developer"},
        })
    
    return events


def create_repository_discovery_results() -> dict[str, list[dict[str, Any]]]:
    """Create synthetic repository discovery results for testing."""
    return {
        "owned_repos": [
            {
                "full_name": "user/main-project",
                "updated_at": "2024-01-15T10:00:00Z",
                "private": False,
                "language": "Python",
            },
            {
                "full_name": "user/side-project", 
                "updated_at": "2024-01-10T10:00:00Z",
                "private": True,
                "language": "JavaScript",
            }
        ],
        "event_repos": {
            "user/main-project": {"score": 15, "source": "events"},
            "org/shared-project": {"score": 8, "source": "events"},
            "user/side-project": {"score": 5, "source": "events"},
        },
        "commit_repos": {
            "user/main-project": {"score": 3, "source": "commits", "commit_count": 12},
            "org/shared-project": {"score": 2, "source": "commits", "commit_count": 8},
            "external/contribution": {"score": 1, "source": "commits", "commit_count": 3},
        }
    }


def create_empty_response() -> dict[str, Any]:
    """Create an empty API response for testing edge cases."""
    return {"items": [], "total_count": 0, "incomplete_results": False}


def create_rate_limit_response() -> dict[str, Any]:
    """Create a rate limit error response."""
    return {
        "message": "API rate limit exceeded",
        "documentation_url": "https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting"
    }


def create_malformed_search_response() -> dict[str, Any]:
    """Create a malformed search response to test error handling."""
    return {
        "incomplete_results": True,
        "items": [{"repository": None}],  # Missing expected fields
        "total_count": 1
    }