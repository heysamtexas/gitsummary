"""Rich context gathering engine for GitHub activity analysis.

This module extracts detailed information from GitHub events and manages
token budgets for LLM calls, ensuring we get maximum value within cost constraints.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, TypeVar

from git_summary.github_client import GitHubClient
from git_summary.models import BaseGitHubEvent


class GitHubEventLike(Protocol):
    """Protocol for GitHub event-like objects."""

    id: str
    type: str
    created_at: str
    actor: Any
    repo: Any
    public: bool
    payload: dict[str, Any] | None = None


logger = logging.getLogger(__name__)


def parse_github_datetime(date_str: str) -> datetime:
    """Parse GitHub API datetime string to datetime object."""
    # GitHub API returns datetime in ISO format with Z suffix
    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))


T = TypeVar("T", bound=BaseGitHubEvent)


@dataclass
class TokenBudget:
    """Token budget management for LLM calls."""

    max_tokens: int
    used_tokens: int = 0
    reserved_tokens: int = 0

    def can_allocate(self, tokens_needed: int) -> bool:
        """Check if tokens can be allocated within budget."""
        return (
            self.used_tokens + self.reserved_tokens + tokens_needed
        ) <= self.max_tokens

    def reserve(self, tokens: int) -> bool:
        """Reserve tokens for future use."""
        if self.can_allocate(tokens):
            self.reserved_tokens += tokens
            return True
        return False

    def consume_reserved(self, tokens: int) -> None:
        """Convert reserved tokens to used tokens."""
        if tokens <= self.reserved_tokens:
            self.reserved_tokens -= tokens
            self.used_tokens += tokens
        else:
            raise ValueError(
                f"Cannot consume {tokens} tokens, only {self.reserved_tokens} reserved"
            )

    def consume(self, tokens: int) -> bool:
        """Directly consume tokens if available."""
        if self.can_allocate(tokens):
            self.used_tokens += tokens
            return True
        return False

    @property
    def remaining(self) -> int:
        """Calculate remaining token budget."""
        return self.max_tokens - self.used_tokens - self.reserved_tokens

    @property
    def utilization(self) -> float:
        """Calculate budget utilization percentage."""
        if self.max_tokens == 0:
            return 0.0
        return (self.used_tokens + self.reserved_tokens) / self.max_tokens


@dataclass
class RichContext:
    """Rich context data extracted from GitHub events."""

    # Basic metrics
    total_events: int = 0
    date_range: tuple[datetime, datetime] | None = None
    repositories: list[str] = field(default_factory=list)

    # Detailed event breakdowns
    commits: list[dict[str, Any]] = field(default_factory=list)
    pull_requests: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    releases: list[dict[str, Any]] = field(default_factory=list)
    reviews: list[dict[str, Any]] = field(default_factory=list)

    # Rich content
    commit_messages: list[str] = field(default_factory=list)
    issue_titles: list[str] = field(default_factory=list)
    pr_titles: list[str] = field(default_factory=list)
    release_notes: list[str] = field(default_factory=list)

    # Token usage tracking
    estimated_tokens: int = 0
    content_priority: dict[str, int] = field(default_factory=dict)

    def add_commit(self, event: GitHubEventLike, details: dict[str, Any]) -> None:
        """Add commit information to context."""
        self.commits.append(
            {
                "sha": details.get("sha", "unknown"),
                "message": details.get("message", ""),
                "repository": event.repo.name if event.repo else "unknown",
                "timestamp": parse_github_datetime(event.created_at),
                "author": details.get("author", {}).get("name", "unknown"),
                "additions": details.get("stats", {}).get("additions", 0),
                "deletions": details.get("stats", {}).get("deletions", 0),
                "files_changed": len(details.get("files", [])),
            }
        )

        if message := details.get("message", ""):
            self.commit_messages.append(message)

    def add_pull_request(self, event: GitHubEventLike, details: dict[str, Any]) -> None:
        """Add pull request information to context."""
        self.pull_requests.append(
            {
                "number": details.get("number", 0),
                "title": details.get("title", ""),
                "state": details.get("state", "unknown"),
                "repository": event.repo.name if event.repo else "unknown",
                "timestamp": parse_github_datetime(event.created_at),
                "author": details.get("user", {}).get("login", "unknown"),
                "merged": details.get("merged", False),
                "additions": details.get("additions", 0),
                "deletions": details.get("deletions", 0),
                "changed_files": details.get("changed_files", 0),
                "comments": details.get("comments", 0),
                "commits": details.get("commits", 0),
            }
        )

        if title := details.get("title", ""):
            self.pr_titles.append(title)

    def add_issue(self, event: GitHubEventLike, details: dict[str, Any]) -> None:
        """Add issue information to context."""
        self.issues.append(
            {
                "number": details.get("number", 0),
                "title": details.get("title", ""),
                "state": details.get("state", "unknown"),
                "repository": event.repo.name if event.repo else "unknown",
                "timestamp": parse_github_datetime(event.created_at),
                "author": details.get("user", {}).get("login", "unknown"),
                "labels": [
                    label.get("name", "") for label in details.get("labels", [])
                ],
                "comments": details.get("comments", 0),
                "assignees": [
                    user.get("login", "") for user in details.get("assignees", [])
                ],
            }
        )

        if title := details.get("title", ""):
            self.issue_titles.append(title)

    def add_release(self, event: GitHubEventLike, details: dict[str, Any]) -> None:
        """Add release information to context."""
        self.releases.append(
            {
                "tag_name": details.get("tag_name", ""),
                "name": details.get("name", ""),
                "repository": event.repo.name if event.repo else "unknown",
                "timestamp": parse_github_datetime(event.created_at),
                "author": details.get("author", {}).get("login", "unknown"),
                "prerelease": details.get("prerelease", False),
                "draft": details.get("draft", False),
                "assets_count": len(details.get("assets", [])),
            }
        )

        if body := details.get("body", ""):
            self.release_notes.append(body)

    def estimate_token_usage(self) -> int:
        """Estimate total token usage for this context."""
        # Rough estimation: 4 characters per token
        total_chars = 0

        # Count character content
        total_chars += sum(len(msg) for msg in self.commit_messages)
        total_chars += sum(len(title) for title in self.issue_titles)
        total_chars += sum(len(title) for title in self.pr_titles)
        total_chars += sum(len(notes) for notes in self.release_notes)

        # Add structured data (JSON-like representation) only if there's content
        if self.commits:
            total_chars += len(str(self.commits))
        if self.pull_requests:
            total_chars += len(str(self.pull_requests))
        if self.issues:
            total_chars += len(str(self.issues))
        if self.releases:
            total_chars += len(str(self.releases))

        self.estimated_tokens = total_chars // 4
        return self.estimated_tokens

    def prioritize_content(self, budget: TokenBudget) -> "RichContext":
        """Create a prioritized version of context that fits within token budget."""
        # Calculate current estimated tokens
        current_tokens = self.estimate_token_usage()

        if current_tokens <= budget.remaining:
            # Content fits within budget
            return self

        # Create prioritized copy
        prioritized = RichContext(
            total_events=self.total_events,
            date_range=self.date_range,
            repositories=self.repositories.copy(),
        )

        # Priority order: releases > PRs > commits > issues
        available_tokens = budget.remaining

        # 1. Always include releases (highest value) but account for their cost
        prioritized.releases = self.releases.copy()
        prioritized.release_notes = self.release_notes.copy()

        # Account for release tokens
        release_tokens = 0
        if self.releases:
            release_tokens += sum(len(str(release)) for release in self.releases) // 4
        if self.release_notes:
            release_tokens += sum(len(note) for note in self.release_notes) // 4

        available_tokens = max(0, available_tokens - release_tokens)

        # 2. Include PRs up to remaining budget
        if available_tokens > 20:  # Only if we have reasonable tokens left
            pr_tokens = sum(len(str(pr)) for pr in self.pull_requests) // 4
            pr_tokens += sum(len(title) for title in self.pr_titles) // 4

            if pr_tokens <= available_tokens:
                prioritized.pull_requests = self.pull_requests.copy()
                prioritized.pr_titles = self.pr_titles.copy()
                available_tokens -= pr_tokens
            else:
                # Include partial PRs
                running_tokens = 0
                for _, (pr, title) in enumerate(
                    zip(
                        self.pull_requests,
                        self.pr_titles + [""] * len(self.pull_requests),
                        strict=False,
                    )
                ):
                    pr_token_cost = (len(str(pr)) + len(title)) // 4
                    if running_tokens + pr_token_cost <= available_tokens:
                        prioritized.pull_requests.append(pr)
                        if title:
                            prioritized.pr_titles.append(title)
                        running_tokens += pr_token_cost
                    else:
                        break
                available_tokens -= running_tokens

        # 3. Include commits up to remaining budget
        if available_tokens > 20:  # Only if we have reasonable tokens left
            commit_tokens = sum(len(str(commit)) for commit in self.commits) // 4
            commit_tokens += sum(len(msg) for msg in self.commit_messages) // 4

            if commit_tokens <= available_tokens:
                prioritized.commits = self.commits.copy()
                prioritized.commit_messages = self.commit_messages.copy()
                available_tokens -= commit_tokens
            else:
                # Include partial commits (prioritize recent ones)
                running_tokens = 0
                for _, (commit, msg) in enumerate(
                    zip(
                        self.commits,
                        self.commit_messages + [""] * len(self.commits),
                        strict=False,
                    )
                ):
                    commit_token_cost = (len(str(commit)) + len(msg)) // 4
                    if running_tokens + commit_token_cost <= available_tokens:
                        prioritized.commits.append(commit)
                        if msg:
                            prioritized.commit_messages.append(msg)
                        running_tokens += commit_token_cost
                    else:
                        break
                available_tokens -= running_tokens

        # 4. Include issues with remaining budget
        if available_tokens > 20:  # Only if we have reasonable tokens left
            issue_tokens = sum(len(str(issue)) for issue in self.issues) // 4
            issue_tokens += sum(len(title) for title in self.issue_titles) // 4

            if issue_tokens <= available_tokens:
                prioritized.issues = self.issues.copy()
                prioritized.issue_titles = self.issue_titles.copy()
            else:
                # Include partial issues
                running_tokens = 0
                for _, (issue, title) in enumerate(
                    zip(
                        self.issues,
                        self.issue_titles + [""] * len(self.issues),
                        strict=False,
                    )
                ):
                    issue_token_cost = (len(str(issue)) + len(title)) // 4
                    if running_tokens + issue_token_cost <= available_tokens:
                        prioritized.issues.append(issue)
                        if title:
                            prioritized.issue_titles.append(title)
                        running_tokens += issue_token_cost
                    else:
                        break

        # Recalculate token estimate for prioritized content
        prioritized.estimate_token_usage()

        logger.info(
            f"Prioritized context: {current_tokens} → {prioritized.estimated_tokens} tokens "
            f"({len(prioritized.releases)} releases, {len(prioritized.pull_requests)} PRs, "
            f"{len(prioritized.commits)} commits, {len(prioritized.issues)} issues)"
        )

        return prioritized


class ContextGatheringEngine:
    """Engine for gathering rich context from GitHub events."""

    def __init__(self, github_client: GitHubClient, default_budget: int = 8000):
        """Initialize the context gathering engine.

        Args:
            github_client: GitHub API client for fetching additional data
            default_budget: Default token budget for context gathering
        """
        self.github_client = github_client
        self.default_budget = default_budget

    async def gather_context(
        self, events: list[GitHubEventLike], budget: TokenBudget | None = None
    ) -> RichContext:
        """Gather rich context from GitHub events.

        Args:
            events: List of GitHub events to analyze
            budget: Token budget for context gathering

        Returns:
            Rich context data within token budget
        """
        if budget is None:
            budget = TokenBudget(max_tokens=self.default_budget)

        logger.info(
            f"Gathering context from {len(events)} events with budget of {budget.max_tokens} tokens"
        )

        context = RichContext(total_events=len(events))

        if events:
            # Set date range
            timestamps = [
                parse_github_datetime(event.created_at)
                for event in events
                if event.created_at
            ]
            if timestamps:
                context.date_range = (min(timestamps), max(timestamps))

            # Extract unique repositories
            repos = {event.repo.name for event in events if event.repo}
            context.repositories = sorted(repos)

        # Process events by type for rich context
        await self._process_events_by_type(events, context, budget)

        # Prioritize content to fit budget
        final_context = context.prioritize_content(budget)

        logger.info(
            f"Context gathering complete: {final_context.estimated_tokens} tokens used, "
            f"{budget.remaining} tokens remaining"
        )

        return final_context

    async def _process_events_by_type(
        self, events: list[GitHubEventLike], context: RichContext, budget: TokenBudget
    ) -> None:
        """Process events by type to extract rich context."""

        # Group events by type for efficient processing
        events_by_type: dict[str, list[GitHubEventLike]] = {}
        for event in events:
            event_type = event.type
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(event)

        # Process each event type
        for event_type, type_events in events_by_type.items():
            logger.debug(f"Processing {len(type_events)} {event_type} events")

            if event_type == "PushEvent":
                await self._process_push_events(type_events, context, budget)
            elif event_type == "PullRequestEvent":
                await self._process_pr_events(type_events, context, budget)
            elif event_type == "IssuesEvent":
                await self._process_issue_events(type_events, context, budget)
            elif event_type == "ReleaseEvent":
                await self._process_release_events(type_events, context, budget)
            elif event_type == "PullRequestReviewEvent":
                await self._process_review_events(type_events, context, budget)
            # Add more event types as needed

    async def _process_push_events(
        self, events: list[GitHubEventLike], context: RichContext, budget: TokenBudget
    ) -> None:
        """Process push events to extract commit information."""
        for event in events:
            # For BaseGitHubEvent, we'll work with the data we have access to
            # In a real implementation, you'd use the EventFactory to get the correct typed event

            # For testing purposes, we'll simulate extracting commit data
            # This would normally come from the specific event type's payload
            if hasattr(event, "payload") and event.payload:
                commits = event.payload.get("commits", [])
                for commit_data in commits:
                    # Extract basic commit info from payload
                    commit_details = {
                        "sha": commit_data.get("sha", ""),
                        "message": commit_data.get("message", ""),
                        "author": commit_data.get("author", {}),
                        "url": commit_data.get("url", ""),
                    }

                    # For budget efficiency, we'll use payload data primarily
                    context.add_commit(event, commit_details)

    async def _process_pr_events(
        self, events: list[GitHubEventLike], context: RichContext, budget: TokenBudget
    ) -> None:
        """Process pull request events."""
        for event in events:
            if not hasattr(event, "payload") or not event.payload:
                continue

            pr_data = event.payload.get("pull_request", {})
            if pr_data:
                context.add_pull_request(event, pr_data)

    async def _process_issue_events(
        self, events: list[GitHubEventLike], context: RichContext, budget: TokenBudget
    ) -> None:
        """Process issue events."""
        for event in events:
            if not hasattr(event, "payload") or not event.payload:
                continue

            issue_data = event.payload.get("issue", {})
            if issue_data:
                context.add_issue(event, issue_data)

    async def _process_release_events(
        self, events: list[GitHubEventLike], context: RichContext, budget: TokenBudget
    ) -> None:
        """Process release events."""
        for event in events:
            if not hasattr(event, "payload") or not event.payload:
                continue

            release_data = event.payload.get("release", {})
            if release_data:
                context.add_release(event, release_data)

    async def _process_review_events(
        self, events: list[GitHubEventLike], context: RichContext, budget: TokenBudget
    ) -> None:
        """Process pull request review events."""
        for event in events:
            if not hasattr(event, "payload") or not event.payload:
                continue

            review_data = event.payload.get("review", {})
            pr_data = event.payload.get("pull_request", {})

            if review_data and pr_data:
                context.reviews.append(
                    {
                        "review_id": review_data.get("id", 0),
                        "state": review_data.get("state", ""),
                        "body": review_data.get("body", ""),
                        "pr_number": pr_data.get("number", 0),
                        "pr_title": pr_data.get("title", ""),
                        "repository": event.repo.name if event.repo else "unknown",
                        "timestamp": parse_github_datetime(event.created_at),
                        "reviewer": review_data.get("user", {}).get("login", "unknown"),
                    }
                )
