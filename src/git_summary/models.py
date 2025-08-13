"""Pydantic models for GitHub API data structures."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, field_validator


class Actor(BaseModel):
    """GitHub actor (user) model."""

    id: int | None = None
    login: str
    display_login: str | None = None
    gravatar_id: str | None = None
    url: str | None = None
    avatar_url: str | None = None


class Repository(BaseModel):
    """GitHub repository model."""

    id: int | None = None
    name: str
    full_name: str | None = None
    url: str | None = None
    description: str | None = None
    private: bool = False
    owner: Actor | None = None
    html_url: str | None = None
    clone_url: str | None = None
    git_url: str | None = None
    ssh_url: str | None = None
    stargazers_count: int | None = None
    watchers_count: int | None = None
    language: str | None = None
    has_issues: bool | None = None
    has_projects: bool | None = None
    has_wiki: bool | None = None
    has_pages: bool | None = None
    forks_count: int | None = None
    archived: bool | None = None
    disabled: bool | None = None
    open_issues_count: int | None = None
    license: dict[str, Any] | None = None
    forks: int | None = None
    open_issues: int | None = None
    watchers: int | None = None
    default_branch: str | None = None


class BaseGitHubEvent(BaseModel):
    """Base GitHub event model with common fields."""

    id: str
    type: str
    actor: Actor
    repo: Repository
    created_at: str
    public: bool
    org: Actor | None = None

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, v: str) -> str:
        """Validate created_at is a valid ISO datetime string."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"Invalid datetime format: {v}") from e
        return v


class Commit(BaseModel):
    """GitHub commit model."""

    sha: str
    author: dict[str, str]
    message: str
    distinct: bool
    url: str
    tree: dict[str, str] | None = None
    committer: dict[str, str] | None = None
    added: list[str] | None = None
    removed: list[str] | None = None
    modified: list[str] | None = None


class PushEventPayload(BaseModel):
    """Payload for PushEvent."""

    repository_id: int | None = None
    push_id: int
    size: int
    distinct_size: int
    ref: str
    head: str
    before: str
    commits: list[Commit]


class PushEvent(BaseGitHubEvent):
    """GitHub PushEvent model."""

    type: Literal["PushEvent"]
    payload: PushEventPayload


class Issue(BaseModel):
    """GitHub issue model."""

    id: int
    number: int
    title: str
    user: Actor
    labels: list[dict[str, Any]]
    state: str
    locked: bool
    assignee: Actor | None = None
    assignees: list[Actor]
    milestone: dict[str, Any] | None = None
    comments: int
    created_at: str
    updated_at: str
    closed_at: str | None = None
    author_association: str
    active_lock_reason: str | None = None
    draft: bool | None = None
    pull_request: dict[str, Any] | None = None
    body: str | None = None
    reactions: dict[str, Any] | None = None
    timeline_url: str | None = None
    performed_via_github_app: dict[str, Any] | None = None
    state_reason: str | None = None
    url: str
    repository_url: str
    labels_url: str
    comments_url: str
    events_url: str
    html_url: str
    node_id: str


class IssuesEventPayload(BaseModel):
    """Payload for IssuesEvent."""

    action: str
    issue: Issue
    repository: Repository | None = None
    sender: Actor | None = None


class IssuesEvent(BaseGitHubEvent):
    """GitHub IssuesEvent model."""

    type: Literal["IssuesEvent"]
    payload: IssuesEventPayload


class PullRequest(BaseModel):
    """GitHub pull request model."""

    id: int
    number: int
    title: str
    user: Actor
    state: str
    locked: bool
    assignee: Actor | None = None
    assignees: list[Actor]
    requested_reviewers: list[Actor]
    requested_teams: list[dict[str, Any]]
    labels: list[dict[str, Any]]
    milestone: dict[str, Any] | None = None
    draft: bool
    commits_url: str
    review_comments_url: str
    review_comment_url: str
    comments_url: str
    statuses_url: str
    head: dict[str, Any]
    base: dict[str, Any]
    links: dict[str, Any]
    author_association: str
    auto_merge: dict[str, Any] | None = None
    active_lock_reason: str | None = None
    merged: bool | None = None
    mergeable: bool | None = None
    rebaseable: bool | None = None
    mergeable_state: str | None = None
    merged_by: Actor | None = None
    comments: int
    review_comments: int
    maintainer_can_modify: bool
    commits: int
    additions: int
    deletions: int
    changed_files: int
    url: str
    html_url: str
    diff_url: str
    patch_url: str
    issue_url: str
    node_id: str
    created_at: str
    updated_at: str
    closed_at: str | None = None
    merged_at: str | None = None
    merge_commit_sha: str | None = None
    body: str | None = None


class PullRequestEventPayload(BaseModel):
    """Payload for PullRequestEvent."""

    action: str
    number: int
    pull_request: PullRequest
    repository: Repository | None = None
    sender: Actor | None = None


class PullRequestEvent(BaseGitHubEvent):
    """GitHub PullRequestEvent model."""

    type: Literal["PullRequestEvent"]
    payload: PullRequestEventPayload


class CreateEventPayload(BaseModel):
    """Payload for CreateEvent."""

    ref: str | None = None
    ref_type: str
    master_branch: str
    description: str | None = None
    pusher_type: str


class CreateEvent(BaseGitHubEvent):
    """GitHub CreateEvent model."""

    type: Literal["CreateEvent"]
    payload: CreateEventPayload


class DeleteEventPayload(BaseModel):
    """Payload for DeleteEvent."""

    ref: str
    ref_type: str
    pusher_type: str


class DeleteEvent(BaseGitHubEvent):
    """GitHub DeleteEvent model."""

    type: Literal["DeleteEvent"]
    payload: DeleteEventPayload


class ForkEventPayload(BaseModel):
    """Payload for ForkEvent."""

    forkee: Repository


class ForkEvent(BaseGitHubEvent):
    """GitHub ForkEvent model."""

    type: Literal["ForkEvent"]
    payload: ForkEventPayload


class WatchEventPayload(BaseModel):
    """Payload for WatchEvent."""

    action: str


class WatchEvent(BaseGitHubEvent):
    """GitHub WatchEvent model."""

    type: Literal["WatchEvent"]
    payload: WatchEventPayload


class Release(BaseModel):
    """GitHub release model."""

    id: int
    tag_name: str
    target_commitish: str
    name: str | None = None
    draft: bool
    author: Actor
    prerelease: bool
    created_at: str
    published_at: str | None = None
    url: str
    html_url: str
    assets_url: str
    upload_url: str
    tarball_url: str | None = None
    zipball_url: str | None = None
    discussion_url: str | None = None
    body: str | None = None
    reactions: dict[str, Any] | None = None
    node_id: str


class ReleaseEventPayload(BaseModel):
    """Payload for ReleaseEvent."""

    action: str
    release: Release
    repository: Repository | None = None
    sender: Actor | None = None


class ReleaseEvent(BaseGitHubEvent):
    """GitHub ReleaseEvent model."""

    type: Literal["ReleaseEvent"]
    payload: ReleaseEventPayload


class Comment(BaseModel):
    """GitHub comment model."""

    id: int
    user: Actor
    created_at: str
    updated_at: str
    author_association: str
    body: str
    reactions: dict[str, Any] | None = None
    url: str
    html_url: str
    issue_url: str | None = None
    node_id: str
    performed_via_github_app: dict[str, Any] | None = None


class IssueCommentEventPayload(BaseModel):
    """Payload for IssueCommentEvent."""

    action: str
    issue: Issue
    comment: Comment
    repository: Repository | None = None
    sender: Actor | None = None


class IssueCommentEvent(BaseGitHubEvent):
    """GitHub IssueCommentEvent model."""

    type: Literal["IssueCommentEvent"]
    payload: IssueCommentEventPayload


class Review(BaseModel):
    """GitHub pull request review model."""

    id: int
    user: Actor
    body: str | None = None
    commit_id: str
    submitted_at: str | None = None
    state: str
    html_url: str
    pull_request_url: str
    author_association: str
    links: dict[str, Any]
    node_id: str


class PullRequestReviewEventPayload(BaseModel):
    """Payload for PullRequestReviewEvent."""

    action: str
    review: Review
    pull_request: PullRequest
    repository: Repository | None = None
    sender: Actor | None = None


class PullRequestReviewEvent(BaseGitHubEvent):
    """GitHub PullRequestReviewEvent model."""

    type: Literal["PullRequestReviewEvent"]
    payload: PullRequestReviewEventPayload


class PullRequestReviewComment(BaseModel):
    """GitHub pull request review comment model."""

    id: int
    diff_hunk: str
    path: str
    position: int | None = None
    original_position: int
    commit_id: str
    original_commit_id: str
    user: Actor
    body: str
    created_at: str
    updated_at: str
    html_url: str
    url: str
    author_association: str
    links: dict[str, Any]
    start_line: int | None = None
    original_start_line: int | None = None
    start_side: str | None = None
    line: int | None = None
    original_line: int | None = None
    side: str
    in_reply_to_id: int | None = None
    pull_request_review_id: int | None = None
    reactions: dict[str, Any] | None = None
    node_id: str


class PullRequestReviewCommentEventPayload(BaseModel):
    """Payload for PullRequestReviewCommentEvent."""

    action: str
    comment: PullRequestReviewComment
    pull_request: PullRequest
    repository: Repository | None = None
    sender: Actor | None = None


class PullRequestReviewCommentEvent(BaseGitHubEvent):
    """GitHub PullRequestReviewCommentEvent model."""

    type: Literal["PullRequestReviewCommentEvent"]
    payload: PullRequestReviewCommentEventPayload


class MemberEventPayload(BaseModel):
    """Payload for MemberEvent."""

    action: str
    member: Actor
    changes: dict[str, Any] | None = None


class MemberEvent(BaseGitHubEvent):
    """GitHub MemberEvent model."""

    type: Literal["MemberEvent"]
    payload: MemberEventPayload


class PublicEventPayload(BaseModel):
    """Payload for PublicEvent (repository made public)."""

    # PublicEvent typically has minimal payload


class PublicEvent(BaseGitHubEvent):
    """GitHub PublicEvent model."""

    type: Literal["PublicEvent"]
    payload: PublicEventPayload


class GollumEventPayload(BaseModel):
    """Payload for GollumEvent (wiki page updates)."""

    pages: list[dict[str, Any]]


class GollumEvent(BaseGitHubEvent):
    """GitHub GollumEvent model."""

    type: Literal["GollumEvent"]
    payload: GollumEventPayload


class UnknownEventPayload(BaseModel):
    """Payload for unknown event types (fallback)."""

    model_config = {"extra": "allow"}  # Allow any extra fields


class UnknownEvent(BaseGitHubEvent):
    """Fallback for unknown or unsupported event types."""

    payload: UnknownEventPayload


# Union type for all GitHub events
GitHubEvent = (
    PushEvent
    | IssuesEvent
    | PullRequestEvent
    | CreateEvent
    | DeleteEvent
    | ForkEvent
    | WatchEvent
    | ReleaseEvent
    | IssueCommentEvent
    | PullRequestReviewEvent
    | PullRequestReviewCommentEvent
    | MemberEvent
    | PublicEvent
    | GollumEvent
    | UnknownEvent
)


class EventFactory:
    """Factory for creating GitHub event instances from API responses."""

    _event_types: dict[str, type[BaseGitHubEvent]] = {
        "PushEvent": PushEvent,
        "IssuesEvent": IssuesEvent,
        "PullRequestEvent": PullRequestEvent,
        "CreateEvent": CreateEvent,
        "DeleteEvent": DeleteEvent,
        "ForkEvent": ForkEvent,
        "WatchEvent": WatchEvent,
        "ReleaseEvent": ReleaseEvent,
        "IssueCommentEvent": IssueCommentEvent,
        "PullRequestReviewEvent": PullRequestReviewEvent,
        "PullRequestReviewCommentEvent": PullRequestReviewCommentEvent,
        "MemberEvent": MemberEvent,
        "PublicEvent": PublicEvent,
        "GollumEvent": GollumEvent,
    }

    @classmethod
    def create_event(cls, event_data: dict[str, Any]) -> BaseGitHubEvent:
        """Create a GitHub event instance from API response data.

        Args:
            event_data: Raw event data from GitHub API

        Returns:
            Typed GitHub event instance

        Raises:
            ValidationError: If event data is invalid
        """
        event_type = event_data.get("type")

        # Try to create the specific event type
        if event_type and event_type in cls._event_types:
            event_class = cls._event_types[event_type]
            try:
                return event_class(**event_data)
            except Exception:
                # If specific event creation fails, fall back to UnknownEvent
                pass

        # Fallback to UnknownEvent for unsupported or invalid events
        # Create minimal payload to ensure it works
        fallback_data = event_data.copy()
        if "payload" not in fallback_data:
            fallback_data["payload"] = {}

        return UnknownEvent(**fallback_data)

    @classmethod
    def get_supported_event_types(cls) -> list[str]:
        """Get list of supported event types."""
        return list(cls._event_types.keys())


# Output schema models for the summary command
class ActivitySummary(BaseModel):
    """Summary statistics for GitHub activity."""

    total_events: int
    repositories_active: int
    event_breakdown: dict[str, int]
    most_active_repository: str | None = None
    most_common_event_type: str | None = None


class DailyRollup(BaseModel):
    """Daily activity rollup."""

    date: str  # ISO date format (YYYY-MM-DD)
    events: int
    repositories: list[str]
    event_types: dict[str, int]

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Validate date is in YYYY-MM-DD format."""
        try:
            datetime.fromisoformat(v)
        except ValueError as e:
            raise ValueError(f"Invalid date format: {v}") from e
        return v


class RepositoryBreakdown(BaseModel):
    """Per-repository activity breakdown."""

    events: int
    event_types: dict[str, int]
    last_activity: str
    first_activity: str | None = None


class ActivityPeriod(BaseModel):
    """Time period for activity analysis."""

    start: str  # ISO datetime
    end: str  # ISO datetime

    @field_validator("start", "end")
    @classmethod
    def validate_datetime(cls, v: str) -> str:
        """Validate datetime is valid ISO format."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"Invalid datetime format: {v}") from e
        return v


class DetailedEvent(BaseModel):
    """Simplified event model for detailed output."""

    type: str
    created_at: str
    repository: str
    actor: str
    details: dict[str, Any]


class GitHubActivityReport(BaseModel):
    """Complete GitHub activity report schema."""

    user: str
    period: ActivityPeriod
    summary: ActivitySummary
    daily_rollups: list[DailyRollup]
    repository_breakdown: dict[str, RepositoryBreakdown]
    detailed_events: list[DetailedEvent]

    model_config = {"json_encoders": {datetime: lambda dt: dt.isoformat()}}
