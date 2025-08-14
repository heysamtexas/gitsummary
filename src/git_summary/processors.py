"""Event processors for GitHub activity data analysis."""

import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from git_summary.models import (
    ActivityPeriod,
    ActivitySummary,
    BaseGitHubEvent,
    DailyRollup,
    DetailedEvent,
    GitHubActivityReport,
    RepositoryBreakdown,
)

logger = logging.getLogger(__name__)


class EventProcessor:
    """Processor for analyzing GitHub events and generating activity summaries."""

    def __init__(self) -> None:
        """Initialize the event processor."""
        pass

    def process(
        self,
        events: list[BaseGitHubEvent],
        username: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> GitHubActivityReport:
        """Process events and generate activity report.

        Args:
            events: List of GitHub events to process
            username: GitHub username for the report
            start_date: Start of analysis period (optional)
            end_date: End of analysis period (optional)

        Returns:
            Complete activity report with summaries and breakdowns

        Raises:
            ValueError: If events list is invalid or processing fails
        """
        logger.info(f"Processing {len(events)} events for user '{username}'")

        # Handle empty dataset
        if not events:
            logger.info("No events to process, returning empty report")
            return self._create_empty_report(username, start_date, end_date)

        # Determine period from events if not provided
        if start_date is None or end_date is None:
            event_dates = [
                datetime.fromisoformat(event.created_at.replace("Z", "+00:00"))
                for event in events
            ]
            if start_date is None:
                start_date = min(event_dates)
            if end_date is None:
                end_date = max(event_dates)

        # Process events into components
        event_breakdown = self._count_events_by_type(events)
        daily_rollups = self._create_daily_rollups(events)
        repository_breakdown = self._create_repository_breakdown(events)
        detailed_events = self._create_detailed_events(events)

        # Calculate summary statistics
        summary = self._calculate_summary_statistics(
            events, event_breakdown, repository_breakdown
        )

        # Create period info
        period = ActivityPeriod(
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        )

        logger.info(
            f"Processed {summary.total_events} events across {summary.repositories_active} repositories"
        )

        return GitHubActivityReport(
            user=username,
            period=period,
            summary=summary,
            daily_rollups=daily_rollups,
            repository_breakdown=repository_breakdown,
            detailed_events=detailed_events,
        )

    def _create_empty_report(
        self,
        username: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> GitHubActivityReport:
        """Create an empty report for when there are no events."""
        now = datetime.now(UTC)
        if start_date is None:
            start_date = now
        if end_date is None:
            end_date = now

        return GitHubActivityReport(
            user=username,
            period=ActivityPeriod(
                start=start_date.isoformat(),
                end=end_date.isoformat(),
            ),
            summary=ActivitySummary(
                total_events=0,
                repositories_active=0,
                event_breakdown={},
                most_active_repository=None,
                most_common_event_type=None,
            ),
            daily_rollups=[],
            repository_breakdown={},
            detailed_events=[],
        )

    def _count_events_by_type(self, events: list[BaseGitHubEvent]) -> dict[str, int]:
        """Count events by their type."""
        event_counts: dict[str, int] = defaultdict(int)
        for event in events:
            event_counts[event.type] += 1
        return dict(event_counts)

    def _create_daily_rollups(self, events: list[BaseGitHubEvent]) -> list[DailyRollup]:
        """Create daily rollups with event counts per day."""
        daily_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "events": 0,
                "repositories": set(),
                "event_types": defaultdict(int),
            }
        )

        # Group events by date
        for event in events:
            # Parse event date and extract date component
            event_datetime = datetime.fromisoformat(
                event.created_at.replace("Z", "+00:00")
            )
            date_str = event_datetime.date().isoformat()

            daily_data[date_str]["events"] += 1
            daily_data[date_str]["repositories"].add(event.repo.name)
            daily_data[date_str]["event_types"][event.type] += 1

        # Convert to DailyRollup objects
        rollups = []
        for date_str in sorted(daily_data.keys()):
            data = daily_data[date_str]
            rollups.append(
                DailyRollup(
                    date=date_str,
                    events=data["events"],
                    repositories=sorted(data["repositories"]),
                    event_types=dict(data["event_types"]),
                )
            )

        return rollups

    def _create_repository_breakdown(
        self, events: list[BaseGitHubEvent]
    ) -> dict[str, RepositoryBreakdown]:
        """Create per-repository activity breakdown."""
        repo_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "events": 0,
                "event_types": defaultdict(int),
                "timestamps": [],
            }
        )

        # Group events by repository
        for event in events:
            repo_name = event.repo.name
            repo_data[repo_name]["events"] += 1
            repo_data[repo_name]["event_types"][event.type] += 1
            repo_data[repo_name]["timestamps"].append(event.created_at)

        # Convert to RepositoryBreakdown objects
        breakdown = {}
        for repo_name, data in repo_data.items():
            # Sort timestamps to find first and last activity
            timestamps = sorted(data["timestamps"])
            breakdown[repo_name] = RepositoryBreakdown(
                events=data["events"],
                event_types=dict(data["event_types"]),
                first_activity=timestamps[0] if timestamps else None,
                last_activity=timestamps[-1] if timestamps else "",
            )

        return breakdown

    def _create_detailed_events(
        self, events: list[BaseGitHubEvent]
    ) -> list[DetailedEvent]:
        """Create simplified detailed events for output."""
        detailed = []
        for event in events:
            # Extract relevant details from payload
            details = self._extract_event_details(event)

            detailed.append(
                DetailedEvent(
                    type=event.type,
                    created_at=event.created_at,
                    repository=event.repo.name,
                    actor=event.actor.login,
                    details=details,
                )
            )

        return detailed

    def _extract_event_details(self, event: Any) -> dict[str, Any]:
        """Extract relevant details from event payload based on event type."""
        details: dict[str, Any] = {}

        if event.type == "PushEvent":
            payload = event.payload
            if hasattr(payload, "commits") and payload.commits:
                details["commits_count"] = len(payload.commits)
                details["ref"] = getattr(payload, "ref", "unknown")
                # Extract commit messages (first 3 commits for brevity)
                commit_messages = []
                for commit in payload.commits[:3]:
                    if hasattr(commit, "message"):
                        commit_messages.append(commit.message.strip())
                if commit_messages:
                    details["commit_messages"] = commit_messages
                if len(payload.commits) > 3:
                    details["more_commits"] = len(payload.commits) - 3
        elif event.type == "IssuesEvent":
            payload = event.payload
            if hasattr(payload, "action"):
                details["action"] = payload.action
            if hasattr(payload, "issue") and hasattr(payload.issue, "number"):
                details["issue_number"] = payload.issue.number
        elif event.type == "PullRequestEvent":
            payload = event.payload
            if hasattr(payload, "action"):
                details["action"] = payload.action
            if hasattr(payload, "pull_request") and hasattr(
                payload.pull_request, "number"
            ):
                details["pr_number"] = payload.pull_request.number
        elif event.type == "CreateEvent":
            payload = event.payload
            if hasattr(payload, "ref_type"):
                details["ref_type"] = payload.ref_type
            if hasattr(payload, "ref") and payload.ref:
                details["ref"] = payload.ref
        elif event.type == "DeleteEvent":
            payload = event.payload
            if hasattr(payload, "ref_type"):
                details["ref_type"] = payload.ref_type
            if hasattr(payload, "ref"):
                details["ref"] = payload.ref
        elif event.type == "ForkEvent":
            payload = event.payload
            if hasattr(payload, "forkee") and hasattr(payload.forkee, "full_name"):
                details["forked_to"] = payload.forkee.full_name
        elif event.type == "WatchEvent":
            payload = event.payload
            if hasattr(payload, "action"):
                details["action"] = payload.action
        elif event.type == "ReleaseEvent":
            payload = event.payload
            if hasattr(payload, "action"):
                details["action"] = payload.action
            if hasattr(payload, "release"):
                release = payload.release
                if hasattr(release, "tag_name"):
                    details["version"] = release.tag_name
                if hasattr(release, "name"):
                    details["release_name"] = release.name
                if hasattr(release, "body") and release.body:
                    # Truncate release notes to first 200 chars
                    details["release_notes"] = release.body[:200].strip()
                    if len(release.body) > 200:
                        details["release_notes"] += "..."
                if hasattr(release, "prerelease"):
                    details["prerelease"] = release.prerelease

        return details

    def _calculate_summary_statistics(
        self,
        events: list[BaseGitHubEvent],
        event_breakdown: dict[str, int],
        repository_breakdown: dict[str, RepositoryBreakdown],
    ) -> ActivitySummary:
        """Calculate overall summary statistics."""
        total_events = len(events)
        repositories_active = len(repository_breakdown)

        # Find most common event type
        most_common_event_type = None
        if event_breakdown:
            most_common_event_type = max(
                event_breakdown.keys(), key=lambda k: event_breakdown[k]
            )

        # Find most active repository
        most_active_repository = None
        if repository_breakdown:
            most_active_repository = max(
                repository_breakdown.keys(),
                key=lambda repo: repository_breakdown[repo].events,
            )

        return ActivitySummary(
            total_events=total_events,
            repositories_active=repositories_active,
            event_breakdown=event_breakdown,
            most_active_repository=most_active_repository,
            most_common_event_type=most_common_event_type,
        )
