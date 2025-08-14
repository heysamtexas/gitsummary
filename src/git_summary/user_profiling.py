"""User profiling and automation detection for adaptive analysis strategies."""

import logging
from collections import Counter
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from git_summary.github_client import GitHubClient
from git_summary.models import BaseGitHubEvent

logger = logging.getLogger(__name__)


class UserProfile(BaseModel):
    """User activity profile with automation detection metrics."""

    username: str
    total_events: int
    issue_event_ratio: float
    single_repo_dominance: float
    dominant_repo: str | None
    high_frequency_score: float
    events_per_hour: float
    is_heavily_automated: bool
    classification: str  # "normal", "power", "heavy-automation"
    confidence_score: float
    automation_indicators: dict[str, Any]


class AutomationDetector:
    """Detects automation pollution in user event streams and classifies users."""

    def __init__(self, github_client: GitHubClient) -> None:
        """Initialize the automation detector.

        Args:
            github_client: GitHub API client for fetching user events
        """
        self.github_client = github_client

        # Thresholds for automation detection
        self.ISSUE_RATIO_THRESHOLD = 0.70  # >70% issue events indicates bot activity
        self.SINGLE_REPO_DOMINANCE_THRESHOLD = 0.80  # >80% events from one repo
        self.HIGH_FREQUENCY_THRESHOLD = 5.0  # >5 events/hour on average

        # Classification confidence thresholds
        self.HIGH_CONFIDENCE = 0.8
        self.MEDIUM_CONFIDENCE = 0.6

    async def classify_user(self, username: str, days: int = 7) -> UserProfile:
        """Classify a user's activity profile based on automation detection heuristics.

        Args:
            username: GitHub username to analyze
            days: Number of days to analyze (default 7 for quick profiling)

        Returns:
            UserProfile with classification and automation indicators
        """
        logger.info(f"Classifying user '{username}' based on {days} days of activity")

        # Fetch user events (limited to first 300 for quick profiling)
        events = []
        event_count = 0
        max_events = 300

        async for event_batch in self.github_client.get_user_events_paginated(
            username,
            per_page=100,
            max_pages=3,  # 3 pages = max 300 events
        ):
            events.extend(event_batch)
            event_count += len(event_batch)
            if event_count >= max_events:
                break

        # If we got less than expected, user might be inactive or new
        if len(events) < 10:
            logger.info(
                f"User '{username}' has very few events ({len(events)}), classifying as normal"
            )
            return UserProfile(
                username=username,
                total_events=len(events),
                issue_event_ratio=0.0,
                single_repo_dominance=0.0,
                dominant_repo=None,
                high_frequency_score=0.0,
                events_per_hour=0.0,
                is_heavily_automated=False,
                classification="normal",
                confidence_score=1.0,
                automation_indicators={"reason": "insufficient_activity"},
            )

        # Calculate automation indicators
        automation_indicators = self._calculate_automation_indicators(events, days)

        # Determine classification and confidence
        classification, confidence = self._determine_classification(
            automation_indicators
        )

        # Create profile
        profile = UserProfile(
            username=username,
            total_events=len(events),
            issue_event_ratio=automation_indicators["issue_event_ratio"],
            single_repo_dominance=automation_indicators["single_repo_dominance"],
            dominant_repo=automation_indicators["dominant_repo"],
            high_frequency_score=automation_indicators["high_frequency_score"],
            events_per_hour=automation_indicators["events_per_hour"],
            is_heavily_automated=classification == "heavy-automation",
            classification=classification,
            confidence_score=confidence,
            automation_indicators=automation_indicators,
        )

        logger.info(
            f"User '{username}' classified as '{classification}' "
            f"(confidence: {confidence:.2f}, automated: {profile.is_heavily_automated})"
        )

        return profile

    def _calculate_automation_indicators(
        self, events: list[BaseGitHubEvent], days: int
    ) -> dict[str, Any]:
        """Calculate various automation indicators from user events.

        Args:
            events: List of GitHub events
            days: Number of days the events span

        Returns:
            Dictionary of automation indicators
        """
        if not events:
            return {
                "issue_event_ratio": 0.0,
                "single_repo_dominance": 0.0,
                "dominant_repo": None,
                "high_frequency_score": 0.0,
                "events_per_hour": 0.0,
                "event_type_distribution": {},
                "repository_distribution": {},
                "time_pattern_analysis": {},
            }

        # Count event types
        event_types = Counter(event.type for event in events)
        total_events = len(events)

        # Calculate issue event ratio (IssuesEvent + IssueCommentEvent)
        issue_events = event_types.get("IssuesEvent", 0) + event_types.get(
            "IssueCommentEvent", 0
        )
        issue_event_ratio = issue_events / total_events if total_events > 0 else 0.0

        # Count repository distribution
        repo_distribution = Counter(event.repo.name for event in events)

        # Calculate single repo dominance
        if repo_distribution:
            most_active_repo, most_active_count = repo_distribution.most_common(1)[0]
            single_repo_dominance = most_active_count / total_events
            dominant_repo = most_active_repo
        else:
            single_repo_dominance = 0.0
            dominant_repo = None

        # Calculate time-based frequency metrics
        if len(events) >= 2:
            # Get time span of events
            event_times = []
            for event in events:
                try:
                    # Parse GitHub timestamp
                    event_time = datetime.fromisoformat(
                        event.created_at.replace("Z", "+00:00")
                    )
                    event_times.append(event_time)
                except (ValueError, AttributeError):
                    continue

            if len(event_times) >= 2:
                event_times.sort()
                time_span = (
                    event_times[-1] - event_times[0]
                ).total_seconds() / 3600  # hours
                events_per_hour = len(event_times) / time_span if time_span > 0 else 0.0
            else:
                events_per_hour = 0.0
        else:
            events_per_hour = 0.0

        # High frequency score (normalized against threshold)
        high_frequency_score = min(events_per_hour / self.HIGH_FREQUENCY_THRESHOLD, 1.0)

        # Time pattern analysis (detect suspicious timing patterns)
        time_pattern_analysis = self._analyze_time_patterns(events)

        return {
            "issue_event_ratio": issue_event_ratio,
            "single_repo_dominance": single_repo_dominance,
            "dominant_repo": dominant_repo,
            "high_frequency_score": high_frequency_score,
            "events_per_hour": events_per_hour,
            "event_type_distribution": dict(event_types),
            "repository_distribution": dict(repo_distribution.most_common(10)),
            "time_pattern_analysis": time_pattern_analysis,
        }

    def _analyze_time_patterns(self, events: list[BaseGitHubEvent]) -> dict[str, Any]:
        """Analyze temporal patterns that might indicate automation.

        Args:
            events: List of GitHub events

        Returns:
            Dictionary with time pattern analysis
        """
        if len(events) < 5:
            return {"pattern_detected": False, "reason": "insufficient_events"}

        event_times = []
        for event in events:
            try:
                event_time = datetime.fromisoformat(
                    event.created_at.replace("Z", "+00:00")
                )
                event_times.append(event_time)
            except (ValueError, AttributeError):
                continue

        if len(event_times) < 5:
            return {
                "pattern_detected": False,
                "reason": "insufficient_valid_timestamps",
            }

        event_times.sort()

        # Check for suspiciously regular intervals
        intervals = []
        for i in range(1, len(event_times)):
            interval = (event_times[i] - event_times[i - 1]).total_seconds()
            intervals.append(interval)

        # Check if many intervals are very similar (potential automation)
        if len(intervals) >= 3:
            # Calculate coefficient of variation for intervals
            import statistics

            mean_interval = statistics.mean(intervals)
            if mean_interval > 0:
                stdev_interval = (
                    statistics.stdev(intervals) if len(intervals) > 1 else 0
                )
                cv = stdev_interval / mean_interval

                # Low coefficient of variation indicates regular timing
                regular_timing = cv < 0.3  # Less than 30% variation

                return {
                    "pattern_detected": regular_timing,
                    "mean_interval_seconds": mean_interval,
                    "coefficient_of_variation": cv,
                    "regular_timing": regular_timing,
                }

        return {"pattern_detected": False, "reason": "irregular_timing"}

    def _determine_classification(
        self, indicators: dict[str, Any]
    ) -> tuple[str, float]:
        """Determine user classification and confidence score.

        Args:
            indicators: Automation indicators dictionary

        Returns:
            Tuple of (classification, confidence_score)
        """
        # Extract key metrics
        issue_ratio = indicators["issue_event_ratio"]
        repo_dominance = indicators["single_repo_dominance"]
        frequency_score = indicators["high_frequency_score"]

        # Count automation flags
        automation_flags = 0
        flag_details = []

        # Flag 1: High issue event ratio
        if issue_ratio > self.ISSUE_RATIO_THRESHOLD:
            automation_flags += 1
            flag_details.append(f"high_issue_ratio({issue_ratio:.2f})")

        # Flag 2: Single repository dominance
        if repo_dominance > self.SINGLE_REPO_DOMINANCE_THRESHOLD:
            automation_flags += 1
            flag_details.append(f"repo_dominance({repo_dominance:.2f})")

        # Flag 3: High frequency activity
        if frequency_score > 0.8:  # >80% of threshold
            automation_flags += 1
            flag_details.append(f"high_frequency({frequency_score:.2f})")

        # Flag 4: Regular timing patterns
        time_patterns = indicators.get("time_pattern_analysis", {})
        if time_patterns.get("regular_timing", False):
            automation_flags += 1
            flag_details.append("regular_timing")

        logger.debug(f"Automation flags: {automation_flags}, details: {flag_details}")

        # Determine classification based on flags
        if automation_flags >= 2:
            classification = "heavy-automation"
            confidence = min(0.6 + (automation_flags * 0.15), 1.0)
        elif automation_flags == 1:
            # Single flag might indicate power user or light automation
            if issue_ratio > 0.5 or repo_dominance > 0.7:
                classification = "power"
                confidence = 0.7
            else:
                classification = "normal"
                confidence = 0.8
        else:
            classification = "normal"
            confidence = 0.9

        return classification, confidence
