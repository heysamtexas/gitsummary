"""Library API for git-summary.

This module provides the main entry points for external projects wanting to use
git-summary as a library rather than a CLI tool.
"""

from __future__ import annotations

# Runtime imports (needed for Pydantic models)
from datetime import datetime
from enum import Enum
from functools import cached_property
from typing import Any, Protocol

from pydantic import BaseModel, field_validator

from git_summary.models import GitHubActivityReport  # noqa: TCH001


# Exception hierarchy for clear error handling
class GitHubAnalysisError(Exception):
    """Base exception for GitHub analysis errors."""


class InvalidTokenError(GitHubAnalysisError):
    """Raised when GitHub token is invalid or expired."""


class UserNotFoundError(GitHubAnalysisError):
    """Raised when GitHub user doesn't exist."""


class RateLimitError(GitHubAnalysisError):
    """Raised when GitHub API rate limit is exceeded."""


class ConfigurationError(GitHubAnalysisError):
    """Raised when analysis configuration is invalid."""


class ProgressCallback(Protocol):
    """Protocol for progress callback functions."""

    def __call__(self, current: int, total: int, message: str) -> None:
        """Progress callback signature.

        Args:
            current: Current progress value
            total: Total expected value
            message: Progress description
        """


class AnalysisStrategy(Enum):
    """Analysis strategy options."""

    AUTO = "auto"  # Let the system choose optimal strategy
    INTELLIGENCE_GUIDED = "intelligence_guided"  # Fast, top repositories
    MULTI_SOURCE = "multi_source"  # Comprehensive, all activity


class AnalysisConfig(BaseModel):
    """Configuration for GitHub activity analysis.

    This immutable configuration object specifies all parameters for analysis.
    Use the builder pattern for convenient construction.
    """

    # Core analysis parameters
    days: int = 7
    max_events: int | None = None
    strategy: AnalysisStrategy = AnalysisStrategy.AUTO
    max_repos: int | None = None

    # Event filtering
    include_events: list[str] | None = None
    exclude_repos: list[str] | None = None

    # AI analysis options (if using AI features)
    ai_model: str | None = None
    ai_persona: str | None = None
    ai_token_budget: int = 200

    # Repository filtering
    repo_filter: list[str] | None = None

    @field_validator("days")
    @classmethod
    def validate_days(cls, v: int) -> int:
        """Validate days within GitHub API constraints.

        GitHub public events are limited to ~90 days of history.
        """
        if v < 1:
            raise ValueError("days must be at least 1")
        if v > 90:
            raise ValueError("days cannot exceed 90 (GitHub API limitation)")
        return v

    @field_validator("max_events")
    @classmethod
    def validate_max_events(cls, v: int | None) -> int | None:
        """Validate max_events if provided."""
        if v is not None and v < 1:
            raise ValueError("max_events must be at least 1")
        return v

    @field_validator("ai_token_budget")
    @classmethod
    def validate_token_budget(cls, v: int) -> int:
        """Validate AI token budget within realistic bounds.

        - Minimum 50: Basic analysis
        - Maximum 1000: Comprehensive analysis without rate limit risk
        """
        if v < 50:
            raise ValueError("ai_token_budget must be at least 50 for basic analysis")
        if v > 1000:
            raise ValueError("ai_token_budget cannot exceed 1000 to avoid rate limits")
        return v

    @classmethod
    def builder(cls) -> AnalysisConfigBuilder:
        """Create a new configuration builder."""
        return AnalysisConfigBuilder()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(exclude_none=True)


class AnalysisConfigBuilder:
    """Builder for AnalysisConfig objects.

    Provides a fluent interface for constructing analysis configurations.
    """

    def __init__(self) -> None:
        """Initialize builder with default values."""
        self._config: dict[str, Any] = {}

    def days(self, days: int) -> AnalysisConfigBuilder:
        """Set number of days to analyze."""
        self._config["days"] = days
        return self

    def max_events(self, max_events: int) -> AnalysisConfigBuilder:
        """Set maximum number of events to process."""
        self._config["max_events"] = max_events
        return self

    def strategy(self, strategy: AnalysisStrategy | str) -> AnalysisConfigBuilder:
        """Set analysis strategy."""
        if isinstance(strategy, str):
            strategy = AnalysisStrategy(strategy)
        self._config["strategy"] = strategy
        return self

    def intelligence_guided(self) -> AnalysisConfigBuilder:
        """Use intelligence-guided analysis (fast, top repos)."""
        return self.strategy(AnalysisStrategy.INTELLIGENCE_GUIDED)

    def comprehensive(self) -> AnalysisConfigBuilder:
        """Use comprehensive multi-source analysis."""
        return self.strategy(AnalysisStrategy.MULTI_SOURCE)

    def max_repos(self, max_repos: int) -> AnalysisConfigBuilder:
        """Set maximum number of repositories to analyze."""
        self._config["max_repos"] = max_repos
        return self

    def include_events(self, event_types: list[str]) -> AnalysisConfigBuilder:
        """Set event types to include."""
        self._config["include_events"] = event_types
        return self

    def exclude_repos(self, repo_names: list[str]) -> AnalysisConfigBuilder:
        """Set repositories to exclude."""
        self._config["exclude_repos"] = repo_names
        return self

    def repo_filter(self, repo_names: list[str]) -> AnalysisConfigBuilder:
        """Set repositories to focus on."""
        self._config["repo_filter"] = repo_names
        return self

    def ai_analysis(
        self,
        model: str = "anthropic/claude-3-7-sonnet-latest",
        persona: str = "tech analyst",
        token_budget: int = 200,
    ) -> AnalysisConfigBuilder:
        """Configure AI-powered analysis."""
        self._config["ai_model"] = model
        self._config["ai_persona"] = persona
        self._config["ai_token_budget"] = token_budget
        return self

    def custom(self) -> AnalysisConfigBuilder:
        """Use custom strategy without automatic configuration."""
        return self.strategy(AnalysisStrategy.AUTO)

    def build(self) -> AnalysisConfig:
        """Build the final configuration object."""
        return AnalysisConfig(**self._config)


class AnalysisResult(BaseModel):
    """Rich result object for GitHub activity analysis.

    Provides convenient access to analysis results with computed properties
    and multiple output formats.
    """

    # Core analysis data
    username: str
    start_date: datetime
    end_date: datetime
    analysis_strategy: str
    execution_time_ms: int | None = None

    # Raw report data (from existing models)
    report: GitHubActivityReport

    # Analysis metadata
    total_api_calls: int = 0
    repositories_discovered: int = 0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def total_events(self) -> int:
        """Total number of events analyzed."""
        return self.report.summary.total_events

    @cached_property
    def total_repositories(self) -> int:
        """Total number of repositories with activity."""
        return self.report.summary.repositories_active

    @cached_property
    def total_commits(self) -> int:
        """Total number of push events (approximates commits)."""
        return self.report.summary.event_breakdown.get("PushEvent", 0)

    @cached_property
    def total_pull_requests(self) -> int:
        """Total number of pull request events."""
        return self.report.summary.event_breakdown.get("PullRequestEvent", 0)

    @cached_property
    def total_issues(self) -> int:
        """Total number of issue events."""
        return self.report.summary.event_breakdown.get("IssuesEvent", 0)

    @cached_property
    def analysis_period_days(self) -> int:
        """Number of days covered by the analysis."""
        return (self.end_date - self.start_date).days

    @cached_property
    def most_active_repository(self) -> str | None:
        """Name of the most active repository."""
        return self.report.summary.most_active_repository

    @cached_property
    def most_common_event_type(self) -> str | None:
        """Most common type of GitHub event."""
        return self.report.summary.most_common_event_type

    @cached_property
    def daily_activity(self) -> list[dict[str, Any]]:
        """Daily activity breakdown as list of dictionaries."""
        return [
            {
                "date": rollup.date,
                "events": rollup.events,
                "repositories": rollup.repositories,
                "commit_count": rollup.event_types.get("PushEvent", 0),
                "pr_count": rollup.event_types.get("PullRequestEvent", 0),
                "issue_count": rollup.event_types.get("IssuesEvent", 0),
            }
            for rollup in self.report.daily_rollups
        ]

    @cached_property
    def repository_activity(self) -> dict[str, dict[str, Any]]:
        """Repository activity breakdown."""
        return {
            repo_name: {
                "events": breakdown.events,
                "event_types": breakdown.event_types,
                "last_activity": breakdown.last_activity,
                "first_activity": breakdown.first_activity,
                "commits": breakdown.event_types.get("PushEvent", 0),
                "pull_requests": breakdown.event_types.get("PullRequestEvent", 0),
                "issues": breakdown.event_types.get("IssuesEvent", 0),
            }
            for repo_name, breakdown in self.report.repository_breakdown.items()
        }

    @property
    def event_breakdown(self) -> dict[str, int]:
        """Breakdown of events by type."""
        return self.report.summary.event_breakdown

    @property
    def performance_metrics(self) -> dict[str, Any]:
        """Analysis performance metrics."""
        metrics = {
            "analysis_strategy": self.analysis_strategy,
            "total_api_calls": self.total_api_calls,
            "repositories_discovered": self.repositories_discovered,
        }

        if self.execution_time_ms:
            metrics["execution_time_seconds"] = self.execution_time_ms / 1000

        return metrics

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "username": self.username,
            "analysis_period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
                "days": self.analysis_period_days,
            },
            "summary": {
                "total_events": self.total_events,
                "total_repositories": self.total_repositories,
                "total_commits": self.total_commits,
                "total_pull_requests": self.total_pull_requests,
                "total_issues": self.total_issues,
                "most_active_repository": self.most_active_repository,
                "most_common_event_type": self.most_common_event_type,
            },
            "daily_activity": self.daily_activity,
            "repository_activity": self.repository_activity,
            "event_breakdown": self.event_breakdown,
            "performance_metrics": self.performance_metrics,
            "raw_report": self.report.model_dump(),
        }

    def to_json(self, **kwargs: Any) -> str:
        """Convert result to JSON string with proper datetime handling."""
        import json

        def json_serializer(obj: Any) -> str:
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)

        return json.dumps(self.to_dict(), indent=2, default=json_serializer, **kwargs)


class GitHubAnalyzer:
    """Main entry point for GitHub activity analysis.

    This is the primary class that external projects should use to analyze
    GitHub user activity.
    """

    def __init__(self, token: str) -> None:
        """Initialize analyzer with GitHub token.

        Args:
            token: GitHub Personal Access Token

        Raises:
            InvalidTokenError: If token format is invalid
        """
        if not token or not token.strip():
            raise InvalidTokenError("GitHub token cannot be empty")

        if not (token.startswith("ghp_") or token.startswith("github_pat_")):
            raise InvalidTokenError("Invalid GitHub token format")

        self.token = token.strip()

    def analyze_user(
        self,
        username: str,
        config: AnalysisConfig | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AnalysisResult:
        """Analyze GitHub user activity.

        Args:
            username: GitHub username to analyze
            config: Analysis configuration (uses defaults if None)
            progress_callback: Optional progress callback

        Returns:
            Rich analysis result object

        Raises:
            UserNotFoundError: If username doesn't exist
            InvalidTokenError: If token is invalid or expired
            RateLimitError: If API rate limit is exceeded
            ConfigurationError: If configuration is invalid
        """
        if not username or not username.strip():
            raise UserNotFoundError("Username cannot be empty")

        if config is None:
            config = AnalysisConfig()

        self.validate_config(config)

        # TODO: This will be implemented after we have the basic structure
        # For now, return a placeholder
        raise NotImplementedError("Analysis implementation coming in next milestone")

    def validate_config(self, config: AnalysisConfig) -> None:
        """Validate configuration against GitHub API constraints.

        Args:
            config: Configuration to validate

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Re-validate at analysis time in case constraints changed
            if config.days > 90:
                raise ConfigurationError(
                    "Cannot analyze more than 90 days due to GitHub API limitations"
                )

            if config.ai_token_budget and config.ai_token_budget > 1000:
                raise ConfigurationError("Token budget too high, risk of rate limiting")

            if config.max_events and config.max_events > 10000:
                raise ConfigurationError("max_events too high, may cause memory issues")

        except ValueError as e:
            raise ConfigurationError(f"Invalid configuration: {e}") from e

    def dry_run(
        self, username: str, config: AnalysisConfig | None = None
    ) -> dict[str, Any]:
        """Validate configuration and estimate API usage without running analysis.

        Args:
            username: GitHub username to analyze
            config: Analysis configuration (uses defaults if None)

        Returns:
            Dictionary with validation results and estimated costs

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if config is None:
            config = AnalysisConfig()

        self.validate_config(config)

        # Estimate API calls based on strategy
        if config.strategy == AnalysisStrategy.INTELLIGENCE_GUIDED:
            estimated_calls = 15 + (config.max_repos or 10) * 2
        elif config.strategy == AnalysisStrategy.MULTI_SOURCE:
            estimated_calls = 60 + (config.max_repos or 25) * 3
        else:  # AUTO
            estimated_calls = 30  # Conservative estimate

        return {
            "valid": True,
            "estimated_api_calls": estimated_calls,
            "estimated_duration_seconds": estimated_calls * 0.5,  # Rough estimate
            "strategy": config.strategy.value,
            "days_analyzed": config.days,
            "warnings": self._get_config_warnings(config),
        }

    def _get_config_warnings(self, config: AnalysisConfig) -> list[str]:
        """Get warnings about potentially problematic configuration."""
        warnings = []

        if config.days > 30:
            warnings.append("Analyzing >30 days may hit GitHub's event history limits")

        if config.max_events and config.max_events > 5000:
            warnings.append("High max_events may cause slow processing")

        if config.ai_token_budget and config.ai_token_budget > 500:
            warnings.append("High AI token budget increases analysis cost")

        return warnings


# Convenience functions for common use cases
def quick_analysis(username: str, token: str, days: int = 7) -> AnalysisResult:
    """Perform a quick analysis with default settings.

    Args:
        username: GitHub username to analyze
        token: GitHub Personal Access Token
        days: Number of days to analyze

    Returns:
        Analysis result
    """
    analyzer = GitHubAnalyzer(token)
    config = AnalysisConfig.builder().days(days).build()
    return analyzer.analyze_user(username, config)


def comprehensive_analysis(username: str, token: str, days: int = 30) -> AnalysisResult:
    """Perform comprehensive analysis with multi-source discovery.

    Args:
        username: GitHub username to analyze
        token: GitHub Personal Access Token
        days: Number of days to analyze

    Returns:
        Analysis result
    """
    analyzer = GitHubAnalyzer(token)
    config = AnalysisConfig.builder().days(days).comprehensive().build()
    return analyzer.analyze_user(username, config)
