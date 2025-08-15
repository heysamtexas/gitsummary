"""Tests for the library API."""

from datetime import UTC, datetime

import pytest

from git_summary.library import (
    AnalysisConfig,
    AnalysisResult,
    AnalysisStrategy,
    ConfigurationError,
    GitHubAnalyzer,
    InvalidTokenError,
    comprehensive_analysis,
    quick_analysis,
)
from git_summary.models import (
    ActivityPeriod,
    ActivitySummary,
    GitHubActivityReport,
)


class TestAnalysisStrategy:
    """Test the AnalysisStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert AnalysisStrategy.AUTO.value == "auto"
        assert AnalysisStrategy.INTELLIGENCE_GUIDED.value == "intelligence_guided"
        assert AnalysisStrategy.MULTI_SOURCE.value == "multi_source"


class TestAnalysisConfig:
    """Test the AnalysisConfig class."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = AnalysisConfig()

        assert config.days == 7
        assert config.max_events is None
        assert config.strategy == AnalysisStrategy.AUTO
        assert config.max_repos is None
        assert config.include_events is None
        assert config.exclude_repos is None
        assert config.ai_model is None
        assert config.ai_persona is None
        assert config.ai_token_budget == 200
        assert config.repo_filter is None

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = AnalysisConfig(
            days=30,
            max_events=1000,
            strategy=AnalysisStrategy.MULTI_SOURCE,
            max_repos=10,
            include_events=["PushEvent", "PullRequestEvent"],
            ai_model="gpt-4",
            ai_persona="data analyst",
            ai_token_budget=500,
        )

        assert config.days == 30
        assert config.max_events == 1000
        assert config.strategy == AnalysisStrategy.MULTI_SOURCE
        assert config.max_repos == 10
        assert config.include_events == ["PushEvent", "PullRequestEvent"]
        assert config.ai_model == "gpt-4"
        assert config.ai_persona == "data analyst"
        assert config.ai_token_budget == 500

    def test_validation_days(self):
        """Test validation of days parameter."""
        # Valid values
        AnalysisConfig(days=1)
        AnalysisConfig(days=90)

        # Invalid values
        with pytest.raises(ValueError, match="days must be at least 1"):
            AnalysisConfig(days=0)

        with pytest.raises(ValueError, match="days cannot exceed 90"):
            AnalysisConfig(days=91)

    def test_validation_max_events(self):
        """Test validation of max_events parameter."""
        # Valid values
        AnalysisConfig(max_events=None)
        AnalysisConfig(max_events=1)
        AnalysisConfig(max_events=10000)

        # Invalid values
        with pytest.raises(ValueError, match="max_events must be at least 1"):
            AnalysisConfig(max_events=0)

    def test_validation_token_budget(self):
        """Test validation of AI token budget."""
        # Valid values
        AnalysisConfig(ai_token_budget=50)
        AnalysisConfig(ai_token_budget=1000)

        # Invalid values
        with pytest.raises(ValueError, match="ai_token_budget must be at least 50"):
            AnalysisConfig(ai_token_budget=49)

        with pytest.raises(ValueError, match="ai_token_budget cannot exceed 1000"):
            AnalysisConfig(ai_token_budget=1001)

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = AnalysisConfig(
            days=14,
            strategy=AnalysisStrategy.MULTI_SOURCE,
            include_events=["PushEvent"],
        )

        result = config.to_dict()

        assert result["days"] == 14
        assert result["strategy"] == AnalysisStrategy.MULTI_SOURCE
        assert result["include_events"] == ["PushEvent"]
        # None values should be excluded
        assert "max_events" not in result


class TestAnalysisConfigBuilder:
    """Test the AnalysisConfigBuilder class."""

    def test_builder_pattern(self):
        """Test fluent builder interface."""
        config = (
            AnalysisConfig.builder()
            .days(21)
            .max_events(500)
            .intelligence_guided()
            .max_repos(15)
            .include_events(["PushEvent", "PullRequestEvent"])
            .exclude_repos(["test/repo"])
            .repo_filter(["important/repo"])
            .ai_analysis("gpt-4", "tech lead", 600)
            .build()
        )

        assert config.days == 21
        assert config.max_events == 500
        assert config.strategy == AnalysisStrategy.INTELLIGENCE_GUIDED
        assert config.max_repos == 15
        assert config.include_events == ["PushEvent", "PullRequestEvent"]
        assert config.exclude_repos == ["test/repo"]
        assert config.repo_filter == ["important/repo"]
        assert config.ai_model == "gpt-4"
        assert config.ai_persona == "tech lead"
        assert config.ai_token_budget == 600

    def test_strategy_shortcuts(self):
        """Test strategy shortcut methods."""
        # Intelligence guided shortcut
        config1 = AnalysisConfig.builder().intelligence_guided().build()
        assert config1.strategy == AnalysisStrategy.INTELLIGENCE_GUIDED

        # Comprehensive shortcut
        config2 = AnalysisConfig.builder().comprehensive().build()
        assert config2.strategy == AnalysisStrategy.MULTI_SOURCE

        # Strategy by value
        config3 = AnalysisConfig.builder().strategy("auto").build()
        assert config3.strategy == AnalysisStrategy.AUTO

    def test_ai_analysis_defaults(self):
        """Test AI analysis configuration with defaults."""
        config = AnalysisConfig.builder().ai_analysis().build()

        assert config.ai_model == "anthropic/claude-3-7-sonnet-latest"
        assert config.ai_persona == "tech analyst"
        assert config.ai_token_budget == 200

    def test_empty_builder(self):
        """Test building with no customizations."""
        config = AnalysisConfig.builder().build()

        # Should have same defaults as AnalysisConfig()
        default_config = AnalysisConfig()
        assert config.days == default_config.days
        assert config.strategy == default_config.strategy


class TestAnalysisResult:
    """Test the AnalysisResult class."""

    def create_sample_result(self) -> AnalysisResult:
        """Create a sample result for testing."""
        # Create sample report data
        summary = ActivitySummary(
            total_events=100,
            repositories_active=5,
            event_breakdown={
                "PushEvent": 60,
                "PullRequestEvent": 25,
                "IssuesEvent": 15,
            },
            most_active_repository="user/main-repo",
            most_common_event_type="PushEvent",
        )

        period = ActivityPeriod(
            start="2024-01-01T00:00:00Z",
            end="2024-01-08T00:00:00Z",
        )

        report = GitHubActivityReport(
            user="testuser",
            period=period,
            summary=summary,
            daily_rollups=[],
            repository_breakdown={},
            detailed_events=[],
        )

        return AnalysisResult(
            username="testuser",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 8, tzinfo=UTC),
            analysis_strategy="intelligence_guided",
            execution_time_ms=5000,
            report=report,
            total_api_calls=25,
            repositories_discovered=5,
        )

    def test_computed_properties(self):
        """Test computed properties of AnalysisResult."""
        result = self.create_sample_result()

        assert result.total_events == 100
        assert result.total_repositories == 5
        assert result.total_commits == 60
        assert result.total_pull_requests == 25
        assert result.total_issues == 15
        assert result.analysis_period_days == 7
        assert result.most_active_repository == "user/main-repo"
        assert result.most_common_event_type == "PushEvent"

    def test_event_breakdown(self):
        """Test event breakdown property."""
        result = self.create_sample_result()

        breakdown = result.event_breakdown
        assert breakdown["PushEvent"] == 60
        assert breakdown["PullRequestEvent"] == 25
        assert breakdown["IssuesEvent"] == 15

    def test_performance_metrics(self):
        """Test performance metrics property."""
        result = self.create_sample_result()

        metrics = result.performance_metrics
        assert metrics["analysis_strategy"] == "intelligence_guided"
        assert metrics["total_api_calls"] == 25
        assert metrics["repositories_discovered"] == 5
        assert metrics["execution_time_seconds"] == 5.0

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = self.create_sample_result()

        data = result.to_dict()

        assert data["username"] == "testuser"
        assert data["analysis_period"]["days"] == 7
        assert data["summary"]["total_events"] == 100
        assert data["summary"]["total_commits"] == 60
        assert "performance_metrics" in data
        assert "raw_report" in data

    def test_to_json(self):
        """Test converting result to JSON."""
        result = self.create_sample_result()

        json_str = result.to_json()

        # Should be valid JSON
        import json

        data = json.loads(json_str)

        assert data["username"] == "testuser"
        assert data["summary"]["total_events"] == 100


class TestGitHubAnalyzer:
    """Test the GitHubAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        token = "ghp_test123"
        analyzer = GitHubAnalyzer(token)

        assert analyzer.token == token

    def test_analyzer_invalid_token(self):
        """Test analyzer with invalid token."""
        with pytest.raises(InvalidTokenError):
            GitHubAnalyzer("")

        with pytest.raises(InvalidTokenError):
            GitHubAnalyzer("invalid_token_format")

    def test_analyzer_dry_run(self):
        """Test analyzer dry run functionality."""
        analyzer = GitHubAnalyzer("ghp_test123")
        config = AnalysisConfig.builder().days(7).intelligence_guided().build()

        result = analyzer.dry_run("testuser", config)

        assert result["valid"] is True
        assert "estimated_api_calls" in result
        assert "strategy" in result
        assert result["strategy"] == "intelligence_guided"

    def test_analyze_user_not_implemented(self):
        """Test that analyze_user raises NotImplementedError."""
        analyzer = GitHubAnalyzer("ghp_test_token")

        with pytest.raises(NotImplementedError):
            analyzer.analyze_user("testuser")

    def test_validate_config(self):
        """Test configuration validation."""
        analyzer = GitHubAnalyzer("ghp_test_token")

        # Valid config should not raise
        valid_config = AnalysisConfig(days=30, ai_token_budget=500)
        analyzer.validate_config(valid_config)

        # Invalid config should raise ConfigurationError when validating
        # (since Pydantic validation happens at construction time)
        with pytest.raises(ConfigurationError):
            # Create a config that passes Pydantic validation but fails business logic
            invalid_config = AnalysisConfig(
                days=89, max_events=15000
            )  # max_events too high
            analyzer.validate_config(invalid_config)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_quick_analysis_not_implemented(self):
        """Test quick_analysis raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            quick_analysis("testuser", "ghp_test_token")

    def test_comprehensive_analysis_not_implemented(self):
        """Test comprehensive_analysis raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            comprehensive_analysis("testuser", "ghp_test_token")
