"""Test the public library API imports and basic functionality."""

import pytest


class TestPublicAPIImports:
    """Test that the public API can be imported correctly."""

    def test_basic_import(self):
        """Test basic package import."""
        import git_summary

        assert hasattr(git_summary, "__version__")
        assert hasattr(git_summary, "__all__")

    def test_core_classes_import(self):
        """Test core library classes can be imported."""
        from git_summary import (
            AnalysisConfig,
            AnalysisStrategy,
            GitHubAnalyzer,
        )

        # Should be able to create instances
        analyzer = GitHubAnalyzer("ghp_test_token")
        assert analyzer.token == "ghp_test_token"

        config = AnalysisConfig()
        assert config.days == 7  # Default value

        assert AnalysisStrategy.AUTO is not None

    def test_builder_pattern_import(self):
        """Test builder pattern works through public API."""
        from git_summary import AnalysisConfig, AnalysisStrategy

        config = AnalysisConfig.builder().days(14).intelligence_guided().build()

        assert config.days == 14
        assert config.strategy == AnalysisStrategy.INTELLIGENCE_GUIDED

    def test_exception_hierarchy_import(self):
        """Test exception classes can be imported and used."""
        from git_summary import (
            ConfigurationError,
            GitHubAnalysisError,
            InvalidTokenError,
            RateLimitError,
            UserNotFoundError,
        )

        # Test inheritance hierarchy
        assert issubclass(InvalidTokenError, GitHubAnalysisError)
        assert issubclass(UserNotFoundError, GitHubAnalysisError)
        assert issubclass(RateLimitError, GitHubAnalysisError)
        assert issubclass(ConfigurationError, GitHubAnalysisError)

        # Test exception creation
        error = InvalidTokenError("test message")
        assert str(error) == "test message"

    def test_convenience_functions_import(self):
        """Test convenience functions can be imported."""
        from git_summary import comprehensive_analysis, quick_analysis

        # Should be callable (even if not implemented)
        assert callable(quick_analysis)
        assert callable(comprehensive_analysis)

        # Should raise NotImplementedError for now
        with pytest.raises(NotImplementedError):
            quick_analysis("testuser", "ghp_test_token")

        with pytest.raises(NotImplementedError):
            comprehensive_analysis("testuser", "ghp_test_token")

    def test_progress_callback_import(self):
        """Test ProgressCallback protocol can be imported."""
        from git_summary import ProgressCallback

        # Should be a protocol/type
        assert ProgressCallback is not None

        # Should be able to create a function that matches the protocol
        def progress_handler(current: int, total: int, message: str) -> None:
            pass

        # Type checker should accept this (runtime test)
        handler: ProgressCallback = progress_handler
        assert callable(handler)

    def test_all_exports_defined(self):
        """Test that all items in __all__ are actually available."""
        import git_summary

        for export_name in git_summary.__all__:
            assert hasattr(git_summary, export_name), f"Missing export: {export_name}"

    def test_api_documentation_examples(self):
        """Test the examples from the module docstring work."""
        from git_summary import AnalysisConfig, GitHubAnalyzer

        # Example 1: Quick analysis
        analyzer = GitHubAnalyzer(token="ghp_test_token")
        assert analyzer.token == "ghp_test_token"  # Use the analyzer
        # Note: analyze_user would raise NotImplementedError

        # Example 2: Custom configuration
        config = AnalysisConfig.builder().days(30).comprehensive().build()
        assert config.days == 30

        # Examples should be syntactically correct
        from git_summary import comprehensive_analysis, quick_analysis

        assert callable(quick_analysis)
        assert callable(comprehensive_analysis)


class TestPublicAPIUsability:
    """Test that the public API is easy to use correctly."""

    def test_analyzer_token_validation(self):
        """Test that GitHubAnalyzer validates tokens properly."""
        from git_summary import GitHubAnalyzer, InvalidTokenError

        # Valid token format should work
        analyzer = GitHubAnalyzer("ghp_valid_token_format")
        assert analyzer.token == "ghp_valid_token_format"

        # Invalid token format should raise clear error
        with pytest.raises(InvalidTokenError, match="GitHub token cannot be empty"):
            GitHubAnalyzer("")

        with pytest.raises(InvalidTokenError, match="Invalid GitHub token format"):
            GitHubAnalyzer("invalid_format")

    def test_config_validation_through_api(self):
        """Test that configuration validation works through public API."""
        from git_summary import AnalysisConfig

        # Valid configurations should work
        config = AnalysisConfig(days=30, ai_token_budget=500)
        assert config.days == 30

        # Invalid configurations should raise validation errors
        with pytest.raises(ValueError, match="days cannot exceed 90"):
            AnalysisConfig(days=100)

        with pytest.raises(ValueError, match="ai_token_budget cannot exceed 1000"):
            AnalysisConfig(ai_token_budget=2000)

    def test_dry_run_functionality(self):
        """Test dry-run functionality through public API."""
        from git_summary import AnalysisConfig, GitHubAnalyzer

        analyzer = GitHubAnalyzer("ghp_test_token")
        config = AnalysisConfig.builder().days(7).intelligence_guided().build()

        result = analyzer.dry_run("testuser", config)

        assert result["valid"] is True
        assert "estimated_api_calls" in result
        assert "strategy" in result
        assert result["strategy"] == "intelligence_guided"
