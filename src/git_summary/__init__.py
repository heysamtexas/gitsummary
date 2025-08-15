"""GitHub Activity Tracker - A comprehensive tool for analyzing GitHub user activity.

Library API for external projects:

    from git_summary import GitHubAnalyzer, AnalysisConfig

    # Quick analysis
    analyzer = GitHubAnalyzer(token="ghp_...")
    result = analyzer.analyze_user("username")

    # Custom configuration
    config = AnalysisConfig.builder().days(30).comprehensive().build()
    result = analyzer.analyze_user("username", config)

    # Convenience functions
    from git_summary import quick_analysis, comprehensive_analysis

    result = quick_analysis("username", "ghp_...")
    result = comprehensive_analysis("username", "ghp_...")
"""

__version__ = "0.1.0"

# Main library API - primary entry points for external projects
# For backwards compatibility and internal use
from git_summary.config import Config
from git_summary.library import (
    AnalysisConfig,
    AnalysisResult,
    AnalysisStrategy,
    ConfigurationError,
    # Exception hierarchy
    GitHubAnalysisError,
    # Core classes
    GitHubAnalyzer,
    InvalidTokenError,
    ProgressCallback,
    RateLimitError,
    UserNotFoundError,
    comprehensive_analysis,
    # Convenience functions
    quick_analysis,
)

__all__ = [
    # Core API
    "GitHubAnalyzer",
    "AnalysisConfig",
    "AnalysisResult",
    "AnalysisStrategy",
    "ProgressCallback",
    # Exceptions
    "GitHubAnalysisError",
    "InvalidTokenError",
    "UserNotFoundError",
    "RateLimitError",
    "ConfigurationError",
    # Convenience functions
    "quick_analysis",
    "comprehensive_analysis",
    # Legacy/internal
    "Config",
    # Metadata
    "__version__",
]
