"""CLI tests that actually test CLI behavior, not mock behavior.

Following Guilfoyle's philosophy: Mock at boundaries, test behavior.
"""
from unittest.mock import patch

from typer.testing import CliRunner

from git_summary.cli import app

runner = CliRunner()


def test_version():
    """Test the version command works."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "git-summary" in result.stdout


def test_auth_status():
    """Test auth-status command works."""
    result = runner.invoke(app, ["auth-status"])
    assert result.exit_code == 0
    assert "Authentication Status" in result.stdout


@patch("git_summary.cli._analyze_user_activity")
def test_summary_with_token(mock_analyze):
    """Test that CLI invokes the analysis function with correct parameters."""
    # Mock the async analysis function
    mock_analyze.return_value = None

    result = runner.invoke(
        app, ["summary", "testuser", "--token", "fake_token", "--days", "7"]
    )

    # Should not crash and should call analysis
    assert result.exit_code == 0
    mock_analyze.assert_called_once()

    # Verify it was called with the right parameters
    call_args = mock_analyze.call_args[0]
    assert call_args[0] == "testuser"  # username
    assert call_args[1] == "fake_token"  # token
    assert call_args[2] == 7  # days


def test_summary_shows_mvp_warnings():
    """Test that CLI shows MVP warnings for unimplemented features."""
    with patch("git_summary.cli._analyze_user_activity") as mock_analyze:
        mock_analyze.return_value = None

        result = runner.invoke(
            app,
            [
                "summary",
                "testuser",
                "--token",
                "fake_token",
                "--exclude-repos",
                "test-repo",
            ],
        )

        assert result.exit_code == 0
        assert "not implemented in this MVP" in result.stdout
