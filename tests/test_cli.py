"""Tests for the CLI module."""

from typer.testing import CliRunner

from git_summary.cli import app

runner = CliRunner()


def test_version():
    """Test the version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "git-summary" in result.stdout


def test_summary_with_token():
    """Test summary command with token."""
    result = runner.invoke(app, ["summary", "testuser", "--token", "fake_token"])
    assert result.exit_code == 0
    assert "Analyzing GitHub activity" in result.stdout


def test_auth_status():
    """Test auth-status command."""
    result = runner.invoke(app, ["auth-status"])
    assert result.exit_code == 0
    assert "Authentication Status" in result.stdout


def test_summary_with_filtering_placeholders():
    """Test summary command with MVP filtering options (should show warnings)."""
    result = runner.invoke(
        app,
        [
            "summary",
            "testuser",
            "--token",
            "fake_token",
            "--include-events",
            "PushEvent",
            "--exclude-repos",
            "test-repo",
        ],
    )
    assert result.exit_code == 0
    assert "not implemented in this MVP" in result.stdout
