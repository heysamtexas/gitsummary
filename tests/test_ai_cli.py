"""Tests for AI CLI command functionality."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from git_summary.cli import app

runner = CliRunner()


class TestAISummaryCommand:
    """Test the ai-summary CLI command."""

    def test_ai_summary_help(self):
        """Test that ai-summary command shows help."""
        result = runner.invoke(app, ["ai-summary", "--help"])
        assert result.exit_code == 0
        assert "Generate an AI-powered summary of GitHub activity" in result.stdout
        assert "--model" in result.stdout
        assert "--persona" in result.stdout
        assert "--token-budget" in result.stdout
        assert "--estimate-cost" in result.stdout

    @patch("git_summary.cli._generate_ai_summary")
    def test_ai_summary_with_username_and_token(self, mock_generate):
        """Test ai-summary command with username and token provided."""
        mock_generate.return_value = None

        result = runner.invoke(
            app,
            [
                "ai-summary",
                "testuser",
                "--token",
                "fake_token",
                "--model",
                "gpt-4o-mini",
                "--persona",
                "tech analyst",
                "--days",
                "3",
            ],
        )

        assert result.exit_code == 0
        mock_generate.assert_called_once()

        # Check the arguments passed to the async function
        call_args = mock_generate.call_args[0]
        assert call_args[0] == "testuser"  # username
        assert call_args[1] == "fake_token"  # token
        assert call_args[2] == "gpt-4o-mini"  # model
        assert call_args[3] == "tech analyst"  # persona
        assert call_args[4] == 3  # days

    @patch("git_summary.cli._generate_ai_summary")
    def test_ai_summary_with_cost_estimation(self, mock_generate):
        """Test ai-summary command with cost estimation flag."""
        mock_generate.return_value = None

        result = runner.invoke(
            app, ["ai-summary", "testuser", "--token", "fake_token", "--estimate-cost"]
        )

        assert result.exit_code == 0
        mock_generate.assert_called_once()

        # Check that estimate_cost flag is passed correctly
        call_args = mock_generate.call_args[0]
        assert call_args[8] is True  # estimate_cost parameter

    @patch("git_summary.cli._generate_ai_summary")
    def test_ai_summary_with_output_file(self, mock_generate):
        """Test ai-summary command with output file specified."""
        mock_generate.return_value = None

        result = runner.invoke(
            app,
            [
                "ai-summary",
                "testuser",
                "--token",
                "fake_token",
                "--output",
                "summary.md",
            ],
        )

        assert result.exit_code == 0
        mock_generate.assert_called_once()

        # Check that output file is passed correctly
        call_args = mock_generate.call_args[0]
        assert call_args[5] == "summary.md"  # output parameter

    @patch("git_summary.cli._generate_ai_summary")
    def test_ai_summary_with_custom_token_budget(self, mock_generate):
        """Test ai-summary command with custom token budget."""
        mock_generate.return_value = None

        result = runner.invoke(
            app,
            [
                "ai-summary",
                "testuser",
                "--token",
                "fake_token",
                "--token-budget",
                "5000",
            ],
        )

        assert result.exit_code == 0
        mock_generate.assert_called_once()

        # Check that token budget is passed correctly
        call_args = mock_generate.call_args[0]
        assert call_args[7] == 5000  # token_budget parameter

    def test_ai_summary_no_username_interactive_input(self):
        """Test ai-summary command prompts for username when not provided."""
        # This test simulates user canceling the prompt
        result = runner.invoke(app, ["ai-summary"], input="\n")

        # Should fail when no username provided
        assert result.exit_code == 1
        assert "Username is required" in result.stdout

    @patch("git_summary.cli.Config")
    def test_ai_summary_no_token_fails(self, mock_config_class):
        """Test that ai-summary fails gracefully when no token is available."""
        mock_config = Mock()
        mock_config.get_token.return_value = None
        mock_config_class.return_value = mock_config

        result = runner.invoke(
            app,
            ["ai-summary", "testuser"],
            input="n\n",  # Decline to enter token
        )

        assert result.exit_code == 1
        assert "GitHub token is required" in result.stdout

    def test_ai_summary_keyboard_interrupt(self):
        """Test ai-summary handles keyboard interrupt gracefully."""
        with patch("git_summary.cli.asyncio.run") as mock_asyncio:
            mock_asyncio.side_effect = KeyboardInterrupt()

            result = runner.invoke(
                app, ["ai-summary", "testuser", "--token", "fake_token"]
            )

            assert result.exit_code == 1
            assert "Operation cancelled by user" in result.stdout

    def test_ai_summary_general_exception(self):
        """Test ai-summary handles general exceptions gracefully."""
        with patch("git_summary.cli.asyncio.run") as mock_asyncio:
            mock_asyncio.side_effect = Exception("Test error")

            result = runner.invoke(
                app, ["ai-summary", "testuser", "--token", "fake_token"]
            )

            assert result.exit_code == 1
            assert "Error: Test error" in result.stdout


class TestAISummaryHelpers:
    """Test helper functions for AI summary CLI."""

    def test_display_cost_estimate(self):
        """Test cost estimate display function."""
        from git_summary.cli import _display_cost_estimate

        cost_info = {
            "cost_estimate": {
                "total_cost": 0.0042,
                "input_tokens": 1500,
                "output_tokens": 300,
            },
            "context_tokens": 1200,
            "total_events": 25,
            "model": "gpt-4o-mini",
            "event_breakdown": {
                "commits": 10,
                "pull_requests": 5,
                "issues": 8,
                "releases": 2,
                "reviews": 0,
            },
        }

        # Should not raise an exception
        _display_cost_estimate(cost_info)

    def test_display_ai_summary(self):
        """Test AI summary display function."""
        from git_summary.cli import _display_ai_summary

        summary_result = {
            "summary": "This is a test AI-generated summary of GitHub activity.",
            "persona_used": "tech_analyst",
            "model_used": "gpt-4o-mini",
            "metadata": {
                "total_events": 15,
                "tokens_used": 1200,
                "repositories": ["owner/repo1", "owner/repo2"],
                "date_range": (datetime.now(UTC), datetime.now(UTC)),
            },
        }

        # Should not raise an exception
        _display_ai_summary(summary_result)

    def test_save_ai_summary_to_file_json(self, tmp_path):
        """Test saving AI summary to JSON file."""
        from git_summary.cli import _save_ai_summary_to_file

        summary_result = {
            "summary": "Test summary",
            "persona_used": "tech_analyst",
            "model_used": "gpt-4o-mini",
            "metadata": {
                "total_events": 10,
                "tokens_used": 800,
                "repositories": ["test/repo"],
            },
        }

        output_file = tmp_path / "test_summary.json"
        _save_ai_summary_to_file(summary_result, str(output_file))

        assert output_file.exists()

        import json

        with open(output_file) as f:
            saved_data = json.load(f)

        assert saved_data["summary"] == "Test summary"
        assert saved_data["persona_used"] == "tech_analyst"

    def test_save_ai_summary_to_file_markdown(self, tmp_path):
        """Test saving AI summary to Markdown file."""
        from git_summary.cli import _save_ai_summary_to_file

        summary_result = {
            "summary": "Test summary content",
            "persona_used": "tech_analyst",
            "model_used": "gpt-4o-mini",
            "metadata": {
                "total_events": 10,
                "tokens_used": 800,
                "repositories": ["test/repo"],
                "date_range": "2024-01-01 to 2024-01-07",
            },
        }

        output_file = tmp_path / "test_summary.md"
        _save_ai_summary_to_file(summary_result, str(output_file))

        assert output_file.exists()

        content = output_file.read_text()
        assert "# GitHub Activity Summary" in content
        assert "Test summary content" in content
        assert "tech_analyst" in content
        assert "gpt-4o-mini" in content

    def test_save_ai_summary_to_file_auto_extension(self, tmp_path):
        """Test saving AI summary automatically adds .md extension."""
        from git_summary.cli import _save_ai_summary_to_file

        summary_result = {
            "summary": "Test summary",
            "persona_used": "tech_analyst",
            "model_used": "gpt-4o-mini",
            "metadata": {"total_events": 5, "tokens_used": 400, "repositories": []},
        }

        output_file = tmp_path / "test_summary"  # No extension
        _save_ai_summary_to_file(summary_result, str(output_file))

        # Should create .md file
        md_file = tmp_path / "test_summary.md"
        assert md_file.exists()

        content = md_file.read_text()
        assert "# GitHub Activity Summary" in content
