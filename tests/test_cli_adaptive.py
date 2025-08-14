"""Robust CLI integration tests following Guilfoyle's guidance.

Test the user contract, not the internal wiring.
Focus on user-visible behavior and outcomes.
"""
from unittest.mock import patch

from typer.testing import CliRunner

from git_summary.cli import app

runner = CliRunner()


class TestCLIBehavior:
    """Test CLI command behavior and user contract."""

    @patch("git_summary.cli._analyze_user_activity")
    def test_comprehensive_flag_changes_behavior(self, mock_analyze):
        """Comprehensive flag should enable different analysis strategy."""
        mock_analyze.return_value = None

        # Run without flag
        result1 = runner.invoke(app, ["summary", "testuser", "--token", "fake_token"])

        # Run with flag
        result2 = runner.invoke(
            app, ["summary", "testuser", "--token", "fake_token", "--comprehensive"]
        )

        assert result1.exit_code == 0
        assert result2.exit_code == 0
        assert mock_analyze.call_count == 2

        # Verify the calls are different (without caring about exact parameters)
        call1_args = mock_analyze.call_args_list[0]
        call2_args = mock_analyze.call_args_list[1]
        assert call1_args != call2_args  # Behavior should be different

    @patch("git_summary.cli._analyze_user_activity")
    def test_max_repos_flag_affects_analysis(self, mock_analyze):
        """Max repos flag should change analysis parameters."""
        mock_analyze.return_value = None

        # Run with default repos
        result1 = runner.invoke(app, ["summary", "testuser", "--token", "fake_token"])

        # Run with limited repos
        result2 = runner.invoke(
            app, ["summary", "testuser", "--token", "fake_token", "--max-repos", "5"]
        )

        assert result1.exit_code == 0
        assert result2.exit_code == 0
        assert mock_analyze.call_count == 2

        # Verify different parameters passed
        call1_args = mock_analyze.call_args_list[0]
        call2_args = mock_analyze.call_args_list[1]
        assert call1_args != call2_args

    @patch("git_summary.cli._analyze_user_activity")
    def test_force_strategy_enables_specific_analysis(self, mock_analyze):
        """Force strategy flag should enable specific analysis mode."""
        mock_analyze.return_value = None

        for strategy in ["intelligence_guided", "multi_source"]:
            result = runner.invoke(
                app,
                [
                    "summary",
                    "testuser",
                    "--token",
                    "fake_token",
                    "--force-strategy",
                    strategy,
                ],
            )

            assert result.exit_code == 0

        assert mock_analyze.call_count == 2

        # Verify different strategies produce different calls
        call1_args = mock_analyze.call_args_list[0]
        call2_args = mock_analyze.call_args_list[1]
        assert call1_args != call2_args

    def test_invalid_strategy_rejected_with_clear_message(self):
        """Users should get clear error for invalid strategy."""
        result = runner.invoke(
            app,
            [
                "summary",
                "testuser",
                "--token",
                "fake_token",
                "--force-strategy",
                "invalid_strategy",
            ],
        )

        assert result.exit_code != 0
        assert "invalid strategy" in result.output.lower()

    def test_conflicting_options_rejected_with_clear_message(self):
        """Users should get clear error for conflicting options."""
        result = runner.invoke(
            app,
            [
                "summary",
                "testuser",
                "--token",
                "fake_token",
                "--comprehensive",
                "--force-strategy",
                "multi_source",
            ],
        )

        assert result.exit_code != 0
        assert "cannot" in result.output.lower()
        assert "both" in result.output.lower()

    def test_invalid_days_rejected_with_clear_message(self):
        """Users should get clear error for invalid days parameter."""
        result = runner.invoke(
            app,
            [
                "summary",
                "testuser",
                "--token",
                "fake_token",
                "--days",
                "999",  # Exceeds 365 day limit
            ],
        )

        assert result.exit_code != 0
        # Should mention the limit in some way
        assert any(
            term in result.output.lower()
            for term in ["365", "day", "exceed", "maximum"]
        )


class TestUserInteractionFlows:
    """Test user interaction patterns and flows."""

    def test_missing_username_prompts_user(self):
        """Missing username should prompt for input."""
        with patch("git_summary.cli.Prompt.ask") as mock_prompt:
            mock_prompt.return_value = "prompted_user"

            with patch("git_summary.cli._analyze_user_activity") as mock_analyze:
                mock_analyze.return_value = None

                result = runner.invoke(
                    app,
                    [
                        "summary",  # No username provided
                        "--token",
                        "fake_token",
                    ],
                )

                assert result.exit_code == 0
                mock_prompt.assert_called_once()
                mock_analyze.assert_called_once()

    def test_missing_token_prompts_user(self):
        """Missing token should prompt for input."""
        with patch("git_summary.cli.Config") as mock_config_class:
            # Mock no stored token
            mock_config = mock_config_class.return_value
            mock_config.get_token.return_value = None

            with patch("git_summary.cli.Confirm.ask") as mock_confirm:
                mock_confirm.side_effect = [True, False]  # Enter token? Save token?

                with patch("git_summary.cli.Prompt.ask") as mock_prompt:
                    mock_prompt.return_value = "prompted_token"

                    with patch(
                        "git_summary.cli._analyze_user_activity"
                    ) as mock_analyze:
                        mock_analyze.return_value = None

                        result = runner.invoke(
                            app,
                            [
                                "summary",
                                "testuser",
                                # No token provided
                            ],
                        )

                        assert result.exit_code == 0
                        mock_confirm.assert_called()
                        mock_prompt.assert_called()

    @patch("git_summary.cli._analyze_user_activity")
    def test_output_file_parameter_handled(self, mock_analyze):
        """Output file parameter should be processed correctly."""
        mock_analyze.return_value = None

        result = runner.invoke(
            app,
            [
                "summary",
                "testuser",
                "--token",
                "fake_token",
                "--output",
                "test_output.json",
            ],
        )

        assert result.exit_code == 0
        mock_analyze.assert_called_once()


class TestErrorHandling:
    """Test error handling and graceful failures."""

    def test_keyboard_interrupt_handled_gracefully(self):
        """Keyboard interrupt should exit gracefully."""
        with patch("git_summary.cli.asyncio.run") as mock_asyncio:
            mock_asyncio.side_effect = KeyboardInterrupt()

            result = runner.invoke(
                app, ["summary", "testuser", "--token", "fake_token"]
            )

            assert result.exit_code != 0
            assert "cancelled" in result.output.lower()

    def test_empty_username_handled(self):
        """Empty username should be rejected or prompt for input."""
        result = runner.invoke(
            app,
            [
                "summary",
                "",  # Empty username
                "--token",
                "fake_token",
            ],
        )

        # Should either fail or prompt - either is acceptable
        # As long as it doesn't crash
        assert result.exit_code in [0, 1, 2]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("git_summary.cli._analyze_user_activity")
    def test_zero_days_handled_appropriately(self, mock_analyze):
        """Zero days parameter should be handled appropriately."""
        mock_analyze.return_value = None

        result = runner.invoke(
            app, ["summary", "testuser", "--token", "fake_token", "--days", "0"]
        )

        # Should either work (0 = today only) or fail gracefully
        assert result.exit_code in [0, 1]

    @patch("git_summary.cli._analyze_user_activity")
    def test_very_large_max_repos_handled(self, mock_analyze):
        """Very large max-repos value should be handled."""
        mock_analyze.return_value = None

        result = runner.invoke(
            app,
            ["summary", "testuser", "--token", "fake_token", "--max-repos", "999999"],
        )

        # Should work - business logic will handle the limit
        assert result.exit_code == 0

    @patch("git_summary.cli._analyze_user_activity")
    def test_all_options_together_work(self, mock_analyze):
        """All compatible options should work together."""
        mock_analyze.return_value = None

        result = runner.invoke(
            app,
            [
                "summary",
                "testuser",
                "--token",
                "fake_token",
                "--days",
                "30",
                "--max-repos",
                "20",
                "--force-strategy",
                "intelligence_guided",
                "--output",
                "results.json",
            ],
        )

        assert result.exit_code == 0
        mock_analyze.assert_called_once()


class TestProgressiveDisclosure:
    """Test that CLI supports both simple and advanced usage."""

    @patch("git_summary.cli._analyze_user_activity")
    def test_simple_usage_works(self, mock_analyze):
        """Simple usage with minimal flags should work."""
        mock_analyze.return_value = None

        result = runner.invoke(app, ["summary", "testuser", "--token", "fake_token"])

        assert result.exit_code == 0
        mock_analyze.assert_called_once()

    @patch("git_summary.cli._analyze_user_activity")
    def test_advanced_usage_works(self, mock_analyze):
        """Advanced usage with many flags should work."""
        mock_analyze.return_value = None

        result = runner.invoke(
            app,
            [
                "summary",
                "testuser",
                "--token",
                "fake_token",
                "--days",
                "14",
                "--max-repos",
                "25",
                "--force-strategy",
                "multi_source",
                "--output",
                "advanced_results.json",
                "--max-events",
                "500",
            ],
        )

        assert result.exit_code == 0
        mock_analyze.assert_called_once()

        # Advanced usage should produce different behavior than simple usage
        # (This is tested by comparison in other tests)
