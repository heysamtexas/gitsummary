"""Tests for Config class progress callbacks."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from git_summary.config import Config
from git_summary.progress import ProgressEvent, ProgressEventType


class TestConfigCallbacks:
    """Test Config class progress callback functionality."""

    def test_config_without_callback(self):
        """Test that Config works without a callback."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("git_summary.config.Path.home", return_value=Path(temp_dir)),
        ):
            config = Config()

            # Should not raise an exception
            config.set_token("test_token")
            token = config.get_token()
            assert token == "test_token"

            config.remove_token()
            assert config.get_token() is None

    def test_config_with_callback(self):
        """Test that Config calls progress callbacks correctly."""
        events = []

        def test_callback(event: ProgressEvent) -> None:
            events.append(event)

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("git_summary.config.Path.home", return_value=Path(temp_dir)),
        ):
            config = Config(progress_callback=test_callback)

            # Test token operations
            config.set_token("test_token")
            config.remove_token()

            # Test AI API key operations
            config.set_ai_api_key("openai", "test_key")
            config.remove_ai_api_key("openai")
            config.remove_ai_api_key("nonexistent")  # Should trigger info event

            # Verify callbacks were called
            assert len(events) == 5

            # Check event types
            assert events[0].event_type == ProgressEventType.COMPLETED
            assert "Token stored securely" in events[0].message

            assert events[1].event_type == ProgressEventType.COMPLETED
            assert "Token removed from local storage" in events[1].message

            assert events[2].event_type == ProgressEventType.COMPLETED
            assert "Openai API key stored securely" in events[2].message

            assert events[3].event_type == ProgressEventType.COMPLETED
            assert "Openai API key removed from local storage" in events[3].message

            assert events[4].event_type == ProgressEventType.INFO
            assert "No nonexistent API key found to remove" in events[4].message

    def test_config_maintains_functionality(self):
        """Test that Config maintains all original functionality."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("git_summary.config.Path.home", return_value=Path(temp_dir)),
        ):
            config = Config()

            # Test GitHub token functionality
            assert config.get_token() is None
            config.set_token("gh_token_123")
            assert config.get_token() == "gh_token_123"

            # Test AI API key functionality
            assert config.get_ai_api_key("openai") is None
            config.set_ai_api_key("openai", "sk-123")
            assert config.get_ai_api_key("openai") == "sk-123"

            # Test config info
            info = config.get_config_info()
            assert info["has_token"] is True
            assert info["ai_api_keys"]["openai"] is True
            assert info["ai_api_keys"]["anthropic"] is False

            # Test cleanup
            config.remove_token()
            config.remove_ai_api_key("openai")

            assert config.get_token() is None
            assert config.get_ai_api_key("openai") is None
