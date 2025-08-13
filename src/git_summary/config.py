"""Configuration management for git-summary."""

import json
from pathlib import Path
from typing import Any

from rich import print


class Config:
    """Manage git-summary configuration and token storage."""

    def __init__(self) -> None:
        """Initialize config with default paths."""
        self.config_dir = Path.home() / ".git-summary"
        self.config_file = self.config_dir / "config.json"
        self._ensure_config_dir()

    def _ensure_config_dir(self) -> None:
        """Create config directory if it doesn't exist."""
        self.config_dir.mkdir(exist_ok=True)
        # Set restrictive permissions on the config directory
        self.config_dir.chmod(0o700)

    def get_token(self) -> str | None:
        """Get stored GitHub token.

        Returns:
            GitHub token if stored, None otherwise
        """
        if not self.config_file.exists():
            return None

        try:
            with self.config_file.open() as f:
                config_data: dict[str, Any] = json.load(f)
                return config_data.get("github_token")
        except (json.JSONDecodeError, OSError):
            return None

    def set_token(self, token: str) -> None:
        """Store GitHub token securely.

        Args:
            token: GitHub Personal Access Token to store
        """
        config_data = self._load_config()
        config_data["github_token"] = token

        with self.config_file.open("w") as f:
            json.dump(config_data, f, indent=2)

        # Set restrictive permissions on the config file
        self.config_file.chmod(0o600)
        print(f"[green]✓[/green] Token stored securely in {self.config_file}")

    def remove_token(self) -> None:
        """Remove stored GitHub token."""
        config_data = self._load_config()
        config_data.pop("github_token", None)

        if config_data:
            with self.config_file.open("w") as f:
                json.dump(config_data, f, indent=2)
        else:
            # Remove empty config file
            self.config_file.unlink(missing_ok=True)

        print("[green]✓[/green] Token removed from local storage")

    def _load_config(self) -> dict[str, Any]:
        """Load existing config or return empty dict."""
        if not self.config_file.exists():
            return {}

        try:
            with self.config_file.open() as f:
                return json.load(f)  # type: ignore[no-any-return]
        except (json.JSONDecodeError, OSError):
            return {}

    def get_config_info(self) -> dict[str, Any]:
        """Get information about current configuration.

        Returns:
            Dictionary with config status information
        """
        has_token = self.get_token() is not None
        config_exists = self.config_file.exists()

        return {
            "config_file": str(self.config_file),
            "config_exists": config_exists,
            "has_token": has_token,
            "config_dir_permissions": oct(self.config_dir.stat().st_mode)[-3:]
            if self.config_dir.exists()
            else None,
            "config_file_permissions": oct(self.config_file.stat().st_mode)[-3:]
            if config_exists
            else None,
        }
