"""Progress callback system for git-summary library.

This module provides a clean interface for progress updates that allows
external projects to handle progress according to their own UI frameworks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProgressEventType(Enum):
    """Types of progress events that can be emitted."""

    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    ERROR = "error"
    INFO = "info"


@dataclass
class ProgressEvent:
    """A progress event containing update information.

    Attributes:
        event_type: The type of progress event
        message: Human-readable description of the current progress
        current: Current progress value (optional)
        total: Total expected progress value (optional)
        metadata: Additional context data (optional)
    """

    event_type: ProgressEventType
    message: str
    current: int | None = None
    total: int | None = None
    metadata: dict[str, Any] | None = None

    @property
    def progress_percentage(self) -> float | None:
        """Calculate progress percentage if current and total are available."""
        if self.current is not None and self.total is not None and self.total > 0:
            return (self.current / self.total) * 100
        return None


# Type alias for progress callback functions
ProgressCallback = Callable[[ProgressEvent], None]


class ProgressNotifier:
    """Helper class for emitting progress events."""

    def __init__(self, callback: ProgressCallback | None = None) -> None:
        """Initialize with optional progress callback.

        Args:
            callback: Function to call when progress events occur
        """
        self.callback = callback

    def notify(self, event: ProgressEvent) -> None:
        """Emit a progress event if callback is available.

        Args:
            event: The progress event to emit
        """
        if self.callback:
            self.callback(event)

    def started(self, message: str, **kwargs: Any) -> None:
        """Emit a STARTED progress event.

        Args:
            message: Description of what is starting
            **kwargs: Additional metadata
        """
        self.notify(
            ProgressEvent(
                event_type=ProgressEventType.STARTED,
                message=message,
                metadata=kwargs or None,
            )
        )

    def progress(
        self,
        message: str,
        current: int | None = None,
        total: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Emit a PROGRESS progress event.

        Args:
            message: Description of current progress
            current: Current progress value
            total: Total expected progress value
            **kwargs: Additional metadata
        """
        self.notify(
            ProgressEvent(
                event_type=ProgressEventType.PROGRESS,
                message=message,
                current=current,
                total=total,
                metadata=kwargs or None,
            )
        )

    def completed(self, message: str, **kwargs: Any) -> None:
        """Emit a COMPLETED progress event.

        Args:
            message: Description of what completed
            **kwargs: Additional metadata
        """
        self.notify(
            ProgressEvent(
                event_type=ProgressEventType.COMPLETED,
                message=message,
                metadata=kwargs or None,
            )
        )

    def error(self, message: str, **kwargs: Any) -> None:
        """Emit an ERROR progress event.

        Args:
            message: Description of the error
            **kwargs: Additional metadata (e.g., exception details)
        """
        self.notify(
            ProgressEvent(
                event_type=ProgressEventType.ERROR,
                message=message,
                metadata=kwargs or None,
            )
        )

    def info(self, message: str, **kwargs: Any) -> None:
        """Emit an INFO progress event.

        Args:
            message: Informational message
            **kwargs: Additional metadata
        """
        self.notify(
            ProgressEvent(
                event_type=ProgressEventType.INFO,
                message=message,
                metadata=kwargs or None,
            )
        )


def create_no_op_callback() -> ProgressCallback:
    """Create a no-op progress callback that does nothing.

    Returns:
        A callback function that ignores all progress events
    """

    def no_op_callback(_event: ProgressEvent) -> None:
        pass

    return no_op_callback
