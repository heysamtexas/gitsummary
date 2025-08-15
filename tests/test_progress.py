"""Tests for the progress callback system."""


from git_summary.progress import (
    ProgressEvent,
    ProgressEventType,
    ProgressNotifier,
    create_no_op_callback,
)


class TestProgressEvent:
    """Test the ProgressEvent data class."""

    def test_basic_event_creation(self):
        """Test creating a basic progress event."""
        event = ProgressEvent(
            event_type=ProgressEventType.STARTED, message="Test message"
        )

        assert event.event_type == ProgressEventType.STARTED
        assert event.message == "Test message"
        assert event.current is None
        assert event.total is None
        assert event.metadata is None

    def test_event_with_progress_values(self):
        """Test creating an event with progress values."""
        event = ProgressEvent(
            event_type=ProgressEventType.PROGRESS,
            message="Processing...",
            current=50,
            total=100,
        )

        assert event.current == 50
        assert event.total == 100
        assert event.progress_percentage == 50.0

    def test_event_with_metadata(self):
        """Test creating an event with metadata."""
        metadata = {"repo": "test/repo", "strategy": "intelligence_guided"}
        event = ProgressEvent(
            event_type=ProgressEventType.INFO,
            message="Strategy selected",
            metadata=metadata,
        )

        assert event.metadata == metadata

    def test_progress_percentage_calculation(self):
        """Test progress percentage calculations."""
        # Test normal case
        event = ProgressEvent(ProgressEventType.PROGRESS, "test", current=25, total=100)
        assert event.progress_percentage == 25.0

        # Test edge cases
        event_no_total = ProgressEvent(ProgressEventType.PROGRESS, "test", current=25)
        assert event_no_total.progress_percentage is None

        event_zero_total = ProgressEvent(
            ProgressEventType.PROGRESS, "test", current=25, total=0
        )
        assert event_zero_total.progress_percentage is None


class TestProgressNotifier:
    """Test the ProgressNotifier helper class."""

    def test_notifier_without_callback(self):
        """Test that notifier works without a callback."""
        notifier = ProgressNotifier()

        # Should not raise an exception
        notifier.started("Test started")
        notifier.progress("Test progress", 1, 10)
        notifier.completed("Test completed")

    def test_notifier_with_callback(self):
        """Test that notifier calls the callback correctly."""
        events = []

        def test_callback(event: ProgressEvent) -> None:
            events.append(event)

        notifier = ProgressNotifier(test_callback)

        notifier.started("Test started")
        notifier.progress("Test progress", 5, 10)
        notifier.completed("Test completed")
        notifier.error("Test error")
        notifier.info("Test info")

        assert len(events) == 5

        # Check event types
        assert events[0].event_type == ProgressEventType.STARTED
        assert events[1].event_type == ProgressEventType.PROGRESS
        assert events[2].event_type == ProgressEventType.COMPLETED
        assert events[3].event_type == ProgressEventType.ERROR
        assert events[4].event_type == ProgressEventType.INFO

        # Check progress event details
        progress_event = events[1]
        assert progress_event.current == 5
        assert progress_event.total == 10
        assert progress_event.progress_percentage == 50.0

    def test_notifier_with_metadata(self):
        """Test that notifier correctly passes metadata."""
        events = []

        def test_callback(event: ProgressEvent) -> None:
            events.append(event)

        notifier = ProgressNotifier(test_callback)

        notifier.info("Test with metadata", extra_info="test_value", count=42)

        assert len(events) == 1
        event = events[0]
        assert event.metadata == {"extra_info": "test_value", "count": 42}


class TestNoOpCallback:
    """Test the no-op callback utility."""

    def test_no_op_callback(self):
        """Test that no-op callback doesn't raise exceptions."""
        callback = create_no_op_callback()

        event = ProgressEvent(ProgressEventType.STARTED, "test")

        # Should not raise any exception
        callback(event)
