"""Adaptive Repository Discovery - Main coordinator for intelligent GitHub analysis.

This module implements the main coordinator that automatically selects the optimal
analysis strategy based on user automation patterns, providing a unified interface
for both intelligence-guided and multi-source discovery approaches.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from typing import Any

from git_summary.github_client import GitHubClient
from git_summary.user_profiling import AutomationDetector, UserProfile
from git_summary.intelligence_guided import IntelligenceGuidedAnalyzer
from git_summary.multi_source_discovery import MultiSourceDiscovery
from git_summary.models import BaseGitHubEvent

logger = logging.getLogger(__name__)


@dataclass
class AnalysisStats:
    """Enhanced analysis statistics with comprehensive metrics."""
    strategy: str
    events_processed: int
    execution_time_ms: float
    api_calls_made: int = 0
    rate_limit_remaining: int | None = None
    cache_hits: int = 0
    cache_misses: int = 0
    errors_encountered: list[str] = field(default_factory=list)
    fallback_used: bool = False
    circuit_breaker_triggered: bool = False


class UserActivity:
    """Container for analyzed user activity data."""
    
    def __init__(
        self,
        username: str,
        profile: UserProfile,
        events: list[BaseGitHubEvent],
        analysis_strategy: str,
        analysis_stats: dict[str, Any],
        execution_time_ms: float,
    ):
        """Initialize user activity container.
        
        Args:
            username: GitHub username analyzed
            profile: User classification profile
            events: Deduplicated events from analysis
            analysis_strategy: Strategy used ('intelligence_guided' or 'multi_source')
            analysis_stats: Strategy-specific analysis statistics
            execution_time_ms: Total analysis execution time in milliseconds
        """
        self.username = username
        self.profile = profile
        self.events = events
        self.analysis_strategy = analysis_strategy
        self.analysis_stats = analysis_stats
        self.execution_time_ms = execution_time_ms
        
        # Derived properties
        self.total_events = len(events)
        self.repositories = list({
            event.repo.name for event in events 
            if event.repo and event.repo.name
        })
        self.event_types = list({event.type for event in events})


class AdaptiveRepositoryDiscovery:
    """Main coordinator for adaptive GitHub repository discovery.
    
    Automatically selects the optimal analysis strategy based on user automation
    patterns and provides a unified interface for GitHub activity analysis.
    """
    
    def __init__(self, github_client: GitHubClient):
        """Initialize the adaptive discovery coordinator.
        
        Args:
            github_client: GitHub API client for making requests
        """
        self.github_client = github_client
        self.automation_detector = AutomationDetector(github_client)
        
        # Strategy instances (created on-demand for optimal resource usage)
        self._intelligence_guided = None
        self._multi_source = None
        
        # Performance tracking
        self.analysis_count = 0
        self.strategy_usage = {"intelligence_guided": 0, "multi_source": 0, "fallback": 0, "emergency": 0}
        
        # Circuit breaker for multi-source strategy (Guilfoyle's fix)
        self.multi_source_failures = 0
        self.multi_source_circuit_open = False
        self.circuit_reset_time = 0
        self.circuit_failure_threshold = 3  # Open circuit after 3 failures
        self.circuit_reset_timeout = 300    # Reset circuit after 5 minutes
        
    def _get_intelligence_guided_analyzer(self) -> IntelligenceGuidedAnalyzer:
        """Get or create intelligence-guided analyzer instance."""
        if self._intelligence_guided is None:
            self._intelligence_guided = IntelligenceGuidedAnalyzer(self.github_client)
        return self._intelligence_guided
        
    def _get_multi_source_analyzer(self) -> MultiSourceDiscovery:
        """Get or create multi-source discovery instance."""
        if self._multi_source is None:
            self._multi_source = MultiSourceDiscovery(self.github_client)
        return self._multi_source
    
    async def analyze_user(
        self, 
        username: str, 
        days: int = 30,
        force_strategy: str | None = None
    ) -> UserActivity:
        """Analyze a GitHub user with adaptive strategy selection.
        
        Args:
            username: GitHub username to analyze
            days: Number of days of activity to analyze
            force_strategy: Force specific strategy ('intelligence_guided' or 'multi_source')
                          Mainly for testing and debugging purposes
        
        Returns:
            UserActivity object containing analysis results and metadata
            
        Raises:
            ValueError: If username is invalid or force_strategy is unknown
            RuntimeError: If all analysis strategies fail
        """
        if not username or not username.strip():
            raise ValueError("Username cannot be empty")
            
        if force_strategy and force_strategy not in ["intelligence_guided", "multi_source"]:
            raise ValueError(f"Unknown strategy: {force_strategy}")
        
        start_time = datetime.now(UTC)
        self.analysis_count += 1
        
        logger.info(f"Starting adaptive analysis for user '{username}' over {days} days")
        
        try:
            # Step 1: Profile the user for automation detection
            logger.debug("Step 1: Profiling user for automation patterns")
            profile = await self.automation_detector.classify_user(username, days=min(days, 7))
            
            # Step 2: Select optimal strategy with validation (Guilfoyle's fix)
            strategy = self._select_strategy(profile, force_strategy)
            logger.info(
                f"Selected strategy '{strategy}' for user '{username}'",
                extra={
                    "username": username,
                    "strategy": strategy,
                    "automation_score": getattr(profile, 'confidence_score', 0),
                    "forced": force_strategy is not None,
                    "days": days,
                    "circuit_breaker_open": self.multi_source_circuit_open
                }
            )
            
            # Step 3: Execute analysis with selected strategy
            logger.debug(f"Step 3: Executing {strategy} analysis")
            events, analysis_stats = await self._execute_strategy(strategy, username, days)
            
            # Step 4: Deduplicate events across sources
            logger.debug("Step 4: Deduplicating events")
            deduplicated_events = self._deduplicate_events(events)
            
            # Calculate execution time and update stats
            execution_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            analysis_stats.execution_time_ms = execution_time
            analysis_stats.events_processed = len(deduplicated_events)  # Update with final count
            
            # Update strategy usage statistics (handle emergency fallback)
            final_strategy = analysis_stats.strategy
            if final_strategy in self.strategy_usage:
                self.strategy_usage[final_strategy] += 1
            else:
                self.strategy_usage["emergency"] += 1
            
            # Create result container with enhanced stats
            user_activity = UserActivity(
                username=username,
                profile=profile,
                events=deduplicated_events,
                analysis_strategy=final_strategy,
                analysis_stats=analysis_stats.__dict__,  # Convert to dict for serialization
                execution_time_ms=execution_time
            )
            
            logger.info(
                f"Analysis complete for '{username}': {len(deduplicated_events)} events "
                f"from {len(user_activity.repositories)} repositories using {strategy} "
                f"strategy in {execution_time:.0f}ms"
            )
            
            return user_activity
            
        except Exception as e:
            logger.error(f"Analysis failed for user '{username}': {e}")
            
            # Attempt fallback if not already using fallback
            if force_strategy != "intelligence_guided":
                logger.warning("Attempting fallback to intelligence-guided strategy")
                try:
                    return await self.analyze_user(username, days, force_strategy="intelligence_guided")
                except Exception as fallback_error:
                    logger.error(f"Fallback strategy also failed: {fallback_error}")
                    self.strategy_usage["fallback"] += 1
            
            # If all strategies fail, raise the original error
            raise RuntimeError(f"All analysis strategies failed for user '{username}': {e}") from e
    
    def _select_strategy(self, profile: UserProfile, force_strategy: str | None = None) -> str:
        """Select optimal analysis strategy with proper validation (Guilfoyle's fix).
        
        Args:
            profile: User automation classification profile
            force_strategy: Optional forced strategy (must be validated)
            
        Returns:
            Strategy name ('intelligence_guided' or 'multi_source')
            
        Raises:
            ValueError: If force_strategy is invalid
        """
        # Validate forced strategy (Guilfoyle's critical fix)
        if force_strategy:
            if force_strategy not in {"intelligence_guided", "multi_source"}:
                raise ValueError(f"Invalid strategy: {force_strategy}")
            logger.debug(f"Using forced strategy: {force_strategy}")
            return force_strategy
        
        # Check circuit breaker for multi-source strategy (Guilfoyle's fix)
        if self._should_skip_multi_source():
            logger.warning(
                "Multi-source strategy circuit breaker is open, forcing intelligence-guided"
            )
            return "intelligence_guided"
        
        # Use multi-source discovery for heavily automated users
        if profile.is_heavily_automated:
            logger.debug(
                f"User classified as heavily automated "
                f"(confidence: {profile.confidence_score:.2f}), using multi-source strategy"
            )
            return "multi_source"
        
        # Use intelligence-guided analysis for normal users (98% of cases)
        logger.debug(
            f"User classified as normal developer "
            f"(confidence: {profile.confidence_score:.2f}), using intelligence-guided strategy"
        )
        return "intelligence_guided"
    
    def _should_skip_multi_source(self) -> bool:
        """Check if multi-source strategy should be skipped due to circuit breaker.
        
        Returns:
            True if multi-source should be skipped
        """
        if not self.multi_source_circuit_open:
            return False
            
        # Check if circuit should be reset (after timeout)
        current_time = time.time()
        if current_time > self.circuit_reset_time:
            self.multi_source_circuit_open = False
            self.multi_source_failures = 0
            logger.info("Multi-source circuit breaker reset - strategy available again")
            return False
            
        return True
    
    async def _execute_strategy(
        self, 
        strategy: str, 
        username: str, 
        days: int
    ) -> tuple[list[BaseGitHubEvent], AnalysisStats]:
        """Execute analysis strategy with multi-level fallback (Guilfoyle's fix).
        
        Args:
            strategy: Strategy to execute
            username: GitHub username
            days: Number of days to analyze
            
        Returns:
            Tuple of (events, analysis_stats)
            
        Raises:
            RuntimeError: If all strategies including emergency fallback fail
        """
        try:
            if strategy == "multi_source":
                return await self._execute_multi_source(username, days)
            elif strategy == "intelligence_guided":
                return await self._execute_intelligence_guided(username, days)
            else:
                raise ValueError(f"Unknown analysis strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Strategy '{strategy}' failed for {username}: {e}")
            
            # Handle circuit breaker for multi-source failures
            if strategy == "multi_source":
                self._handle_multi_source_failure()
                
                # Fallback to intelligence-guided
                logger.warning(f"Multi-source failed, falling back to intelligence-guided: {e}")
                try:
                    events, stats = await self._execute_intelligence_guided(username, days)
                    stats.fallback_used = True
                    stats.errors_encountered.append(f"multi_source_failed: {str(e)}")
                    return events, stats
                except Exception as fallback_error:
                    logger.error(f"Intelligence-guided fallback also failed: {fallback_error}")
                    # Continue to emergency fallback
            
            # Emergency fallback - return minimal viable data
            logger.error(f"All primary strategies failed for {username}, using emergency fallback")
            return self._emergency_fallback(username, days, e)
    
    async def _execute_multi_source(self, username: str, days: int) -> tuple[list[BaseGitHubEvent], AnalysisStats]:
        """Execute multi-source discovery strategy."""
        analyzer = self._get_multi_source_analyzer()
        events = await analyzer.discover_and_fetch(username, days)
        
        # Convert to enhanced stats
        basic_stats = analyzer.get_analysis_stats(events)
        enhanced_stats = AnalysisStats(
            strategy="multi_source",
            events_processed=len(events),
            execution_time_ms=0,  # Will be set by caller
        )
        
        # Reset multi-source failure count on success
        self.multi_source_failures = 0
        
        return events, enhanced_stats
    
    async def _execute_intelligence_guided(self, username: str, days: int) -> tuple[list[BaseGitHubEvent], AnalysisStats]:
        """Execute intelligence-guided analysis strategy."""
        analyzer = self._get_intelligence_guided_analyzer()
        events = await analyzer.discover_and_fetch(username, days)
        
        # Convert to enhanced stats
        basic_stats = analyzer.get_analysis_stats(events)
        enhanced_stats = AnalysisStats(
            strategy="intelligence_guided",
            events_processed=len(events),
            execution_time_ms=0,  # Will be set by caller
        )
        
        return events, enhanced_stats
    
    def _handle_multi_source_failure(self) -> None:
        """Handle multi-source strategy failure for circuit breaker pattern."""
        self.multi_source_failures += 1
        
        if self.multi_source_failures >= self.circuit_failure_threshold:
            self.multi_source_circuit_open = True
            self.circuit_reset_time = time.time() + self.circuit_reset_timeout
            logger.warning(
                f"Multi-source circuit breaker opened after {self.multi_source_failures} failures. "
                f"Will reset in {self.circuit_reset_timeout} seconds."
            )
    
    def _emergency_fallback(self, username: str, days: int, error: Exception) -> tuple[list[BaseGitHubEvent], AnalysisStats]:
        """Return minimal viable response when all strategies fail (Guilfoyle's fix)."""
        logger.critical(f"Emergency fallback activated for {username}")
        
        emergency_stats = AnalysisStats(
            strategy="emergency_fallback",
            events_processed=0,
            execution_time_ms=0,
            errors_encountered=[f"all_strategies_failed: {str(error)}"]
        )
        
        # Return empty events list - at least we don't crash
        return [], emergency_stats
    
    def _deduplicate_events(self, events: list[BaseGitHubEvent]) -> list[BaseGitHubEvent]:
        """Deduplicate events using composite keys (Guilfoyle's fix).
        
        GitHub event IDs are not unique across all endpoints, so we use
        a composite key for reliable deduplication.
        
        Args:
            events: List of events that may contain duplicates
            
        Returns:
            List of unique events sorted chronologically
        """
        if not events:
            return []
        
        seen_keys = set()
        unique_events = []
        
        for event in events:
            dedup_key = self._create_dedup_key(event)
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                unique_events.append(event)
        
        # Sort events chronologically
        unique_events.sort(key=lambda e: e.created_at)
        
        duplicates_removed = len(events) - len(unique_events)
        if duplicates_removed > 0:
            logger.debug(f"Removed {duplicates_removed} duplicate events using composite keys")
        
        return unique_events
    
    def _create_dedup_key(self, event: BaseGitHubEvent) -> str:
        """Create composite key for reliable deduplication (Guilfoyle's fix).
        
        Args:
            event: GitHub event to create key for
            
        Returns:
            Composite key string for deduplication
        """
        # Use multiple fields to create unique identifier
        key_parts = [
            event.type,
            event.created_at,
            event.actor.login if event.actor else "unknown",
            event.repo.name if event.repo else "unknown"
        ]
        
        # Add event-specific identifiers when available
        if hasattr(event, 'id') and event.id:
            key_parts.append(str(event.id))
            
        return "|".join(str(part) for part in key_parts)
    
    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for the coordinator.
        
        Returns:
            Dictionary with performance metrics
        """
        total_analyses = sum(self.strategy_usage.values())
        
        return {
            "total_analyses": total_analyses,
            "strategy_usage": dict(self.strategy_usage),
            "strategy_distribution": {
                strategy: (count / total_analyses * 100) if total_analyses > 0 else 0
                for strategy, count in self.strategy_usage.items()
            },
            "automation_detection_rate": (
                self.strategy_usage["multi_source"] / total_analyses * 100
                if total_analyses > 0 else 0
            ),
            "fallback_rate": (
                self.strategy_usage["fallback"] / total_analyses * 100
                if total_analyses > 0 else 0
            )
        }
    
    def reset_performance_stats(self) -> None:
        """Reset performance tracking statistics."""
        self.analysis_count = 0
        self.strategy_usage = {"intelligence_guided": 0, "multi_source": 0, "fallback": 0}
        logger.info("Performance statistics reset")