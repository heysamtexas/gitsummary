"""AI Summary Orchestrator Pipeline for GitHub activity analysis.

This module orchestrates the complete context → persona → LLM → summary pipeline
for AI-powered activity analysis.
"""

import logging
from typing import Any

from git_summary.ai.client import LLMClient
from git_summary.ai.context import ContextGatheringEngine, GitHubEventLike, RichContext, TokenBudget
from git_summary.ai.personas import PersonaManager
from git_summary.github_client import GitHubClient

logger = logging.getLogger(__name__)


class ActivitySummarizer:
    """Orchestrates the complete AI-powered activity summarization pipeline."""

    def __init__(
        self,
        github_client: GitHubClient,
        llm_client: LLMClient | None = None,
        persona_manager: PersonaManager | None = None,
        default_token_budget: int = 8000,
    ) -> None:
        """Initialize the Activity Summarizer.

        Args:
            github_client: GitHub API client for fetching additional data
            llm_client: LiteLLM client for AI generation (creates default if None)
            persona_manager: Persona manager for different analysis styles (creates default if None)
            default_token_budget: Default token budget for context gathering
        """
        self.github_client = github_client
        self.llm_client = llm_client or LLMClient()
        self.persona_manager = persona_manager or PersonaManager()
        self.context_engine = ContextGatheringEngine(
            github_client, default_budget=default_token_budget
        )
        self.default_token_budget = default_token_budget

        logger.info("ActivitySummarizer initialized with default persona and model")

    async def generate_summary(
        self,
        events: list[GitHubEventLike],
        persona_name: str = "tech analyst",
        token_budget: int | None = None,
        include_context_details: bool = True,
    ) -> dict[str, Any]:
        """Generate an AI-powered summary of GitHub activity.

        Args:
            events: List of GitHub events to analyze
            persona_name: Name of persona to use for analysis
            token_budget: Token budget for context gathering (uses default if None)
            include_context_details: Whether to include detailed context in response

        Returns:
            Dictionary containing the generated summary and metadata

        Raises:
            ValueError: If persona is not found
            RuntimeError: If summary generation fails
        """
        logger.info(
            f"Starting summary generation for {len(events)} events using {persona_name} persona"
        )

        try:
            # Step 1: Gather rich context from events
            budget = TokenBudget(max_tokens=token_budget or self.default_token_budget)
            rich_context = await self.context_engine.gather_context(events, budget)

            logger.info(
                f"Context gathered: {rich_context.estimated_tokens} tokens, "
                f"{len(rich_context.commits)} commits, {len(rich_context.pull_requests)} PRs, "
                f"{len(rich_context.issues)} issues, {len(rich_context.releases)} releases"
            )

            # Step 2: Get persona for analysis style
            persona = self.persona_manager.get_persona(persona_name)
            if not persona:
                raise ValueError(f"Persona '{persona_name}' not found")

            # Step 3: Format context for LLM
            context_data = self._format_context_for_llm(rich_context)
            system_prompt = persona.get_system_prompt()
            user_content = persona.get_summary_instructions(context_data)

            logger.debug(
                f"Generated prompts - System: {len(system_prompt)} chars, User: {len(user_content)} chars"
            )

            # Step 4: Generate summary using LLM
            summary_text = await self.llm_client.generate_summary(
                system_prompt, user_content
            )

            logger.info("Summary generated successfully")

            # Step 5: Prepare response
            response = {
                "summary": summary_text.strip(),
                "persona_used": persona_name,
                "model_used": self.llm_client.model,
                "metadata": {
                    "total_events": rich_context.total_events,
                    "tokens_used": rich_context.estimated_tokens,
                    "date_range": rich_context.date_range,
                    "repositories": rich_context.repositories,
                    "event_breakdown": {
                        "commits": len(rich_context.commits),
                        "pull_requests": len(rich_context.pull_requests),
                        "issues": len(rich_context.issues),
                        "releases": len(rich_context.releases),
                        "reviews": len(rich_context.reviews),
                    },
                },
            }

            # Include detailed context if requested
            if include_context_details:
                response["context"] = {
                    "commits": rich_context.commits,
                    "pull_requests": rich_context.pull_requests,
                    "issues": rich_context.issues,
                    "releases": rich_context.releases,
                    "reviews": rich_context.reviews,
                }

            return response

        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            raise RuntimeError(f"Failed to generate summary: {e}") from e

    def generate_summary_sync(
        self,
        events: list[GitHubEventLike],
        persona_name: str = "tech analyst",
        token_budget: int | None = None,
        include_context_details: bool = True,
    ) -> dict[str, Any]:
        """Synchronous version of generate_summary for non-async contexts.

        Args:
            events: List of GitHub events to analyze
            persona_name: Name of persona to use for analysis
            token_budget: Token budget for context gathering (uses default if None)
            include_context_details: Whether to include detailed context in response

        Returns:
            Dictionary containing the generated summary and metadata

        Raises:
            ValueError: If persona is not found
            RuntimeError: If summary generation fails
        """
        import asyncio

        # Create new event loop if none exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.generate_summary(
                    events, persona_name, token_budget, include_context_details
                )
            )
        except Exception as e:
            logger.error(f"Sync summary generation failed: {e}")
            raise

    async def estimate_cost(
        self,
        events: list[GitHubEventLike],
        persona_name: str = "tech analyst",
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        """Estimate the cost of generating a summary without actually calling the LLM.

        Args:
            events: List of GitHub events to analyze
            persona_name: Name of persona to use for analysis
            token_budget: Token budget for context gathering (uses default if None)

        Returns:
            Dictionary containing cost estimates and token usage information

        Raises:
            ValueError: If persona is not found
        """
        logger.info(
            f"Estimating cost for {len(events)} events using {persona_name} persona"
        )

        try:
            # Step 1: Gather rich context (same as actual generation)
            budget = TokenBudget(max_tokens=token_budget or self.default_token_budget)
            rich_context = await self.context_engine.gather_context(events, budget)

            # Step 2: Get persona
            persona = self.persona_manager.get_persona(persona_name)
            if not persona:
                raise ValueError(f"Persona '{persona_name}' not found")

            # Step 3: Format context and estimate cost
            context_data = self._format_context_for_llm(rich_context)
            system_prompt = persona.get_system_prompt()
            user_content = persona.get_summary_instructions(context_data)

            # Step 4: Get cost estimate from LLM client
            cost_estimate = self.llm_client.estimate_cost(system_prompt, user_content)

            return {
                "cost_estimate": cost_estimate,
                "context_tokens": rich_context.estimated_tokens,
                "total_events": rich_context.total_events,
                "event_breakdown": {
                    "commits": len(rich_context.commits),
                    "pull_requests": len(rich_context.pull_requests),
                    "issues": len(rich_context.issues),
                    "releases": len(rich_context.releases),
                    "reviews": len(rich_context.reviews),
                },
                "persona_used": persona_name,
                "model": self.llm_client.model,
            }

        except ValueError as e:
            logger.error(f"Cost estimation configuration error: {e}")
            raise
        except Exception as e:
            logger.error(f"Cost estimation failed: {e}")
            raise RuntimeError(f"Failed to estimate cost: {e}") from e

    def get_available_personas(self) -> list[str]:
        """Get list of available persona names.

        Returns:
            List of persona names that can be used for analysis
        """
        personas = self.persona_manager.list_personas()
        return [persona.name.lower().replace(" ", "_") for persona in personas]

    def _format_context_for_llm(self, rich_context: RichContext) -> dict[str, Any]:
        """Format rich context data for LLM consumption.

        Args:
            rich_context: RichContext object with gathered GitHub data

        Returns:
            Dictionary formatted for persona consumption
        """
        return {
            "total_events": rich_context.total_events,
            "date_range": rich_context.date_range,
            "repositories": rich_context.repositories,
            "commits": rich_context.commits,
            "commit_messages": rich_context.commit_messages,
            "pull_requests": rich_context.pull_requests,
            "pr_titles": rich_context.pr_titles,
            "issues": rich_context.issues,
            "issue_titles": rich_context.issue_titles,
            "releases": rich_context.releases,
            "release_notes": rich_context.release_notes,
            "reviews": rich_context.reviews,
            "estimated_tokens": rich_context.estimated_tokens,
        }
