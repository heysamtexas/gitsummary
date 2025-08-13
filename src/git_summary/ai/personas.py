"""AI personas for different types of GitHub activity analysis.

This module provides different AI personalities that can analyze the same
GitHub data from different perspectives, giving users variety in their
daily work summaries.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BasePersona(ABC):
    """Base class for AI personas that analyze GitHub activity."""

    def __init__(self, name: str, description: str):
        """Initialize the persona.

        Args:
            name: Unique name for the persona
            description: Human-readable description of the persona's style
        """
        self.name = name
        self.description = description

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt that defines this persona's analysis style.

        Returns:
            System prompt string for the AI model
        """
        pass

    @abstractmethod
    def get_summary_instructions(self, context: dict[str, Any]) -> str:
        """Get specific instructions for summarizing the given context.

        Args:
            context: Rich context data about GitHub activity

        Returns:
            Detailed instructions for analyzing the context
        """
        pass

    def __str__(self) -> str:
        """String representation of the persona."""
        return f"{self.name}: {self.description}"

    def __repr__(self) -> str:
        """Developer representation of the persona."""
        return f"<{self.__class__.__name__}(name='{self.name}')>"


class TechnicalAnalystPersona(BasePersona):
    """Technical analyst persona focused on code quality and engineering practices.

    This persona provides detailed technical analysis of commits, code changes,
    and engineering practices. It focuses on:
    - Code quality and architecture decisions
    - Testing and CI/CD activity
    - Technical debt and refactoring efforts
    - Performance and security considerations
    """

    def __init__(self) -> None:
        super().__init__(
            name="Tech Analyst",
            description="Deep technical analysis focusing on code quality, architecture, and engineering practices",
        )

    def get_system_prompt(self) -> str:
        """Get the system prompt for technical analysis."""
        return """You are a Senior Technical Analyst specializing in code review and engineering practices.

Your role is to analyze GitHub activity from a technical perspective, focusing on:

**Code Quality & Architecture:**
- Review commit patterns for architectural decisions
- Identify refactoring efforts and technical debt reduction
- Assess code organization and design patterns
- Evaluate API design and interface changes

**Engineering Practices:**
- Analyze testing patterns (unit, integration, e2e)
- Review CI/CD pipeline changes and deployments
- Assess code review practices and collaboration
- Identify performance optimizations and security improvements

**Technical Communication:**
- Use precise technical terminology
- Provide actionable insights for engineering improvement
- Structure analysis around technical themes
- Quantify technical metrics when possible

**Analysis Style:**
- Be concise but thorough
- Focus on impact and engineering value
- Highlight both achievements and areas for improvement
- Provide context for technical decisions

Remember: Your audience consists of engineers who value technical depth and actionable insights."""

    def get_summary_instructions(self, context: dict[str, Any]) -> str:
        """Get technical analysis instructions for the given context."""

        # Extract key metrics from context
        total_commits = context.get("commit_count", 0)
        repos_active = len(context.get("repositories", []))
        has_prs = bool(context.get("pull_requests", []))
        has_issues = bool(context.get("issues", []))
        has_releases = bool(context.get("releases", []))

        instructions = f"""Analyze the following GitHub activity data with technical focus:

**Context Overview:**
- {total_commits} commits across {repos_active} repositories
- Pull requests: {"Yes" if has_prs else "No"}
- Issue activity: {"Yes" if has_issues else "No"}
- Releases: {"Yes" if has_releases else "No"}

**Analysis Framework:**

1. **Code Change Analysis** (if commits present):
   - Categorize commits by type (features, bugs, refactoring, docs, tests)
   - Identify architectural changes or major refactoring
   - Assess commit message quality and development practices
   - Note any performance or security-related changes

2. **Collaboration Patterns** (if PRs/issues present):
   - Review code review activity and collaboration patterns
   - Analyze issue resolution and problem-solving approaches
   - Assess documentation and communication quality

3. **Project Health Indicators**:
   - Evaluate testing activity and coverage patterns
   - Review CI/CD pipeline usage and deployment frequency
   - Assess dependency updates and maintenance activity
   - Identify technical debt reduction efforts

4. **Engineering Insights**:
   - Highlight significant technical achievements
   - Identify patterns that suggest good/concerning practices
   - Provide recommendations for engineering improvement
   - Quantify impact where possible

**Output Format:**
Structure your analysis as:
- **Technical Summary**: 2-3 sentence overview of engineering activity
- **Key Technical Themes**: 3-4 main technical focus areas
- **Engineering Highlights**: Notable technical achievements
- **Recommendations**: 1-2 actionable technical suggestions (if applicable)

Keep the analysis under 400 words, focusing on technical substance over verbosity."""

        return instructions


class PersonaManager:
    """Manager for AI personas used in GitHub activity analysis."""

    def __init__(self) -> None:
        """Initialize the persona manager with available personas."""
        self._personas: dict[str, BasePersona] = {}
        self._register_default_personas()

    def _register_default_personas(self) -> None:
        """Register the default set of personas."""
        technical_analyst = TechnicalAnalystPersona()
        self._personas[technical_analyst.name.lower()] = technical_analyst

        logger.info(f"Registered {len(self._personas)} default personas")

    def get_persona(self, name: str) -> BasePersona:
        """Get a persona by name.

        Args:
            name: Name of the persona (case-insensitive)

        Returns:
            The requested persona

        Raises:
            ValueError: If persona not found
        """
        normalized_name = name.lower().strip()

        if normalized_name not in self._personas:
            available = list(self._personas.keys())
            raise ValueError(
                f"Persona '{name}' not found. Available personas: {available}"
            )

        return self._personas[normalized_name]

    def list_personas(self) -> list[BasePersona]:
        """Get list of all available personas.

        Returns:
            List of available personas
        """
        return list(self._personas.values())

    def get_default_persona(self) -> BasePersona:
        """Get the default persona for analysis.

        Returns:
            Default persona (Technical Analyst)
        """
        return self.get_persona("tech analyst")

    def register_persona(self, persona: BasePersona) -> None:
        """Register a new persona.

        Args:
            persona: Persona to register
        """
        normalized_name = persona.name.lower().strip()
        self._personas[normalized_name] = persona
        logger.info(f"Registered persona: {persona.name}")
