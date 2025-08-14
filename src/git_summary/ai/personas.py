"""AI personas for different types of GitHub activity analysis.

This module provides different AI personalities that can analyze the same
GitHub data from different perspectives, giving users variety in their
daily work summaries.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml

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


class ConfigurablePersona(BasePersona):
    """Persona loaded from YAML configuration."""

    def __init__(self, yaml_path: Path):
        """Initialize persona from YAML configuration.

        Args:
            yaml_path: Path to YAML configuration file

        Raises:
            ValueError: If YAML is invalid or missing required fields
            FileNotFoundError: If YAML file doesn't exist
        """
        config = self._load_yaml_config(yaml_path)
        super().__init__(config["name"], config["description"])

        self.yaml_path = yaml_path
        self.config = config
        self.system_prompt_template = config["system_prompt"]
        self.analysis_framework = config["analysis_framework"]
        self.context_processing = config.get("context_processing", {})
        self.output_format = config.get("output_format", {})

        logger.info(f"Loaded YAML persona '{self.name}' from {yaml_path.name}")

    def _load_yaml_config(self, yaml_path: Path) -> dict[str, Any]:
        """Load and validate YAML persona configuration.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Validated configuration dictionary

        Raises:
            ValueError: If YAML is invalid or missing required fields
            FileNotFoundError: If file doesn't exist
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Persona YAML file not found: {yaml_path}")

        try:
            with open(yaml_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {yaml_path}: {e}")

        if not isinstance(config, dict):
            raise ValueError(f"YAML root must be a dictionary in {yaml_path}")

        # Validate required fields
        required_fields = ["name", "description", "system_prompt", "analysis_framework"]
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in {yaml_path}: {missing_fields}"
            )

        # Validate analysis_framework structure
        if "sections" not in config["analysis_framework"]:
            raise ValueError(
                f"analysis_framework must contain 'sections' array in {yaml_path}"
            )

        sections = config["analysis_framework"]["sections"]
        if not isinstance(sections, list) or not sections:
            raise ValueError(
                f"analysis_framework.sections must be non-empty array in {yaml_path}"
            )

        # Validate each section has required fields
        for i, section in enumerate(sections):
            if not isinstance(section, dict):
                raise ValueError(f"Section {i} must be a dictionary in {yaml_path}")
            if "name" not in section or "description" not in section:
                raise ValueError(
                    f"Section {i} missing 'name' or 'description' in {yaml_path}"
                )

        return config

    def get_system_prompt(self) -> str:
        """Get the system prompt from YAML configuration."""
        return self.system_prompt_template.strip()

    def get_summary_instructions(self, context: dict[str, Any]) -> str:
        """Generate analysis instructions based on YAML configuration.

        Args:
            context: Rich context data about GitHub activity

        Returns:
            Formatted instructions for the AI model
        """
        instructions = "Analyze the following GitHub activity data:\n\n"

        # Add context overview
        total_commits = context.get("commit_count", 0)
        repos_active = len(context.get("repositories", []))
        has_prs = bool(context.get("pull_requests", []))
        has_issues = bool(context.get("issues", []))
        has_releases = bool(context.get("releases", []))

        instructions += "**Context Overview:**\n"
        instructions += (
            f"- {total_commits} commits across {repos_active} repositories\n"
        )
        instructions += f"- Pull requests: {'Yes' if has_prs else 'No'}\n"
        instructions += f"- Issue activity: {'Yes' if has_issues else 'No'}\n"
        instructions += f"- Releases: {'Yes' if has_releases else 'No'}\n\n"

        # Generate analysis framework from YAML
        instructions += "**Analysis Framework:**\n"
        for i, section in enumerate(self.analysis_framework["sections"], 1):
            instructions += f"{i}. **{section['name']}**: {section['description']}\n"

            # Add section-specific requirements
            if "max_length" in section:
                instructions += f"   (Max {section['max_length']} characters)\n"
            elif "max_items" in section:
                instructions += f"   (Max {section['max_items']} items)\n"

            if section.get("format") == "bullet_list":
                instructions += "   (Format as bullet list)\n"
            elif section.get("format") == "table":
                instructions += "   (Format as table)\n"

            if section.get("optional"):
                instructions += "   (Optional - include only if relevant)\n"

        instructions += "\n"

        # Add GitHub activity data
        instructions += self._format_github_data(context)

        # Add output format requirements
        if self.output_format:
            instructions += "\n**Output Requirements:**\n"
            if "max_words" in self.output_format:
                instructions += (
                    f"- Keep analysis under {self.output_format['max_words']} words\n"
                )
            if "tone" in self.output_format:
                instructions += f"- Use {self.output_format['tone']} tone\n"
            if "audience" in self.output_format:
                instructions += f"- Target audience: {self.output_format['audience']}\n"
            if self.output_format.get("include_metrics"):
                instructions += "- Include quantitative metrics when possible\n"

        return instructions

    def _format_github_data(self, context: dict[str, Any]) -> str:
        """Format GitHub activity data according to context processing rules."""
        data_output = "**GitHub Activity Data:**\n\n"

        # Get processing limits from config
        commit_config = self.context_processing.get("commit_analysis", {})
        pr_config = self.context_processing.get("pr_analysis", {})
        issue_config = self.context_processing.get("issue_analysis", {})

        max_commits = commit_config.get("max_commits_displayed", 10)
        max_prs = pr_config.get("max_prs_displayed", 5)
        max_issues = issue_config.get("max_issues_displayed", 5)

        # Add repositories info
        repos = context.get("repositories", [])
        if repos:
            data_output += (
                f"**Active Repositories ({len(repos)}):** {', '.join(repos)}\n\n"
            )

        # Add commit data
        commits = context.get("commits", [])
        commit_messages = context.get("commit_messages", [])
        if commits:
            data_output += f"**Commits ({len(commits)} total):**\n"
            for i, commit in enumerate(commits[:max_commits]):
                msg = commit_messages[i] if i < len(commit_messages) else "No message"
                data_output += f"- {commit.get('repository', 'unknown')}: {msg}\n"

                if commit_config.get("include_line_changes") and (
                    commit.get("additions", 0) or commit.get("deletions", 0)
                ):
                    data_output += f"  (+{commit.get('additions', 0)}, -{commit.get('deletions', 0)} lines)\n"

            if len(commits) > max_commits:
                data_output += f"... and {len(commits) - max_commits} more commits\n"
            data_output += "\n"

        # Add PR data
        prs = context.get("pull_requests", [])
        pr_titles = context.get("pr_titles", [])
        if prs:
            data_output += f"**Pull Requests ({len(prs)} total):**\n"
            for i, pr in enumerate(prs[:max_prs]):
                title = (
                    pr_titles[i] if i < len(pr_titles) else pr.get("title", "No title")
                )
                data_output += f"- #{pr.get('number', 'unknown')} ({pr.get('state', 'unknown')}): {title}\n"

            if len(prs) > max_prs:
                data_output += f"... and {len(prs) - max_prs} more pull requests\n"
            data_output += "\n"

        # Add issue data
        issues = context.get("issues", [])
        issue_titles = context.get("issue_titles", [])
        if issues:
            data_output += f"**Issues ({len(issues)} total):**\n"
            for i, issue in enumerate(issues[:max_issues]):
                title = (
                    issue_titles[i]
                    if i < len(issue_titles)
                    else issue.get("title", "No title")
                )
                data_output += f"- #{issue.get('number', 'unknown')} ({issue.get('state', 'unknown')}): {title}\n"

            if len(issues) > max_issues:
                data_output += f"... and {len(issues) - max_issues} more issues\n"
            data_output += "\n"

        # Add release data
        releases = context.get("releases", [])
        release_notes = context.get("release_notes", [])
        if releases:
            data_output += f"**Releases ({len(releases)} total):**\n"
            for i, release in enumerate(releases[:3]):
                notes = release_notes[i] if i < len(release_notes) else "No notes"
                tag_name = (
                    release.tag_name if hasattr(release, "tag_name") else "unknown"
                )
                data_output += (
                    f"- {tag_name}: {notes[:100]}{'...' if len(notes) > 100 else ''}\n"
                )

            if len(releases) > 3:
                data_output += f"... and {len(releases) - 3} more releases\n"

        return data_output


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

Keep the analysis under 400 words, focusing on technical substance over verbosity.

**GitHub Activity Data:**

"""

        # Append the actual context data
        instructions += f"""
**Repositories Active:** {repos_active}
**Repositories:** {', '.join(context.get('repositories', []))}

"""

        # Add commit data if present
        commits = context.get("commits", [])
        commit_messages = context.get("commit_messages", [])
        if commits:
            instructions += f"**Commits ({len(commits)} total):**\n"
            for i, commit in enumerate(commits[:10]):  # Show up to 10 commits
                msg = commit_messages[i] if i < len(commit_messages) else "No message"
                instructions += f"- {commit.get('repository', 'unknown')}: {msg}\n"
                if commit.get("additions", 0) or commit.get("deletions", 0):
                    instructions += f"  (+{commit.get('additions', 0)}, -{commit.get('deletions', 0)} lines)\n"
            if len(commits) > 10:
                instructions += f"... and {len(commits) - 10} more commits\n"
            instructions += "\n"

        # Add PR data if present
        prs = context.get("pull_requests", [])
        pr_titles = context.get("pr_titles", [])
        if prs:
            instructions += f"**Pull Requests ({len(prs)} total):**\n"
            for i, pr in enumerate(prs[:5]):  # Show up to 5 PRs
                title = (
                    pr_titles[i] if i < len(pr_titles) else pr.get("title", "No title")
                )
                instructions += f"- #{pr.get('number', 'unknown')} ({pr.get('state', 'unknown')}): {title}\n"
            if len(prs) > 5:
                instructions += f"... and {len(prs) - 5} more pull requests\n"
            instructions += "\n"

        # Add issue data if present
        issues = context.get("issues", [])
        issue_titles = context.get("issue_titles", [])
        if issues:
            instructions += f"**Issues ({len(issues)} total):**\n"
            for i, issue in enumerate(issues[:5]):  # Show up to 5 issues
                title = (
                    issue_titles[i]
                    if i < len(issue_titles)
                    else issue.get("title", "No title")
                )
                instructions += f"- #{issue.get('number', 'unknown')} ({issue.get('state', 'unknown')}): {title}\n"
            if len(issues) > 5:
                instructions += f"... and {len(issues) - 5} more issues\n"
            instructions += "\n"

        # Add release data if present
        releases = context.get("releases", [])
        release_notes = context.get("release_notes", [])
        if releases:
            instructions += f"**Releases ({len(releases)} total):**\n"
            for i, release in enumerate(releases[:3]):  # Show up to 3 releases
                notes = release_notes[i] if i < len(release_notes) else "No notes"
                tag_name = (
                    release.tag_name if hasattr(release, "tag_name") else "unknown"
                )
                instructions += (
                    f"- {tag_name}: {notes[:100]}{'...' if len(notes) > 100 else ''}\n"
                )
            if len(releases) > 3:
                instructions += f"... and {len(releases) - 3} more releases\n"

        return instructions


class PersonaManager:
    """Enhanced manager supporting both coded and YAML personas."""

    def __init__(self, personas_dir: Path | None = None) -> None:
        """Initialize the persona manager with available personas.

        Args:
            personas_dir: Directory containing YAML persona files (defaults to built-in personas/ subdirectory)
        """
        self._personas: dict[str, BasePersona] = {}
        self.built_in_personas_dir = personas_dir or Path(__file__).parent / "personas"
        self.user_personas_dir = Path.home() / ".git-summary" / "personas"

        self._register_default_personas()
        self._load_built_in_yaml_personas()
        self._load_user_yaml_personas()

    def _register_default_personas(self) -> None:
        """Register the default set of built-in personas."""
        technical_analyst = TechnicalAnalystPersona()
        self._personas[technical_analyst.name.lower()] = technical_analyst

        logger.info(f"Registered {len(self._personas)} built-in personas")

    def _load_built_in_yaml_personas(self) -> None:
        """Load built-in personas from YAML files in the package directory."""
        self._load_yaml_personas_from_dir(self.built_in_personas_dir, "built-in")

    def _load_user_yaml_personas(self) -> None:
        """Load user-custom personas from YAML files in ~/.git-summary/personas/."""
        self._load_yaml_personas_from_dir(self.user_personas_dir, "user")

    def _load_yaml_personas_from_dir(
        self, personas_dir: Path, source_type: str
    ) -> None:
        """Load personas from YAML files in the specified directory.

        Args:
            personas_dir: Directory to load personas from
            source_type: Type of personas being loaded (for logging)
        """
        if not personas_dir.exists():
            logger.debug(
                f"{source_type.title()} personas directory not found: {personas_dir}"
            )
            return

        yaml_files = list(personas_dir.glob("*.yaml")) + list(
            personas_dir.glob("*.yml")
        )
        if not yaml_files:
            logger.debug(f"No YAML persona files found in {personas_dir}")
            return

        yaml_count = 0
        for yaml_file in yaml_files:
            try:
                persona = ConfigurablePersona(yaml_file)
                self.register_persona(persona)
                yaml_count += 1
                logger.info(
                    f"Loaded {source_type} YAML persona: {persona.name} from {yaml_file.name}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load {source_type} persona from {yaml_file}: {e}"
                )

        if yaml_count > 0:
            logger.info(f"Successfully loaded {yaml_count} {source_type} YAML personas")

    def reload_yaml_personas(self) -> None:
        """Reload all YAML personas from disk.

        Useful for development and testing when persona files change.
        Preserves built-in personas and only reloads YAML-based ones.
        """
        # Remove existing YAML personas
        yaml_personas = [
            name
            for name, persona in self._personas.items()
            if isinstance(persona, ConfigurablePersona)
        ]
        for name in yaml_personas:
            del self._personas[name]
            logger.debug(f"Removed YAML persona: {name}")

        # Reload from both directories
        self._load_built_in_yaml_personas()
        self._load_user_yaml_personas()
        logger.info("Reloaded all YAML personas from built-in and user directories")

    def create_persona_template(
        self,
        name: str,
        output_path: Path | None = None,
        template_type: str = "basic",
        save_to_user_dir: bool = True,
    ) -> Path:
        """Create a new YAML persona template file.

        Args:
            name: Name for the new persona
            output_path: Where to save the template (defaults to user personas directory)
            template_type: Type of template ('basic', 'advanced', 'technical')
            save_to_user_dir: If True and output_path is None, saves to ~/.git-summary/personas/

        Returns:
            Path to the created template file

        Raises:
            ValueError: If template already exists or invalid template_type
        """
        if output_path is None:
            filename = name.lower().replace(" ", "_") + ".yaml"
            if save_to_user_dir:
                # Ensure user personas directory exists
                self.user_personas_dir.mkdir(parents=True, exist_ok=True)
                output_path = self.user_personas_dir / filename
            else:
                output_path = self.built_in_personas_dir / filename

        if output_path.exists():
            raise ValueError(f"Template file already exists: {output_path}")

        template = self._get_persona_template(name, template_type)

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                template,
                f,
                default_flow_style=False,
                sort_keys=False,
                width=80,
                indent=2,
            )

        logger.info(f"Created persona template: {output_path}")
        return output_path

    def _get_persona_template(self, name: str, template_type: str) -> dict[str, Any]:
        """Generate persona template based on type."""
        if template_type == "basic":
            return {
                "name": name,
                "description": f"AI persona for {name.lower()} analysis",
                "version": "1.0",
                "author": "user",
                "system_prompt": f"You are a {name} specializing in GitHub activity analysis.\n\nFocus on providing insights relevant to {name.lower()} perspectives and priorities.\nBe clear, actionable, and insightful in your analysis.",
                "analysis_framework": {
                    "sections": [
                        {
                            "name": "Summary",
                            "description": "Brief overview of key activity and trends",
                            "max_length": 200,
                        },
                        {
                            "name": "Key Insights",
                            "description": "Main findings and important patterns",
                            "format": "bullet_list",
                            "max_items": 4,
                        },
                        {
                            "name": "Recommendations",
                            "description": "Actionable suggestions for improvement",
                            "format": "bullet_list",
                            "max_items": 3,
                            "optional": True,
                        },
                    ]
                },
                "context_processing": {
                    "commit_analysis": {
                        "max_commits_displayed": 8,
                        "include_line_changes": False,
                    },
                    "pr_analysis": {"max_prs_displayed": 5},
                    "issue_analysis": {"max_issues_displayed": 5},
                },
                "output_format": {
                    "max_words": 300,
                    "tone": "professional",
                    "audience": "general",
                    "include_metrics": True,
                },
            }
        elif template_type == "technical":
            return {
                "name": name,
                "description": f"Technical {name.lower()} focused on engineering practices and code quality",
                "version": "1.0",
                "author": "user",
                "system_prompt": f"You are a Senior {name} with deep technical expertise.\n\nAnalyze GitHub activity from a technical perspective, focusing on:\n- Code quality and architecture decisions\n- Engineering practices and workflows\n- Technical debt and optimization opportunities\n- Development velocity and quality metrics",
                "analysis_framework": {
                    "sections": [
                        {
                            "name": "Technical Summary",
                            "description": "Overview of engineering activity and technical focus",
                            "max_length": 150,
                        },
                        {
                            "name": "Engineering Highlights",
                            "description": "Notable technical achievements and improvements",
                            "format": "bullet_list",
                            "max_items": 4,
                        },
                        {
                            "name": "Code Quality Insights",
                            "description": "Observations about code quality and engineering practices",
                            "format": "bullet_list",
                            "max_items": 3,
                        },
                        {
                            "name": "Technical Recommendations",
                            "description": "Actionable suggestions for engineering improvement",
                            "format": "bullet_list",
                            "max_items": 2,
                            "optional": True,
                        },
                    ]
                },
                "context_processing": {
                    "commit_analysis": {
                        "max_commits_displayed": 12,
                        "include_line_changes": True,
                        "categorize_by": [
                            "features",
                            "bugs",
                            "refactoring",
                            "tests",
                            "docs",
                        ],
                    },
                    "pr_analysis": {
                        "max_prs_displayed": 6,
                        "include_review_comments": True,
                    },
                    "issue_analysis": {
                        "max_issues_displayed": 6,
                        "focus_on_technical": True,
                    },
                },
                "output_format": {
                    "max_words": 400,
                    "tone": "technical",
                    "audience": "engineers",
                    "include_metrics": True,
                },
            }
        else:
            raise ValueError(f"Unknown template type: {template_type}")

    def list_personas_by_type(self) -> dict[str, list[BasePersona]]:
        """Get personas grouped by type and source.

        Returns:
            Dictionary with persona lists organized by source type
        """
        built_in = []
        package_yaml = []
        user_yaml = []

        for persona in self._personas.values():
            if isinstance(persona, ConfigurablePersona):
                # Check if it's from user directory or package directory
                if str(persona.yaml_path).startswith(str(self.user_personas_dir)):
                    user_yaml.append(persona)
                else:
                    package_yaml.append(persona)
            else:
                built_in.append(persona)

        return {
            "built_in": built_in,
            "yaml": package_yaml + user_yaml,  # For backward compatibility
            "package_yaml": package_yaml,
            "user_yaml": user_yaml,
        }

    def get_persona_info(self, name: str) -> dict[str, Any]:
        """Get detailed information about a persona.

        Args:
            name: Name of the persona

        Returns:
            Dictionary with persona information

        Raises:
            ValueError: If persona not found
        """
        persona = self.get_persona(name)

        info = {
            "name": persona.name,
            "description": persona.description,
            "type": "yaml" if isinstance(persona, ConfigurablePersona) else "built_in",
        }

        if isinstance(persona, ConfigurablePersona):
            info.update(
                {
                    "yaml_path": str(persona.yaml_path),
                    "version": persona.config.get("version", "unknown"),
                    "author": persona.config.get("author", "unknown"),
                    "sections": len(persona.analysis_framework.get("sections", [])),
                    "max_words": persona.output_format.get("max_words", "unlimited"),
                    "tone": persona.output_format.get("tone", "not specified"),
                    "audience": persona.output_format.get("audience", "general"),
                }
            )

        return info

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
