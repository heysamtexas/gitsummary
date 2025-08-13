"""Tests for AI persona system."""

from typing import Any

import pytest

from git_summary.ai.personas import BasePersona, PersonaManager, TechnicalAnalystPersona


class MockPersona(BasePersona):
    """Mock persona for testing purposes."""

    def get_system_prompt(self) -> str:
        return "You are a test persona."

    def get_summary_instructions(self, context: dict[str, Any]) -> str:
        return "Test instructions for analysis."


class TestBasePersona:
    """Test the base persona interface."""

    def test_persona_initialization(self):
        """Test persona initialization with name and description."""
        persona = MockPersona("Test", "A test persona")
        assert persona.name == "Test"
        assert persona.description == "A test persona"

    def test_persona_string_representation(self):
        """Test string representations of personas."""
        persona = MockPersona("Test", "A test persona")
        assert str(persona) == "Test: A test persona"
        assert "MockPersona" in repr(persona)
        assert "Test" in repr(persona)

    def test_abstract_methods_implemented(self):
        """Test that concrete personas implement required methods."""
        persona = MockPersona("Test", "A test persona")
        assert persona.get_system_prompt() == "You are a test persona."
        assert "Test instructions" in persona.get_summary_instructions({})


class TestTechnicalAnalystPersona:
    """Test the Technical Analyst persona."""

    def setup_method(self):
        """Set up test persona."""
        self.persona = TechnicalAnalystPersona()

    def test_persona_properties(self):
        """Test persona name and description."""
        assert self.persona.name == "Tech Analyst"
        assert "technical analysis" in self.persona.description.lower()
        assert "code quality" in self.persona.description.lower()

    def test_system_prompt_content(self):
        """Test that system prompt contains technical focus areas."""
        prompt = self.persona.get_system_prompt()

        # Check for key technical analysis areas
        assert "Technical Analyst" in prompt
        assert "code quality" in prompt.lower() or "Code Quality" in prompt
        assert "architecture" in prompt.lower()
        assert "engineering" in prompt.lower()
        assert "testing" in prompt.lower()

    def test_summary_instructions_basic_context(self):
        """Test summary instructions with basic context."""
        context = {
            "commit_count": 5,
            "repositories": ["repo1", "repo2"],
            "pull_requests": [],
            "issues": [],
            "releases": [],
        }

        instructions = self.persona.get_summary_instructions(context)

        # Check that context metrics are reflected
        assert "5 commits" in instructions
        assert "2 repositories" in instructions
        assert "Pull requests: No" in instructions

    def test_summary_instructions_rich_context(self):
        """Test summary instructions with rich context data."""
        context = {
            "commit_count": 12,
            "repositories": ["repo1", "repo2", "repo3"],
            "pull_requests": [{"id": 1}],
            "issues": [{"id": 1}],
            "releases": [{"id": 1}],
        }

        instructions = self.persona.get_summary_instructions(context)

        # Check that all activity types are detected
        assert "12 commits" in instructions
        assert "3 repositories" in instructions
        assert "Pull requests: Yes" in instructions
        assert "Issue activity: Yes" in instructions
        assert "Releases: Yes" in instructions

    def test_instructions_contain_analysis_framework(self):
        """Test that instructions provide clear analysis framework."""
        context = {"commit_count": 3, "repositories": ["test"]}
        instructions = self.persona.get_summary_instructions(context)

        # Check for structured analysis sections
        assert "Code Change Analysis" in instructions
        assert "Collaboration Patterns" in instructions
        assert "Project Health" in instructions
        assert "Engineering Insights" in instructions

        # Check for output format guidance
        assert "Technical Summary" in instructions
        assert "Key Technical Themes" in instructions


class TestPersonaManager:
    """Test the persona manager."""

    def setup_method(self):
        """Set up test manager."""
        self.manager = PersonaManager()

    def test_manager_initialization(self):
        """Test that manager initializes with default personas."""
        personas = self.manager.list_personas()
        assert len(personas) > 0

        # Should include technical analyst
        names = [p.name for p in personas]
        assert "Tech Analyst" in names

    def test_get_persona_by_name(self):
        """Test retrieving personas by name."""
        # Test exact name
        persona = self.manager.get_persona("Tech Analyst")
        assert isinstance(persona, TechnicalAnalystPersona)
        assert persona.name == "Tech Analyst"

        # Test case insensitive lookup
        persona = self.manager.get_persona("TECH ANALYST")
        assert isinstance(persona, TechnicalAnalystPersona)

        # Test with extra whitespace
        persona = self.manager.get_persona("  tech analyst  ")
        assert isinstance(persona, TechnicalAnalystPersona)

    def test_get_nonexistent_persona(self):
        """Test error when requesting non-existent persona."""
        with pytest.raises(ValueError, match="not found"):
            self.manager.get_persona("NonExistent Persona")

    def test_get_default_persona(self):
        """Test getting the default persona."""
        default = self.manager.get_default_persona()
        assert isinstance(default, TechnicalAnalystPersona)
        assert default.name == "Tech Analyst"

    def test_list_personas(self):
        """Test listing all available personas."""
        personas = self.manager.list_personas()
        assert isinstance(personas, list)
        assert len(personas) >= 1

        # All items should be personas
        for persona in personas:
            assert isinstance(persona, BasePersona)

    def test_register_custom_persona(self):
        """Test registering a custom persona."""
        custom_persona = MockPersona("Custom", "A custom test persona")

        # Register the persona
        self.manager.register_persona(custom_persona)

        # Should be retrievable
        retrieved = self.manager.get_persona("Custom")
        assert retrieved is custom_persona
        assert retrieved.name == "Custom"

        # Should appear in listing
        all_personas = self.manager.list_personas()
        assert custom_persona in all_personas

    def test_persona_name_normalization(self):
        """Test that persona names are normalized for lookup."""
        # Register with mixed case
        custom_persona = MockPersona("Mixed Case Name", "Test")
        self.manager.register_persona(custom_persona)

        # Should be retrievable with different casing
        assert self.manager.get_persona("mixed case name") is custom_persona
        assert self.manager.get_persona("MIXED CASE NAME") is custom_persona
        assert self.manager.get_persona("  Mixed Case Name  ") is custom_persona
