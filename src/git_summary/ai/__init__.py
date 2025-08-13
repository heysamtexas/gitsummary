"""AI integration module for git-summary.

This module provides AI-powered analysis and summarization capabilities
using LiteLLM for multi-provider support, with different personas for
varied analysis perspectives.
"""

from .client import LLMClient
from .personas import BasePersona, PersonaManager, TechnicalAnalystPersona

__all__ = ["LLMClient", "BasePersona", "TechnicalAnalystPersona", "PersonaManager"]
