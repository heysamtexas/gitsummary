"""AI integration module for git-summary.

This module provides AI-powered analysis and summarization capabilities
using LiteLLM for multi-provider support, with different personas for
varied analysis perspectives and rich context gathering for detailed analysis.
"""

from .client import LLMClient
from .context import ContextGatheringEngine, RichContext, TokenBudget
from .orchestrator import ActivitySummarizer
from .personas import BasePersona, PersonaManager, TechnicalAnalystPersona

__all__ = [
    "LLMClient",
    "BasePersona",
    "TechnicalAnalystPersona",
    "PersonaManager",
    "ContextGatheringEngine",
    "RichContext",
    "TokenBudget",
    "ActivitySummarizer",
]
