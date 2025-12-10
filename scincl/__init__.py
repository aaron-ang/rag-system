"""
SciNCL - Neighborhood Contrastive Learning for Scientific Documents
"""

from scincl.core import SciNCLIngestion, SciNCLRetrieval, BedrockLLMAssistant
from scincl.utils import load_or_create_artifacts

__all__ = [
    "SciNCLIngestion",
    "SciNCLRetrieval",
    "BedrockLLMAssistant",
    "load_or_create_artifacts",
]
