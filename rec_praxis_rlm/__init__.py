"""REC Praxis RLM - Retrieval-Enhanced Context for Praxis Reinforcement Learning Memory.

The package exposes a curated public API, but avoids importing heavy optional
dependencies (FAISS, sentence-transformers, DSPy, MLflow) at import time.
Public symbols are loaded lazily on first access via ``__getattr__``.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

try:
    __version__ = version("rec-praxis-rlm")
except PackageNotFoundError:  # pragma: no cover
    # Fallback for editable/source checkouts where metadata isn't installed.
    __version__ = "0.9.2"

# Public API surface (lazy-loaded below).
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Core
    "ProceduralMemory": ("rec_praxis_rlm.memory", "ProceduralMemory"),
    "Experience": ("rec_praxis_rlm.memory", "Experience"),
    "RLMContext": ("rec_praxis_rlm.rlm", "RLMContext"),
    "SearchMatch": ("rec_praxis_rlm.rlm", "SearchMatch"),
    "ExecutionResult": ("rec_praxis_rlm.rlm", "ExecutionResult"),
    # Configuration
    "MemoryConfig": ("rec_praxis_rlm.config", "MemoryConfig"),
    "ReplConfig": ("rec_praxis_rlm.config", "ReplConfig"),
    "PlannerConfig": ("rec_praxis_rlm.config", "PlannerConfig"),
    # Planner / tools
    "PraxisRLMPlanner": ("rec_praxis_rlm.dspy_agent", "PraxisRLMPlanner"),
    # Semantic memory
    "FactStore": ("rec_praxis_rlm.fact_store", "FactStore"),
    "Fact": ("rec_praxis_rlm.fact_store", "Fact"),
    # Telemetry
    "setup_mlflow_tracing": ("rec_praxis_rlm.telemetry", "setup_mlflow_tracing"),
    "add_telemetry_hook": ("rec_praxis_rlm.telemetry", "add_telemetry_hook"),
    "emit_event": ("rec_praxis_rlm.telemetry", "emit_event"),
    # Metrics
    "memory_retrieval_quality": ("rec_praxis_rlm.metrics", "memory_retrieval_quality"),
    "SemanticF1Score": ("rec_praxis_rlm.metrics", "SemanticF1Score"),
    # Types
    "Severity": ("rec_praxis_rlm.types", "Severity"),
    "OWASPCategory": ("rec_praxis_rlm.types", "OWASPCategory"),
    "Finding": ("rec_praxis_rlm.types", "Finding"),
    "CVEFinding": ("rec_praxis_rlm.types", "CVEFinding"),
    "SecretFinding": ("rec_praxis_rlm.types", "SecretFinding"),
    "AuditReport": ("rec_praxis_rlm.types", "AuditReport"),
    # Compression
    "ObservationCompressor": ("rec_praxis_rlm.compression", "ObservationCompressor"),
    "LLMProvider": ("rec_praxis_rlm.compression", "LLMProvider"),
    "OpenAIProvider": ("rec_praxis_rlm.compression", "OpenAIProvider"),
    # Privacy
    "PrivacyRedactor": ("rec_praxis_rlm.privacy", "PrivacyRedactor"),
    "RedactionPattern": ("rec_praxis_rlm.privacy", "RedactionPattern"),
    "classify_privacy_level": ("rec_praxis_rlm.privacy", "classify_privacy_level"),
    "redact_secrets": ("rec_praxis_rlm.privacy", "redact_secrets"),
    # Concepts
    "ConceptTagger": ("rec_praxis_rlm.concepts", "ConceptTagger"),
    # Experience Classification
    "ExperienceClassifier": ("rec_praxis_rlm.experience_classifier", "ExperienceClassifier"),
    # Endless Mode
    "EndlessAgent": ("rec_praxis_rlm.endless_mode", "EndlessAgent"),
    "TokenBudget": ("rec_praxis_rlm.endless_mode", "TokenBudget"),
    "CompressionConfig": ("rec_praxis_rlm.endless_mode", "CompressionConfig"),
}


def __getattr__(name: str) -> Any:
    """Lazily import public API symbols on first access."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'rec_praxis_rlm' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_IMPORTS.keys()))


if TYPE_CHECKING:  # pragma: no cover
    # Import for type checkers only.
    from rec_praxis_rlm.memory import ProceduralMemory, Experience
    from rec_praxis_rlm.rlm import RLMContext, SearchMatch, ExecutionResult
    from rec_praxis_rlm.config import MemoryConfig, ReplConfig, PlannerConfig
    from rec_praxis_rlm.dspy_agent import PraxisRLMPlanner
    from rec_praxis_rlm.fact_store import FactStore, Fact
    from rec_praxis_rlm.telemetry import setup_mlflow_tracing, add_telemetry_hook, emit_event
    from rec_praxis_rlm.metrics import memory_retrieval_quality, SemanticF1Score
    from rec_praxis_rlm.types import (
        Severity,
        OWASPCategory,
        Finding,
        CVEFinding,
        SecretFinding,
        AuditReport,
    )
    from rec_praxis_rlm.compression import ObservationCompressor, LLMProvider, OpenAIProvider
    from rec_praxis_rlm.privacy import PrivacyRedactor, RedactionPattern, classify_privacy_level, redact_secrets
    from rec_praxis_rlm.concepts import ConceptTagger
    from rec_praxis_rlm.experience_classifier import ExperienceClassifier
    from rec_praxis_rlm.endless_mode import EndlessAgent, TokenBudget, CompressionConfig

__all__ = [
    # Version
    "__version__",
    # Core classes
    "ProceduralMemory",
    "Experience",
    "RLMContext",
    "SearchMatch",
    "ExecutionResult",
    "PraxisRLMPlanner",
    "FactStore",
    "Fact",
    # Configuration
    "MemoryConfig",
    "ReplConfig",
    "PlannerConfig",
    # Telemetry
    "setup_mlflow_tracing",
    "add_telemetry_hook",
    "emit_event",
    # Metrics
    "memory_retrieval_quality",
    "SemanticF1Score",
    # Types
    "Severity",
    "OWASPCategory",
    "Finding",
    "CVEFinding",
    "SecretFinding",
    "AuditReport",
    # Compression
    "ObservationCompressor",
    "LLMProvider",
    "OpenAIProvider",
    # Privacy
    "PrivacyRedactor",
    "RedactionPattern",
    "classify_privacy_level",
    "redact_secrets",
    # Concepts
    "ConceptTagger",
    # Experience Classification
    "ExperienceClassifier",
    # Endless Mode
    "EndlessAgent",
    "TokenBudget",
    "CompressionConfig",
]
