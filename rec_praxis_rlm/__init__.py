"""REC Praxis RLM - Retrieval-Enhanced Context for Praxis Reinforcement Learning Memory."""

__version__ = "0.9.1"

from rec_praxis_rlm.memory import ProceduralMemory, Experience
from rec_praxis_rlm.rlm import RLMContext, SearchMatch, ExecutionResult
from rec_praxis_rlm.config import MemoryConfig, ReplConfig, PlannerConfig
from rec_praxis_rlm.dspy_agent import PraxisRLMPlanner
from rec_praxis_rlm.fact_store import FactStore, Fact
from rec_praxis_rlm.telemetry import (
    setup_mlflow_tracing,
    add_telemetry_hook,
    emit_event,
)
from rec_praxis_rlm.metrics import (
    memory_retrieval_quality,
    SemanticF1Score,
)
from rec_praxis_rlm.types import (
    Severity,
    OWASPCategory,
    Finding,
    CVEFinding,
    SecretFinding,
    AuditReport,
)
from rec_praxis_rlm.compression import (
    ObservationCompressor,
    LLMProvider,
    OpenAIProvider,
)

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
]
