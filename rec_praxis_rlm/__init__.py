"""REC Praxis RLM - Retrieval-Enhanced Context for Praxis Reinforcement Learning Memory."""

__version__ = "0.1.0"

from rec_praxis_rlm.memory import ProceduralMemory, Experience
from rec_praxis_rlm.rlm import RLMContext, SearchMatch, ExecutionResult
from rec_praxis_rlm.config import MemoryConfig, ReplConfig, PlannerConfig
from rec_praxis_rlm.dspy_agent import PraxisRLMPlanner
from rec_praxis_rlm.telemetry import (
    setup_mlflow_tracing,
    add_telemetry_hook,
    emit_event,
)
from rec_praxis_rlm.metrics import (
    memory_retrieval_quality,
    SemanticF1Score,
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
]
