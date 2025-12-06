"""Exception hierarchy for rec_praxis_rlm package."""


class RecPraxisRLMError(Exception):
    """Base exception for all rec_praxis_rlm errors."""

    pass


# Memory-related exceptions
class MemoryError(RecPraxisRLMError):
    """Base exception for procedural memory errors."""

    pass


class StorageError(MemoryError):
    """Exception raised for storage-related failures (JSONL I/O)."""

    pass


class EmbeddingError(MemoryError):
    """Exception raised for embedding computation failures."""

    pass


class RetrievalError(MemoryError):
    """Exception raised for experience retrieval failures."""

    pass


# RLM context-related exceptions
class RLMError(RecPraxisRLMError):
    """Base exception for RLM context errors."""

    pass


class DocumentNotFoundError(RLMError):
    """Exception raised when requested document ID is not found."""

    pass


class SearchError(RLMError):
    """Exception raised for search operation failures (grep, regex)."""

    pass


class ExecutionError(RLMError):
    """Exception raised for safe code execution failures."""

    pass


# Planner-related exceptions
class PlannerError(RecPraxisRLMError):
    """Base exception for planner errors."""

    pass


class ToolCallError(PlannerError):
    """Exception raised for tool call failures in ReAct agent."""

    pass


class LMError(PlannerError):
    """Exception raised for language model API failures."""

    pass


class OptimizationError(PlannerError):
    """Exception raised for optimizer compilation failures."""

    pass
