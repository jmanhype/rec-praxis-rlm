"""Unit tests for exception hierarchy."""
import pytest

from rec_praxis_rlm.exceptions import (
    RecPraxisRLMError,
    MemoryError,
    StorageError,
    EmbeddingError,
    RetrievalError,
    RLMError,
    DocumentNotFoundError,
    SearchError,
    ExecutionError,
    PlannerError,
    ToolCallError,
    LMError,
    OptimizationError,
)


class TestBaseException:
    """Tests for base exception class."""

    def test_base_exception_is_exception(self) -> None:
        """Test that RecPraxisRLMError inherits from Exception."""
        assert issubclass(RecPraxisRLMError, Exception)

    def test_base_exception_with_message(self) -> None:
        """Test creating base exception with message."""
        error = RecPraxisRLMError("test error")
        assert str(error) == "test error"


class TestMemoryErrors:
    """Tests for memory-related exceptions."""

    def test_memory_error_inherits_from_base(self) -> None:
        """Test that MemoryError inherits from RecPraxisRLMError."""
        assert issubclass(MemoryError, RecPraxisRLMError)

    def test_storage_error(self) -> None:
        """Test StorageError."""
        assert issubclass(StorageError, MemoryError)
        error = StorageError("Failed to write to storage")
        assert "storage" in str(error).lower()

    def test_embedding_error(self) -> None:
        """Test EmbeddingError."""
        assert issubclass(EmbeddingError, MemoryError)
        error = EmbeddingError("Failed to compute embedding")
        assert "embedding" in str(error).lower()

    def test_retrieval_error(self) -> None:
        """Test RetrievalError."""
        assert issubclass(RetrievalError, MemoryError)
        error = RetrievalError("Failed to retrieve experiences")
        assert "retrieve" in str(error).lower()


class TestRLMErrors:
    """Tests for RLM context-related exceptions."""

    def test_rlm_error_inherits_from_base(self) -> None:
        """Test that RLMError inherits from RecPraxisRLMError."""
        assert issubclass(RLMError, RecPraxisRLMError)

    def test_document_not_found_error(self) -> None:
        """Test DocumentNotFoundError."""
        assert issubclass(DocumentNotFoundError, RLMError)
        error = DocumentNotFoundError("Document 'test' not found")
        assert "not found" in str(error).lower()

    def test_search_error(self) -> None:
        """Test SearchError."""
        assert issubclass(SearchError, RLMError)
        error = SearchError("Regex pattern timeout")
        assert "timeout" in str(error).lower()

    def test_execution_error(self) -> None:
        """Test ExecutionError."""
        assert issubclass(ExecutionError, RLMError)
        error = ExecutionError("Code execution failed")
        assert "execution" in str(error).lower()


class TestPlannerErrors:
    """Tests for planner-related exceptions."""

    def test_planner_error_inherits_from_base(self) -> None:
        """Test that PlannerError inherits from RecPraxisRLMError."""
        assert issubclass(PlannerError, RecPraxisRLMError)

    def test_tool_call_error(self) -> None:
        """Test ToolCallError."""
        assert issubclass(ToolCallError, PlannerError)
        error = ToolCallError("Tool call failed")
        assert "tool" in str(error).lower()

    def test_lm_error(self) -> None:
        """Test LMError."""
        assert issubclass(LMError, PlannerError)
        error = LMError("Language model API error")
        assert "error" in str(error).lower()

    def test_optimization_error(self) -> None:
        """Test OptimizationError."""
        assert issubclass(OptimizationError, PlannerError)
        error = OptimizationError("Optimization failed")
        assert "optimization" in str(error).lower()
