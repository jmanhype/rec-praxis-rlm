"""Unit tests for DSPy tool wrappers."""
import pytest
from unittest.mock import Mock, MagicMock, patch
import time

from rec_praxis_rlm.tools import create_recall_tool, create_search_tool, create_exec_tool
from rec_praxis_rlm.memory import ProceduralMemory, Experience
from rec_praxis_rlm.rlm import RLMContext
from rec_praxis_rlm.config import MemoryConfig, ReplConfig


class TestCreateRecallTool:
    """Tests for create_recall_tool wrapper."""

    def test_creates_callable_tool(self) -> None:
        """Test that create_recall_tool returns a callable."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        tool = create_recall_tool(memory)

        assert callable(tool)

    def test_tool_calls_memory_recall(self) -> None:
        """Test that tool delegates to memory.recall()."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))

        # Store some experiences
        memory.store(
            Experience(
                env_features=["test"],
                goal="test goal",
                action="test action",
                result="test result",
                success=True,
                timestamp=time.time(),
            )
        )

        tool = create_recall_tool(memory)

        # Call the tool
        result = tool(env_features=["test"], goal="test goal", top_k=3)

        # Should return experiences
        assert isinstance(result, list)

    def test_tool_signature(self) -> None:
        """Test that tool has correct signature for DSPy."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        tool = create_recall_tool(memory)

        # Tool should accept env_features, goal, top_k
        # This is validated by the fact that it works with these params
        result = tool(env_features=["a"], goal="b", top_k=1)
        assert isinstance(result, list)


class TestCreateSearchTool:
    """Tests for create_search_tool wrapper."""

    def test_creates_callable_tool(self) -> None:
        """Test that create_search_tool returns a callable."""
        ctx = RLMContext(ReplConfig())
        tool = create_search_tool(ctx)

        assert callable(tool)

    def test_tool_calls_context_grep(self) -> None:
        """Test that tool delegates to context.grep()."""
        ctx = RLMContext(ReplConfig())
        ctx.add_document("test_doc", "Line 1\nERROR: test\nLine 3")

        tool = create_search_tool(ctx)

        # Call the tool
        result = tool(pattern="ERROR", doc_id="test_doc")

        # Should return search matches
        assert isinstance(result, list)
        if len(result) > 0:
            # If matches found, should have match attributes
            assert hasattr(result[0], 'match_text')

    def test_tool_signature(self) -> None:
        """Test that tool has correct signature for DSPy."""
        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc", "test")

        tool = create_search_tool(ctx)

        # Tool should accept pattern, doc_id
        result = tool(pattern="test", doc_id="doc")
        assert isinstance(result, list)


class TestCreateExecTool:
    """Tests for create_exec_tool wrapper."""

    def test_creates_callable_tool(self) -> None:
        """Test that create_exec_tool returns a callable."""
        ctx = RLMContext(ReplConfig())
        tool = create_exec_tool(ctx)

        assert callable(tool)

    def test_tool_calls_context_safe_exec(self) -> None:
        """Test that tool delegates to context.safe_exec()."""
        ctx = RLMContext(ReplConfig())

        tool = create_exec_tool(ctx)

        # Call the tool
        result = tool(code="2 + 2")

        # Should return execution result
        assert hasattr(result, 'success')
        assert hasattr(result, 'output')

    def test_tool_signature(self) -> None:
        """Test that tool has correct signature for DSPy."""
        ctx = RLMContext(ReplConfig())
        tool = create_exec_tool(ctx)

        # Tool should accept code, context_vars
        result = tool(code="x + y", context_vars={"x": 1, "y": 2})
        assert hasattr(result, 'success')

    def test_tool_returns_string_representation(self) -> None:
        """Test that tool can return string for DSPy compatibility."""
        ctx = RLMContext(ReplConfig())
        tool = create_exec_tool(ctx)

        result = tool(code="2 + 2")

        # Result should be convertible to string for DSPy
        str_result = str(result)
        assert isinstance(str_result, str)
        assert len(str_result) > 0
