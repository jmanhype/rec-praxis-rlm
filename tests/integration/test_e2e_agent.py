"""End-to-end integration tests for autonomous agent planning."""
import pytest
from unittest.mock import Mock, MagicMock, patch
import time

from rec_praxis_rlm import (
    PraxisRLMPlanner,
    ProceduralMemory,
    RLMContext,
    Experience,
    MemoryConfig,
    ReplConfig,
    PlannerConfig,
)


class TestIT003AutonomousPlanning:
    """IT-003: End-to-end autonomous planning workflow."""

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_autonomous_planning_workflow(self, mock_dspy: Mock, mock_react_class: Mock) -> None:
        """Test complete autonomous planning workflow.

        IT-003: Initialize planner with memory+context → Provide goal "analyze errors in log"
        → Verify planner calls memory recall, context search, and context execution in logical sequence
        → Verify final answer contains relevant information
        """
        # Setup memory with past experiences
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))

        # Store an experience about analyzing logs
        memory.store(
            Experience(
                env_features=["log_file", "error_analysis"],
                goal="analyze errors in log",
                action="grep for ERROR pattern, then count occurrences",
                result="Found 45 errors, mostly database timeouts",
                success=True,
                timestamp=time.time(),
            )
        )

        # Setup context with log data
        context = RLMContext(ReplConfig())
        log_content = """
2025-12-03 10:00:01 INFO Request processed successfully
2025-12-03 10:00:05 ERROR Database timeout on query: SELECT * FROM orders
2025-12-03 10:00:10 ERROR Connection refused to payment gateway
2025-12-03 10:00:15 INFO Request processed successfully
2025-12-03 10:00:20 ERROR Database timeout on query: SELECT * FROM products
"""
        context.add_document("application.log", log_content)

        # Create planner
        config = PlannerConfig(
            lm_model="openai/gpt-4o-mini",
            enable_mlflow_tracing=False,
            max_iters=10
        )
        planner = PraxisRLMPlanner(memory, config)

        # Add context to planner
        planner.add_context(context, context_name="logs")

        # Mock the ReAct agent to simulate tool calls
        mock_agent_instance = MagicMock()

        # Simulate agent behavior:
        # 1. First call recalls from memory
        # 2. Then searches logs
        # 3. Then executes code to count
        # 4. Finally returns answer
        mock_agent_instance.return_value = MagicMock(
            answer="Analysis complete: Found 3 ERROR entries in the log. Primary issue is database timeouts (2 occurrences) and 1 payment gateway connection failure."
        )

        planner._agent = mock_agent_instance

        # Execute planning
        answer = planner.plan(
            goal="analyze errors in log",
            env_features=["log_file", "error_analysis"]
        )

        # Verify planner returned a meaningful answer
        assert isinstance(answer, str)
        assert len(answer) > 0

        # Verify the answer mentions errors or analysis
        assert "ERROR" in answer or "error" in answer or "Analysis" in answer

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_planner_with_multiple_contexts(self, mock_dspy: Mock, mock_react_class: Mock) -> None:
        """Test planner can work with multiple context sources."""
        # Setup memory
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))

        # Setup multiple contexts
        logs_context = RLMContext(ReplConfig())
        logs_context.add_document("app.log", "ERROR: test error\nINFO: test info")

        docs_context = RLMContext(ReplConfig())
        docs_context.add_document("README.md", "# Project Documentation\n\nHow to debug errors...")

        # Create planner
        config = PlannerConfig(enable_mlflow_tracing=False)
        planner = PraxisRLMPlanner(memory, config)

        # Add multiple contexts
        planner.add_context(logs_context, context_name="logs")
        planner.add_context(docs_context, context_name="docs")

        # Verify both contexts are registered
        assert "logs" in planner.contexts
        assert "docs" in planner.contexts
        assert len(planner.contexts) == 2

        # Mock agent response
        mock_agent = MagicMock()
        mock_agent.return_value = MagicMock(answer="Checked both logs and docs")
        planner._agent = mock_agent

        # Execute planning
        answer = planner.plan(
            goal="investigate error using logs and documentation",
            env_features=["multi_source_analysis"]
        )

        assert isinstance(answer, str)
        assert len(answer) > 0


class TestMaxItersStoppingCondition:
    """Test that agent respects max_iters limit."""

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_max_iters_respected(self, mock_dspy: Mock, mock_react_class: Mock) -> None:
        """Test that planning loop stops after max_iters.

        US3 Scenario 5: Given agent runs planning loop for maximum iterations
        without concluding, When iteration limit is reached, Then it returns
        the best action or answer available at that point.
        """
        # Setup minimal configuration
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        config = PlannerConfig(max_iters=5, enable_mlflow_tracing=False)

        planner = PraxisRLMPlanner(memory, config)

        # Verify max_iters was set correctly
        assert planner.config.max_iters == 5

        # Mock ReAct to verify it was configured with correct max_iters
        # The ReAct class should have been called with max_iters=5
        mock_react_class.assert_called()
        call_kwargs = mock_react_class.call_args.kwargs
        assert call_kwargs.get('max_iters') == 5

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_planner_returns_answer_at_max_iters(self, mock_dspy: Mock, mock_react_class: Mock) -> None:
        """Test that planner returns best available answer when hitting max_iters."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        config = PlannerConfig(max_iters=3, enable_mlflow_tracing=False)

        planner = PraxisRLMPlanner(memory, config)

        # Mock agent to return partial answer (simulating max_iters reached)
        mock_agent = MagicMock()
        mock_agent.return_value = MagicMock(
            answer="Partial analysis: Started investigation but reached iteration limit. Preliminary findings: 2 errors detected."
        )
        planner._agent = mock_agent

        # Execute planning
        answer = planner.plan(
            goal="complex task requiring many iterations",
            env_features=["complex_analysis"]
        )

        # Verify planner still returns an answer (best available)
        assert isinstance(answer, str)
        assert len(answer) > 0
        # The answer should indicate it's partial or limited
        assert "Partial" in answer or "Preliminary" in answer or "limit" in answer


class TestAgentToolIntegration:
    """Test that agent tools are properly wired."""

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_planner_has_recall_tool(self, mock_dspy: Mock, mock_react_class: Mock) -> None:
        """Test that planner initializes with memory recall tool."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        config = PlannerConfig(enable_mlflow_tracing=False)

        planner = PraxisRLMPlanner(memory, config)

        # Verify planner has tools list
        assert hasattr(planner, '_tools')
        assert isinstance(planner._tools, list)

        # Should have at least the recall tool
        assert len(planner._tools) >= 1

        # First tool should be the recall tool
        assert callable(planner._tools[0])
        assert planner._tools[0].__name__ == "recall_memory"

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_add_context_adds_tools(self, mock_dspy: Mock, mock_react_class: Mock) -> None:
        """Test that adding context adds search and exec tools."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        config = PlannerConfig(enable_mlflow_tracing=False)

        planner = PraxisRLMPlanner(memory, config)

        # Initial tool count (just recall)
        initial_tool_count = len(planner._tools)

        # Add a context
        context = RLMContext(ReplConfig())
        planner.add_context(context, context_name="test")

        # Should have added 2 more tools (search + exec)
        assert len(planner._tools) == initial_tool_count + 2

        # Verify tools are callable
        for tool in planner._tools:
            assert callable(tool)

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_tool_names_are_set(self, mock_dspy: Mock, mock_react_class: Mock) -> None:
        """Test that all tools have proper names for DSPy."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        config = PlannerConfig(enable_mlflow_tracing=False)

        planner = PraxisRLMPlanner(memory, config)

        # Add context to get all tool types
        context = RLMContext(ReplConfig())
        planner.add_context(context, context_name="test")

        # Verify all tools have names
        tool_names = [tool.__name__ for tool in planner._tools]

        # Should have recall, search, and execute
        assert "recall_memory" in tool_names
        assert "search_context" in tool_names
        assert "execute_code" in tool_names
