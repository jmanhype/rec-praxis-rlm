"""Unit tests for PraxisRLMPlanner DSPy agent."""
import pytest
from unittest.mock import Mock, MagicMock, patch
import time

from rec_praxis_rlm.dspy_agent import PraxisRLMPlanner
from rec_praxis_rlm.memory import ProceduralMemory, Experience
from rec_praxis_rlm.rlm import RLMContext
from rec_praxis_rlm.config import MemoryConfig, ReplConfig, PlannerConfig


class TestPraxisRLMPlannerInit:
    """Tests for PraxisRLMPlanner initialization."""

    @patch('rec_praxis_rlm.dspy_agent.dspy', None)
    def test_init_raises_import_error_when_dspy_not_available(self) -> None:
        """Test that __init__ raises ImportError when DSPy not available."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        config = PlannerConfig()

        with pytest.raises(ImportError, match="dspy not installed"):
            PraxisRLMPlanner(memory, config)

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_init_configures_dspy_lm(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test that __init__ configures DSPy LM."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        config = PlannerConfig(lm_model="openai/gpt-4o-mini")

        planner = PraxisRLMPlanner(memory, config)

        # Should have configured DSPy
        assert planner.config == config
        assert planner.memory == memory

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    @patch('rec_praxis_rlm.dspy_agent.mlflow')
    def test_init_enables_mlflow_tracing(self, mock_mlflow: Mock, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test that MLflow tracing is enabled if configured."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        config = PlannerConfig(
            lm_model="openai/gpt-4o-mini", enable_mlflow_tracing=True
        )

        planner = PraxisRLMPlanner(memory, config)

        # MLflow autolog should be called
        mock_mlflow.dspy.autolog.assert_called()

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_init_initializes_tool_list(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test that tool list is initialized."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        config = PlannerConfig()

        planner = PraxisRLMPlanner(memory, config)

        # Should have empty contexts initially
        assert planner.contexts == {}


class TestPraxisRLMPlannerAddContext:
    """Tests for add_context method."""

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_add_context_registers_context(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test that add_context registers RLMContext."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        planner = PraxisRLMPlanner(memory, PlannerConfig())

        ctx = RLMContext(ReplConfig())
        planner.add_context(ctx, context_name="logs")

        # Context should be registered
        assert "logs" in planner.contexts
        assert planner.contexts["logs"] == ctx

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_add_multiple_contexts(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test that multiple contexts can be added."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        planner = PraxisRLMPlanner(memory, PlannerConfig())

        ctx1 = RLMContext(ReplConfig())
        ctx2 = RLMContext(ReplConfig())

        planner.add_context(ctx1, context_name="logs")
        planner.add_context(ctx2, context_name="docs")

        assert len(planner.contexts) == 2
        assert "logs" in planner.contexts
        assert "docs" in planner.contexts


class TestPraxisRLMPlannerPlan:
    """Tests for plan method."""

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_plan_returns_string(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test that plan() returns a string answer."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        planner = PraxisRLMPlanner(memory, PlannerConfig())

        # Mock the ReAct agent to return a result
        mock_agent = MagicMock()
        mock_agent.return_value = MagicMock(answer="Test answer")
        planner._agent = mock_agent

        answer = planner.plan(goal="test goal", env_features=["test"])

        assert isinstance(answer, str)
        assert len(answer) > 0

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_plan_emits_telemetry(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test that plan() emits telemetry events."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        planner = PraxisRLMPlanner(memory, PlannerConfig())

        # Mock agent
        mock_agent = MagicMock()
        mock_agent.return_value = MagicMock(answer="Test answer")
        planner._agent = mock_agent

        with patch('rec_praxis_rlm.dspy_agent.emit_event') as mock_emit:
            answer = planner.plan(goal="test goal", env_features=["test"])

            # Should have emitted telemetry
            mock_emit.assert_called()


class TestPraxisRLMPlannerAsync:
    """Tests for async methods."""

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    @pytest.mark.asyncio
    async def test_aplan_delegates_to_plan(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test that aplan() delegates to plan()."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        planner = PraxisRLMPlanner(memory, PlannerConfig())

        # Mock agent
        mock_agent = MagicMock()
        mock_agent.return_value = MagicMock(answer="Async answer")
        planner._agent = mock_agent

        answer = await planner.aplan(goal="test goal", env_features=["test"])

        assert isinstance(answer, str)


class TestPraxisRLMPlannerOptimize:
    """Tests for optimize method."""

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_optimize_with_miprov2(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test optimize() with MIPROv2 optimizer."""
        with patch('dspy.teleprompt.MIPROv2') as mock_mipro:
            memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
            config = PlannerConfig(optimizer="miprov2")
            planner = PraxisRLMPlanner(memory, config)

            # Mock optimizer and compilation
            mock_optimizer_instance = MagicMock()
            mock_optimized_agent = MagicMock()
            mock_optimizer_instance.compile.return_value = mock_optimized_agent
            mock_mipro.return_value = mock_optimizer_instance

            # Mock metric
            mock_metric = MagicMock()
            trainset = [{"example": 1}]

            optimized_planner = planner.optimize(trainset, mock_metric)

            # Should have created MIPROv2 optimizer
            mock_mipro.assert_called_once()
            # Should have compiled
            mock_optimizer_instance.compile.assert_called_once()
            # Should return a new planner
            assert isinstance(optimized_planner, PraxisRLMPlanner)

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_optimize_with_simba(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test optimize() with SIMBA optimizer."""
        with patch('dspy.teleprompt.SIMBA') as mock_simba:
            memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
            config = PlannerConfig(optimizer="simba")
            planner = PraxisRLMPlanner(memory, config)

            # Mock optimizer and compilation
            mock_optimizer_instance = MagicMock()
            mock_optimized_agent = MagicMock()
            mock_optimizer_instance.compile.return_value = mock_optimized_agent
            mock_simba.return_value = mock_optimizer_instance

            # Mock metric
            mock_metric = MagicMock()
            trainset = [{"example": 1}]

            optimized_planner = planner.optimize(trainset, mock_metric)

            # Should have created SIMBA optimizer
            mock_simba.assert_called_once()
            # Should have compiled
            mock_optimizer_instance.compile.assert_called_once()
            # Should return a new planner
            assert isinstance(optimized_planner, PraxisRLMPlanner)

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_optimize_with_unsupported_optimizer_raises_error(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test optimize() raises ValueError for unsupported optimizer."""
        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        config = PlannerConfig(optimizer="unsupported")
        planner = PraxisRLMPlanner(memory, config)

        mock_metric = MagicMock()
        trainset = [{"example": 1}]

        with pytest.raises(ValueError, match="Unsupported optimizer"):
            planner.optimize(trainset, mock_metric)

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_optimize_preserves_contexts_and_tools(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test optimize() preserves contexts and tools in optimized planner."""
        with patch('dspy.teleprompt.MIPROv2') as mock_mipro:
            memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
            config = PlannerConfig(optimizer="miprov2")
            planner = PraxisRLMPlanner(memory, config)

            # Add a context
            ctx = RLMContext(ReplConfig())
            planner.add_context(ctx, "test_ctx")

            # Mock optimizer
            mock_optimizer_instance = MagicMock()
            mock_optimized_agent = MagicMock()
            mock_optimizer_instance.compile.return_value = mock_optimized_agent
            mock_mipro.return_value = mock_optimizer_instance

            mock_metric = MagicMock()
            trainset = [{"example": 1}]

            optimized_planner = planner.optimize(trainset, mock_metric)

            # Optimized planner should have the same contexts
            assert "test_ctx" in optimized_planner.contexts
            assert optimized_planner.contexts["test_ctx"] == ctx


class TestPraxisRLMPlannerSaveLoad:
    """Tests for save/load methods."""

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_save_creates_file(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test that save() creates a file."""
        import tempfile
        import os

        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        planner = PraxisRLMPlanner(memory, PlannerConfig())

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "planner.json")
            planner.save(save_path)

            # File should exist
            assert os.path.exists(save_path)

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_load_restores_planner(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test that load() restores a saved planner."""
        import tempfile
        import os

        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        planner1 = PraxisRLMPlanner(memory, PlannerConfig())

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "planner.json")
            planner1.save(save_path)

            # Load into new planner
            planner2 = PraxisRLMPlanner.load(save_path, memory)

            assert planner2 is not None
            assert isinstance(planner2, PraxisRLMPlanner)


class TestPraxisRLMPlannerAsync:
    """Tests for async planning methods."""

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_aplan_uses_threadpool(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test that aplan() runs in thread pool without blocking event loop."""
        import asyncio

        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))

        # Set up mock agent that returns a result with answer attribute
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "Test plan output"
        mock_agent_instance.return_value = mock_result
        mock_react.return_value = mock_agent_instance

        planner = PraxisRLMPlanner(memory, PlannerConfig())

        async def test_async():
            result = await planner.aplan(
                goal="Test goal",
                env_features=["feature1", "feature2"]
            )
            return result

        result = asyncio.run(test_async())
        assert result == "Test plan output"

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_aplan_concurrent_execution(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test that multiple aplan() calls can run concurrently."""
        import asyncio

        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))

        # Mock ReAct agent to return different plans
        mock_agent_instance = MagicMock()
        call_count = [0]

        def mock_agent_call(*args, **kwargs):
            mock_result = MagicMock()
            mock_result.answer = f"Plan {call_count[0]}"
            call_count[0] += 1
            return mock_result

        mock_agent_instance.side_effect = mock_agent_call
        mock_react.return_value = mock_agent_instance

        planner = PraxisRLMPlanner(memory, PlannerConfig())

        async def run_concurrent_plans():
            # Run 3 concurrent planning tasks
            tasks = [
                planner.aplan(goal=f"Goal {i}", env_features=[f"feature{i}"])
                for i in range(3)
            ]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run_concurrent_plans())

        # All 3 should complete successfully
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    @patch('rec_praxis_rlm.dspy_agent.ReAct')
    @patch('rec_praxis_rlm.dspy_agent.dspy')
    def test_aplan_passes_parameters_correctly(self, mock_dspy: Mock, mock_react: Mock) -> None:
        """Test that aplan() correctly passes parameters to plan()."""
        import asyncio

        memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))

        # Mock ReAct agent
        mock_agent_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "Detailed plan"
        mock_agent_instance.return_value = mock_result
        mock_react.return_value = mock_agent_instance

        planner = PraxisRLMPlanner(memory, PlannerConfig())

        async def test_params():
            result = await planner.aplan(
                goal="Build auth system",
                env_features=["Python", "FastAPI", "JWT"]
            )
            return result

        result = asyncio.run(test_params())
        assert result == "Detailed plan"

        # Verify agent was called with correct question format
        mock_agent_instance.assert_called_once()
        call_args = mock_agent_instance.call_args
        assert "Build auth system" in call_args[1]["question"]
        assert "Python" in call_args[1]["question"]
