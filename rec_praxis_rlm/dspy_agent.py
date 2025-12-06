"""DSPy-based autonomous planning agent for RLM workflows."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TYPE_CHECKING
import json
from pathlib import Path

from rec_praxis_rlm.memory import ProceduralMemory
from rec_praxis_rlm.rlm import RLMContext
from rec_praxis_rlm.config import PlannerConfig
from rec_praxis_rlm.tools import create_recall_tool, create_search_tool, create_exec_tool
from rec_praxis_rlm.telemetry import emit_event

# Type stubs for optional dependencies
if TYPE_CHECKING:
    import dspy
    from dspy import ReAct
    import mlflow
    import mlflow.dspy
else:
    # Try to import dspy, but don't fail if it's not available
    # This allows tests to mock it
    try:
        import dspy
        from dspy import ReAct
    except ImportError:
        dspy = None  # type: ignore[assignment]
        ReAct = None  # type: ignore[assignment,misc]

    # Try to import mlflow, but it's optional
    try:
        import mlflow
        import mlflow.dspy
    except ImportError:  # pragma: no cover
        mlflow = None  # type: ignore[assignment]


class PraxisRLMPlanner:
    """Autonomous planning agent using DSPy ReAct with memory and context tools.

    Combines procedural memory recall with document inspection and safe code execution
    to autonomously plan solutions to goals.

    Attributes:
        memory: ProceduralMemory instance for experience recall
        config: PlannerConfig with LM and optimizer settings
        contexts: Dictionary of registered RLMContext instances
    """

    def __init__(self, memory: ProceduralMemory, config: PlannerConfig):
        """Initialize the planner with memory and configuration.

        Args:
            memory: ProceduralMemory instance for experience recall
            config: PlannerConfig with LM and optimizer settings

        Raises:
            ImportError: If DSPy is not installed
        """
        # Check if dspy is available (either real or mocked)
        if dspy is None:
            raise ImportError("dspy not installed. Install with: pip install dspy")

        self.memory = memory
        self.config = config
        self.contexts: dict[str, RLMContext] = {}

        # Configure DSPy LM
        lm = dspy.LM(config.lm_model, temperature=config.temperature)
        dspy.configure(lm=lm)

        # Enable MLflow tracing if configured
        if config.enable_mlflow_tracing and mlflow is not None:
            mlflow.dspy.autolog()

        # Create initial tool list with memory recall
        self._tools: list[Callable] = [create_recall_tool(memory)]

        # Initialize ReAct agent with signature
        # DSPy 3.0 requires a signature string: "input_field -> output_field"
        self._agent = ReAct(
            signature="question -> answer",
            tools=self._tools,
            max_iters=config.max_iters
        )

        # Initialize ThreadPoolExecutor for async operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="planner-async")

    def add_context(self, context: RLMContext, context_name: str) -> None:
        """Register an RLMContext and add its tools to the agent.

        Args:
            context: RLMContext instance to register
            context_name: Name to register the context under
        """
        # Register context
        self.contexts[context_name] = context

        # Create tools for this context
        search_tool = create_search_tool(context)
        exec_tool = create_exec_tool(context)

        # Add tools to the agent's tool list
        self._tools.extend([search_tool, exec_tool])

        # Reinitialize agent with updated tools
        self._agent = ReAct(
            signature="question -> answer",
            tools=self._tools,
            max_iters=self.config.max_iters
        )

    def plan(self, goal: str, env_features: list[str]) -> str:
        """Generate a plan for the given goal using ReAct reasoning.

        Args:
            goal: The goal to plan for
            env_features: Environmental features describing the context

        Returns:
            String answer with the generated plan
        """
        # Emit telemetry event
        emit_event(
            "planner.plan",
            {
                "goal": goal,
                "env_features": env_features,
                "num_contexts": len(self.contexts),
            },
        )

        # Call ReAct agent
        env_str = ", ".join(env_features)
        question = (
            f"Goal: {goal}\n" f"Environment: {env_str}\n" f"Generate a plan to achieve this goal."
        )
        result = self._agent(question=question)

        # Extract answer
        answer = result.answer if hasattr(result, "answer") else str(result)

        return answer

    async def aplan(self, goal: str, env_features: list[str]) -> str:
        """Async version of plan().

        Uses ThreadPoolExecutor to run the synchronous plan() in a thread pool,
        preventing blocking of the event loop.

        Args:
            goal: The goal to plan for
            env_features: Environmental features describing the context

        Returns:
            String answer with the generated plan
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.plan, goal, env_features)

    def optimize(self, trainset: list[Any], metric: Callable, **kwargs: Any) -> "PraxisRLMPlanner":
        """Optimize the planner using DSPy compiler.

        Args:
            trainset: Training examples for optimization
            metric: Metric function to optimize for
            **kwargs: Additional arguments for the optimizer

        Returns:
            Optimized PraxisRLMPlanner instance
        """
        # Import optimizer based on config
        if self.config.optimizer == "miprov2":
            from dspy.teleprompt import MIPROv2

            optimizer = MIPROv2(metric=metric, auto=self.config.optimizer_auto_level, **kwargs)
        elif self.config.optimizer == "simba":
            from dspy.teleprompt import SIMBA

            optimizer = SIMBA(metric=metric, auto=self.config.optimizer_auto_level, **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        # Compile the agent
        optimized_agent = optimizer.compile(
            self._agent,
            trainset=trainset,
        )

        # Create new planner with optimized agent
        optimized_planner = PraxisRLMPlanner(self.memory, self.config)
        optimized_planner._agent = optimized_agent
        optimized_planner.contexts = self.contexts
        optimized_planner._tools = self._tools

        return optimized_planner

    def save(self, path: str) -> None:
        """Save planner state to a file.

        Args:
            path: Path to save the planner state
        """
        state = {
            "config": self.config.model_dump(),
            "context_names": list(self.contexts.keys()),
        }

        # Save state to JSON
        Path(path).write_text(json.dumps(state, indent=2))

        # Save DSPy program state
        # Note: DSPy programs can be saved using their save() method
        # This is a simplified version that saves config only

    @classmethod
    def load(cls, path: str, memory: ProceduralMemory) -> "PraxisRLMPlanner":
        """Load planner state from a file.

        Args:
            path: Path to the saved planner state
            memory: ProceduralMemory instance to use

        Returns:
            Restored PraxisRLMPlanner instance
        """
        # Load state from JSON
        state = json.loads(Path(path).read_text())

        # Restore config
        config = PlannerConfig(**state["config"])

        # Create planner
        planner = cls(memory, config)

        # Note: Contexts need to be re-registered by the caller
        # since they contain runtime state

        return planner
