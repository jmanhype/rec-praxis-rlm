"""DSPy tool wrappers for memory recall, context search, and code execution."""

from typing import Callable, Optional, Any

from rec_praxis_rlm.memory import ProceduralMemory
from rec_praxis_rlm.rlm import RLMContext, SearchMatch, ExecutionResult


def create_recall_tool(memory: ProceduralMemory) -> Callable:
    """Create a DSPy-compatible tool for memory recall.

    Args:
        memory: ProceduralMemory instance

    Returns:
        Callable tool function that recalls experiences
    """

    def recall_tool(env_features: list[str], goal: str, top_k: int = 6) -> list:
        """Recall similar experiences from procedural memory.

        Args:
            env_features: Environmental features to match
            goal: Goal description to match
            top_k: Number of top experiences to return

        Returns:
            List of Experience objects
        """
        return memory.recall(env_features=env_features, goal=goal, top_k=top_k)

    # Add metadata for DSPy
    recall_tool.__name__ = "recall_memory"
    recall_tool.__doc__ = (
        "Recall similar past experiences from memory based on environmental features and goal"
    )

    return recall_tool


def create_search_tool(context: RLMContext) -> Callable:
    """Create a DSPy-compatible tool for context search.

    Args:
        context: RLMContext instance

    Returns:
        Callable tool function that searches documents
    """

    def search_tool(
        pattern: str, doc_id: Optional[str] = None, max_matches: int = 10
    ) -> list[SearchMatch]:
        """Search documents for pattern using regex.

        Args:
            pattern: Regular expression pattern to search
            doc_id: Optional document ID to search (searches all if None)
            max_matches: Maximum number of matches to return

        Returns:
            List of SearchMatch objects
        """
        return context.grep(pattern=pattern, doc_id=doc_id, max_matches=max_matches)

    # Add metadata for DSPy
    search_tool.__name__ = "search_context"
    search_tool.__doc__ = "Search documents for pattern using regex, returns matches with context"

    return search_tool


def create_exec_tool(context: RLMContext) -> Callable:
    """Create a DSPy-compatible tool for safe code execution.

    Args:
        context: RLMContext instance

    Returns:
        Callable tool function that executes code safely
    """

    def exec_tool(code: str, context_vars: Optional[dict[str, Any]] = None) -> ExecutionResult:
        """Execute Python code safely in sandboxed environment.

        Args:
            code: Python code to execute
            context_vars: Variables to inject into execution context

        Returns:
            ExecutionResult with output, error, and metadata
        """
        if context_vars is None:
            context_vars = {}

        return context.safe_exec(code=code, context_vars=context_vars)

    # Add metadata for DSPy
    exec_tool.__name__ = "execute_code"
    exec_tool.__doc__ = "Execute Python code safely to transform or analyze data"

    return exec_tool
