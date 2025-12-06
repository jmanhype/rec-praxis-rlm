"""Quickstart example for rec_praxis_rlm.

Demonstrates all three core capabilities:
1. Procedural memory storage and retrieval
2. Context manipulation (search, extract, execute)
3. Autonomous planning with integrated tools

Run: python examples/quickstart.py
"""
import time
from rec_praxis_rlm import (
    ProceduralMemory,
    Experience,
    RLMContext,
    PraxisRLMPlanner,
    MemoryConfig,
    ReplConfig,
    PlannerConfig,
)


def demo_memory():
    """Demonstrate procedural memory - User Story 1."""
    print("\n=== User Story 1: Procedural Memory ===")

    # Initialize memory
    memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))

    # Store experiences
    memory.store(Experience(
        env_features=["web", "sidebar", "article"],
        goal="extract product prices",
        action="use CSS selector 'main .price'",
        result="Successfully extracted 25 prices",
        success=True,
        timestamp=time.time(),
    ))

    memory.store(Experience(
        env_features=["web", "sidebar", "pagination"],
        goal="extract product listings",
        action="iterate through paginated results",
        result="Extracted 100 products across 5 pages",
        success=True,
        timestamp=time.time(),
    ))

    # Retrieve similar experiences
    results = memory.recall(
        env_features=["web", "sidebar"],
        goal="extract data",
        top_k=2
    )

    print(f"Stored {memory.size()} experiences")
    print(f"Retrieved {len(results)} relevant experiences")
    for exp in results:
        print(f"  - {exp.goal}: {exp.action}")


def demo_context():
    """Demonstrate context manipulation - User Story 2."""
    print("\n=== User Story 2: Context Manipulation ===")

    # Create context with log data
    ctx = RLMContext(ReplConfig())

    log_data = """
2025-12-03 10:00:01 INFO Request processed
2025-12-03 10:00:05 ERROR Database timeout
2025-12-03 10:00:10 ERROR Connection refused
2025-12-03 10:00:15 INFO Request processed
"""
    ctx.add_document("app.log", log_data)

    # Search for patterns
    errors = ctx.grep(r"ERROR", doc_id="app.log")
    print(f"Found {len(errors)} error entries")

    # Execute safe code
    result = ctx.safe_exec(
        code="log_text.count('ERROR')",
        context_vars={"log_text": log_data}
    )
    print(f"Counted errors via execution: {result.output.strip()}")

    # Extract lines
    first_lines = ctx.head("app.log", n_lines=2)
    print(f"First 2 lines: {len(first_lines.splitlines())} lines")


def demo_planning():
    """Demonstrate autonomous planning - User Story 3."""
    print("\n=== User Story 3: Autonomous Planning ===")

    # This demo uses mocked DSPy components - see integration tests for full example
    print("Note: Autonomous planning requires DSPy and LM API keys.")
    print("See tests/integration/test_e2e_agent.py for complete examples.")
    print("Key capabilities:")
    print("  - Agent recalls past experiences from memory")
    print("  - Agent searches context documents")
    print("  - Agent executes code for transformations")
    print("  - Agent returns final answers autonomously")


if __name__ == "__main__":
    print("rec_praxis_rlm Quickstart")
    print("=" * 50)

    demo_memory()
    demo_context()
    demo_planning()

    print("\n" + "=" * 50)
    print("✅ Quickstart complete!")
    print("\nNext steps:")
    print("  - See examples/web_agent.py for web scraping use case")
    print("  - See examples/log_analyzer.py for log analysis use case")
    print("  - See examples/optimization.py for DSPy optimizer usage")
