"""Optimization example using DSPy MIPROv2.

Demonstrates how to optimize an agent's performance using DSPy's built-in optimizers.

Note: This example requires DSPy and an LM API key to run.

Run: OPENAI_API_KEY=your_key python examples/optimization.py
"""
from rec_praxis_rlm import (
    ProceduralMemory,
    RLMContext,
    PraxisRLMPlanner,
    MemoryConfig,
    ReplConfig,
    PlannerConfig,
    memory_retrieval_quality,
)


def demonstrate_optimization():
    """Show how to optimize agent performance."""
    print("=== Agent Optimization with DSPy ===\n")

    print("This example demonstrates:")
    print("1. Creating a planner with memory and context")
    print("2. Defining training examples")
    print("3. Using MIPROv2 optimizer to improve performance")
    print("4. Evaluating before/after metrics")
    print()

    print("Note: Full optimization requires:")
    print("  - DSPy installation (pip install dspy-ai)")
    print("  - LM API key (export OPENAI_API_KEY=...)")
    print("  - Training dataset with ground truth")
    print()

    # Example training data structure
    print("Example training data format:")
    training_example = {
        "goal": "analyze database errors",
        "env_features": ["production", "high_traffic"],
        "expected_action": "recall similar incidents and apply fix",
        "expected_quality": 0.8,
    }
    print(f"  {training_example}")
    print()

    # Example metric usage
    print("Example metric evaluation:")
    from rec_praxis_rlm import Experience
    import time

    example = {
        "env_features": ["web", "sidebar"],
        "goal": "extract data",
        "expected_success_rate": 1.0,
    }

    prediction = [
        Experience(
            env_features=["web", "sidebar", "article"],
            goal="extract product data",
            action="use selector",
            result="success",
            success=True,
            timestamp=time.time(),
        )
    ]

    score = memory_retrieval_quality(example, prediction, trace=None)
    print(f"  Memory retrieval quality score: {score:.2f}")
    print()

    # Optimization workflow
    print("Optimization workflow:")
    print("  1. Create planner with initial configuration")
    print("  2. Prepare training examples (10-50 examples)")
    print("  3. Define metric function (e.g., memory_retrieval_quality)")
    print("  4. Call planner.optimize(trainset, metric)")
    print("  5. Evaluate optimized planner on test set")
    print("  6. Save optimized planner: planner.save('optimized.json')")
    print()

    print("Example code:")
    print("""
    # Initialize planner
    memory = ProceduralMemory(MemoryConfig())
    planner = PraxisRLMPlanner(memory, PlannerConfig(
        optimizer="miprov2",
        optimizer_auto_level="medium"
    ))

    # Prepare training data
    trainset = [
        # ... your training examples ...
    ]

    # Define custom metric
    def evaluate(example, prediction, trace):
        # Your evaluation logic
        return score

    # Optimize
    optimized_planner = planner.optimize(trainset, metric=evaluate)

    # Save optimized version
    optimized_planner.save("optimized_planner.json")
    """)


if __name__ == "__main__":
    demonstrate_optimization()
    print("\n✅ Optimization example complete!")
    print("\nNext steps:")
    print("  - Collect training examples from your use case")
    print("  - Define domain-specific metrics")
    print("  - Run optimization with MIPROv2 or SIMBA")
    print("  - Evaluate improvements on held-out test set")
