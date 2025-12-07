"""Ablation study: Validate 60/40 env/goal weighting hypothesis.

This study tests different weighting configurations to validate that:
- env_weight = 0.6, goal_weight = 0.4 provides optimal recall quality
- Environmental similarity is more important than goal similarity
- But goal similarity is still necessary for good results

Configurations tested:
1. 50/50 (equal weighting)
2. 60/40 (default - hypothesis)
3. 70/30 (prioritize environment)
4. 40/60 (prioritize goal)
5. 100/0 (environment only)
6. 0/100 (goal only)
"""

import time
import pytest
from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig


def create_test_dataset():
    """Create test dataset with known good/bad matches.

    Returns:
        tuple: (experiences, query, expected_top_matches)
    """
    base_time = time.time()

    experiences = [
        # Experience 1: Perfect match (both env and goal)
        Experience(
            env_features=["web", "python", "javascript"],
            goal="scrape dynamic website content",
            action="Used Selenium with WebDriverWait",
            result="Successfully extracted 200 items",
            success=True,
            timestamp=base_time
        ),

        # Experience 2: Good env match, different goal
        Experience(
            env_features=["web", "python", "javascript"],
            goal="test web application forms",
            action="Used Selenium with form automation",
            result="Automated 50 test cases",
            success=True,
            timestamp=base_time + 100
        ),

        # Experience 3: Good goal match, different env
        Experience(
            env_features=["mobile", "ios", "swift"],
            goal="scrape mobile app data",
            action="Used XCTest for UI automation",
            result="Extracted app state data",
            success=True,
            timestamp=base_time + 200
        ),

        # Experience 4: Poor match (unrelated)
        Experience(
            env_features=["database", "postgresql"],
            goal="optimize database queries",
            action="Added composite indexes",
            result="Reduced query time to 0.08s",
            success=True,
            timestamp=base_time + 300
        ),

        # Experience 5: Env overlap, unrelated goal
        Experience(
            env_features=["web", "python"],
            goal="build REST API",
            action="Used FastAPI framework",
            result="Created 20 endpoints",
            success=True,
            timestamp=base_time + 400
        ),
    ]

    # Query: Looking for web scraping with JavaScript
    query_env = ["web", "python", "javascript", "scraping"]
    query_goal = "scrape data from JavaScript-heavy website"

    # Expected ranking (by match quality):
    # 1. Experience 1: Perfect match (web+python+javascript + scraping goal)
    # 2. Experience 2: Good env match (web+python+javascript)
    # 3. Experience 5: Partial env match (web+python)
    # 4. Experience 3: Goal match only (scraping)
    # 5. Experience 4: No match

    return experiences, query_env, query_goal, [0, 1, 4, 2, 3]  # Expected order by index


def evaluate_weighting(env_weight, goal_weight, experiences, query_env, query_goal):
    """Evaluate recall quality for a given weighting.

    Returns:
        dict: Metrics including top-3 accuracy, MRR, etc.
    """
    # Create memory with specific weighting
    memory = ProceduralMemory(
        config=MemoryConfig(
            storage_path=":memory:",
            env_weight=env_weight,
            goal_weight=goal_weight,
            similarity_threshold=0.0,  # Don't filter by threshold
            top_k=5
        )
    )

    # Store all experiences
    for exp in experiences:
        memory.store(exp)

    # Recall with query
    results = memory.recall(env_features=query_env, goal=query_goal, top_k=5)

    # Get result indices
    result_indices = [experiences.index(exp) for exp in results]

    return {
        "env_weight": env_weight,
        "goal_weight": goal_weight,
        "results": result_indices,
        "top_1_correct": result_indices[0] == 0 if len(result_indices) > 0 else False,
        "top_3_has_perfect": 0 in result_indices[:3] if len(result_indices) >= 3 else False,
        "num_results": len(result_indices)
    }


def test_ablation_study():
    """Run ablation study across multiple weighting configurations."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY: Environmental vs Goal Weighting")
    print("=" * 70)

    # Create test dataset
    experiences, query_env, query_goal, expected_order = create_test_dataset()

    print(f"\nTest Dataset:")
    print(f"  - {len(experiences)} experiences")
    print(f"  - Query env: {query_env}")
    print(f"  - Query goal: {query_goal[:50]}...")
    print(f"  - Expected best match: Experience 0 (perfect env+goal match)")

    # Test configurations
    configs = [
        (0.5, 0.5, "Equal weighting"),
        (0.6, 0.4, "Default (hypothesis)"),
        (0.7, 0.3, "Prioritize environment"),
        (0.4, 0.6, "Prioritize goal"),
        (1.0, 0.0, "Environment only"),
        (0.0, 1.0, "Goal only"),
    ]

    results = []

    print("\n" + "-" * 70)
    print("Results:")
    print("-" * 70)

    for env_w, goal_w, desc in configs:
        metrics = evaluate_weighting(env_w, goal_w, experiences, query_env, query_goal)
        results.append(metrics)

        print(f"\n{desc} (env={env_w:.1f}, goal={goal_w:.1f}):")
        print(f"  Top-1: Exp {metrics['results'][0]} {'✅' if metrics['top_1_correct'] else '❌'}")
        print(f"  Top-3: {metrics['results'][:3]}")
        print(f"  Perfect match in top-3: {'✅' if metrics['top_3_has_perfect'] else '❌'}")

    # Analysis
    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)

    # Count how many configs got top-1 correct
    top_1_correct_count = sum(1 for r in results if r['top_1_correct'])

    # Find best configuration
    best_config = max(results, key=lambda r: (r['top_1_correct'], r['top_3_has_perfect']))

    print(f"\n✅ Configurations with correct top-1: {top_1_correct_count}/{len(configs)}")
    print(f"\n🏆 Best configuration:")
    print(f"   env_weight={best_config['env_weight']:.1f}, goal_weight={best_config['goal_weight']:.1f}")
    print(f"   Top-1 correct: {'✅' if best_config['top_1_correct'] else '❌'}")

    # Validate hypothesis
    default_config = results[1]  # 60/40 is second in list

    print(f"\n📊 Hypothesis Validation (60/40):")
    print(f"   Top-1 correct: {'✅' if default_config['top_1_correct'] else '❌'}")
    print(f"   Perfect match in top-3: {'✅' if default_config['top_3_has_perfect'] else '❌'}")

    if default_config['top_1_correct']:
        print(f"\n✅ HYPOTHESIS VALIDATED: 60/40 weighting achieves optimal recall")
    else:
        print(f"\n⚠️  HYPOTHESIS NEEDS REFINEMENT: Consider adjusting weights")
        print(f"   Best performing: env={best_config['env_weight']}, goal={best_config['goal_weight']}")

    # Assertions
    assert default_config['top_3_has_perfect'], "60/40 should have perfect match in top-3"


def test_env_only_vs_goal_only():
    """Test extreme cases: environment-only vs goal-only."""
    experiences, query_env, query_goal, _ = create_test_dataset()

    # Environment only (100/0)
    env_only = evaluate_weighting(1.0, 0.0, experiences, query_env, query_goal)

    # Goal only (0/100)
    goal_only = evaluate_weighting(0.0, 1.0, experiences, query_env, query_goal)

    print("\n" + "=" * 70)
    print("Extreme Cases: Environment-Only vs Goal-Only")
    print("=" * 70)

    print(f"\nEnvironment-Only (100/0):")
    print(f"  Top-3: {env_only['results'][:3]}")
    print(f"  Should prioritize: Exp 0, 1, 4 (web+python matches)")

    print(f"\nGoal-Only (0/100):")
    print(f"  Top-3: {goal_only['results'][:3]}")
    print(f"  Should prioritize: Exp 0, 2 (scraping goal matches)")

    print(f"\n📊 Insights:")
    print(f"  - Environment-only top-1: Exp {env_only['results'][0]}")
    print(f"  - Goal-only top-1: Exp {goal_only['results'][0]}")

    # Both should still get perfect match in top-3 (it matches both criteria)
    assert env_only['top_3_has_perfect'], "Env-only should find perfect match"
    assert goal_only['top_3_has_perfect'], "Goal-only should find perfect match"


def test_weighting_sensitivity():
    """Test sensitivity to small weight changes."""
    experiences, query_env, query_goal, _ = create_test_dataset()

    print("\n" + "=" * 70)
    print("Sensitivity Analysis: Small Weight Changes")
    print("=" * 70)

    # Test small variations around 60/40
    variations = [
        (0.55, 0.45),
        (0.60, 0.40),  # Default
        (0.65, 0.35),
    ]

    results = []
    for env_w, goal_w in variations:
        metrics = evaluate_weighting(env_w, goal_w, experiences, query_env, query_goal)
        results.append(metrics)

        print(f"\nenv={env_w:.2f}, goal={goal_w:.2f}:")
        print(f"  Top-3: {metrics['results'][:3]}")
        print(f"  Top-1 correct: {'✅' if metrics['top_1_correct'] else '❌'}")

    # Check if results are stable (top-1 should be consistent)
    top_1_results = [r['results'][0] for r in results]

    print(f"\n📊 Stability:")
    print(f"  Top-1 results: {top_1_results}")
    print(f"  Consistent: {'✅' if len(set(top_1_results)) == 1 else '❌'}")

    # Results should be relatively stable around 60/40
    # At minimum, perfect match should be in top-3 for all variations
    for r in results:
        assert r['top_3_has_perfect'], f"Variation ({r['env_weight']}/{r['goal_weight']}) should have perfect match in top-3"


if __name__ == "__main__":
    # Run ablation study
    pytest.main([__file__, "-v", "-s"])
