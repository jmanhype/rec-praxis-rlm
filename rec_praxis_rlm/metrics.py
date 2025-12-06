"""Custom metrics for evaluating memory retrieval and agent performance."""

from typing import Any, Optional

from rec_praxis_rlm.memory import Experience


def memory_retrieval_quality(
    example: dict[str, Any], prediction: Any, trace: Optional[Any] = None
) -> float:
    """Evaluate memory retrieval quality using hybrid scoring.

    Metric combines:
    - Environmental feature similarity (40%)
    - Goal text similarity (30%)
    - Success rate of retrieved experiences (30%)

    Args:
        example: Ground truth example with 'env_features', 'goal', 'expected_success_rate'
        prediction: Retrieved experiences (list of Experience objects)
        trace: Optional execution trace (not used currently)

    Returns:
        Score between 0.0 and 1.0, where 1.0 is perfect retrieval
    """
    if not isinstance(prediction, list):
        return 0.0

    if len(prediction) == 0:
        return 0.0

    # Extract ground truth
    expected_env = set(example.get("env_features", []))
    expected_goal = example.get("goal", "")
    expected_success_rate = example.get("expected_success_rate", 1.0)

    # Compute environmental similarity (Jaccard)
    env_similarities = []
    for exp in prediction:
        if isinstance(exp, Experience):
            retrieved_env = set(exp.env_features)
            if len(expected_env) == 0 and len(retrieved_env) == 0:
                env_sim = 1.0
            else:
                env_sim = len(expected_env.intersection(retrieved_env)) / len(
                    expected_env.union(retrieved_env)
                )
            env_similarities.append(env_sim)

    avg_env_sim = sum(env_similarities) / len(env_similarities) if env_similarities else 0.0

    # Compute goal similarity (simple token overlap for now)
    goal_similarities = []
    expected_tokens = set(expected_goal.lower().split())
    for exp in prediction:
        if isinstance(exp, Experience):
            retrieved_tokens = set(exp.goal.lower().split())
            if len(expected_tokens) == 0 and len(retrieved_tokens) == 0:
                goal_sim = 1.0
            else:
                goal_sim = len(expected_tokens.intersection(retrieved_tokens)) / len(
                    expected_tokens.union(retrieved_tokens)
                )
            goal_similarities.append(goal_sim)

    avg_goal_sim = sum(goal_similarities) / len(goal_similarities) if goal_similarities else 0.0

    # Compute success rate
    success_count = sum(1 for exp in prediction if isinstance(exp, Experience) and exp.success)
    actual_success_rate = success_count / len(prediction)
    success_score = 1.0 - abs(actual_success_rate - expected_success_rate)

    # Weighted combination
    score = (0.4 * avg_env_sim) + (0.3 * avg_goal_sim) + (0.3 * success_score)

    return score


class SemanticF1Score:
    """DSPy-compatible metric wrapper for semantic F1 scoring.

    Evaluates retrieval quality using precision and recall of relevant experiences.
    """

    def __init__(self, relevance_threshold: float = 0.7):
        """Initialize metric.

        Args:
            relevance_threshold: Minimum similarity score to consider an experience relevant
        """
        self.relevance_threshold = relevance_threshold

    def __call__(
        self, example: dict[str, Any], prediction: Any, trace: Optional[Any] = None
    ) -> float:
        """Compute semantic F1 score.

        Args:
            example: Ground truth with 'relevant_experience_ids' list
            prediction: Retrieved experiences
            trace: Optional execution trace

        Returns:
            F1 score between 0.0 and 1.0
        """
        if not isinstance(prediction, list):
            return 0.0

        if len(prediction) == 0:
            return 0.0

        # Get ground truth relevant IDs
        relevant_ids = set(example.get("relevant_experience_ids", []))

        if len(relevant_ids) == 0:
            return 0.0

        # Get predicted IDs
        predicted_ids = set()
        for exp in prediction:
            if isinstance(exp, Experience):
                # Use hash of experience as ID (env_features + goal)
                exp_id = hash((tuple(exp.env_features), exp.goal))
                predicted_ids.add(exp_id)

        # Compute precision and recall
        true_positives = len(relevant_ids.intersection(predicted_ids))

        if len(predicted_ids) == 0:
            precision = 0.0
        else:
            precision = true_positives / len(predicted_ids)

        # relevant_ids cannot be empty here (we return early at line 123-124 if it is)
        recall = true_positives / len(relevant_ids)

        # Compute F1
        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)

        return f1
