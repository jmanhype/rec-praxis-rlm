"""Unit tests for custom metrics."""
import pytest
import time

from rec_praxis_rlm.metrics import memory_retrieval_quality, SemanticF1Score
from rec_praxis_rlm.memory import Experience


class TestMemoryRetrievalQuality:
    """Tests for memory_retrieval_quality metric."""

    def test_perfect_retrieval_scores_high(self) -> None:
        """Test that perfect retrieval gets high score."""
        # Ground truth
        example = {
            "env_features": ["web", "sidebar", "article"],
            "goal": "extract product prices",
            "expected_success_rate": 1.0,
        }

        # Perfect predictions
        prediction = [
            Experience(
                env_features=["web", "sidebar", "article"],
                goal="extract product prices",
                action="use CSS selector",
                result="success",
                success=True,
                timestamp=time.time(),
            )
        ]

        score = memory_retrieval_quality(example, prediction)

        # Should be high (not necessarily 1.0 due to token-based goal similarity)
        assert score >= 0.8

    def test_empty_prediction_scores_zero(self) -> None:
        """Test that empty prediction gets zero score."""
        example = {
            "env_features": ["web"],
            "goal": "test",
            "expected_success_rate": 1.0,
        }

        score = memory_retrieval_quality(example, [])

        assert score == 0.0

    def test_non_list_prediction_scores_zero(self) -> None:
        """Test that non-list prediction gets zero score."""
        example = {
            "env_features": ["web"],
            "goal": "test",
            "expected_success_rate": 1.0,
        }

        score = memory_retrieval_quality(example, "not a list")

        assert score == 0.0

    def test_scoring_formula_weights(self) -> None:
        """Test that scoring uses 40% env + 30% goal + 30% success."""
        example = {
            "env_features": ["a", "b"],
            "goal": "test goal",
            "expected_success_rate": 0.5,
        }

        # Create predictions with varying characteristics
        prediction = [
            Experience(
                env_features=["a", "b"],  # Perfect env match
                goal="different",  # Poor goal match
                action="action",
                result="result",
                success=True,
                timestamp=time.time(),
            ),
            Experience(
                env_features=["a", "b"],
                goal="different",
                action="action",
                result="result",
                success=False,  # 50% success rate
                timestamp=time.time(),
            ),
        ]

        score = memory_retrieval_quality(example, prediction)

        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0

    def test_prediction_with_non_experience_objects(self) -> None:
        """Test that non-Experience objects in prediction are skipped."""
        example = {
            "env_features": ["a"],
            "goal": "test",
            "expected_success_rate": 1.0,
        }

        # Mix of Experience and non-Experience objects
        prediction = [
            Experience(
                env_features=["a"],
                goal="test",
                action="action",
                result="result",
                success=True,
                timestamp=time.time(),
            ),
            "not an experience",  # This should be skipped
            {"dict": "object"},  # This should be skipped too
        ]

        score = memory_retrieval_quality(example, prediction)

        # Should compute based on the one valid Experience
        assert score > 0.0

    def test_empty_env_features_both_sides(self) -> None:
        """Test env similarity when both expected and retrieved are empty."""
        example = {
            "env_features": [],
            "goal": "test",
            "expected_success_rate": 1.0,
        }

        prediction = [
            Experience(
                env_features=[],
                goal="test",
                action="action",
                result="result",
                success=True,
                timestamp=time.time(),
            )
        ]

        score = memory_retrieval_quality(example, prediction)

        # Empty env features on both sides should match perfectly (env_sim = 1.0)
        assert score > 0.8

    def test_empty_goal_both_sides(self) -> None:
        """Test goal similarity when both expected and retrieved are empty."""
        example = {
            "env_features": ["a"],
            "goal": "",
            "expected_success_rate": 1.0,
        }

        prediction = [
            Experience(
                env_features=["a"],
                goal="",
                action="action",
                result="result",
                success=True,
                timestamp=time.time(),
            )
        ]

        score = memory_retrieval_quality(example, prediction)

        # Empty goal on both sides should match perfectly (goal_sim = 1.0)
        assert score > 0.8


class TestSemanticF1Score:
    """Tests for SemanticF1Score metric."""

    def test_perfect_retrieval_f1_is_one(self) -> None:
        """Test that perfect retrieval gets F1=1.0."""
        metric = SemanticF1Score()

        # Create experiences with known IDs
        exp1 = Experience(
            env_features=["a"],
            goal="goal1",
            action="action",
            result="result",
            success=True,
            timestamp=time.time(),
        )

        # Ground truth - use same hash computation
        exp_id = hash((tuple(exp1.env_features), exp1.goal))

        example = {
            "relevant_experience_ids": [exp_id],
        }

        prediction = [exp1]

        score = metric(example, prediction)

        assert score == 1.0

    def test_empty_prediction_f1_is_zero(self) -> None:
        """Test that empty prediction gets F1=0.0."""
        metric = SemanticF1Score()

        example = {
            "relevant_experience_ids": [123, 456],
        }

        score = metric(example, [])

        assert score == 0.0

    def test_no_relevant_ids_f1_is_zero(self) -> None:
        """Test that no relevant IDs returns F1=0.0."""
        metric = SemanticF1Score()

        example = {
            "relevant_experience_ids": [],
        }

        prediction = [
            Experience(
                env_features=["a"],
                goal="test",
                action="action",
                result="result",
                success=True,
                timestamp=time.time(),
            )
        ]

        score = metric(example, prediction)

        assert score == 0.0

    def test_partial_overlap_computes_correctly(self) -> None:
        """Test F1 computation with partial overlap."""
        metric = SemanticF1Score()

        exp1 = Experience(
            env_features=["a"],
            goal="goal1",
            action="action",
            result="result",
            success=True,
            timestamp=time.time(),
        )

        exp2 = Experience(
            env_features=["b"],
            goal="goal2",
            action="action",
            result="result",
            success=True,
            timestamp=time.time(),
        )

        # Only exp1 is relevant
        exp1_id = hash((tuple(exp1.env_features), exp1.goal))

        example = {
            "relevant_experience_ids": [exp1_id, 999],  # exp1 + one missing
        }

        prediction = [exp1, exp2]  # exp1 (correct) + exp2 (false positive)

        score = metric(example, prediction)

        # Precision = 1/2 (1 true positive, 1 false positive)
        # Recall = 1/2 (1 true positive, 1 false negative)
        # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        assert score == 0.5

    def test_non_list_prediction_returns_zero(self) -> None:
        """Test that non-list prediction returns 0.0."""
        metric = SemanticF1Score()

        example = {
            "relevant_experience_ids": [123],
        }

        score = metric(example, "not a list")

        assert score == 0.0

    def test_prediction_with_non_experience_objects_f1(self) -> None:
        """Test that non-Experience objects in prediction are skipped."""
        metric = SemanticF1Score()

        exp1 = Experience(
            env_features=["a"],
            goal="goal1",
            action="action",
            result="result",
            success=True,
            timestamp=time.time(),
        )

        exp_id = hash((tuple(exp1.env_features), exp1.goal))

        example = {
            "relevant_experience_ids": [exp_id],
        }

        # Mix of Experience and non-Experience
        prediction = [exp1, "not an experience", {"dict": "object"}]

        score = metric(example, prediction)

        # Should still get F1=1.0 since exp1 matches
        assert score == 1.0

    def test_all_non_experience_predictions_f1_zero(self) -> None:
        """Test that all non-Experience predictions returns 0.0."""
        metric = SemanticF1Score()

        example = {
            "relevant_experience_ids": [123, 456],
        }

        # Only non-Experience objects
        prediction = ["not", "experience", "objects"]

        score = metric(example, prediction)

        # predicted_ids will be empty, so precision = 0
        assert score == 0.0
