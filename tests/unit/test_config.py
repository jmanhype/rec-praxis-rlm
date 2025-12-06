"""Unit tests for configuration models."""
import pytest
from pydantic import ValidationError

from rec_praxis_rlm.config import MemoryConfig, ReplConfig, PlannerConfig


class TestMemoryConfig:
    """Tests for MemoryConfig Pydantic model."""

    def test_defaults(self) -> None:
        """Test default values are correctly set."""
        config = MemoryConfig()
        assert config.storage_path == "./memory.jsonl"
        assert config.top_k == 6
        assert config.similarity_threshold == 0.5
        assert config.env_weight == 0.6
        assert config.goal_weight == 0.4
        assert config.require_success is False
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding_api_fallback is None
        assert config.result_size_limit == 50000

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = MemoryConfig(
            storage_path="./custom_memory.jsonl",
            top_k=10,
            similarity_threshold=0.7,
            env_weight=0.7,
            goal_weight=0.3,
            require_success=True,
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            embedding_api_fallback="openai",
            result_size_limit=100000,
        )
        assert config.storage_path == "./custom_memory.jsonl"
        assert config.top_k == 10
        assert config.similarity_threshold == 0.7
        assert config.env_weight == 0.7
        assert config.goal_weight == 0.3
        assert config.require_success is True
        assert config.embedding_model == "sentence-transformers/all-mpnet-base-v2"
        assert config.embedding_api_fallback == "openai"
        assert config.result_size_limit == 100000

    def test_weight_sum_validation(self) -> None:
        """Test that env_weight + goal_weight must sum to 1.0."""
        # Valid: sum to 1.0
        config = MemoryConfig(env_weight=0.6, goal_weight=0.4)
        assert config.env_weight + config.goal_weight == 1.0

        # Valid: different weights that sum to 1.0
        config2 = MemoryConfig(env_weight=0.3, goal_weight=0.7)
        assert config2.env_weight + config2.goal_weight == 1.0

        # Invalid: sum > 1.0
        with pytest.raises(ValueError, match="env_weight \\+ goal_weight must sum to 1.0"):
            MemoryConfig(env_weight=0.7, goal_weight=0.5)

        # Invalid: sum < 1.0
        with pytest.raises(ValueError, match="env_weight \\+ goal_weight must sum to 1.0"):
            MemoryConfig(env_weight=0.3, goal_weight=0.3)

    def test_top_k_constraints(self) -> None:
        """Test top_k must be between 1 and 100."""
        # Valid: within range
        config = MemoryConfig(top_k=1)
        assert config.top_k == 1

        config = MemoryConfig(top_k=100)
        assert config.top_k == 100

        config = MemoryConfig(top_k=50)
        assert config.top_k == 50

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            MemoryConfig(top_k=0)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            MemoryConfig(top_k=101)

    def test_similarity_threshold_constraints(self) -> None:
        """Test similarity_threshold must be between 0.0 and 1.0."""
        # Valid: at boundaries
        config = MemoryConfig(similarity_threshold=0.0)
        assert config.similarity_threshold == 0.0

        config = MemoryConfig(similarity_threshold=1.0)
        assert config.similarity_threshold == 1.0

        # Valid: in range
        config = MemoryConfig(similarity_threshold=0.75)
        assert config.similarity_threshold == 0.75

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            MemoryConfig(similarity_threshold=-0.1)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            MemoryConfig(similarity_threshold=1.1)

    def test_env_weight_constraints(self) -> None:
        """Test env_weight must be between 0.0 and 1.0."""
        # Valid range (note: must also satisfy sum constraint)
        config = MemoryConfig(env_weight=0.0, goal_weight=1.0)
        assert config.env_weight == 0.0

        config = MemoryConfig(env_weight=1.0, goal_weight=0.0)
        assert config.env_weight == 1.0

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            MemoryConfig(env_weight=-0.1, goal_weight=1.1)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            MemoryConfig(env_weight=1.1, goal_weight=-0.1)

    def test_goal_weight_constraints(self) -> None:
        """Test goal_weight must be between 0.0 and 1.0."""
        # Valid range (note: must also satisfy sum constraint)
        config = MemoryConfig(env_weight=1.0, goal_weight=0.0)
        assert config.goal_weight == 0.0

        config = MemoryConfig(env_weight=0.0, goal_weight=1.0)
        assert config.goal_weight == 1.0

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            MemoryConfig(env_weight=1.1, goal_weight=-0.1)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            MemoryConfig(env_weight=-0.1, goal_weight=1.1)


class TestReplConfig:
    """Tests for ReplConfig Pydantic model."""

    def test_defaults(self) -> None:
        """Test default values are correctly set."""
        config = ReplConfig()
        assert config.max_output_chars == 10000
        assert config.max_search_matches == 100
        assert config.search_context_chars == 200
        assert config.execution_timeout_seconds == 5.0
        assert config.enable_sandbox is True
        assert config.log_executions is True
        assert config.allowed_builtins == [
            "len",
            "range",
            "sum",
            "max",
            "min",
            "abs",
            "round",
            "sorted",
            "enumerate",
            "zip",
            "map",
            "filter",
            "all",
            "any",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
        ]

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ReplConfig(
            max_output_chars=50000,
            max_search_matches=200,
            search_context_chars=500,
            execution_timeout_seconds=10.0,
            enable_sandbox=True,
            log_executions=True,
            allowed_builtins=["len", "sum", "max"],
        )
        assert config.max_output_chars == 50000
        assert config.max_search_matches == 200
        assert config.search_context_chars == 500
        assert config.execution_timeout_seconds == 10.0
        assert config.enable_sandbox is True
        assert config.log_executions is True
        assert config.allowed_builtins == ["len", "sum", "max"]

    def test_timeout_constraints(self) -> None:
        """Test execution_timeout_seconds must be between 0.1 and 60.0."""
        # Valid: at boundaries
        config = ReplConfig(execution_timeout_seconds=0.1)
        assert config.execution_timeout_seconds == 0.1

        config = ReplConfig(execution_timeout_seconds=60.0)
        assert config.execution_timeout_seconds == 60.0

        # Valid: in range
        config = ReplConfig(execution_timeout_seconds=5.0)
        assert config.execution_timeout_seconds == 5.0

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            ReplConfig(execution_timeout_seconds=0.05)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            ReplConfig(execution_timeout_seconds=61.0)

    def test_allowed_builtins_validation(self) -> None:
        """Test allowed_builtins accepts list of strings."""
        config = ReplConfig(allowed_builtins=["len", "sum"])
        assert config.allowed_builtins == ["len", "sum"]

        # Empty list is valid
        config = ReplConfig(allowed_builtins=[])
        assert config.allowed_builtins == []


class TestPlannerConfig:
    """Tests for PlannerConfig Pydantic model."""

    def test_defaults(self) -> None:
        """Test default values are correctly set."""
        config = PlannerConfig()
        assert config.lm_model == "openai/gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_iters == 10
        assert config.enable_mlflow_tracing is True
        assert config.log_traces_from_compile is False
        assert config.optimizer == "miprov2"
        assert config.optimizer_auto_level == "medium"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = PlannerConfig(
            lm_model="anthropic/claude-3-5-sonnet-20241022",
            temperature=0.1,
            max_iters=15,
            enable_mlflow_tracing=True,
            log_traces_from_compile=True,
            optimizer="simba",
            optimizer_auto_level="heavy",
        )
        assert config.lm_model == "anthropic/claude-3-5-sonnet-20241022"
        assert config.temperature == 0.1
        assert config.max_iters == 15
        assert config.enable_mlflow_tracing is True
        assert config.log_traces_from_compile is True
        assert config.optimizer == "simba"
        assert config.optimizer_auto_level == "heavy"

    def test_temperature_constraints(self) -> None:
        """Test temperature must be between 0.0 and 2.0."""
        # Valid: at boundaries
        config = PlannerConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = PlannerConfig(temperature=2.0)
        assert config.temperature == 2.0

        # Valid: in range
        config = PlannerConfig(temperature=0.7)
        assert config.temperature == 0.7

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            PlannerConfig(temperature=-0.1)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            PlannerConfig(temperature=2.1)

    def test_max_iters_constraints(self) -> None:
        """Test max_iters must be between 1 and 50."""
        # Valid: at boundaries
        config = PlannerConfig(max_iters=1)
        assert config.max_iters == 1

        config = PlannerConfig(max_iters=50)
        assert config.max_iters == 50

        # Valid: in range
        config = PlannerConfig(max_iters=10)
        assert config.max_iters == 10

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            PlannerConfig(max_iters=0)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            PlannerConfig(max_iters=51)

    def test_optimizer_auto_level_validation(self) -> None:
        """Test optimizer_auto_level accepts valid values."""
        valid_levels = ["light", "medium", "heavy"]
        for level in valid_levels:
            config = PlannerConfig(optimizer_auto_level=level)
            assert config.optimizer_auto_level == level

        # Invalid value
        with pytest.raises(ValidationError):
            PlannerConfig(optimizer_auto_level="invalid")
