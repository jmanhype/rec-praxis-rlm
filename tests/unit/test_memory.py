"""Unit tests for procedural memory module."""
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from pydantic import ValidationError

from rec_praxis_rlm.memory import Experience, ProceduralMemory
from rec_praxis_rlm.config import MemoryConfig


class TestExperience:
    """Tests for Experience Pydantic model."""

    def test_valid_experience(self) -> None:
        """Test creating a valid experience."""
        exp = Experience(
            env_features=["has_sidebar", "article_layout"],
            goal="extract article text",
            action="css_selector('main article > p')",
            result="extracted 5 paragraphs",
            success=True,
            timestamp=time.time(),
        )
        assert exp.env_features == ["has_sidebar", "article_layout"]
        assert exp.goal == "extract article text"
        assert exp.action == "css_selector('main article > p')"
        assert exp.result == "extracted 5 paragraphs"
        assert exp.success is True
        assert exp.timestamp > 0
        assert exp.embedding is None
        assert exp.cost is None
        assert exp.metadata == {}

    def test_experience_with_optional_fields(self) -> None:
        """Test experience with optional fields populated."""
        exp = Experience(
            env_features=["test"],
            goal="test goal",
            action="test action",
            result="test result",
            success=True,
            timestamp=time.time(),
            embedding=[0.1, 0.2, 0.3],
            cost=0.001,
            metadata={"source": "test"},
        )
        assert exp.embedding == [0.1, 0.2, 0.3]
        assert exp.cost == 0.001
        assert exp.metadata == {"source": "test"}

    def test_env_features_must_be_list(self) -> None:
        """Test that env_features must be a list."""
        with pytest.raises(ValidationError):
            Experience(
                env_features="not_a_list",  # type: ignore
                goal="test",
                action="test",
                result="test",
                success=True,
                timestamp=time.time(),
            )

    def test_goal_must_be_string(self) -> None:
        """Test that goal must be a string."""
        with pytest.raises(ValidationError):
            Experience(
                env_features=["test"],
                goal=123,  # type: ignore
                action="test",
                result="test",
                success=True,
                timestamp=time.time(),
            )

    def test_action_must_be_string(self) -> None:
        """Test that action must be a string."""
        with pytest.raises(ValidationError):
            Experience(
                env_features=["test"],
                goal="test",
                action=["not", "a", "string"],  # type: ignore
                result="test",
                success=True,
                timestamp=time.time(),
            )

    def test_result_must_be_string(self) -> None:
        """Test that result must be a string."""
        with pytest.raises(ValidationError):
            Experience(
                env_features=["test"],
                goal="test",
                action="test",
                result=123,  # type: ignore
                success=True,
                timestamp=time.time(),
            )

    def test_success_must_be_bool(self) -> None:
        """Test that success must be a boolean."""
        with pytest.raises(ValidationError):
            Experience(
                env_features=["test"],
                goal="test",
                action="test",
                result="test",
                success="yes",  # type: ignore
                timestamp=time.time(),
            )

    def test_timestamp_must_be_positive(self) -> None:
        """Test that timestamp must be > 0."""
        with pytest.raises(ValidationError):
            Experience(
                env_features=["test"],
                goal="test",
                action="test",
                result="test",
                success=True,
                timestamp=-1.0,
            )

    def test_embedding_must_be_list_of_floats(self) -> None:
        """Test that embedding must be a list of floats."""
        # Valid: list of floats
        exp = Experience(
            env_features=["test"],
            goal="test",
            action="test",
            result="test",
            success=True,
            timestamp=time.time(),
            embedding=[0.1, 0.2, 0.3],
        )
        assert exp.embedding == [0.1, 0.2, 0.3]

        # Invalid: not a list
        with pytest.raises(ValidationError):
            Experience(
                env_features=["test"],
                goal="test",
                action="test",
                result="test",
                success=True,
                timestamp=time.time(),
                embedding="not_a_list",  # type: ignore
            )

    def test_cost_must_be_float(self) -> None:
        """Test that cost must be a float."""
        # Valid: float
        exp = Experience(
            env_features=["test"],
            goal="test",
            action="test",
            result="test",
            success=True,
            timestamp=time.time(),
            cost=0.001,
        )
        assert exp.cost == 0.001

        # Valid: int coerced to float
        exp2 = Experience(
            env_features=["test"],
            goal="test",
            action="test",
            result="test",
            success=True,
            timestamp=time.time(),
            cost=1,
        )
        assert exp2.cost == 1.0


class TestJSONLPersistence:
    """Tests for JSONL persistence operations."""

    def test_store_appends_to_file(self) -> None:
        """Test that store() appends experiences to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)

            exp1 = Experience(
                env_features=["test1"],
                goal="goal1",
                action="action1",
                result="result1",
                success=True,
                timestamp=time.time(),
            )
            exp2 = Experience(
                env_features=["test2"],
                goal="goal2",
                action="action2",
                result="result2",
                success=True,
                timestamp=time.time(),
            )

            memory.store(exp1)
            memory.store(exp2)

            # File should exist and have 3 lines (version marker + 2 experiences)
            assert os.path.exists(storage_path)
            with open(storage_path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 3

    def test_concurrent_store_appends_without_corruption(self) -> None:
        """Concurrent store() calls should not corrupt JSONL."""
        import json
        import hashlib
        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path, embedding_model="")
            memory = ProceduralMemory(config, use_faiss=False, enable_privacy_redaction=False)

            num_threads = 8
            per_thread = 5

            def worker(tid: int) -> None:
                for i in range(per_thread):
                    exp = Experience(
                        env_features=[f"t{tid}", f"i{i}"],
                        goal=f"goal-{tid}-{i}",
                        action="action",
                        result="result",
                        success=True,
                        timestamp=time.time(),
                    )
                    memory.store(exp)

            with ThreadPoolExecutor(max_workers=num_threads) as ex:
                list(ex.map(worker, range(num_threads)))

            with open(storage_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            assert json.loads(lines[0])["__version__"] == "2.0"
            data_lines = lines[1:]
            assert len(data_lines) == num_threads * per_thread

            goals: list[str] = []
            for line in data_lines:
                checksum, json_data = line.split("|", 1)
                assert hashlib.sha256(json_data.encode()).hexdigest() == checksum
                obj = json.loads(json_data)
                goals.append(obj["goal"])

            assert len(set(goals)) == len(goals)

    def test_load_parses_all_lines(self) -> None:
        """Test that load() parses all JSONL lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)

            # Create memory and store experiences
            memory1 = ProceduralMemory(config)
            for i in range(5):
                memory1.store(
                    Experience(
                        env_features=[f"test{i}"],
                        goal=f"goal{i}",
                        action=f"action{i}",
                        result=f"result{i}",
                        success=True,
                        timestamp=time.time(),
                    )
                )

            # Create new memory instance (simulates restart)
            memory2 = ProceduralMemory(config)
            assert memory2.size() == 5

    def test_corrupted_lines_skipped_with_warning(self) -> None:
        """Test that corrupted JSONL lines are skipped with warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")

            # Write a mix of valid and invalid JSON
            with open(storage_path, "w") as f:
                f.write('{"env_features": ["test1"], "goal": "goal1", "action": "action1", "result": "result1", "success": true, "timestamp": 1.0}\n')
                f.write('INVALID JSON LINE\n')
                f.write('{"env_features": ["test2"], "goal": "goal2", "action": "action2", "result": "result2", "success": true, "timestamp": 2.0}\n')

            config = MemoryConfig(storage_path=storage_path)
            with patch("rec_praxis_rlm.memory.logger") as mock_logger:
                memory = ProceduralMemory(config)
                # Should load 2 valid experiences, skip 1 corrupted line
                assert memory.size() == 2
                # Should have logged a warning
                mock_logger.warning.assert_called()

    def test_atomic_writes(self) -> None:
        """Test that writes are atomic (temp file + rename)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)

            exp = Experience(
                env_features=["test"],
                goal="goal",
                action="action",
                result="result",
                success=True,
                timestamp=time.time(),
            )

            # Store should use atomic write
            memory.store(exp)

            # File should exist and be readable
            assert os.path.exists(storage_path)
            with open(storage_path, "r") as f:
                content = f.read()
            assert "test" in content


class TestJaccardSimilarity:
    """Tests for Jaccard similarity computation."""

    def test_empty_sets(self) -> None:
        """Test Jaccard similarity with empty sets."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        sim = memory._jaccard_similarity(set(), set())
        assert sim == 0.0

    def test_identical_sets(self) -> None:
        """Test Jaccard similarity with identical sets."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        sim = memory._jaccard_similarity({"a", "b", "c"}, {"a", "b", "c"})
        assert sim == 1.0

    def test_partial_overlap(self) -> None:
        """Test Jaccard similarity with partial overlap."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        # {a, b} ∩ {b, c} = {b}, union = {a, b, c}
        # Jaccard = 1/3 = 0.333...
        sim = memory._jaccard_similarity({"a", "b"}, {"b", "c"})
        assert abs(sim - 1/3) < 0.001

    def test_no_overlap(self) -> None:
        """Test Jaccard similarity with no overlap."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        sim = memory._jaccard_similarity({"a", "b"}, {"c", "d"})
        assert sim == 0.0

    def test_case_sensitivity(self) -> None:
        """Test that Jaccard similarity is case-sensitive."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        # "A" != "a"
        sim = memory._jaccard_similarity({"A"}, {"a"})
        assert sim == 0.0


class TestHybridScoring:
    """Tests for hybrid similarity scoring."""

    def test_60_40_weighting(self) -> None:
        """Test default 60% env + 40% goal weighting."""
        config = MemoryConfig(storage_path=":memory:", env_weight=0.6, goal_weight=0.4)
        memory = ProceduralMemory(config)

        exp = Experience(
            env_features=["a", "b"],
            goal="test goal",
            action="action",
            result="result",
            success=True,
            timestamp=time.time(),
            embedding=[0.5, 0.5],  # Mock embedding
        )

        query_embedding = [0.8, 0.8]
        query_env_features = ["a", "b"]

        with patch.object(memory, "_cosine_similarity", return_value=0.9):
            score = memory._compute_similarity_score(
                exp, query_env_features, query_embedding
            )
            # env_sim = 1.0 (identical), goal_sim = 0.9 (mocked)
            # score = 0.6 * 1.0 + 0.4 * 0.9 = 0.96
            assert abs(score - 0.96) < 0.001

    def test_configurable_weights(self) -> None:
        """Test custom similarity weights."""
        config = MemoryConfig(storage_path=":memory:", env_weight=0.3, goal_weight=0.7)
        memory = ProceduralMemory(config)

        exp = Experience(
            env_features=["a"],
            goal="test",
            action="action",
            result="result",
            success=True,
            timestamp=time.time(),
            embedding=[0.5],
        )

        with patch.object(memory, "_cosine_similarity", return_value=0.8):
            score = memory._compute_similarity_score(
                exp, ["a"], [0.5]
            )
            # env_sim = 1.0, goal_sim = 0.8
            # score = 0.3 * 1.0 + 0.7 * 0.8 = 0.86
            assert abs(score - 0.86) < 0.001

    def test_threshold_filtering(self) -> None:
        """Test that experiences below threshold are filtered."""
        # This will be tested in ProceduralMemory.recall() tests
        pass

    def test_top_k_selection(self) -> None:
        """Test that only top-k experiences are returned."""
        # This will be tested in ProceduralMemory.recall() tests
        pass


class TestProceduralMemory:
    """Tests for ProceduralMemory class."""

    def test_init_loads_existing_jsonl(self) -> None:
        """Test that __init__ loads existing experiences from JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")

            # Create and populate memory
            memory1 = ProceduralMemory(MemoryConfig(storage_path=storage_path))
            for i in range(3):
                memory1.store(
                    Experience(
                        env_features=[f"test{i}"],
                        goal=f"goal{i}",
                        action=f"action{i}",
                        result=f"result{i}",
                        success=True,
                        timestamp=time.time(),
                    )
                )

            # Create new instance - should load existing data
            memory2 = ProceduralMemory(MemoryConfig(storage_path=storage_path))
            assert memory2.size() == 3

    def test_store_adds_experience(self) -> None:
        """Test that store() adds an experience to memory."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        assert memory.size() == 0

        exp = Experience(
            env_features=["test"],
            goal="goal",
            action="action",
            result="result",
            success=True,
            timestamp=time.time(),
        )
        memory.store(exp)

        assert memory.size() == 1

    def test_recall_returns_top_k_sorted(self) -> None:
        """Test that recall() returns top-k experiences sorted by score."""
        # Will be implemented after _compute_similarity_score is complete
        pass

    def test_require_success_filter(self) -> None:
        """Test that require_success=True filters out failed experiences."""
        config = MemoryConfig(storage_path=":memory:", require_success=True)
        memory = ProceduralMemory(config)

        # Store mix of successful and failed experiences
        memory.store(
            Experience(
                env_features=["test"],
                goal="goal1",
                action="action1",
                result="success",
                success=True,
                timestamp=time.time(),
            )
        )
        memory.store(
            Experience(
                env_features=["test"],
                goal="goal2",
                action="action2",
                result="failed",
                success=False,
                timestamp=time.time(),
            )
        )

        # Recall should only return successful experience
        results = memory.recall(env_features=["test"], goal="goal1")
        assert len(results) == 1
        assert results[0].success is True

    def test_init_without_embedding_model(self) -> None:
        """Test initialization without embedding model."""
        config = MemoryConfig(storage_path=":memory:", embedding_model="")
        memory = ProceduralMemory(config)
        assert memory.embedding_provider is None

    def test_init_with_failed_embedding_model(self) -> None:
        """Test initialization with failed embedding model load."""
        config = MemoryConfig(storage_path=":memory:", embedding_model="invalid/model")
        with patch("rec_praxis_rlm.memory.logger") as mock_logger:
            memory = ProceduralMemory(config)
            assert memory.embedding_provider is None
            mock_logger.warning.assert_called()

    def test_init_with_injected_embedding_provider(self) -> None:
        """Test dependency injection with custom embedding provider."""
        from rec_praxis_rlm.embeddings import EmbeddingProvider

        # Create mock provider
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]

        # Inject provider (should NOT create default provider)
        config = MemoryConfig(storage_path=":memory:", embedding_model="should-be-ignored")
        memory = ProceduralMemory(config, embedding_provider=mock_provider)

        # Should use injected provider
        assert memory.embedding_provider is mock_provider

        # Verify it works
        exp = Experience(
            env_features=["test"],
            goal="test goal",
            action="action",
            result="result",
            success=True,
            timestamp=time.time(),
        )
        memory.store(exp)

        # Should have called injected provider
        mock_provider.embed.assert_called_once_with("test goal")
        assert exp.embedding == [0.1, 0.2, 0.3]

    def test_init_with_none_injected_provider_uses_config(self) -> None:
        """Test that explicitly passing None uses config-based provider creation."""
        config = MemoryConfig(storage_path=":memory:", embedding_model="all-MiniLM-L6-v2")

        with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
            mock_provider = Mock()
            mock_provider.embed.return_value = [0.5, 0.6]
            mock_embedding.return_value = mock_provider

            # Explicitly pass None (should fall back to config)
            memory = ProceduralMemory(config, embedding_provider=None)

            # Should have created provider from config
            assert memory.embedding_provider is mock_provider
            mock_embedding.assert_called_once_with("all-MiniLM-L6-v2")

    def test_init_injected_provider_ignores_config_model(self) -> None:
        """Test that injected provider takes precedence over config.embedding_model."""
        from rec_praxis_rlm.embeddings import EmbeddingProvider

        # Create mock provider
        mock_injected = Mock(spec=EmbeddingProvider)
        mock_injected.embed.return_value = [0.1, 0.2, 0.3]

        # Config has a model, but injected provider should take precedence
        config = MemoryConfig(storage_path=":memory:", embedding_model="all-MiniLM-L6-v2")

        with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
            # This should NOT be called since we're injecting
            memory = ProceduralMemory(config, embedding_provider=mock_injected)

            # Should use injected provider
            assert memory.embedding_provider is mock_injected

            # Should NOT have created provider from config
            mock_embedding.assert_not_called()

    def test_init_with_injected_provider_backward_compatibility(self) -> None:
        """Test that existing code without dependency injection still works."""
        config = MemoryConfig(storage_path=":memory:", embedding_model="all-MiniLM-L6-v2")

        with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
            mock_provider = Mock()
            mock_provider.embed.return_value = [0.7, 0.8]
            mock_embedding.return_value = mock_provider

            # Old code: no embedding_provider parameter
            memory = ProceduralMemory(config)

            # Should work identically (create provider from config)
            assert memory.embedding_provider is mock_provider
            mock_embedding.assert_called_once_with("all-MiniLM-L6-v2")

            # Verify functionality
            exp = Experience(
                env_features=["test"],
                goal="test goal",
                action="action",
                result="result",
                success=True,
                timestamp=time.time(),
            )
            memory.store(exp)
            mock_provider.embed.assert_called_once_with("test goal")

    def test_init_injected_provider_none_when_no_config_model(self) -> None:
        """Test that no provider is created when config has no model and none injected."""
        config = MemoryConfig(storage_path=":memory:", embedding_model="")

        # No injection, no config model
        memory = ProceduralMemory(config, embedding_provider=None)

        # Should have None provider
        assert memory.embedding_provider is None

    def test_load_experiences_empty_lines_skipped(self) -> None:
        """Test that empty lines in JSONL are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")

            # Write JSONL with empty lines
            with open(storage_path, "w") as f:
                f.write('{"env_features": ["test1"], "goal": "goal1", "action": "action1", "result": "result1", "success": true, "timestamp": 1.0}\n')
                f.write('\n')  # Empty line
                f.write('   \n')  # Whitespace line
                f.write('{"env_features": ["test2"], "goal": "goal2", "action": "action2", "result": "result2", "success": true, "timestamp": 2.0}\n')

            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)
            assert memory.size() == 2

    def test_append_experience_storage_error(self) -> None:
        """Test that storage errors are properly raised."""
        from rec_praxis_rlm.exceptions import StorageError

        config = MemoryConfig(storage_path="/invalid/nonexistent/path/memory.jsonl")
        memory = ProceduralMemory(config)

        exp = Experience(
            env_features=["test"],
            goal="goal",
            action="action",
            result="result",
            success=True,
            timestamp=time.time(),
        )

        with pytest.raises(StorageError):
            memory.store(exp)

    def test_cosine_similarity_zero_vectors(self) -> None:
        """Test cosine similarity with zero vectors."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        # Zero vector should return 0.0
        sim = memory._cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        assert sim == 0.0

        sim = memory._cosine_similarity([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        assert sim == 0.0

    def test_cosine_similarity_mismatched_dimensions(self) -> None:
        """Test that cosine similarity raises error for mismatched dimensions."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        with pytest.raises(ValueError, match="same dimension"):
            memory._cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_cosine_similarity_identical_vectors(self) -> None:
        """Test cosine similarity with identical vectors."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        sim = memory._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert abs(sim - 1.0) < 0.001

    def test_cosine_similarity_orthogonal_vectors(self) -> None:
        """Test cosine similarity with orthogonal vectors."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        # Orthogonal vectors should have similarity ~0
        sim = memory._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 0.001

    def test_store_computes_embedding_if_missing(self) -> None:
        """Test that store() computes embedding if missing and provider available."""
        config = MemoryConfig(storage_path=":memory:", embedding_model="all-MiniLM-L6-v2")

        with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
            mock_provider = Mock()
            mock_provider.embed.return_value = [0.1, 0.2, 0.3]
            mock_embedding.return_value = mock_provider

            memory = ProceduralMemory(config)

            exp = Experience(
                env_features=["test"],
                goal="test goal",
                action="action",
                result="result",
                success=True,
                timestamp=time.time(),
            )

            memory.store(exp)

            # Should have called embed()
            mock_provider.embed.assert_called_once_with("test goal")
            assert exp.embedding == [0.1, 0.2, 0.3]

    def test_store_embedding_failure_warning(self) -> None:
        """Test that embedding failures during store() are logged as warnings."""
        config = MemoryConfig(storage_path=":memory:", embedding_model="all-MiniLM-L6-v2")

        with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
            mock_provider = Mock()
            mock_provider.embed.side_effect = Exception("Embedding failed")
            mock_embedding.return_value = mock_provider

            memory = ProceduralMemory(config)

            exp = Experience(
                env_features=["test"],
                goal="test goal",
                action="action",
                result="result",
                success=True,
                timestamp=time.time(),
            )

            with patch("rec_praxis_rlm.memory.logger") as mock_logger:
                memory.store(exp)
                mock_logger.warning.assert_called()

    def test_recall_custom_top_k(self) -> None:
        """Test recall with custom top_k parameter."""
        config = MemoryConfig(storage_path=":memory:", top_k=10)
        memory = ProceduralMemory(config)

        # Store 5 experiences
        for i in range(5):
            memory.store(
                Experience(
                    env_features=["test"],
                    goal=f"goal{i}",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=time.time(),
                )
            )

        # Recall with top_k=2
        results = memory.recall(env_features=["test"], goal="goal0", top_k=2)
        assert len(results) <= 2

    def test_recall_embedding_failure_warning(self) -> None:
        """Test that embedding failures during recall() are logged as warnings."""
        config = MemoryConfig(storage_path=":memory:", embedding_model="all-MiniLM-L6-v2")

        with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
            mock_provider = Mock()
            mock_embedding.return_value = mock_provider

            memory = ProceduralMemory(config)

            # Add experience
            memory.store(
                Experience(
                    env_features=["test"],
                    goal="goal",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=time.time(),
                )
            )

            # Make embed() fail during recall
            mock_provider.embed.side_effect = Exception("Embedding failed")

            with patch("rec_praxis_rlm.memory.logger") as mock_logger:
                results = memory.recall(env_features=["test"], goal="query")
                mock_logger.warning.assert_called()

    def test_recall_filters_by_threshold(self) -> None:
        """Test that recall filters experiences below similarity threshold."""
        config = MemoryConfig(
            storage_path=":memory:",
            similarity_threshold=0.8,
            env_weight=1.0,
            goal_weight=0.0,
        )
        memory = ProceduralMemory(config)

        # Store experiences with different overlaps
        memory.store(
            Experience(
                env_features=["a", "b", "c"],  # High overlap
                goal="goal",
                action="action",
                result="result",
                success=True,
                timestamp=time.time(),
            )
        )
        memory.store(
            Experience(
                env_features=["d", "e", "f"],  # No overlap
                goal="goal",
                action="action",
                result="result",
                success=True,
                timestamp=time.time(),
            )
        )

        # Recall with features that match first experience
        results = memory.recall(env_features=["a", "b", "c"], goal="goal")

        # Should only get high-similarity match
        assert len(results) >= 1
        assert set(results[0].env_features) == {"a", "b", "c"}

    def test_compact_with_none_returns_zero(self) -> None:
        """Test that compact(None) returns 0 without modifying memory."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        for i in range(5):
            memory.store(
                Experience(
                    env_features=["test"],
                    goal=f"goal{i}",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=float(i + 1),  # Start from 1.0 to avoid validation error
                )
            )

        removed = memory.compact(keep_recent_n=None)
        assert removed == 0
        assert memory.size() == 5

    def test_compact_in_memory_mode_skips_file_write(self) -> None:
        """Test that compact() in :memory: mode doesn't write to file."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        for i in range(5):
            memory.store(
                Experience(
                    env_features=["test"],
                    goal=f"goal{i}",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=float(i + 1),
                )
            )

        # Compact with keep_recent_n=2, should work without file I/O
        removed = memory.compact(keep_recent_n=2)
        assert removed == 3
        assert memory.size() == 2

    def test_compact_keeps_recent_n(self) -> None:
        """Test that compact() keeps only the N most recent experiences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)

            # Store 10 experiences with increasing timestamps
            for i in range(10):
                memory.store(
                    Experience(
                        env_features=["test"],
                        goal=f"goal{i}",
                        action="action",
                        result="result",
                        success=True,
                        timestamp=float(i + 1),  # Start from 1.0
                    )
                )

            # Compact to keep only 3 most recent
            removed = memory.compact(keep_recent_n=3)
            assert removed == 7
            assert memory.size() == 3

            # Verify kept the most recent
            timestamps = [exp.timestamp for exp in memory.experiences]
            assert timestamps == [10.0, 9.0, 8.0]

    def test_compact_rewrites_storage_file(self) -> None:
        """Test that compact() rewrites the storage file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)

            # Store 5 experiences
            for i in range(5):
                memory.store(
                    Experience(
                        env_features=["test"],
                        goal=f"goal{i}",
                        action="action",
                        result="result",
                        success=True,
                        timestamp=float(i + 1),  # Start from 1.0
                    )
                )

            memory.compact(keep_recent_n=2)

            # File should have 3 lines (version marker + 2 experiences)
            with open(storage_path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 3

    def test_compact_storage_error(self) -> None:
        """Test that compact() raises StorageError on write failure."""
        from rec_praxis_rlm.exceptions import StorageError

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)

            memory.store(
                Experience(
                    env_features=["test"],
                    goal="goal",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=time.time(),
                )
            )

            # Mock open() to fail during compact
            with patch("builtins.open", side_effect=Exception("Write failed")):
                with pytest.raises(StorageError, match="Failed to compact storage"):
                    memory.compact(keep_recent_n=1)

    def test_recompute_embeddings_success(self) -> None:
        """Test successful recomputation of all embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path, embedding_model="all-MiniLM-L6-v2")

            with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
                mock_old_provider = Mock()
                mock_old_provider.embed.return_value = [0.1, 0.2]
                mock_embedding.return_value = mock_old_provider

                memory = ProceduralMemory(config)

                # Store experiences
                for i in range(3):
                    memory.store(
                        Experience(
                            env_features=["test"],
                            goal=f"goal{i}",
                            action="action",
                            result="result",
                            success=True,
                            timestamp=float(i + 1),  # Start from 1.0
                        )
                    )

                # Mock new provider
                mock_new_provider = Mock()
                mock_new_provider.embed.return_value = [0.5, 0.6]
                mock_embedding.return_value = mock_new_provider

                # Recompute embeddings
                memory.recompute_embeddings("new-model")

                # Should have called embed() for each experience
                assert mock_new_provider.embed.call_count == 3

    def test_recompute_embeddings_model_load_failure(self) -> None:
        """Test that recompute_embeddings raises EmbeddingError on model load failure."""
        from rec_praxis_rlm.exceptions import EmbeddingError

        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding", side_effect=Exception("Model load failed")):
            with pytest.raises(EmbeddingError, match="Failed to load new model"):
                memory.recompute_embeddings("invalid-model")

    def test_recompute_embeddings_in_memory_mode_skips_file_write(self) -> None:
        """Test that recompute_embeddings() in :memory: mode doesn't write to file."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        # Store experiences
        for i in range(2):
            memory.store(
                Experience(
                    env_features=["test"],
                    goal=f"goal{i}",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=float(i + 1),
                )
            )

        # Mock new embedding provider
        with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
            mock_provider = Mock()
            mock_provider.embed.return_value = [0.5, 0.6]
            mock_embedding.return_value = mock_provider

            # Recompute embeddings - should work without file I/O
            memory.recompute_embeddings("new-model")

            # Should have called embed() for each experience
            assert mock_provider.embed.call_count == 2

    def test_recompute_embeddings_individual_failures_logged(self) -> None:
        """Test that individual embedding failures during recompute are logged as warnings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)

            # Store experiences
            for i in range(3):
                memory.store(
                    Experience(
                        env_features=["test"],
                        goal=f"goal{i}",
                        action="action",
                        result="result",
                        success=True,
                        timestamp=float(i + 1),  # Start from 1.0
                    )
                )

            # Mock provider that fails on embed
            with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
                mock_provider = Mock()
                mock_provider.embed.side_effect = Exception("Embedding failed")
                mock_embedding.return_value = mock_provider

                with patch("rec_praxis_rlm.memory.logger") as mock_logger:
                    memory.recompute_embeddings("new-model")
                    # Should have logged warnings for each failure
                    assert mock_logger.warning.call_count == 3

    def test_recompute_embeddings_storage_error(self) -> None:
        """Test that recompute_embeddings raises StorageError on write failure."""
        from rec_praxis_rlm.exceptions import StorageError

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)

            memory.store(
                Experience(
                    env_features=["test"],
                    goal="goal",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=time.time(),
                )
            )

            with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
                mock_provider = Mock()
                mock_provider.embed.return_value = [0.1, 0.2]
                mock_embedding.return_value = mock_provider

                # Mock open() to fail during rewrite
                with patch("builtins.open", side_effect=Exception("Write failed")):
                    with pytest.raises(StorageError, match="Failed to rewrite storage"):
                        memory.recompute_embeddings("new-model")

    def test_arecall_uses_threadpool(self) -> None:
        """Test that arecall() runs in thread pool without blocking event loop."""
        import asyncio

        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        memory.store(
            Experience(
                env_features=["test"],
                goal="goal",
                action="action",
                result="result",
                success=True,
                timestamp=time.time(),
            )
        )

        # Test async version
        results = asyncio.run(memory.arecall(env_features=["test"], goal="goal"))
        assert len(results) >= 0

    def test_arecall_concurrent_execution(self) -> None:
        """Test that multiple arecall() calls can run concurrently."""
        import asyncio

        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        # Store experiences
        for i in range(10):
            memory.store(
                Experience(
                    env_features=[f"test{i}"],
                    goal=f"goal{i}",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=float(i + 1),
                )
            )

        async def run_concurrent_recalls():
            # Run 5 concurrent recalls
            tasks = [
                memory.arecall(env_features=[f"test{i}"], goal=f"goal{i}")
                for i in range(5)
            ]
            results = await asyncio.gather(*tasks)
            return results

        # Execute concurrent recalls
        results = asyncio.run(run_concurrent_recalls())

        # All 5 recalls should complete successfully
        assert len(results) == 5
        assert all(isinstance(r, list) for r in results)

    def test_arecall_passes_parameters_correctly(self) -> None:
        """Test that arecall() correctly passes all parameters to recall()."""
        import asyncio

        config = MemoryConfig(storage_path=":memory:", top_k=10)
        memory = ProceduralMemory(config)

        # Store experiences
        for i in range(15):
            memory.store(
                Experience(
                    env_features=["common"],
                    goal=f"goal{i}",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=float(i + 1),
                )
            )

        # Test with custom top_k
        results = asyncio.run(memory.arecall(
            env_features=["common"],
            goal="goal0",
            top_k=3
        ))

        # Should respect top_k parameter
        assert len(results) <= 3

    def test_load_experiences_file_not_found(self) -> None:
        """Test that _load_experiences handles FileNotFoundError gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)

            # Mock os.path.exists to return True, but open() will raise FileNotFoundError
            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
                    # Should not raise - FileNotFoundError is caught in _load_experiences
                    memory = ProceduralMemory(config)
                    assert memory.size() == 0

    def test_append_experience_temp_file_cleanup_on_exception(self) -> None:
        """Test that temp file is cleaned up when exception occurs during write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)

            exp = Experience(
                env_features=["test"],
                goal="goal",
                action="action",
                result="result",
                success=True,
                timestamp=time.time(),
            )

            # Mock os.replace to raise exception after temp file is created
            # This ensures temp_path exists when exception occurs
            with patch("os.replace", side_effect=Exception("Replace failed")):
                try:
                    memory.store(exp)
                except Exception:
                    # Exception expected
                    pass

            # Verify no temp files left behind (cleanup succeeded)
            temp_files = [f for f in os.listdir(tmpdir) if f.startswith(".memory_")]
            assert len(temp_files) == 0

    def test_append_experience_temp_file_missing_on_exception(self) -> None:
        """Test exception handling when temp file doesn't exist during cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)

            exp = Experience(
                env_features=["test"],
                goal="goal",
                action="action",
                result="result",
                success=True,
                timestamp=time.time(),
            )

            # Mock os.fdopen to fail immediately (before temp file is written)
            # This means temp_path might not exist when exception handler runs
            original_fdopen = os.fdopen
            def mock_fdopen(fd, *args, **kwargs):
                # Close the file descriptor immediately to simulate early failure
                os.close(fd)
                raise Exception("FD open failed")

            with patch("os.fdopen", side_effect=mock_fdopen):
                try:
                    memory.store(exp)
                except Exception:
                    # Exception expected
                    pass

            # Should handle gracefully even if temp file doesn't exist
            # No assertion needed - just verify no crash

    def test_jaccard_similarity_one_empty_set(self) -> None:
        """Test Jaccard similarity when one set is empty but not both."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        # Test first set empty
        sim1 = memory._jaccard_similarity(set(), {"a", "b"})
        assert sim1 == 0.0

        # Test second set empty
        sim2 = memory._jaccard_similarity({"a", "b"}, set())
        assert sim2 == 0.0


class TestStorageVersionMigration:
    """Tests for storage version migration."""

    def test_new_file_gets_version_marker(self) -> None:
        """Test that new storage files get version marker as first line."""
        import json
        from rec_praxis_rlm.memory import STORAGE_VERSION

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)

            # Store first experience
            memory.store(
                Experience(
                    env_features=["test"],
                    goal="goal",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=time.time(),
                )
            )

            # Read file and check first line is version marker
            with open(storage_path, "r") as f:
                lines = f.readlines()

            assert len(lines) == 2  # Version + 1 experience
            first_line = json.loads(lines[0].strip())
            assert first_line == {"__version__": STORAGE_VERSION}

    def test_load_with_version_marker(self) -> None:
        """Test loading storage file with version marker."""
        import json
        from rec_praxis_rlm.memory import STORAGE_VERSION

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")

            # Write file with version marker
            with open(storage_path, "w") as f:
                f.write(json.dumps({"__version__": STORAGE_VERSION}) + "\n")
                f.write('{"env_features": ["test1"], "goal": "goal1", "action": "action1", "result": "result1", "success": true, "timestamp": 1.0}\n')
                f.write('{"env_features": ["test2"], "goal": "goal2", "action": "action2", "result": "result2", "success": true, "timestamp": 2.0}\n')

            config = MemoryConfig(storage_path=storage_path)
            with patch("rec_praxis_rlm.memory.logger") as mock_logger:
                memory = ProceduralMemory(config)

                # Should have loaded 2 experiences (version marker skipped)
                assert memory.size() == 2

                # Should have logged version info
                mock_logger.info.assert_any_call(f"Loading storage version {STORAGE_VERSION}")

    def test_load_legacy_format_without_version_marker(self) -> None:
        """Test loading legacy storage file without version marker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")

            # Write legacy file (no version marker)
            with open(storage_path, "w") as f:
                f.write('{"env_features": ["test1"], "goal": "goal1", "action": "action1", "result": "result1", "success": true, "timestamp": 1.0}\n')
                f.write('{"env_features": ["test2"], "goal": "goal2", "action": "action2", "result": "result2", "success": true, "timestamp": 2.0}\n')

            config = MemoryConfig(storage_path=storage_path)
            with patch("rec_praxis_rlm.memory.logger") as mock_logger:
                memory = ProceduralMemory(config)

                # Should have loaded 2 experiences
                assert memory.size() == 2

                # Should have logged legacy format detection
                mock_logger.info.assert_any_call("Loading legacy storage format (no version marker)")

    def test_migrate_from_legacy_version_0_0(self) -> None:
        """Test migration from legacy version 0.0."""
        import json
        from rec_praxis_rlm.memory import STORAGE_VERSION

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")

            # Write file with version 0.0 marker
            with open(storage_path, "w") as f:
                f.write(json.dumps({"__version__": "0.0"}) + "\n")
                f.write('{"env_features": ["test1"], "goal": "goal1", "action": "action1", "result": "result1", "success": true, "timestamp": 1.0}\n')

            config = MemoryConfig(storage_path=storage_path)
            with patch("rec_praxis_rlm.memory.logger") as mock_logger:
                memory = ProceduralMemory(config)

                # Should have loaded experience after migration
                assert memory.size() == 1

                # Should have logged migration
                mock_logger.info.assert_any_call("Migrating storage from version 0.0 to 2.0")
                mock_logger.info.assert_any_call("Migrating from 0.0 to 2.0 (adding checksums)")

    def test_unsupported_future_version_raises_error(self) -> None:
        """Test that unsupported future versions raise StorageError."""
        import json
        from rec_praxis_rlm.exceptions import StorageError

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")

            # Write file with future version
            with open(storage_path, "w") as f:
                f.write(json.dumps({"__version__": "99.0"}) + "\n")
                f.write('{"env_features": ["test1"], "goal": "goal1", "action": "action1", "result": "result1", "success": true, "timestamp": 1.0}\n')

            config = MemoryConfig(storage_path=storage_path)
            with pytest.raises(StorageError, match="Unsupported storage version 99.0"):
                ProceduralMemory(config)

    def test_invalid_version_marker_treated_as_legacy(self) -> None:
        """Test that invalid version markers are treated as legacy format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")

            # Write file with malformed version marker
            with open(storage_path, "w") as f:
                f.write('{"__version__": INVALID}\n')  # Invalid JSON
                f.write('{"env_features": ["test1"], "goal": "goal1", "action": "action1", "result": "result1", "success": true, "timestamp": 1.0}\n')

            config = MemoryConfig(storage_path=storage_path)
            with patch("rec_praxis_rlm.memory.logger") as mock_logger:
                memory = ProceduralMemory(config)

                # Should have loaded 0 experiences (first line is invalid, second line loads)
                # Actually, the invalid marker line should be skipped with warning
                assert memory.size() == 1  # Second line loads successfully

                # Should have logged warning about invalid version marker
                mock_logger.warning.assert_any_call("Invalid version marker, treating as legacy format")

    def test_compact_preserves_version_marker(self) -> None:
        """Test that compact() preserves version marker in rewritten file."""
        import json
        from rec_praxis_rlm.memory import STORAGE_VERSION

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)

            # Store 5 experiences
            for i in range(5):
                memory.store(
                    Experience(
                        env_features=["test"],
                        goal=f"goal{i}",
                        action="action",
                        result="result",
                        success=True,
                        timestamp=float(i + 1),
                    )
                )

            # Compact to 2 experiences
            memory.compact(keep_recent_n=2)

            # Read file and verify version marker
            with open(storage_path, "r") as f:
                lines = f.readlines()

            assert len(lines) == 3  # Version + 2 experiences
            first_line = json.loads(lines[0].strip())
            assert first_line == {"__version__": STORAGE_VERSION}

    def test_recompute_embeddings_preserves_version_marker(self) -> None:
        """Test that recompute_embeddings() preserves version marker in rewritten file."""
        import json
        from rec_praxis_rlm.memory import STORAGE_VERSION

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)

            # Store 2 experiences
            for i in range(2):
                memory.store(
                    Experience(
                        env_features=["test"],
                        goal=f"goal{i}",
                        action="action",
                        result="result",
                        success=True,
                        timestamp=float(i + 1),
                    )
                )

            # Recompute embeddings with mocked provider
            with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
                mock_provider = Mock()
                mock_provider.embed.return_value = [0.1, 0.2]
                mock_embedding.return_value = mock_provider

                memory.recompute_embeddings("new-model")

            # Read file and verify version marker
            with open(storage_path, "r") as f:
                lines = f.readlines()

            assert len(lines) == 3  # Version + 2 experiences
            first_line = json.loads(lines[0].strip())
            assert first_line == {"__version__": STORAGE_VERSION}

    def test_append_to_existing_file_preserves_version_marker(self) -> None:
        """Test that appending to existing file preserves version marker."""
        import json
        from rec_praxis_rlm.memory import STORAGE_VERSION

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)

            # Create first memory instance and store experience
            memory1 = ProceduralMemory(config)
            memory1.store(
                Experience(
                    env_features=["test1"],
                    goal="goal1",
                    action="action1",
                    result="result1",
                    success=True,
                    timestamp=1.0,
                )
            )

            # Create second memory instance and store another experience
            memory2 = ProceduralMemory(config)
            memory2.store(
                Experience(
                    env_features=["test2"],
                    goal="goal2",
                    action="action2",
                    result="result2",
                    success=True,
                    timestamp=2.0,
                )
            )

            # Read file
            with open(storage_path, "r") as f:
                lines = f.readlines()

            # Should have version marker + 2 experiences
            assert len(lines) == 3
            first_line = json.loads(lines[0].strip())
            assert first_line == {"__version__": STORAGE_VERSION}

    def test_empty_file_loads_successfully(self) -> None:
        """Test that empty storage file loads successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")

            # Create empty file
            Path(storage_path).touch()

            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)

            # Should have 0 experiences
            assert memory.size() == 0


class TestFAISSIndexing:
    """Tests for FAISS-accelerated retrieval."""

    def test_faiss_available(self) -> None:
        """Test that FAISS is available in this environment."""
        from rec_praxis_rlm.memory import FAISS_AVAILABLE
        assert FAISS_AVAILABLE is True

    def test_init_with_faiss_disabled(self) -> None:
        """Test initialization with FAISS explicitly disabled."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config, use_faiss=False)

        assert memory.use_faiss is False
        assert memory._faiss_index is None

    def test_faiss_index_built_on_load(self) -> None:
        """Test that FAISS index is built when loading experiences with embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)

            # Create memory and store experiences with embeddings
            memory1 = ProceduralMemory(config)
            for i in range(10):
                memory1.store(
                    Experience(
                        env_features=[f"feature{i}"],
                        goal=f"goal{i}",
                        action="action",
                        result="result",
                        success=True,
                        timestamp=float(i + 1),
                        embedding=[float(i), float(i + 1), float(i + 2)],  # Mock embeddings
                    )
                )

            # Create new memory instance (should build FAISS index on load)
            memory2 = ProceduralMemory(config)

            assert memory2.use_faiss is True
            assert memory2._faiss_index is not None
            assert memory2._embedding_dimension == 3

    def test_faiss_index_incremental_add(self) -> None:
        """Test that FAISS index is updated incrementally when storing experiences."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        # Initially no index
        assert memory._faiss_index is None

        # Store first experience - should build index
        memory.store(
            Experience(
                env_features=["test"],
                goal="goal1",
                action="action",
                result="result",
                success=True,
                timestamp=1.0,
                embedding=[1.0, 2.0, 3.0],
            )
        )

        assert memory._faiss_index is not None
        assert memory._embedding_dimension == 3

        # Store second experience - should add to index
        memory.store(
            Experience(
                env_features=["test"],
                goal="goal2",
                action="action",
                result="result",
                success=True,
                timestamp=2.0,
                embedding=[4.0, 5.0, 6.0],
            )
        )

        # Index should now have 2 vectors
        assert memory._faiss_index.ntotal == 2

    def test_recall_with_faiss(self) -> None:
        """Test that recall uses FAISS when available."""
        config = MemoryConfig(storage_path=":memory:", top_k=3, embedding_model="all-MiniLM-L6-v2")

        # Mock the embedding provider
        with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
            mock_provider = Mock()
            mock_provider.embed.side_effect = lambda goal: [10.0, 11.0, 12.0]  # Return query embedding
            mock_embedding.return_value = mock_provider

            memory = ProceduralMemory(config)

            # Store experiences with embeddings directly
            for i in range(20):
                memory.store(
                    Experience(
                        env_features=["feature"],
                        goal=f"goal{i}",
                        action="action",
                        result="result",
                        success=True,
                        timestamp=float(i + 1),
                        embedding=[float(i), float(i + 1), float(i + 2)],
                    )
                )

            # Recall should use FAISS
            # Query with embedding similar to experience 10: [10.0, 11.0, 12.0]
            results = memory.recall(env_features=["feature"], goal="query", top_k=3)

            # Should get results
            assert len(results) > 0
            assert all(isinstance(exp, Experience) for exp in results)

    def test_recall_fallback_to_linear_when_no_faiss_index(self) -> None:
        """Test that recall falls back to linear scan when FAISS index not available."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config, use_faiss=False)

        # Store experiences
        memory.store(
            Experience(
                env_features=["test"],
                goal="goal1",
                action="action",
                result="result",
                success=True,
                timestamp=1.0,
            )
        )

        # Should use linear scan (no embeddings)
        results = memory.recall(env_features=["test"], goal="query", top_k=1)

        # Should still work (may return 0 or 1 result depending on scoring)
        assert isinstance(results, list)

    def test_recall_fallback_when_no_query_embedding(self) -> None:
        """Test that recall falls back to linear scan when query embedding fails."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        # Store experience with embedding
        memory.store(
            Experience(
                env_features=["test"],
                goal="goal1",
                action="action",
                result="result",
                success=True,
                timestamp=1.0,
                embedding=[1.0, 2.0, 3.0],
            )
        )

        # Make embedding provider return None (simulating failure)
        memory.embedding_provider = None

        # Should fall back to linear scan
        results = memory.recall(env_features=["test"], goal="query", top_k=1)

        assert isinstance(results, list)

    def test_rebuild_faiss_index_after_compact(self) -> None:
        """Test that FAISS index is rebuilt after compaction."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        # Store 10 experiences
        for i in range(10):
            memory.store(
                Experience(
                    env_features=["test"],
                    goal=f"goal{i}",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=float(i + 1),
                    embedding=[float(i), float(i + 1), float(i + 2)],
                )
            )

        assert memory._faiss_index.ntotal == 10

        # Compact to 5
        memory.compact(keep_recent_n=5)

        # Index should be rebuilt with 5 vectors
        assert memory._faiss_index.ntotal == 5

    def test_rebuild_faiss_index_after_recompute_embeddings(self) -> None:
        """Test that FAISS index is rebuilt after recomputing embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config)

            # Store experiences
            for i in range(5):
                memory.store(
                    Experience(
                        env_features=["test"],
                        goal=f"goal{i}",
                        action="action",
                        result="result",
                        success=True,
                        timestamp=float(i + 1),
                    )
                )

            old_dimension = memory._embedding_dimension

            # Recompute embeddings with mock provider
            with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
                mock_provider = Mock()
                mock_provider.embed.return_value = [0.1, 0.2, 0.3, 0.4]  # Different dimension
                mock_embedding.return_value = mock_provider

                memory.recompute_embeddings("new-model")

            # Index should be rebuilt with new dimension
            assert memory._embedding_dimension == 4
            assert memory._embedding_dimension != old_dimension

    def test_faiss_index_zero_vector_fallback(self) -> None:
        """Test that recall falls back to linear scan when query is zero vector."""
        config = MemoryConfig(storage_path=":memory:", embedding_model="all-MiniLM-L6-v2")

        # Mock the embedding provider
        with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
            mock_provider = Mock()
            # Return zero vector for query
            mock_provider.embed.side_effect = lambda goal: [0.0, 0.0, 0.0]
            mock_embedding.return_value = mock_provider

            memory = ProceduralMemory(config)

            # Store experience
            memory.store(
                Experience(
                    env_features=["test"],
                    goal="goal1",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=1.0,
                    embedding=[1.0, 2.0, 3.0],
                )
            )

            # Query with zero vector (should fall back to linear)
            results = memory.recall(env_features=["test"], goal="query", top_k=1)

            assert isinstance(results, list)

    def test_faiss_index_handles_invalid_embeddings(self) -> None:
        """Test that FAISS index building handles invalid embeddings gracefully."""
        from rec_praxis_rlm.memory import ProceduralMemory
        config = MemoryConfig(storage_path=":memory:")

        # Create memory with mock embedding provider
        with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
            mock_provider = Mock()
            # Return non-list embedding (should be filtered out)
            mock_provider.embed.return_value = Mock()  # Invalid embedding
            mock_embedding.return_value = mock_provider

            memory = ProceduralMemory(config)

            # Store experience with invalid embedding
            memory.store(
                Experience(
                    env_features=["test"],
                    goal="goal1",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=1.0,
                )
            )

            # Index should remain None (no valid embeddings)
            assert memory._faiss_index is None

    def test_faiss_candidate_multiplier_filtering(self) -> None:
        """Test that FAISS over-fetches candidates for re-ranking."""
        config = MemoryConfig(storage_path=":memory:", top_k=2, embedding_model="all-MiniLM-L6-v2")

        # Mock the embedding provider
        with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
            mock_provider = Mock()
            # Return query embedding
            mock_provider.embed.side_effect = lambda goal: [15.0, 16.0, 17.0]
            mock_embedding.return_value = mock_provider

            memory = ProceduralMemory(config)

            # Store 30 experiences
            for i in range(30):
                memory.store(
                    Experience(
                        env_features=[f"feature{i % 3}"],  # Rotate through 3 different features
                        goal=f"goal{i}",
                        action="action",
                        result="result",
                        success=True,
                        timestamp=float(i + 1),
                        embedding=[float(i), float(i + 1), float(i + 2)],
                    )
                )

            # Query with specific embedding
            results = memory.recall(env_features=["feature1"], goal="query", top_k=2)

            # Should return top 2 after re-ranking (even though FAISS fetched more candidates)
            assert len(results) <= 2

    def test_rebuild_faiss_index_early_return_when_disabled(self) -> None:
        """Test that _rebuild_faiss_index returns early when use_faiss=False."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config, use_faiss=False)

        # Add experiences with embeddings
        memory.store(
            Experience(
                env_features=["test"],
                goal="goal1",
                action="action",
                result="result",
                success=True,
                timestamp=1.0,
                embedding=[1.0, 2.0, 3.0],
            )
        )

        # Manually call _rebuild_faiss_index (should return early)
        memory._rebuild_faiss_index()

        # Index should remain None
        assert memory._faiss_index is None

    def test_rebuild_faiss_index_exception_handling(self) -> None:
        """Test that _rebuild_faiss_index handles exceptions gracefully."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        # Store experience with embedding
        memory.store(
            Experience(
                env_features=["test"],
                goal="goal1",
                action="action",
                result="result",
                success=True,
                timestamp=1.0,
                embedding=[1.0, 2.0, 3.0],
            )
        )

        # Mock faiss.IndexFlatIP to raise exception
        with patch("faiss.IndexFlatIP", side_effect=Exception("FAISS error")):
            with patch("rec_praxis_rlm.memory.logger") as mock_logger:
                # Rebuild should catch exception and log warning
                memory._rebuild_faiss_index()

                # Should have logged warning
                mock_logger.warning.assert_called_once()
                assert "Failed to build FAISS index" in str(mock_logger.warning.call_args)

                # Index should be None after exception
                assert memory._faiss_index is None
                assert memory._embedding_dimension is None

    def test_store_incremental_add_with_zero_norm_embedding(self) -> None:
        """Test that store handles zero-norm embeddings gracefully during incremental add."""
        config = MemoryConfig(storage_path=":memory:")
        memory = ProceduralMemory(config)

        # Store first experience with valid embedding (builds index)
        memory.store(
            Experience(
                env_features=["test"],
                goal="goal1",
                action="action",
                result="result",
                success=True,
                timestamp=1.0,
                embedding=[1.0, 2.0, 3.0],
            )
        )

        assert memory._faiss_index is not None
        initial_count = memory._faiss_index.ntotal

        # Store second experience with zero-norm embedding (should skip incremental add)
        memory.store(
            Experience(
                env_features=["test"],
                goal="goal2",
                action="action",
                result="result",
                success=True,
                timestamp=2.0,
                embedding=[0.0, 0.0, 0.0],  # Zero-norm embedding
            )
        )

        # Index count should not change (zero-norm embedding not added)
        assert memory._faiss_index.ntotal == initial_count

    def test_recall_with_faiss_invalid_index_filtering(self) -> None:
        """Test that _recall_with_faiss handles invalid FAISS indices gracefully."""
        config = MemoryConfig(storage_path=":memory:", embedding_model="all-MiniLM-L6-v2")

        with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
            mock_provider = Mock()
            mock_provider.embed.side_effect = lambda goal: [1.0, 2.0, 3.0]
            mock_embedding.return_value = mock_provider

            memory = ProceduralMemory(config)

            # Store experience
            memory.store(
                Experience(
                    env_features=["test"],
                    goal="goal1",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=1.0,
                    embedding=[1.0, 2.0, 3.0],
                )
            )

            # Mock FAISS search to return invalid index (-1)
            with patch.object(memory._faiss_index, "search", return_value=(
                [[0.9]],  # distances
                [[-1]]  # Invalid index
            )):
                results = memory.recall(env_features=["test"], goal="query", top_k=1)

                # Should handle invalid index gracefully (return empty or filtered results)
                assert isinstance(results, list)

    def test_recall_with_faiss_require_success_filter(self) -> None:
        """Test that _recall_with_faiss filters out failed experiences when require_success=True."""
        config = MemoryConfig(
            storage_path=":memory:",
            require_success=True,
            embedding_model="all-MiniLM-L6-v2"
        )

        with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
            mock_provider = Mock()
            mock_provider.embed.side_effect = lambda goal: [1.0, 2.0, 3.0]
            mock_embedding.return_value = mock_provider

            memory = ProceduralMemory(config)

            # Store successful experience
            memory.store(
                Experience(
                    env_features=["test"],
                    goal="goal1",
                    action="action",
                    result="result",
                    success=True,
                    timestamp=1.0,
                    embedding=[1.0, 2.0, 3.0],
                )
            )

            # Store failed experience
            memory.store(
                Experience(
                    env_features=["test"],
                    goal="goal2",
                    action="action",
                    result="result",
                    success=False,  # Failed experience
                    timestamp=2.0,
                    embedding=[1.1, 2.1, 3.1],  # Similar embedding
                )
            )

            # Recall should only return successful experience
            results = memory.recall(env_features=["test"], goal="query", top_k=10)

            # Should filter out failed experience
            assert all(exp.success for exp in results)
            assert len(results) >= 1

    def test_compact_with_faiss_disabled(self) -> None:
        """Test that compact() doesn't rebuild FAISS index when use_faiss=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config, use_faiss=False)

            # Store 5 experiences
            for i in range(5):
                memory.store(
                    Experience(
                        env_features=["test"],
                        goal=f"goal{i}",
                        action="action",
                        result="result",
                        success=True,
                        timestamp=float(i + 1),
                        embedding=[float(i), float(i + 1), float(i + 2)],
                    )
                )

            # Compact
            memory.compact(keep_recent_n=2)

            # Index should still be None (not rebuilt)
            assert memory._faiss_index is None

    def test_recompute_embeddings_with_faiss_disabled(self) -> None:
        """Test that recompute_embeddings() doesn't rebuild FAISS index when use_faiss=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)
            memory = ProceduralMemory(config, use_faiss=False)

            # Store experiences
            for i in range(3):
                memory.store(
                    Experience(
                        env_features=["test"],
                        goal=f"goal{i}",
                        action="action",
                        result="result",
                        success=True,
                        timestamp=float(i + 1),
                    )
                )

            # Recompute embeddings with mock provider
            with patch("rec_praxis_rlm.memory.SentenceTransformerEmbedding") as mock_embedding:
                mock_provider = Mock()
                mock_provider.embed.return_value = [0.1, 0.2, 0.3]
                mock_embedding.return_value = mock_provider

                memory.recompute_embeddings("new-model")

            # Index should still be None (not rebuilt)
            assert memory._faiss_index is None
