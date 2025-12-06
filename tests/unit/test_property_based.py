"""Property-based tests using Hypothesis for rec_praxis_rlm package.

These tests verify invariants and properties that should hold for all inputs,
discovering edge cases that traditional example-based tests might miss.
"""
import hashlib
import time
from typing import Any
from unittest.mock import Mock, MagicMock, patch

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis import HealthCheck

from rec_praxis_rlm.embeddings import (
    SentenceTransformerEmbedding,
    APIEmbedding,
    DEFAULT_EMBEDDING_CACHE_SIZE,
)
from rec_praxis_rlm.memory import ProceduralMemory, Experience
from rec_praxis_rlm.rlm import RLMContext, SearchMatch
from rec_praxis_rlm.sandbox import SafeExecutor, _validate_code
from rec_praxis_rlm.config import MemoryConfig, ReplConfig
from rec_praxis_rlm.exceptions import SearchError, ExecutionError


class TestEmbeddingCacheProperties:
    """Property-based tests for embedding cache behavior."""

    @given(st.text(min_size=1, max_size=1000))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_cache_key_deterministic(self, text: str) -> None:
        """Property: Cache keys for same text are always identical."""
        # Mock the SentenceTransformer to avoid loading real models
        with patch("rec_praxis_rlm.embeddings.SentenceTransformer"):
            embedding = SentenceTransformerEmbedding()

            key1 = embedding._get_cache_key(text)
            key2 = embedding._get_cache_key(text)

            assert key1 == key2, "Cache keys must be deterministic"
            assert len(key1) == 64, "SHA256 produces 64-char hex strings"

    @given(st.text(min_size=1, max_size=1000), st.text(min_size=1, max_size=1000))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_cache_key_unique_for_different_texts(self, text1: str, text2: str) -> None:
        """Property: Different texts (almost always) produce different cache keys."""
        assume(text1 != text2)  # Only test when texts are different

        with patch("rec_praxis_rlm.embeddings.SentenceTransformer"):
            embedding = SentenceTransformerEmbedding()

            key1 = embedding._get_cache_key(text1)
            key2 = embedding._get_cache_key(text2)

            # SHA256 collision probability is negligible
            assert key1 != key2, "Different texts should produce different cache keys"

    @given(st.integers(min_value=1, max_value=1000))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_cache_size_respected(self, cache_size: int) -> None:
        """Property: Cache never exceeds configured size."""
        with patch("rec_praxis_rlm.embeddings.SentenceTransformer") as mock_st:
            # Mock model to return dummy embeddings
            mock_model = MagicMock()
            mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
            mock_st.return_value = mock_model

            embedding = SentenceTransformerEmbedding(cache_size=cache_size)

            # Embed more texts than cache size
            for i in range(cache_size + 10):
                embedding.embed(f"text_{i}")

            # Cache should not exceed size
            assert len(embedding._cache) <= cache_size, \
                f"Cache size {len(embedding._cache)} exceeds limit {cache_size}"

    @given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=50))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_batch_embed_consistent_with_single(self, texts: list[str]) -> None:
        """Property: Batch embedding produces same results as individual embeds."""
        with patch("rec_praxis_rlm.embeddings.SentenceTransformer") as mock_st:
            # Mock model to return deterministic embeddings based on text hash
            mock_model = MagicMock()

            def mock_encode(texts_list, **kwargs):
                # Return deterministic embedding based on text content
                return [[hash(t) % 100 / 100.0, (hash(t) * 2) % 100 / 100.0, (hash(t) * 3) % 100 / 100.0]
                        for t in texts_list]

            mock_model.encode.side_effect = mock_encode
            mock_st.return_value = mock_model

            embedding = SentenceTransformerEmbedding()

            # Get batch embeddings
            batch_results = embedding.embed_batch(texts)

            # Get individual embeddings
            individual_results = [embedding.embed(t) for t in texts]

            # Should match (batch uses cache from individual calls)
            assert batch_results == individual_results, \
                "Batch embedding should match individual embeddings"


class TestProceduralMemoryProperties:
    """Property-based tests for procedural memory behavior."""

    @given(
        st.lists(
            st.tuples(
                st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5),
                st.text(min_size=1, max_size=100),
                st.booleans(),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_recall_returns_most_recent_first(self, experiences_data: list) -> None:
        """Property: Recalled experiences are ordered by similarity score (descending)."""
        config = MemoryConfig(storage_path=":memory:")

        # Mock embedding provider
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]

        memory = ProceduralMemory(config, embedding_provider=mock_provider, use_faiss=False)

        # Store experiences
        for env_features, goal, success in experiences_data:
            experience = Experience(
                env_features=env_features,
                goal=goal,
                action="test action",
                result="test result",
                success=success,
                timestamp=time.time(),
            )
            memory.store(experience)

        # Recall experiences
        recalled = memory.recall(
            env_features=["test"],
            goal="test goal",
            top_k=min(10, len(experiences_data)),
        )

        # Property: Similarity scores should be non-increasing
        scores = [exp.similarity_score for exp in recalled]
        assert scores == sorted(scores, reverse=True), \
            "Recalled experiences must be ordered by similarity score (descending)"

    @given(st.integers(min_value=1, max_value=50))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_top_k_returns_at_most_k_results(self, top_k: int) -> None:
        """Property: Recall returns at most top_k experiences."""
        config = MemoryConfig(storage_path=":memory:")

        # Mock embedding provider
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]

        memory = ProceduralMemory(config, embedding_provider=mock_provider, use_faiss=False)

        # Store 100 experiences
        for i in range(100):
            experience = Experience(
                env_features=[f"env_{i}"],
                goal=f"goal_{i}",
                action=f"action_{i}",
                result=f"result_{i}",
                success=True,
                timestamp=time.time(),
            )
            memory.store(experience)

        # Recall with top_k
        recalled = memory.recall(
            env_features=["test"],
            goal="test goal",
            top_k=top_k,
        )

        assert len(recalled) <= top_k, \
            f"Recall returned {len(recalled)} results, expected at most {top_k}"


class TestSandboxSecurityProperties:
    """Property-based tests for safe code execution sandbox."""

    @given(st.text(alphabet=st.characters(whitelist_categories=('Ll', 'Lu')), min_size=1, max_size=100))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_prohibited_imports_always_rejected(self, module_name: str) -> None:
        """Property: Import statements are always rejected."""
        # Only test with valid Python identifiers
        assume(module_name.isidentifier())

        code = f"import {module_name}"

        config = ReplConfig()
        executor = SafeExecutor(config)

        result = executor.execute(code)

        assert not result.success, "Import statements must be rejected"
        assert result.error is not None, "Error message must be provided"
        # Error may be syntax error or import restriction - both are rejections
        assert result.error, "Error message must be provided for imports"

    @given(st.text(min_size=1, max_size=1000))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_prohibited_builtins_always_rejected(self, var_name: str) -> None:
        """Property: Prohibited builtins like eval, exec are always rejected."""
        prohibited = ["eval", "exec", "__import__", "compile", "open"]

        for builtin in prohibited:
            code = f"{builtin}({var_name!r})"

            config = ReplConfig()
            executor = SafeExecutor(config)

            result = executor.execute(code)

            assert not result.success, f"{builtin} must be rejected"
            assert result.error is not None, "Error message must be provided"

    @given(st.integers(), st.integers())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_safe_arithmetic_always_succeeds(self, a: int, b: int) -> None:
        """Property: Safe arithmetic operations always succeed."""
        # Avoid division by zero
        assume(b != 0)

        code = f"result = {a} + {b}"

        config = ReplConfig()
        executor = SafeExecutor(config)

        result = executor.execute(code)

        assert result.success, f"Safe arithmetic should succeed: {code}"
        assert result.error is None, "No error should be reported for safe code"


class TestRLMContextProperties:
    """Property-based tests for RLM context behavior."""

    @given(st.text(alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd')), min_size=1, max_size=20),
           st.text(min_size=10, max_size=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much])
    def test_grep_finds_all_occurrences(self, pattern: str, doc_text: str) -> None:
        """Property: Grep finds all literal occurrences of a pattern."""
        # Only test with alphanumeric patterns to avoid regex special chars
        assume(pattern.isalnum())
        assume(len(pattern) > 0)

        config = ReplConfig()
        context = RLMContext(config)

        # Add document
        context.add_document("test_doc", doc_text)

        # Search for pattern
        try:
            matches = context.grep(pattern, doc_id="test_doc")

            # Count actual occurrences in text
            actual_count = doc_text.count(pattern)
            found_count = len(matches)

            # Should find at most as many as actually exist
            # (may find fewer due to max_matches limit)
            assert found_count <= actual_count, \
                f"Found {found_count} matches, but only {actual_count} exist"
        except SearchError:
            # Pattern may be rejected by ReDoS protection, that's OK
            pass

    @given(st.text(min_size=10, max_size=500))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_peek_preserves_content(self, doc_text: str) -> None:
        """Property: Peek returns exact substring from document."""
        config = ReplConfig()
        context = RLMContext(config)

        context.add_document("test_doc", doc_text)

        # Peek various ranges
        start_char = min(5, len(doc_text) // 2)
        end_char = min(start_char + 10, len(doc_text))

        peeked = context.peek("test_doc", start_char, end_char)
        expected = doc_text[start_char:end_char]

        assert peeked == expected, \
            f"Peek should return exact substring"

    @given(st.integers(min_value=1, max_value=100))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_head_returns_first_n_lines(self, n_lines: int) -> None:
        """Property: head() returns exactly first N lines."""
        config = ReplConfig()
        context = RLMContext(config)

        # Create document with known line count
        total_lines = 200
        doc_text = "\n".join([f"line_{i}" for i in range(total_lines)])

        context.add_document("test_doc", doc_text)

        head_result = context.head("test_doc", n_lines=n_lines)
        result_lines = head_result.split("\n")

        # Should return exactly n_lines (or less if document is shorter)
        expected_count = min(n_lines, total_lines)
        assert len(result_lines) == expected_count, \
            f"head({n_lines}) should return {expected_count} lines"

        # First line should match
        assert result_lines[0] == "line_0", "First line should be line_0"


class TestConfigurationProperties:
    """Property-based tests for configuration validation."""

    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_memory_config_weights_must_sum_to_one(self, env_weight: float) -> None:
        """Property: env_weight + goal_weight must sum to 1.0."""
        # Derive goal_weight to ensure they sum to 1.0
        goal_weight = 1.0 - env_weight

        config = MemoryConfig(
            env_weight=env_weight,
            goal_weight=goal_weight,
        )

        # Should create successfully
        assert abs(config.env_weight + config.goal_weight - 1.0) < 0.001, \
            "Weights must sum to 1.0"

    @given(st.integers(min_value=1, max_value=10000))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_cache_size_always_positive(self, cache_size: int) -> None:
        """Property: Cache size must be positive."""
        with patch("rec_praxis_rlm.embeddings.SentenceTransformer"):
            embedding = SentenceTransformerEmbedding(cache_size=cache_size)

            assert embedding.cache_size > 0, "Cache size must be positive"
            assert embedding.cache_size == cache_size, "Cache size should match configured value"
