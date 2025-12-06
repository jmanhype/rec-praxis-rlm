"""End-to-end integration tests for procedural memory."""
import time
import tempfile
import os

import pytest

from rec_praxis_rlm.memory import Experience, ProceduralMemory
from rec_praxis_rlm.config import MemoryConfig


class TestIT001StoreAndRetrieve:
    """IT-001: Store 50 experiences and retrieve with new query."""

    def test_store_50_and_retrieve_top_k(self) -> None:
        """Store 50 experiences with varying features, retrieve with new query, verify top-k ranking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path, top_k=6, similarity_threshold=0.0)
            memory = ProceduralMemory(config)

            # Store 50 experiences with different environmental features
            # Create clusters of similar experiences
            experiences = []

            # Cluster 1: Web scraping with sidebar (10 experiences)
            for i in range(10):
                experiences.append(
                    Experience(
                        env_features=["has_sidebar", "article_layout", "pagination"],
                        goal=f"extract article {i}",
                        action=f"css_selector('article#{i}')",
                        result=f"extracted article {i}",
                        success=True,
                        timestamp=time.time() + i,
                    )
                )

            # Cluster 2: Table extraction (10 experiences)
            for i in range(10):
                experiences.append(
                    Experience(
                        env_features=["has_table", "data_heavy", "structured"],
                        goal=f"extract table data {i}",
                        action=f"css_selector('table#{i}')",
                        result=f"extracted table {i}",
                        success=True,
                        timestamp=time.time() + 10 + i,
                    )
                )

            # Cluster 3: Image extraction (10 experiences)
            for i in range(10):
                experiences.append(
                    Experience(
                        env_features=["has_images", "gallery", "lightbox"],
                        goal=f"extract images {i}",
                        action=f"css_selector('img.gallery-{i}')",
                        result=f"extracted {i} images",
                        success=True,
                        timestamp=time.time() + 20 + i,
                    )
                )

            # Cluster 4: Form interactions (10 experiences)
            for i in range(10):
                experiences.append(
                    Experience(
                        env_features=["has_form", "authentication", "validation"],
                        goal=f"submit form {i}",
                        action=f"fill_form('form-{i}')",
                        result=f"form {i} submitted",
                        success=True,
                        timestamp=time.time() + 30 + i,
                    )
                )

            # Cluster 5: Dynamic content (10 experiences)
            for i in range(10):
                experiences.append(
                    Experience(
                        env_features=["dynamic_loading", "javascript_heavy", "spa"],
                        goal=f"wait for content {i}",
                        action=f"wait_for_selector('#content-{i}')",
                        result=f"content {i} loaded",
                        success=True,
                        timestamp=time.time() + 40 + i,
                    )
                )

            # Store all experiences
            for exp in experiences:
                memory.store(exp)

            assert memory.size() == 50

            # Query: Looking for article extraction with sidebar
            # Should match Cluster 1 most closely
            query_env_features = ["has_sidebar", "article_layout", "pagination"]
            query_goal = "extract main article content"

            results = memory.recall(
                env_features=query_env_features,
                goal=query_goal,
                top_k=6,
            )

            # Verify we got 6 results
            assert len(results) == 6

            # Verify results are from Cluster 1 (all should have "has_sidebar")
            for result in results:
                assert "has_sidebar" in result.env_features

            # Verify results are sorted by similarity (all should have identical env_features)
            for result in results:
                assert set(result.env_features) == set(query_env_features)


class TestIT004PersistenceAcrossRestarts:
    """IT-004: Test persistence across process restarts."""

    def test_persistence_across_restart(self) -> None:
        """Store 100 experiences, simulate restart, verify all retrievable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "memory.jsonl")
            config = MemoryConfig(storage_path=storage_path)

            # First "process": Create memory and store 100 experiences
            memory1 = ProceduralMemory(config)

            for i in range(100):
                memory1.store(
                    Experience(
                        env_features=[f"feature_{i % 10}"],
                        goal=f"goal {i}",
                        action=f"action {i}",
                        result=f"result {i}",
                        success=True,
                        timestamp=time.time() + i,
                    )
                )

            assert memory1.size() == 100

            # Simulate process restart: Create new memory instance
            memory2 = ProceduralMemory(config)

            # Verify all 100 experiences loaded
            assert memory2.size() == 100

            # Verify we can retrieve experiences
            results = memory2.recall(
                env_features=["feature_5"],
                goal="goal 25",
                top_k=10,
            )

            # Should get at least some results
            assert len(results) > 0

            # Verify specific experience is retrievable
            matching_results = [r for r in results if r.goal == "goal 25"]
            assert len(matching_results) > 0


class TestIT006HybridScoring:
    """IT-006: Test hybrid scoring with high env overlap + low goal sim vs. vice versa."""

    def test_hybrid_scoring_weighting(self) -> None:
        """Verify that hybrid weighting correctly balances environmental and goal similarity."""
        config = MemoryConfig(
            storage_path=":memory:",
            env_weight=0.6,
            goal_weight=0.4,
            similarity_threshold=0.0,
        )
        memory = ProceduralMemory(config)

        # Experience 1: High environmental similarity, different goal
        exp1 = Experience(
            env_features=["a", "b", "c"],
            goal="completely different goal about tables",
            action="action1",
            result="result1",
            success=True,
            timestamp=time.time(),
        )

        # Experience 2: Low environmental similarity, similar goal
        exp2 = Experience(
            env_features=["x", "y", "z"],
            goal="extract article text",
            action="action2",
            result="result2",
            success=True,
            timestamp=time.time(),
        )

        # Experience 3: Moderate environmental and goal similarity
        exp3 = Experience(
            env_features=["a", "b", "d"],
            goal="extract article content",
            action="action3",
            result="result3",
            success=True,
            timestamp=time.time(),
        )

        memory.store(exp1)
        memory.store(exp2)
        memory.store(exp3)

        # Query: High env similarity with exp1, high goal similarity with exp2
        results = memory.recall(
            env_features=["a", "b", "c"],
            goal="extract article text from page",
            top_k=3,
        )

        # With 60% env weight:
        # - exp1 should score high on env (1.0 * 0.6 = 0.6) but low on goal
        # - exp2 should score low on env (0.0 * 0.6 = 0.0) but high on goal
        # - exp3 should score moderate on both

        # The exact ranking depends on embeddings, but we can verify:
        # 1. All three are returned
        assert len(results) == 3

        # 2. exp1 should be highly ranked due to perfect env match
        exp1_idx = next(i for i, r in enumerate(results) if r.action == "action1")
        # With 60% env weight, perfect env match should rank first or second
        assert exp1_idx <= 1


class TestIT007EmbeddingFallback:
    """IT-007: Test text-based fallback when embeddings disabled."""

    def test_fallback_without_embeddings(self) -> None:
        """Disable embeddings, verify text-based retrieval still works with warning."""
        # Use a non-existent model to force fallback
        config = MemoryConfig(
            storage_path=":memory:",
            embedding_model="nonexistent-model-should-fail",
        )

        # This should log a warning but not crash
        memory = ProceduralMemory(config)

        # Store experiences
        exp1 = Experience(
            env_features=["feature_a"],
            goal="test goal",
            action="action1",
            result="result1",
            success=True,
            timestamp=time.time(),
        )

        exp2 = Experience(
            env_features=["feature_b"],
            goal="different goal",
            action="action2",
            result="result2",
            success=True,
            timestamp=time.time(),
        )

        memory.store(exp1)
        memory.store(exp2)

        # Recall should still work (using env features only)
        results = memory.recall(
            env_features=["feature_a"],
            goal="test goal",
            top_k=2,
        )

        # Should get exp1 first due to env feature match
        assert len(results) >= 1
        assert results[0].env_features == ["feature_a"]
