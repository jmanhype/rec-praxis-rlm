"""Tests for ProceduralMemory + FactStore integration."""

import time
import pytest
from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig, FactStore


def test_memory_with_factstore():
    """Test that ProceduralMemory auto-extracts facts when FactStore is configured."""
    # Create fact store
    fact_store = FactStore(storage_path=":memory:")

    # Create memory with fact store integration
    memory = ProceduralMemory(
        config=MemoryConfig(storage_path=":memory:"),
        fact_store=fact_store
    )

    # Store experience with facts
    memory.store(Experience(
        env_features=["database", "postgresql", "performance"],
        goal="optimize slow query",
        action="Added composite index (user_id, created_at)",
        result="Reduced query time from 5.2s to 0.08s",
        success=True,
        timestamp=time.time()
    ))

    # Check that facts were extracted
    assert fact_store.count_facts() > 0

    # Query for extracted facts (should find time metrics)
    time_facts = fact_store.query("query_time")
    assert len(time_facts) >= 1

    # Verify fact has correct source_id
    assert time_facts[0].source_id is not None
    assert time_facts[0].source_id.startswith("exp_")

    # Verify the fact contains a time value
    assert "s" in time_facts[0].value  # Should contain "s" for seconds


def test_memory_with_acronym_extraction():
    """Test acronym extraction from experiences."""
    fact_store = FactStore(storage_path=":memory:")
    memory = ProceduralMemory(
        config=MemoryConfig(storage_path=":memory:"),
        fact_store=fact_store
    )

    # Store experience with acronym
    memory.store(Experience(
        env_features=["indexing", "similarity"],
        goal="implement fast similarity search",
        action="Used FAISS = Facebook AI Similarity Search for indexing",
        result="Achieved 10x speedup compared to linear search",
        success=True,
        timestamp=time.time()
    ))

    # Check acronym was extracted
    faiss_facts = fact_store.query("FAISS")
    assert len(faiss_facts) >= 1
    assert "Facebook" in faiss_facts[0].value

    # Check metric was extracted
    speedup_facts = fact_store.query("speedup")
    assert len(speedup_facts) >= 1


def test_memory_without_factstore():
    """Test that memory works normally without FactStore (backward compatibility)."""
    # Create memory without fact store
    memory = ProceduralMemory(config=MemoryConfig(storage_path=":memory:"))

    # Store experience
    memory.store(Experience(
        env_features=["test"],
        goal="test goal",
        action="test action",
        result="test result",
        success=True,
        timestamp=time.time()
    ))

    # Should work fine (no errors)
    assert memory.size() == 1


def test_factstore_preserves_experience_source():
    """Test that facts are linked to source experiences."""
    fact_store = FactStore(storage_path=":memory:")
    memory = ProceduralMemory(
        config=MemoryConfig(storage_path=":memory:"),
        fact_store=fact_store
    )

    # Store experience
    exp_time = time.time()
    memory.store(Experience(
        env_features=["config"],
        goal="configure connection pool",
        action="Set max_pool = 50 and min_pool = 10",
        result="Configuration applied successfully",
        success=True,
        timestamp=exp_time
    ))

    # Query facts
    max_pool_facts = fact_store.query("max_pool")
    assert len(max_pool_facts) == 1

    # Verify source_id matches timestamp
    expected_source_id = f"exp_{int(exp_time)}"
    assert max_pool_facts[0].source_id == expected_source_id


def test_multiple_facts_from_single_experience():
    """Test extracting multiple facts from one experience."""
    fact_store = FactStore(storage_path=":memory:")
    memory = ProceduralMemory(
        config=MemoryConfig(storage_path=":memory:"),
        fact_store=fact_store
    )

    # Store rich experience with multiple facts
    memory.store(Experience(
        env_features=["optimization", "caching"],
        goal="improve API performance",
        action="Implemented Redis caching with TTL = 300",
        result="Response time reduced from 2.5s to 0.1s (25x speedup)",
        success=True,
        timestamp=time.time()
    ))

    # Should extract multiple facts: TTL value, response times, speedup
    assert fact_store.count_facts() >= 2

    # Verify different fact types were extracted
    ttl_facts = fact_store.query("TTL")
    time_facts = fact_store.query("0.1s")

    assert len(ttl_facts) >= 1 or len(time_facts) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
