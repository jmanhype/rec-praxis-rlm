"""Tests for FactStore semantic memory."""

import pytest
from rec_praxis_rlm.fact_store import FactStore, Fact


def test_fact_store_init():
    """Test FactStore initialization."""
    store = FactStore(storage_path=":memory:")
    assert store.count_facts() == 0
    store.close()


def test_extract_acronyms():
    """Test acronym extraction."""
    store = FactStore(storage_path=":memory:")

    text = "FAISS = Facebook AI Similarity Search. It provides fast indexing."
    facts = store.extract_facts(text, source_id="test_001")

    assert len(facts) >= 1
    acronym_facts = [f for f in facts if f.category == "Acronym" and f.key == "FAISS"]
    assert len(acronym_facts) == 1
    assert "Facebook AI Similarity Search" in acronym_facts[0].value

    store.close()


def test_extract_stands_for():
    """Test 'stands for' pattern extraction."""
    store = FactStore(storage_path=":memory:")

    text = "DSPy stands for Declarative Self-improving Python."
    facts = store.extract_facts(text, source_id="test_002")

    assert len(facts) >= 1
    dspy_facts = [f for f in facts if f.key == "DSPy"]
    assert len(dspy_facts) == 1
    assert "Declarative Self-improving Python" in dspy_facts[0].value

    store.close()


def test_extract_metrics():
    """Test metric extraction."""
    store = FactStore(storage_path=":memory:")

    text = "Achieved 10x speedup. Query time reduced to 0.08s from 5.2s."
    facts = store.extract_facts(text, source_id="test_003")

    # Should extract: 10x speedup, 0.08s, 5.2s
    assert len(facts) >= 2

    # Check for speedup metric
    speedup_facts = [f for f in facts if f.category == "Metric" and "10x" in f.value]
    assert len(speedup_facts) >= 1

    # Check for time metrics
    time_facts = [f for f in facts if f.category == "Metric" and ("0.08s" in f.value or "5.2s" in f.value)]
    assert len(time_facts) >= 1

    store.close()


def test_extract_key_value_pairs():
    """Test key-value pair extraction."""
    store = FactStore(storage_path=":memory:")

    text = "Set max_pool = 50 and min_pool = 10 for optimal performance."
    facts = store.extract_facts(text, source_id="test_004")

    # Should extract max_pool and min_pool
    max_pool_facts = [f for f in facts if f.key == "max_pool"]
    assert len(max_pool_facts) == 1
    assert "50" in max_pool_facts[0].value

    min_pool_facts = [f for f in facts if f.key == "min_pool"]
    assert len(min_pool_facts) == 1
    assert "10" in min_pool_facts[0].value

    store.close()


def test_query_facts():
    """Test querying facts."""
    store = FactStore(storage_path=":memory:")

    # Add some facts
    store.extract_facts("FAISS = Facebook AI Similarity Search", source_id="s1")
    store.extract_facts("Achieved 10x speedup", source_id="s2")
    store.extract_facts("DSPy stands for Declarative Self-improving Python", source_id="s3")

    # Query by key
    faiss_results = store.query("FAISS")
    assert len(faiss_results) >= 1
    assert faiss_results[0].key == "FAISS"

    # Query by value
    facebook_results = store.query("Facebook")
    assert len(facebook_results) >= 1

    # Query by category
    acronyms = store.query("", category="Acronym")
    assert len(acronyms) >= 2  # FAISS and DSPy

    metrics = store.query("", category="Metric")
    assert len(metrics) >= 1  # 10x speedup

    store.close()


def test_get_by_key():
    """Test exact key lookup."""
    store = FactStore(storage_path=":memory:")

    store.extract_facts("FAISS = Facebook AI Similarity Search", source_id="s1")

    # Exact key match
    fact = store.get_by_key("FAISS")
    assert fact is not None
    assert fact.key == "FAISS"
    assert "Facebook" in fact.value

    # Non-existent key
    fact = store.get_by_key("NONEXISTENT")
    assert fact is None

    store.close()


def test_get_all_by_category():
    """Test retrieving all facts in a category."""
    store = FactStore(storage_path=":memory:")

    store.extract_facts("FAISS = Facebook AI Similarity Search", source_id="s1")
    store.extract_facts("DSPy stands for Declarative Self-improving Python", source_id="s2")
    store.extract_facts("Achieved 10x speedup and 0.08s latency", source_id="s3")

    # Get all acronyms
    acronyms = store.get_all_by_category("Acronym")
    assert len(acronyms) >= 2

    # Get all metrics
    metrics = store.get_all_by_category("Metric")
    assert len(metrics) >= 1

    store.close()


def test_fact_evidence():
    """Test that facts include evidence context."""
    store = FactStore(storage_path=":memory:")

    text = "We use FAISS = Facebook AI Similarity Search for fast indexing."
    facts = store.extract_facts(text, source_id="test_evidence")

    faiss_fact = [f for f in facts if f.key == "FAISS"][0]
    assert faiss_fact.evidence is not None
    assert len(faiss_fact.evidence) > 0
    assert "FAISS" in faiss_fact.evidence

    store.close()


def test_fact_source_id():
    """Test that facts are linked to source experiences."""
    store = FactStore(storage_path=":memory:")

    store.extract_facts("FAISS = Facebook AI Similarity Search", source_id="exp_123")

    fact = store.get_by_key("FAISS")
    assert fact.source_id == "exp_123"

    store.close()


def test_multiple_facts_same_text():
    """Test extracting multiple facts from same text."""
    store = FactStore(storage_path=":memory:")

    text = """
    We use FAISS = Facebook AI Similarity Search for indexing.
    It provides 10x speedup compared to linear search.
    Query time reduced from 5.2s to 0.08s.
    Configuration: max_pool = 50, min_pool = 10.
    """

    facts = store.extract_facts(text, source_id="multi_fact_test")

    # Should extract multiple fact types
    assert len(facts) >= 4

    # Verify we got different categories
    categories = set(f.category for f in facts)
    assert "Acronym" in categories  # FAISS
    assert "Metric" in categories  # 10x, 0.08s, 5.2s

    store.close()


def test_context_manager():
    """Test FactStore as context manager."""
    with FactStore(storage_path=":memory:") as store:
        store.extract_facts("FAISS = Facebook AI Similarity Search", source_id="ctx_test")
        assert store.count_facts() >= 1

    # Store should be closed after exiting context
    # (Can't easily test this without implementation details)


def test_temporal_ordering():
    """Test that newer facts are returned first."""
    store = FactStore(storage_path=":memory:")

    # Add facts over time (simulated)
    import time

    store.extract_facts("endpoint = /api/v1/users", source_id="old")
    time.sleep(0.01)  # Small delay
    store.extract_facts("endpoint = /api/v2/users", source_id="newer")
    time.sleep(0.01)
    store.extract_facts("endpoint = /api/v3/users", source_id="newest")

    # Query should return newest first
    results = store.query("endpoint")
    assert len(results) == 3
    assert "v3" in results[0].value  # Most recent
    assert results[0].source_id == "newest"

    store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
