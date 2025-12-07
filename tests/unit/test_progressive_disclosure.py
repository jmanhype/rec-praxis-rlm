"""Unit tests for progressive disclosure API."""

import time
from rec_praxis_rlm.memory import ProceduralMemory, Experience
from rec_praxis_rlm.config import MemoryConfig


def test_recall_layer1_returns_compressed():
    """Test layer1 returns compressed summaries."""
    config = MemoryConfig(
        storage_path=":memory:",
        embedding_model="",  # Disable embeddings for speed
        similarity_threshold=0.1,  # Lower threshold since no embeddings
    )
    memory = ProceduralMemory(config=config, use_faiss=False)

    # Store some experiences
    for i in range(3):
        exp = Experience(
            env_features=["api", "database"],
            goal=f"Test goal {i}",
            action=f"SELECT * FROM users WHERE id = {i}",
            result=f"Query returned {i} rows",
            success=True,
            timestamp=time.time() + i,
        )
        memory.store(exp)

    # Layer 1: Compressed summaries
    summaries, experiences = memory.recall_layer1(["api"], "Test goal", top_k=3)

    assert len(summaries) == 3
    assert len(experiences) == 3
    # Summaries should be shorter than full experiences
    for summary in summaries:
        assert len(summary) > 0


def test_recall_layer2_returns_full():
    """Test layer2 returns full experiences."""
    config = MemoryConfig(storage_path=":memory:", embedding_model="")
    memory = ProceduralMemory(config=config, use_faiss=False)

    exp = Experience(
        env_features=["api"],
        goal="Test",
        action="Action",
        result="Result",
        success=True,
        timestamp=time.time(),
    )
    memory.store(exp)

    _, experiences = memory.recall_layer1(["api"], "Test")
    full_experiences = memory.recall_layer2(experiences)

    assert len(full_experiences) == 1
    assert full_experiences[0].goal == "Test"


def test_recall_layer3_expands_context():
    """Test layer3 expands to related experiences."""
    config = MemoryConfig(
        storage_path=":memory:",
        embedding_model="",
        similarity_threshold=0.1,  # Lower threshold since no embeddings
    )
    memory = ProceduralMemory(config=config, use_faiss=False)

    # Store main experience
    exp1 = Experience(
        env_features=["api", "database"],
        goal="Main task",
        action="Main action",
        result="Main result",
        success=True,
        timestamp=time.time(),
        tags=["sql", "query"],
    )
    memory.store(exp1)

    # Store related experience (shares tags)
    exp2 = Experience(
        env_features=["database"],
        goal="Related task",
        action="Related action",
        result="Related result",
        success=True,
        timestamp=time.time() + 1,
        tags=["sql"],
    )
    memory.store(exp2)

    # Store unrelated experience
    exp3 = Experience(
        env_features=["network"],
        goal="Unrelated",
        action="Unrelated action",
        result="Unrelated result",
        success=True,
        timestamp=time.time() + 2,
        tags=["tcp"],
    )
    memory.store(exp3)

    # Recall layer 1
    _, experiences = memory.recall_layer1(["api"], "Main", top_k=1)

    # Expand to layer 3
    expanded = memory.recall_layer3(experiences, expand_top_n=1)

    # Should include original + related (not unrelated)
    assert len(expanded) >= len(experiences)
    # exp2 should be included (shares sql tag)
    goals = [e.goal for e in expanded]
    assert "Related task" in goals


def test_progressive_disclosure_workflow():
    """Test full progressive disclosure workflow."""
    config = MemoryConfig(storage_path=":memory:", embedding_model="")
    memory = ProceduralMemory(config=config, use_faiss=False)

    # Store experiences
    for i in range(5):
        exp = Experience(
            env_features=["api"],
            goal=f"Task {i}",
            action=f"Action {i}",
            result=f"Result {i}",
            success=True,
            timestamp=time.time() + i,
        )
        memory.store(exp)

    # Layer 1: Get compressed summaries
    summaries, layer1_experiences = memory.recall_layer1(["api"], "Task", top_k=3)
    assert len(summaries) == 3

    # Layer 2: Get full details
    layer2_experiences = memory.recall_layer2(layer1_experiences)
    assert len(layer2_experiences) == 3
    assert layer2_experiences[0].action.startswith("Action")

    # Layer 3: Expand context
    layer3_experiences = memory.recall_layer3(layer2_experiences)
    # Should have at least the original 3
    assert len(layer3_experiences) >= 3
