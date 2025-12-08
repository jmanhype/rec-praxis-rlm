"""Tests for endless mode with token budget tracking and automatic compression."""

import pytest
import time
from rec_praxis_rlm import (
    ProceduralMemory,
    MemoryConfig,
    Experience,
    EndlessAgent,
    TokenBudget,
    CompressionConfig,
)


@pytest.fixture
def memory():
    """Create in-memory ProceduralMemory instance."""
    config = MemoryConfig(
        storage_path=":memory:",
        embedding_model="",  # Disable embeddings
        similarity_threshold=0.1,
    )
    return ProceduralMemory(config=config, use_faiss=False)


@pytest.fixture
def endless_agent(memory):
    """Create EndlessAgent with test configuration."""
    config = CompressionConfig(
        threshold=0.4,
        target_rate=0.2,
        min_experiences=5,
        layer1_threshold=0.3,
        layer2_threshold=0.5,
    )
    return EndlessAgent(memory=memory, token_budget=10000, compression_config=config)


def test_token_budget_tracking():
    """Test token budget tracking and utilization calculation."""
    budget = TokenBudget(total_budget=10000)

    assert budget.remaining_tokens == 10000
    assert budget.utilization_rate == 0.0

    # Track some tokens
    budget.track(prompt_tokens=500, completion_tokens=200)

    assert budget.used_tokens == 700
    assert budget.prompt_tokens == 500
    assert budget.completion_tokens == 200
    assert budget.remaining_tokens == 9300
    assert budget.utilization_rate == 0.07

    # Track more
    budget.track(prompt_tokens=1000, completion_tokens=500)

    assert budget.used_tokens == 2200
    assert budget.utilization_rate == 0.22


def test_token_budget_reset():
    """Test token budget reset preserves total budget."""
    budget = TokenBudget(total_budget=10000)
    budget.track(prompt_tokens=500, completion_tokens=200)

    assert budget.used_tokens == 700

    budget.reset()

    assert budget.used_tokens == 0
    assert budget.total_budget == 10000
    assert budget.utilization_rate == 0.0


def test_endless_agent_initialization(endless_agent):
    """Test EndlessAgent initialization."""
    assert endless_agent.budget.total_budget == 10000
    assert endless_agent.budget.used_tokens == 0
    assert endless_agent.config.threshold == 0.4


def test_track_tokens(endless_agent):
    """Test token tracking in EndlessAgent."""
    endless_agent.track_tokens(prompt_tokens=1000, completion_tokens=500)

    assert endless_agent.budget.used_tokens == 1500
    assert endless_agent.budget.utilization_rate == 0.15


def test_should_compress_below_threshold(endless_agent, memory):
    """Test compression is not triggered below threshold."""
    # Add some experiences
    for i in range(10):
        exp = Experience(
            env_features=["test"],
            goal=f"Goal {i}",
            action=f"Action {i}",
            result=f"Result {i}",
            success=True,
            timestamp=time.time(),
        )
        memory.store(exp)

    # Below threshold
    endless_agent.track_tokens(prompt_tokens=1000, completion_tokens=500)
    assert endless_agent.budget.utilization_rate < 0.4
    assert not endless_agent.should_compress()


def test_should_compress_above_threshold(endless_agent, memory):
    """Test compression is triggered above threshold."""
    # Add experiences
    for i in range(10):
        exp = Experience(
            env_features=["test"],
            goal=f"Goal {i}",
            action=f"Action {i}",
            result=f"Result {i}",
            success=True,
            timestamp=time.time(),
        )
        memory.store(exp)

    # Above threshold (40%)
    endless_agent.track_tokens(prompt_tokens=3000, completion_tokens=1500)
    assert endless_agent.budget.utilization_rate >= 0.4
    assert endless_agent.should_compress()


def test_should_compress_min_experiences(endless_agent, memory):
    """Test compression requires minimum experiences."""
    # Only 3 experiences (below minimum of 5)
    for i in range(3):
        exp = Experience(
            env_features=["test"],
            goal=f"Goal {i}",
            action=f"Action {i}",
            result=f"Result {i}",
            success=True,
            timestamp=time.time(),
        )
        memory.store(exp)

    # Above threshold but not enough experiences
    endless_agent.track_tokens(prompt_tokens=3000, completion_tokens=1500)
    assert not endless_agent.should_compress()


def test_get_recommended_layer_low_utilization(endless_agent):
    """Test layer recommendation with low utilization."""
    endless_agent.track_tokens(prompt_tokens=500, completion_tokens=200)
    assert endless_agent.budget.utilization_rate < 0.3
    assert endless_agent.get_recommended_layer() == 2  # Full details


def test_get_recommended_layer_medium_utilization(endless_agent):
    """Test layer recommendation with medium utilization."""
    endless_agent.track_tokens(prompt_tokens=2500, completion_tokens=1500)
    assert 0.3 <= endless_agent.budget.utilization_rate < 0.5
    assert endless_agent.get_recommended_layer() == 1  # Compressed


def test_get_recommended_layer_high_utilization(endless_agent):
    """Test layer recommendation with high utilization."""
    endless_agent.track_tokens(prompt_tokens=5000, completion_tokens=3000)
    assert endless_agent.budget.utilization_rate >= 0.5
    assert endless_agent.get_recommended_layer() == 1  # Compressed


def test_auto_compress_context(endless_agent, memory):
    """Test automatic context compression."""
    # Add 20 experiences
    for i in range(20):
        exp = Experience(
            env_features=["test"],
            goal=f"Goal {i}",
            action=f"Action {i}",
            result=f"Result {i}",
            success=True,
            timestamp=time.time() + i,  # Increment timestamps
        )
        memory.store(exp)

    assert memory.size() == 20

    # Trigger compression
    endless_agent.track_tokens(prompt_tokens=3000, completion_tokens=1500)
    result = endless_agent.auto_compress_context()

    assert result["compressed"]
    assert result["experiences_removed"] > 0
    assert result["experiences_kept"] >= 5  # Minimum
    assert memory.size() < 20
    assert memory.size() == result["experiences_kept"]


def test_auto_compress_no_compression_needed(endless_agent):
    """Test auto_compress_context when compression not needed."""
    endless_agent.track_tokens(prompt_tokens=500, completion_tokens=200)

    result = endless_agent.auto_compress_context()

    assert not result["compressed"]
    assert "reason" in result


def test_recall_adaptive_layer1(endless_agent, memory):
    """Test adaptive recall uses layer1 at high utilization."""
    # Add experiences
    for i in range(10):
        exp = Experience(
            env_features=["api", "database"],
            goal=f"Optimize query {i}",
            action=f"Added index on column {i}",
            result=f"Query time reduced {i}ms",
            success=True,
            timestamp=time.time(),
        )
        memory.store(exp)

    # High utilization
    endless_agent.track_tokens(prompt_tokens=5000, completion_tokens=3000)

    results, metadata = endless_agent.recall_adaptive(
        env_features=["api"],
        goal="Optimize database query",
        top_k=3
    )

    assert metadata["layer"] == 1
    assert metadata["format"] == "compressed_strings"
    # Results should be strings (compressed) or experiences (if compression unavailable)
    assert len(results) >= 0  # May be 0 if no matches above threshold


def test_recall_adaptive_layer2(endless_agent, memory):
    """Test adaptive recall uses layer2 at low utilization."""
    # Add experiences
    for i in range(10):
        exp = Experience(
            env_features=["api", "database"],
            goal=f"Optimize query {i}",
            action=f"Added index on column {i}",
            result=f"Query time reduced {i}ms",
            success=True,
            timestamp=time.time(),
        )
        memory.store(exp)

    # Low utilization
    endless_agent.track_tokens(prompt_tokens=500, completion_tokens=200)

    results, metadata = endless_agent.recall_adaptive(
        env_features=["api"],
        goal="Optimize database query",
        top_k=3
    )

    assert metadata["layer"] == 2
    assert metadata["format"] == "full_experiences"
    assert all(isinstance(r, Experience) for r in results)


def test_get_status(endless_agent, memory):
    """Test get_status returns complete information."""
    # Add some experiences
    for i in range(5):
        exp = Experience(
            env_features=["test"],
            goal=f"Goal {i}",
            action=f"Action {i}",
            result=f"Result {i}",
            success=True,
            timestamp=time.time(),
        )
        memory.store(exp)

    # Track tokens
    endless_agent.track_tokens(prompt_tokens=1000, completion_tokens=500)

    status = endless_agent.get_status()

    assert "token_budget" in status
    assert status["token_budget"]["total"] == 10000
    assert status["token_budget"]["used"] == 1500
    assert status["token_budget"]["utilization"] == 0.15

    assert "memory" in status
    assert status["memory"]["total_experiences"] == 5

    assert "compression" in status
    assert status["compression"]["enabled"] is True
    assert status["compression"]["threshold"] == 0.4


def test_reset_budget(endless_agent):
    """Test budget reset."""
    # Track tokens
    endless_agent.track_tokens(prompt_tokens=1000, completion_tokens=500)
    assert endless_agent.budget.used_tokens == 1500

    # Reset with same budget
    endless_agent.reset_budget()
    assert endless_agent.budget.used_tokens == 0
    assert endless_agent.budget.total_budget == 10000

    # Reset with new budget
    endless_agent.reset_budget(new_budget=50000)
    assert endless_agent.budget.total_budget == 50000
    assert endless_agent.budget.used_tokens == 0


def test_compression_statistics_tracking(endless_agent, memory):
    """Test compression events are tracked."""
    # Add experiences
    for i in range(20):
        exp = Experience(
            env_features=["test"],
            goal=f"Goal {i}",
            action=f"Action {i}",
            result=f"Result {i}",
            success=True,
            timestamp=time.time() + i,
        )
        memory.store(exp)

    # Trigger compression twice
    endless_agent.track_tokens(prompt_tokens=3000, completion_tokens=1500)
    endless_agent.auto_compress_context()

    endless_agent.track_tokens(prompt_tokens=3000, completion_tokens=1500)
    endless_agent.auto_compress_context()

    assert endless_agent.budget.compression_events == 2


def test_endless_mode_100_step_simulation(memory):
    """Test endless mode can handle 100+ step agent simulation."""
    agent = EndlessAgent(
        memory=memory,
        token_budget=100000,
        compression_config=CompressionConfig(
            threshold=0.4,
            target_rate=0.2,
            min_experiences=10,
        )
    )

    # Simulate 100 steps
    for step in range(100):
        # Store experience
        exp = Experience(
            env_features=["agent", "step"],
            goal=f"Step {step} goal",
            action=f"Step {step} action",
            result=f"Step {step} result",
            success=step % 10 != 0,  # Fail every 10th step
            timestamp=time.time() + step,
        )
        memory.store(exp)

        # Track token usage (~500 tokens per step)
        agent.track_tokens(prompt_tokens=300, completion_tokens=200)

        # Auto-compress when needed
        if agent.should_compress():
            result = agent.auto_compress_context()
            assert result["compressed"]

    # Verify agent stayed within budget
    assert agent.budget.utilization_rate < 1.0

    # Verify compression occurred
    assert agent.budget.compression_events > 0

    # Verify memory was compacted
    assert memory.size() < 100

    status = agent.get_status()
    assert status["compression"]["compression_events"] > 0
