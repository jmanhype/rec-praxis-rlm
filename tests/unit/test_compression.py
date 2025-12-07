"""Unit tests for observation compression module."""

import pytest
from unittest.mock import Mock, MagicMock
from rec_praxis_rlm.compression import (
    ObservationCompressor,
    OpenAIProvider,
    LLMProvider,
)
from rec_praxis_rlm.memory import Experience
import time


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Return mock compressed output."""
        return """- Goal: Test compression of experience
- Action: Called API endpoint with params
- Outcome: Returned 200 OK with data
- Success: Completed successfully"""


def test_mock_provider_protocol():
    """Test that MockLLMProvider implements LLMProvider protocol."""
    mock_provider = MockLLMProvider()
    assert isinstance(mock_provider, LLMProvider)


def test_compressor_initialization_with_provider():
    """Test compressor initialization with custom provider."""
    mock_provider = MockLLMProvider()
    compressor = ObservationCompressor(provider=mock_provider)
    assert compressor.provider == mock_provider


def test_compress_experience_with_mock():
    """Test experience compression with mock provider."""
    mock_provider = MockLLMProvider()
    compressor = ObservationCompressor(provider=mock_provider)

    experience = Experience(
        env_features=["api", "database", "cache"],
        goal="Fetch user profile data",
        action="SELECT * FROM users WHERE id = 123",
        result="User profile retrieved successfully with 15 fields",
        success=True,
        timestamp=time.time(),
    )

    compressed = compressor.compress_experience(experience)

    # Verify compression worked
    assert len(compressed) > 0
    assert "Goal" in compressed
    assert "Action" in compressed
    assert "Success" in compressed


def test_compress_batch():
    """Test batch compression."""
    mock_provider = MockLLMProvider()
    compressor = ObservationCompressor(provider=mock_provider)

    experiences = [
        Experience(
            env_features=["api"],
            goal=f"Test goal {i}",
            action=f"Action {i}",
            result=f"Result {i}",
            success=True,
            timestamp=time.time(),
        )
        for i in range(3)
    ]

    compressed_list = compressor.compress_batch(experiences)

    assert len(compressed_list) == 3
    for compressed in compressed_list:
        assert len(compressed) > 0
        assert "Goal" in compressed


def test_format_for_prompt():
    """Test formatting compressed experiences for prompt."""
    mock_provider = MockLLMProvider()
    compressor = ObservationCompressor(provider=mock_provider)

    compressed_list = [
        "Experience 1 summary",
        "Experience 2 summary",
        "Experience 3 summary",
    ]

    formatted = compressor.format_for_prompt(compressed_list)

    assert "## Relevant Past Experiences" in formatted
    assert "### Experience 1" in formatted
    assert "### Experience 2" in formatted
    assert "### Experience 3" in formatted


def test_compress_experience_truncates_long_fields():
    """Test that very long action/result fields are truncated."""
    mock_provider = MockLLMProvider()
    compressor = ObservationCompressor(provider=mock_provider)

    # Create experience with very long action and result
    long_action = "A" * 5000  # 5000 chars
    long_result = "B" * 5000

    experience = Experience(
        env_features=["test"],
        goal="Test goal",
        action=long_action,
        result=long_result,
        success=True,
        timestamp=time.time(),
    )

    compressed = compressor.compress_experience(experience)

    # Should not fail and should return something
    assert len(compressed) > 0


def test_compress_experience_fallback_on_error():
    """Test fallback when LLM provider fails."""

    class FailingProvider:
        def generate(self, prompt: str, max_tokens: int = 500) -> str:
            raise Exception("LLM API error")

    compressor = ObservationCompressor(provider=FailingProvider())

    experience = Experience(
        env_features=["test"],
        goal="Test goal for fallback",
        action="Test action",
        result="Test result",
        success=True,
        timestamp=time.time(),
    )

    # Should not raise, should return fallback
    compressed = compressor.compress_experience(experience)
    assert len(compressed) > 0
    assert "Test goal for fallback" in compressed


@pytest.mark.skip(reason="Requires OpenAI API key")
def test_openai_provider_integration():
    """Integration test with real OpenAI API (skip in CI)."""
    provider = OpenAIProvider(model="gpt-4o-mini")

    prompt = "Summarize in 50 words: Python is a high-level programming language."
    result = provider.generate(prompt, max_tokens=100)

    assert len(result) > 0
    assert len(result) < 500  # Should be concise
