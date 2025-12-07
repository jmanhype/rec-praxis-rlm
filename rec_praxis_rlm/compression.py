"""Observation compression module for reducing token usage in memory recall.

Inspired by claude-mem's compression strategy, this module compresses experiences
from ~2000-3000 tokens down to ~500 tokens using LLM summarization.

Benefits:
- 80-90% token reduction in recall
- Enable 20 experiences instead of 6 in same token budget
- Preserve essential information (goal, action, outcome)
- Maintain semantic similarity for retrieval

Usage:
    from rec_praxis_rlm.compression import ObservationCompressor

    compressor = ObservationCompressor(model="openai/gpt-4o-mini")

    # Compress single experience
    compressed = compressor.compress_experience(experience)

    # Compress batch for recall
    compressed_list = compressor.compress_batch(experiences)
"""

import logging
from typing import Optional, Protocol, runtime_checkable

from rec_praxis_rlm.memory import Experience

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers supporting text generation."""

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        ...


class OpenAIProvider:
    """OpenAI LLM provider for compression."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """Initialize OpenAI provider.

        Args:
            model: OpenAI model name (default: gpt-4o-mini for cost efficiency)
            api_key: Optional API key (falls back to OPENAI_API_KEY env var)

        Raises:
            ImportError: If openai package not installed
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required for compression. "
                "Install with: pip install openai"
            )

        self.model = model
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # Use default client (reads OPENAI_API_KEY from env)
            self.client = openai.OpenAI()

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using OpenAI API.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise summarizer. Extract only essential information.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,  # Deterministic for compression
        )

        return response.choices[0].message.content or ""


class ObservationCompressor:
    """Compress experiences using LLM summarization to reduce token usage.

    This class uses an LLM to compress experiences from ~2000-3000 tokens down
    to ~500 tokens while preserving essential information for retrieval.

    Compression strategy:
    1. Extract core elements: goal, action, outcome, key metrics
    2. Remove verbose details, stack traces, redundant information
    3. Preserve semantic meaning for similarity search
    4. Maintain success/failure signal
    """

    COMPRESSION_PROMPT_TEMPLATE = """Compress this agent experience into ~100 words while preserving essential information:

**Environment Features:** {env_features}
**Goal:** {goal}
**Action Taken:** {action}
**Result:** {result}
**Success:** {success}
**Metadata:** {metadata}

Extract ONLY:
1. Core goal (1 sentence)
2. Key action taken (1-2 sentences)
3. Critical outcome/metrics (1-2 sentences)
4. Success/failure reason (1 sentence)

Format as concise bullet points. Omit verbose details, stack traces, full code."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        """Initialize compressor with LLM provider.

        Args:
            provider: Optional LLM provider (for dependency injection)
            model: Model name if using default OpenAI provider
            api_key: API key if using default OpenAI provider

        Raises:
            ImportError: If openai not installed and no provider given
        """
        if provider is not None:
            self.provider = provider
        else:
            # Create default OpenAI provider
            self.provider = OpenAIProvider(model=model, api_key=api_key)

    def compress_experience(
        self, experience: Experience, max_tokens: int = 150
    ) -> str:
        """Compress a single experience to ~500 tokens.

        Args:
            experience: Experience to compress
            max_tokens: Maximum tokens for compressed output

        Returns:
            Compressed experience as string (~500 tokens)

        Raises:
            Exception: If LLM API call fails
        """
        # Format prompt with experience details
        prompt = self.COMPRESSION_PROMPT_TEMPLATE.format(
            env_features=", ".join(experience.env_features),
            goal=experience.goal,
            action=experience.action[:1000],  # Truncate very long actions
            result=experience.result[:1000],  # Truncate very long results
            success="✅ Success" if experience.success else "❌ Failed",
            metadata=str(experience.metadata) if experience.metadata else "None",
        )

        try:
            compressed = self.provider.generate(prompt, max_tokens=max_tokens)
            logger.debug(
                f"Compressed experience from ~{len(prompt)} chars to {len(compressed)} chars"
            )
            return compressed
        except Exception as e:
            logger.error(f"Compression failed: {e}. Returning truncated original.")
            # Fallback: truncate to approximate token limit
            fallback = f"{experience.goal}\n{experience.action[:200]}\n{experience.result[:200]}"
            return fallback[:500]  # Rough token estimate

    def compress_batch(
        self, experiences: list[Experience], max_tokens_per_exp: int = 150
    ) -> list[str]:
        """Compress batch of experiences for recall.

        Args:
            experiences: List of experiences to compress
            max_tokens_per_exp: Maximum tokens per compressed experience

        Returns:
            List of compressed experience strings
        """
        compressed_list = []
        for exp in experiences:
            compressed = self.compress_experience(exp, max_tokens=max_tokens_per_exp)
            compressed_list.append(compressed)

        return compressed_list

    def format_for_prompt(self, compressed_experiences: list[str]) -> str:
        """Format compressed experiences for LLM prompt injection.

        Args:
            compressed_experiences: List of compressed experience strings

        Returns:
            Formatted string ready for prompt injection
        """
        formatted = "## Relevant Past Experiences\n\n"
        for i, exp in enumerate(compressed_experiences, start=1):
            formatted += f"### Experience {i}\n{exp}\n\n"

        return formatted
