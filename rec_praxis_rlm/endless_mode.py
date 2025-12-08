"""Endless mode for long-running agents with automatic context management.

This module provides token budget tracking, automatic compression, and context
window management to enable 100+ step agents without context exhaustion.

Features:
- Token counting and budget tracking
- Automatic context compression when approaching limits
- Progressive disclosure integration
- Context window monitoring
- Compaction triggers based on utilization

Usage:
    from rec_praxis_rlm.endless_mode import EndlessAgent

    agent = EndlessAgent(
        memory=memory,
        token_budget=100000,
        compression_threshold=0.4
    )

    # Track tokens used
    agent.track_tokens(prompt_tokens=500, completion_tokens=200)

    # Get recommended recall layer based on budget
    layer = agent.get_recommended_layer()

    # Auto-compress if needed
    if agent.should_compress():
        compressed = agent.auto_compress_context()
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

from rec_praxis_rlm.memory import ProceduralMemory, Experience

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Token budget tracker for context window management.

    Attributes:
        total_budget: Total token budget for the session
        used_tokens: Tokens used so far
        prompt_tokens: Tokens used in prompts
        completion_tokens: Tokens used in completions
        compression_events: Number of compression operations performed
    """
    total_budget: int = 100000
    used_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    compression_events: int = 0

    @property
    def remaining_tokens(self) -> int:
        """Calculate remaining token budget."""
        return max(0, self.total_budget - self.used_tokens)

    @property
    def utilization_rate(self) -> float:
        """Calculate budget utilization rate (0.0 to 1.0)."""
        return self.used_tokens / self.total_budget if self.total_budget > 0 else 0.0

    def track(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Track token usage.

        Args:
            prompt_tokens: Tokens used in prompt
            completion_tokens: Tokens used in completion
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.used_tokens = self.prompt_tokens + self.completion_tokens

    def reset(self) -> None:
        """Reset token counters (keeps budget)."""
        self.used_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0


@dataclass
class CompressionConfig:
    """Configuration for automatic compression.

    Attributes:
        threshold: Utilization threshold to trigger compression (0.0-1.0)
        target_rate: Target utilization after compression (0.0-1.0)
        min_experiences: Minimum experiences before enabling compression
        layer1_threshold: Use layer1 (compressed) above this utilization
        layer2_threshold: Use layer2 (full) below this utilization
        layer3_enabled: Enable layer3 (expanded) context
    """
    threshold: float = 0.4  # Compress when >40% budget used
    target_rate: float = 0.2  # Compress down to ~20% usage
    min_experiences: int = 10  # Need at least 10 experiences
    layer1_threshold: float = 0.3  # Use compressed summaries above 30%
    layer2_threshold: float = 0.5  # Use full details below 50%
    layer3_enabled: bool = False  # Disable expanded context by default


class EndlessAgent:
    """Agent with endless mode support via automatic context management.

    This class wraps ProceduralMemory with token budget tracking and automatic
    compression to enable long-running agents (100+ steps) without context exhaustion.

    Key features:
    - Token budget tracking with utilization monitoring
    - Automatic compression when approaching budget limits
    - Progressive disclosure layer selection based on budget
    - Context compaction triggers
    - Dumb zone avoidance (stay under 40% utilization)
    """

    def __init__(
        self,
        memory: ProceduralMemory,
        token_budget: int = 100000,
        compression_config: Optional[CompressionConfig] = None,
    ):
        """Initialize endless mode agent.

        Args:
            memory: ProceduralMemory instance
            token_budget: Total token budget for session
            compression_config: Optional compression configuration
        """
        self.memory = memory
        self.budget = TokenBudget(total_budget=token_budget)
        self.config = compression_config or CompressionConfig()

        logger.info(
            f"Initialized EndlessAgent with {token_budget} token budget, "
            f"compression threshold={self.config.threshold}"
        )

    def track_tokens(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Track token usage in the session.

        Args:
            prompt_tokens: Tokens used in prompt
            completion_tokens: Tokens used in completion
        """
        self.budget.track(prompt_tokens, completion_tokens)

        # Log warnings when approaching budget
        utilization = self.budget.utilization_rate
        if utilization >= 0.8:
            logger.warning(
                f"Token budget 80% exhausted "
                f"({self.budget.used_tokens}/{self.budget.total_budget} tokens)"
            )
        elif utilization >= self.config.threshold:
            logger.info(
                f"Token budget {utilization*100:.1f}% used, "
                f"consider compression or compaction"
            )

    def should_compress(self) -> bool:
        """Check if automatic compression should be triggered.

        Returns:
            True if compression should be performed
        """
        # Check utilization threshold
        if self.budget.utilization_rate < self.config.threshold:
            return False

        # Check minimum experience count
        if self.memory.size() < self.config.min_experiences:
            return False

        return True

    def get_recommended_layer(self) -> Literal[1, 2, 3]:
        """Get recommended progressive disclosure layer based on budget.

        Returns:
            1 (compressed), 2 (full), or 3 (expanded)
        """
        utilization = self.budget.utilization_rate

        # High utilization: use compressed summaries (layer1)
        if utilization >= self.config.layer1_threshold:
            return 1

        # Low utilization: use full details (layer2)
        if utilization <= self.config.layer2_threshold:
            return 2

        # Very low utilization + layer3 enabled: use expanded (layer3)
        if utilization <= 0.2 and self.config.layer3_enabled:
            return 3

        # Default: full details
        return 2

    def auto_compress_context(self) -> dict:
        """Automatically compress context to reduce token usage.

        This compacts memory by removing old experiences, targeting the
        configured target utilization rate.

        Returns:
            Dictionary with compression statistics
        """
        if not self.should_compress():
            return {
                "compressed": False,
                "reason": "Compression not needed",
                "utilization_before": self.budget.utilization_rate,
            }

        # Calculate how many experiences to keep
        current_size = self.memory.size()
        target_size = int(current_size * (self.config.target_rate / self.budget.utilization_rate))
        keep_n = max(self.config.min_experiences, target_size)

        logger.info(
            f"Auto-compressing memory: {current_size} -> {keep_n} experiences "
            f"(target utilization: {self.config.target_rate*100:.1f}%)"
        )

        # Compact memory
        removed = self.memory.compact(keep_recent_n=keep_n)

        # Estimate token savings (rough heuristic: ~1000 tokens per experience)
        estimated_tokens_saved = removed * 1000

        # Update budget (reduce used tokens)
        old_utilization = self.budget.utilization_rate
        self.budget.used_tokens = max(
            0,
            self.budget.used_tokens - estimated_tokens_saved
        )
        new_utilization = self.budget.utilization_rate

        self.budget.compression_events += 1

        logger.info(
            f"Compression complete: removed {removed} experiences, "
            f"saved ~{estimated_tokens_saved} tokens, "
            f"utilization {old_utilization*100:.1f}% -> {new_utilization*100:.1f}%"
        )

        return {
            "compressed": True,
            "experiences_removed": removed,
            "experiences_kept": keep_n,
            "estimated_tokens_saved": estimated_tokens_saved,
            "utilization_before": old_utilization,
            "utilization_after": new_utilization,
            "compression_events": self.budget.compression_events,
        }

    def recall_adaptive(
        self,
        env_features: list[str],
        goal: str,
        top_k: Optional[int] = None,
    ) -> tuple[list[str] | list[Experience], dict]:
        """Adaptive recall that automatically selects the best layer.

        Automatically chooses progressive disclosure layer based on current
        token budget utilization.

        Args:
            env_features: Environmental features to match
            goal: Goal to match
            top_k: Number of experiences to return

        Returns:
            Tuple of (experiences, metadata)
            experiences: Either compressed strings (layer1) or Experience objects (layer2/3)
            metadata: Dictionary with layer used and budget info
        """
        layer = self.get_recommended_layer()

        if layer == 1:
            # Use compressed summaries
            compressed, experiences = self.memory.recall_layer1(env_features, goal, top_k)
            return compressed, {
                "layer": 1,
                "count": len(compressed),
                "utilization": self.budget.utilization_rate,
                "format": "compressed_strings",
            }
        elif layer == 2:
            # Use full experiences
            experiences = self.memory.recall(env_features, goal, top_k)
            return experiences, {
                "layer": 2,
                "count": len(experiences),
                "utilization": self.budget.utilization_rate,
                "format": "full_experiences",
            }
        else:
            # Use expanded context (layer3)
            base_experiences = self.memory.recall(env_features, goal, top_k)
            expanded = self.memory.recall_layer3(base_experiences)
            return expanded, {
                "layer": 3,
                "count": len(expanded),
                "base_count": len(base_experiences),
                "utilization": self.budget.utilization_rate,
                "format": "expanded_experiences",
            }

    def get_status(self) -> dict:
        """Get current status of endless mode agent.

        Returns:
            Dictionary with budget, memory, and compression status
        """
        return {
            "token_budget": {
                "total": self.budget.total_budget,
                "used": self.budget.used_tokens,
                "remaining": self.budget.remaining_tokens,
                "utilization": self.budget.utilization_rate,
                "prompt_tokens": self.budget.prompt_tokens,
                "completion_tokens": self.budget.completion_tokens,
            },
            "memory": {
                "total_experiences": self.memory.size(),
                "has_embeddings": any(exp.embedding is not None for exp in self.memory.experiences),
                "has_faiss": self.memory.use_faiss,
            },
            "compression": {
                "enabled": True,
                "threshold": self.config.threshold,
                "target_rate": self.config.target_rate,
                "should_compress": self.should_compress(),
                "compression_events": self.budget.compression_events,
                "recommended_layer": self.get_recommended_layer(),
            },
        }

    def reset_budget(self, new_budget: Optional[int] = None) -> None:
        """Reset token budget for new session.

        Args:
            new_budget: Optional new budget (keeps existing if None)
        """
        if new_budget is not None:
            self.budget.total_budget = new_budget

        self.budget.reset()
        logger.info(f"Reset token budget to {self.budget.total_budget}")
