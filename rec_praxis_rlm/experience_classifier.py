"""Experience type classification for procedural memory.

Auto-classifies experiences into 4 types based on goal/action/result text:
- learn: New pattern/knowledge acquisition
- recover: Error recovery/debugging
- optimize: Performance/efficiency improvements
- explore: Experimentation/discovery

Usage:
    from rec_praxis_rlm.experience_classifier import ExperienceClassifier

    classifier = ExperienceClassifier()

    # Classify experience
    experience_type = classifier.classify(
        goal="Fix timeout error in API call",
        action="Added retry logic with exponential backoff",
        result="API calls now succeed consistently",
        success=True,
    )
    # Returns: "recover"
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Keywords for each experience type
LEARN_KEYWORDS = [
    "learn", "understand", "discover", "find", "new", "first time",
    "explore", "investigate", "research", "study", "analyze",
    "pattern", "example", "tutorial", "guide", "documentation",
]

RECOVER_KEYWORDS = [
    "fix", "error", "bug", "issue", "problem", "fail", "crash",
    "debug", "troubleshoot", "resolve", "repair", "correct",
    "exception", "warning", "broken", "not working", "failing",
]

OPTIMIZE_KEYWORDS = [
    "optimize", "improve", "faster", "slower", "performance",
    "refactor", "cleanup", "simplify", "reduce", "increase",
    "efficiency", "cache", "benchmark", "latency", "throughput",
]

EXPLORE_KEYWORDS = [
    "explore", "experiment", "try", "test", "prototype",
    "investigation", "research", "alternative", "different approach",
    "what if", "could", "might", "maybe", "attempt",
]


class ExperienceClassifier:
    """Classify experiences into types for better recall relevance.

    Uses heuristic keyword matching to classify experiences into:
    - learn: Acquiring new knowledge/patterns
    - recover: Error recovery/debugging
    - optimize: Performance improvements
    - explore: Experimentation/discovery
    """

    def __init__(self):
        """Initialize experience classifier."""
        pass

    def classify(
        self,
        goal: str,
        action: str,
        result: str,
        success: bool,
    ) -> str:
        """Classify experience type based on text and outcome.

        Args:
            goal: Goal description
            action: Action taken
            result: Result of action
            success: Whether action was successful

        Returns:
            Experience type: "learn", "recover", "optimize", or "explore"
        """
        # Combine all text for analysis (lowercase for matching)
        combined_text = f"{goal} {action} {result}".lower()

        # Count keyword matches for each type
        scores = {
            "recover": self._count_matches(combined_text, RECOVER_KEYWORDS),
            "optimize": self._count_matches(combined_text, OPTIMIZE_KEYWORDS),
            "explore": self._count_matches(combined_text, EXPLORE_KEYWORDS),
            "learn": self._count_matches(combined_text, LEARN_KEYWORDS),
        }

        # Apply success/failure heuristics
        if not success:
            # Failed experiences are often recovery attempts
            scores["recover"] += 2
        else:
            # Successful experiences weighted toward learning
            scores["learn"] += 1

        # Return type with highest score (default to "learn")
        max_score = max(scores.values())
        if max_score == 0:
            # No keywords matched, default to "learn"
            return "learn"

        # Return highest scoring type
        for exp_type, score in scores.items():
            if score == max_score:
                return exp_type

        return "learn"  # Fallback

    def _count_matches(self, text: str, keywords: list[str]) -> int:
        """Count how many keywords appear in text.

        Args:
            text: Text to search (already lowercased)
            keywords: List of keywords to match

        Returns:
            Number of keyword matches
        """
        count = 0
        for keyword in keywords:
            if keyword in text:
                count += 1
        return count

    def classify_experience(self, experience):
        """Classify an Experience object and update its experience_type field.

        Args:
            experience: Experience object to classify (modified in-place)

        Returns:
            The modified Experience object with experience_type populated
        """
        # Import here to avoid circular dependency
        from rec_praxis_rlm.memory import Experience

        # Classify based on text fields
        exp_type = self.classify(
            goal=experience.goal,
            action=experience.action,
            result=experience.result,
            success=experience.success,
        )

        # Update experience
        experience.experience_type = exp_type

        logger.debug(f"Classified experience as '{exp_type}'")

        return experience
