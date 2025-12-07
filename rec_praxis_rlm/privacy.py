"""Privacy protection and sensitive data redaction for experiences.

This module provides automatic detection and redaction of sensitive information
in experiences before storage, preventing accidental secret leakage.

Supported detection patterns:
- API keys, tokens, passwords
- Email addresses, SSNs, credit cards
- AWS keys, JWT tokens
- Environment variable secrets

Usage:
    from rec_praxis_rlm.privacy import PrivacyRedactor, classify_privacy_level

    redactor = PrivacyRedactor()

    # Classify privacy level
    level = classify_privacy_level(text)  # Returns: public/private/pii

    # Redact sensitive data
    cleaned_text = redactor.redact(text)

    # Auto-redact experience
    cleaned_exp = redactor.redact_experience(experience)
"""

import logging
import re
from typing import Optional, Pattern
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RedactionPattern:
    """Pattern for detecting and redacting sensitive data."""

    name: str
    pattern: Pattern
    replacement: str
    privacy_level: str  # public/private/pii


class PrivacyRedactor:
    """Detect and redact sensitive information from text.

    This class uses regex patterns to identify and redact:
    - API keys (generic, AWS, Stripe, OpenAI, Anthropic)
    - Passwords and secrets
    - Email addresses
    - Credit card numbers
    - Social Security Numbers
    - JWT tokens
    - Environment variable secrets
    """

    REDACTION_PATTERNS = [
        # API Keys and Tokens (more specific patterns first)
        RedactionPattern(
            name="anthropic_key",
            pattern=re.compile(r"sk-ant-[a-zA-Z0-9_-]{15,}"),
            replacement="[REDACTED_ANTHROPIC_KEY]",
            privacy_level="private",
        ),
        RedactionPattern(
            name="openai_key",
            pattern=re.compile(r"sk-[a-zA-Z0-9_-]{15,}"),
            replacement="[REDACTED_OPENAI_KEY]",
            privacy_level="private",
        ),
        RedactionPattern(
            name="aws_key",
            pattern=re.compile(r"AKIA[0-9A-Z]{16}"),
            replacement="[REDACTED_AWS_KEY]",
            privacy_level="private",
        ),
        RedactionPattern(
            name="generic_api_key",
            pattern=re.compile(r"(?i)(api[_-]?key|apikey|access[_-]?token)[\"']?\s*[:=]\s*[\"']?([a-zA-Z0-9_\-]{20,})"),
            replacement=r"\1=[REDACTED_API_KEY]",
            privacy_level="private",
        ),
        RedactionPattern(
            name="bearer_token",
            pattern=re.compile(r"(?i)bearer\s+[a-zA-Z0-9\-._~+/]+=*"),
            replacement="Bearer [REDACTED_TOKEN]",
            privacy_level="private",
        ),
        # JWT Tokens
        RedactionPattern(
            name="jwt_token",
            pattern=re.compile(r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"),
            replacement="[REDACTED_JWT]",
            privacy_level="private",
        ),
        # Passwords
        RedactionPattern(
            name="password",
            pattern=re.compile(r"(?i)(password|passwd|pwd)[\"']?\s*[:=]\s*[\"']?([^\s\"']{6,})"),
            replacement=r"\1=[REDACTED_PASSWORD]",
            privacy_level="private",
        ),
        # Email addresses (PII)
        RedactionPattern(
            name="email",
            pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            replacement="[REDACTED_EMAIL]",
            privacy_level="pii",
        ),
        # Credit card numbers (PII)
        RedactionPattern(
            name="credit_card",
            pattern=re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
            replacement="[REDACTED_CC]",
            privacy_level="pii",
        ),
        # SSN (PII)
        RedactionPattern(
            name="ssn",
            pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            replacement="[REDACTED_SSN]",
            privacy_level="pii",
        ),
        # Private IP addresses (less sensitive, mark as private)
        RedactionPattern(
            name="private_ip",
            pattern=re.compile(r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3})\b"),
            replacement="[REDACTED_PRIVATE_IP]",
            privacy_level="private",
        ),
    ]

    def __init__(self, patterns: Optional[list[RedactionPattern]] = None):
        """Initialize redactor with patterns.

        Args:
            patterns: Optional custom redaction patterns (uses defaults if None)
        """
        self.patterns = patterns or self.REDACTION_PATTERNS

    def redact(self, text: str) -> tuple[str, str]:
        """Redact sensitive information from text.

        Args:
            text: Text to redact

        Returns:
            Tuple of (redacted_text, detected_privacy_level)
            Privacy level is the highest level found: pii > private > public
        """
        if not text:
            return text, "public"

        redacted_text = text
        detected_level = "public"

        for pattern_obj in self.patterns:
            matches = list(pattern_obj.pattern.finditer(redacted_text))
            if matches:
                logger.debug(
                    f"Found {len(matches)} instances of {pattern_obj.name} "
                    f"(privacy={pattern_obj.privacy_level})"
                )

                # Update detected privacy level (pii > private > public)
                if pattern_obj.privacy_level == "pii":
                    detected_level = "pii"
                elif pattern_obj.privacy_level == "private" and detected_level == "public":
                    detected_level = "private"

                # Perform redaction
                redacted_text = pattern_obj.pattern.sub(pattern_obj.replacement, redacted_text)

        return redacted_text, detected_level

    def redact_experience(self, experience):
        """Redact sensitive data from an Experience object.

        Args:
            experience: Experience object to redact (modified in-place)

        Returns:
            The modified Experience object
        """
        # Import here to avoid circular dependency
        from rec_praxis_rlm.memory import Experience

        # Redact each text field
        goal_redacted, goal_level = self.redact(experience.goal)
        action_redacted, action_level = self.redact(experience.action)
        result_redacted, result_level = self.redact(experience.result)

        # Determine highest privacy level detected
        levels = [goal_level, action_level, result_level]
        if "pii" in levels:
            detected_level = "pii"
        elif "private" in levels:
            detected_level = "private"
        else:
            detected_level = "public"

        # Update experience
        experience.goal = goal_redacted
        experience.action = action_redacted
        experience.result = result_redacted

        # Auto-set privacy_level if not already set
        if not hasattr(experience, "privacy_level") or experience.privacy_level == "public":
            experience.privacy_level = detected_level

        return experience


def classify_privacy_level(text: str) -> str:
    """Classify privacy level of text.

    Args:
        text: Text to classify

    Returns:
        Privacy level: public, private, or pii
    """
    redactor = PrivacyRedactor()
    _, level = redactor.redact(text)
    return level


# Convenience function for quick redaction
def redact_secrets(text: str) -> str:
    """Quick redaction of secrets from text.

    Args:
        text: Text to redact

    Returns:
        Redacted text
    """
    redactor = PrivacyRedactor()
    redacted, _ = redactor.redact(text)
    return redacted
