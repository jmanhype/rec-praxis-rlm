"""Unit tests for privacy redaction module."""

import pytest
from rec_praxis_rlm.privacy import (
    PrivacyRedactor,
    RedactionPattern,
    classify_privacy_level,
    redact_secrets,
)
from rec_praxis_rlm.memory import Experience
import time
import re


def test_redact_openai_key():
    """Test redaction of OpenAI API keys."""
    redactor = PrivacyRedactor()
    text = "My API key is sk-proj-abc123xyz456 and it's secret"

    redacted, level = redactor.redact(text)

    assert "sk-proj-abc123xyz456" not in redacted
    assert "[REDACTED_OPENAI_KEY]" in redacted
    assert level == "private"


def test_redact_anthropic_key():
    """Test redaction of Anthropic API keys."""
    redactor = PrivacyRedactor()
    text = "API key: sk-ant-api03-abc123xyz456def789ghi012"

    redacted, level = redactor.redact(text)

    assert "sk-ant-api03-abc123xyz456def789ghi012" not in redacted
    assert "[REDACTED_ANTHROPIC_KEY]" in redacted
    assert level == "private"


def test_redact_aws_key():
    """Test redaction of AWS access keys."""
    redactor = PrivacyRedactor()
    text = "AWS key AKIAIOSFODNN7EXAMPLE is here"

    redacted, level = redactor.redact(text)

    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert "[REDACTED_AWS_KEY]" in redacted
    assert level == "private"


def test_redact_bearer_token():
    """Test redaction of Bearer tokens."""
    redactor = PrivacyRedactor()
    text = "Authorization: Bearer abc123.xyz456.def789"

    redacted, level = redactor.redact(text)

    assert "abc123" not in redacted
    assert "Bearer [REDACTED_TOKEN]" in redacted
    assert level == "private"


def test_redact_jwt():
    """Test redaction of JWT tokens."""
    redactor = PrivacyRedactor()
    text = "JWT: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

    redacted, level = redactor.redact(text)

    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted
    assert "[REDACTED_JWT]" in redacted
    assert level == "private"


def test_redact_password():
    """Test redaction of passwords."""
    redactor = PrivacyRedactor()
    text = "password=mySecret123 in config"

    redacted, level = redactor.redact(text)

    assert "mySecret123" not in redacted
    assert "password=[REDACTED_PASSWORD]" in redacted
    assert level == "private"


def test_redact_email():
    """Test redaction of email addresses (PII)."""
    redactor = PrivacyRedactor()
    text = "Contact me at user@example.com for details"

    redacted, level = redactor.redact(text)

    assert "user@example.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert level == "pii"


def test_redact_credit_card():
    """Test redaction of credit card numbers (PII)."""
    redactor = PrivacyRedactor()
    text = "Card number: 4532-1234-5678-9010"

    redacted, level = redactor.redact(text)

    assert "4532-1234-5678-9010" not in redacted
    assert "[REDACTED_CC]" in redacted
    assert level == "pii"


def test_redact_ssn():
    """Test redaction of SSNs (PII)."""
    redactor = PrivacyRedactor()
    text = "SSN: 123-45-6789"

    redacted, level = redactor.redact(text)

    assert "123-45-6789" not in redacted
    assert "[REDACTED_SSN]" in redacted
    assert level == "pii"


def test_redact_private_ip():
    """Test redaction of private IP addresses."""
    redactor = PrivacyRedactor()
    text = "Server at 192.168.1.100 and 10.0.0.50"

    redacted, level = redactor.redact(text)

    assert "192.168.1.100" not in redacted
    assert "10.0.0.50" not in redacted
    assert "[REDACTED_PRIVATE_IP]" in redacted
    assert level == "private"


def test_privacy_level_hierarchy():
    """Test that PII level takes precedence over private."""
    redactor = PrivacyRedactor()
    text = "API key: sk-abc123xyz456def789 and email user@example.com"

    redacted, level = redactor.redact(text)

    # Both should be redacted
    assert "sk-abc123xyz456def789" not in redacted
    assert "user@example.com" not in redacted

    # Level should be PII (higher than private)
    assert level == "pii"


def test_redact_experience():
    """Test redacting entire Experience object."""
    redactor = PrivacyRedactor()

    experience = Experience(
        env_features=["api", "database"],
        goal="Fetch user profile",
        action="API call with key sk-proj-abc123xyz456def789",
        result="User data: user@example.com retrieved",
        success=True,
        timestamp=time.time(),
    )

    redacted_exp = redactor.redact_experience(experience)

    # Check redactions
    assert "sk-proj-abc123xyz456def789" not in redacted_exp.action
    assert "[REDACTED_OPENAI_KEY]" in redacted_exp.action
    assert "user@example.com" not in redacted_exp.result
    assert "[REDACTED_EMAIL]" in redacted_exp.result

    # Check privacy level auto-set to highest (PII)
    assert redacted_exp.privacy_level == "pii"


def test_redact_experience_public():
    """Test that public data gets public privacy level."""
    redactor = PrivacyRedactor()

    experience = Experience(
        env_features=["api"],
        goal="Test public data",
        action="Simple action without secrets",
        result="Result without sensitive data",
        success=True,
        timestamp=time.time(),
    )

    redacted_exp = redactor.redact_experience(experience)

    # No redactions should occur
    assert redacted_exp.goal == "Test public data"
    assert redacted_exp.action == "Simple action without secrets"

    # Privacy level should be public
    assert redacted_exp.privacy_level == "public"


def test_classify_privacy_level():
    """Test privacy level classification function."""
    assert classify_privacy_level("Plain public text") == "public"
    assert classify_privacy_level("API key sk-abc123xyz456def789") == "private"
    assert classify_privacy_level("Email: user@example.com") == "pii"


def test_redact_secrets_convenience():
    """Test convenience function for quick redaction."""
    text = "API key sk-abc123xyz456def789 and password=secret123"
    redacted = redact_secrets(text)

    assert "sk-abc123xyz456def789" not in redacted
    assert "secret123" not in redacted
    assert "[REDACTED_OPENAI_KEY]" in redacted


def test_custom_redaction_pattern():
    """Test adding custom redaction patterns."""
    custom_pattern = RedactionPattern(
        name="custom_id",
        pattern=re.compile(r"ID-\d{6}"),
        replacement="[REDACTED_ID]",
        privacy_level="private",
    )

    redactor = PrivacyRedactor(patterns=[custom_pattern])
    text = "User ID-123456 accessed the system"

    redacted, level = redactor.redact(text)

    assert "ID-123456" not in redacted
    assert "[REDACTED_ID]" in redacted
    assert level == "private"


def test_empty_text():
    """Test redaction of empty text."""
    redactor = PrivacyRedactor()
    redacted, level = redactor.redact("")

    assert redacted == ""
    assert level == "public"


def test_multiple_secrets_same_type():
    """Test redacting multiple secrets of the same type."""
    redactor = PrivacyRedactor()
    text = "Keys: sk-abc123xyz456def789 and sk-xyz789abc123def456"

    redacted, level = redactor.redact(text)

    # Both should be redacted
    assert "sk-abc123xyz456def789" not in redacted
    assert "sk-xyz789abc123def456" not in redacted
    assert redacted.count("[REDACTED_OPENAI_KEY]") == 2
