"""Production-ready agents for CLI tools and IDE integrations.

These agents implement the interface contract defined in rec_praxis_rlm.types
and are designed for use in pre-commit hooks, CI/CD pipelines, and IDE extensions.
"""

from rec_praxis_rlm.agents.code_review import CodeReviewAgent
from rec_praxis_rlm.agents.security_audit import SecurityAuditAgent
from rec_praxis_rlm.agents.dependency_scan import DependencyScanAgent
from rec_praxis_rlm.agents.test_generation import TestGenerationAgent

__all__ = [
    "CodeReviewAgent",
    "SecurityAuditAgent",
    "DependencyScanAgent",
    "TestGenerationAgent",
]
