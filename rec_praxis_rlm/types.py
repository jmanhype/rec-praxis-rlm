"""Type definitions for rec-praxis-rlm agents and CLI tools.

This module defines the interface contract that all agents must implement
to work with the CLI tools and IDE integrations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Severity(Enum):
    """Severity levels for code review findings."""
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class OWASPCategory(Enum):
    """OWASP Top 10 security categories."""
    A01_BROKEN_ACCESS_CONTROL = "A01:2021 - Broken Access Control"
    A02_CRYPTOGRAPHIC_FAILURES = "A02:2021 - Cryptographic Failures"
    A03_INJECTION = "A03:2021 - Injection"
    A04_INSECURE_DESIGN = "A04:2021 - Insecure Design"
    A05_SECURITY_MISCONFIGURATION = "A05:2021 - Security Misconfiguration"
    A06_VULNERABLE_COMPONENTS = "A06:2021 - Vulnerable and Outdated Components"
    A07_IDENTIFICATION_FAILURES = "A07:2021 - Identification and Authentication Failures"
    A08_DATA_INTEGRITY_FAILURES = "A08:2021 - Software and Data Integrity Failures"
    A09_LOGGING_FAILURES = "A09:2021 - Security Logging and Monitoring Failures"
    A10_SSRF = "A10:2021 - Server-Side Request Forgery"


@dataclass
class Finding:
    """A code review or security finding.

    This is the interface contract that all agent implementations must follow.
    CLI tools expect findings to have these exact fields.
    """
    file_path: str
    severity: Severity
    title: str
    description: str
    remediation: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    owasp_category: Optional[OWASPCategory] = None
    cwe_id: Optional[str] = None
    confidence: Optional[float] = None  # 0.0-1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert finding to JSON-serializable dict."""
        return {
            "file": self.file_path,
            "line": self.line_number,
            "column": self.column_number,
            "severity": self.severity.name,
            "title": self.title,
            "description": self.description,
            "remediation": self.remediation,
            "owasp": self.owasp_category.value if self.owasp_category else None,
            "cwe": self.cwe_id,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class CVEFinding:
    """A CVE vulnerability finding in dependencies."""
    package_name: str
    installed_version: str
    severity: Severity
    cve_id: str
    description: str
    remediation: str
    fixed_version: Optional[str] = None
    cvss_score: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert CVE finding to JSON-serializable dict."""
        return {
            "type": "CVE",
            "package": self.package_name,
            "version": self.installed_version,
            "severity": self.severity.name,
            "cve_id": self.cve_id,
            "title": f"{self.cve_id} in {self.package_name}",
            "description": self.description,
            "remediation": self.remediation,
            "fixed_version": self.fixed_version,
            "cvss_score": self.cvss_score,
        }


@dataclass
class SecretFinding:
    """A hardcoded secret finding."""
    file_path: str
    line_number: int
    secret_type: str  # "API Key", "Password", "Private Key", etc.
    severity: Severity
    description: str
    remediation: str
    matched_text: Optional[str] = None  # Redacted preview

    def to_dict(self) -> dict:
        """Convert secret finding to JSON-serializable dict."""
        return {
            "type": "Secret",
            "file": self.file_path,
            "line": self.line_number,
            "severity": self.severity.name,
            "title": f"{self.secret_type} detected",
            "description": self.description,
            "remediation": self.remediation,
            "matched_text": self.matched_text,
        }


@dataclass
class AuditReport:
    """Security audit report containing multiple findings."""
    findings: list[Finding]
    summary: str
    files_scanned: int
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int

    def to_dict(self) -> dict:
        """Convert audit report to JSON-serializable dict."""
        return {
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "files_scanned": self.files_scanned,
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "high_issues": self.high_issues,
            "medium_issues": self.medium_issues,
            "low_issues": self.low_issues,
        }
