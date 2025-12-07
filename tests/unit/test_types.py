"""Unit tests for rec_praxis_rlm.types module."""

import pytest
from rec_praxis_rlm.types import (
    Severity,
    Finding,
    CVEFinding,
    SecretFinding,
    OWASPCategory,
    AuditReport,
)


class TestSeverity:
    """Test cases for Severity enum."""

    def test_severity_enum_values(self):
        """Verify all severity levels exist with correct values."""
        assert Severity.INFO.value == 0
        assert Severity.LOW.value == 1
        assert Severity.MEDIUM.value == 2
        assert Severity.HIGH.value == 3
        assert Severity.CRITICAL.value == 4

    def test_severity_ordering(self):
        """Verify severity levels can be compared by value."""
        assert Severity.INFO.value < Severity.LOW.value
        assert Severity.LOW.value < Severity.MEDIUM.value
        assert Severity.MEDIUM.value < Severity.HIGH.value
        assert Severity.HIGH.value < Severity.CRITICAL.value

    def test_severity_from_name(self):
        """Verify severity can be accessed by name."""
        assert Severity["CRITICAL"] == Severity.CRITICAL
        assert Severity["INFO"] == Severity.INFO


class TestFinding:
    """Test cases for Finding dataclass."""

    def test_finding_creation(self):
        """Verify Finding can be instantiated with required fields."""
        finding = Finding(
            file_path="test.py",
            severity=Severity.HIGH,
            title="Test Issue",
            description="This is a test",
            remediation="Fix it"
        )
        assert finding.file_path == "test.py"
        assert finding.severity == Severity.HIGH
        assert finding.title == "Test Issue"

    def test_finding_optional_fields(self):
        """Verify Finding optional fields default to None."""
        finding = Finding(
            file_path="test.py",
            severity=Severity.HIGH,
            title="Test",
            description="Test",
            remediation="Fix"
        )
        assert finding.line_number is None
        assert finding.column_number is None
        assert finding.owasp_category is None
        assert finding.cwe_id is None
        assert finding.confidence is None

    def test_finding_with_all_fields(self):
        """Verify Finding with all optional fields populated."""
        finding = Finding(
            file_path="test.py",
            line_number=42,
            column_number=10,
            severity=Severity.CRITICAL,
            title="SQL Injection",
            description="SQL injection vulnerability",
            remediation="Use parameterized queries",
            owasp_category=OWASPCategory.A03_INJECTION,
            cwe_id="CWE-89",
            confidence=0.95
        )
        assert finding.line_number == 42
        assert finding.column_number == 10
        assert finding.owasp_category == OWASPCategory.A03_INJECTION
        assert finding.cwe_id == "CWE-89"
        assert finding.confidence == 0.95

    def test_finding_to_dict(self):
        """Verify Finding.to_dict() produces correct JSON structure."""
        finding = Finding(
            file_path="app.py",
            line_number=100,
            severity=Severity.HIGH,
            title="Hardcoded Password",
            description="Password in source",
            remediation="Use env vars",
            owasp_category=OWASPCategory.A07_IDENTIFICATION_FAILURES,
            cwe_id="CWE-798",
            confidence=0.99
        )

        result = finding.to_dict()

        assert result["file"] == "app.py"
        assert result["line"] == 100
        assert result["severity"] == "HIGH"
        assert result["title"] == "Hardcoded Password"
        assert result["description"] == "Password in source"
        assert result["remediation"] == "Use env vars"
        assert result["owasp"] == "A07:2021 - Identification and Authentication Failures"
        assert result["cwe"] == "CWE-798"
        assert result["confidence"] == 0.99

    def test_finding_to_dict_with_none_values(self):
        """Verify Finding.to_dict() handles None values correctly."""
        finding = Finding(
            file_path="test.py",
            severity=Severity.LOW,
            title="Minor Issue",
            description="Not important",
            remediation="Optional fix"
        )

        result = finding.to_dict()

        assert result["file"] == "test.py"
        assert result["line"] is None
        assert result["owasp"] is None
        assert result["cwe"] is None
        assert result["confidence"] is None


class TestCVEFinding:
    """Test cases for CVEFinding dataclass."""

    def test_cve_finding_creation(self):
        """Verify CVEFinding can be instantiated."""
        finding = CVEFinding(
            package_name="requests",
            installed_version="2.25.0",
            cve_id="CVE-2021-1234",
            severity=Severity.HIGH,
            description="Security vulnerability",
            remediation="Upgrade to 2.26.0",
            fixed_version="2.26.0"
        )
        assert finding.package_name == "requests"
        assert finding.cve_id == "CVE-2021-1234"
        assert finding.fixed_version == "2.26.0"

    def test_cve_finding_no_fixed_version(self):
        """Verify CVEFinding works without fixed_version."""
        finding = CVEFinding(
            package_name="old-lib",
            installed_version="1.0.0",
            cve_id="CVE-2020-9999",
            severity=Severity.CRITICAL,
            description="No fix available",
            remediation="Consider alternative package"
        )
        assert finding.fixed_version is None

    def test_cve_finding_to_dict(self):
        """Verify CVEFinding.to_dict() produces correct structure."""
        finding = CVEFinding(
            package_name="django",
            installed_version="3.0.0",
            cve_id="CVE-2021-5678",
            severity=Severity.CRITICAL,
            description="Critical vulnerability",
            remediation="Upgrade to 3.2.0",
            fixed_version="3.2.0"
        )

        result = finding.to_dict()

        assert result["package"] == "django"
        assert result["version"] == "3.0.0"
        assert result["cve_id"] == "CVE-2021-5678"
        assert result["severity"] == "CRITICAL"
        assert result["description"] == "Critical vulnerability"
        assert result["fixed_version"] == "3.2.0"
        assert result["type"] == "CVE"


class TestSecretFinding:
    """Test cases for SecretFinding dataclass."""

    def test_secret_finding_creation(self):
        """Verify SecretFinding can be instantiated."""
        finding = SecretFinding(
            file_path="config.py",
            line_number=42,
            secret_type="API Key",
            severity=Severity.CRITICAL,
            description="Hardcoded API key detected",
            remediation="Use environment variables"
        )
        assert finding.file_path == "config.py"
        assert finding.line_number == 42
        assert finding.secret_type == "API Key"

    def test_secret_finding_to_dict(self):
        """Verify SecretFinding.to_dict() produces correct structure."""
        finding = SecretFinding(
            file_path="app.py",
            line_number=10,
            secret_type="AWS Access Key",
            severity=Severity.CRITICAL,
            description="AWS credentials exposed",
            remediation="Rotate keys and use IAM roles"
        )

        result = finding.to_dict()

        assert result["file"] == "app.py"
        assert result["line"] == 10
        assert result["title"] == "AWS Access Key detected"
        assert result["severity"] == "CRITICAL"
        assert result["type"] == "Secret"


class TestOWASPCategory:
    """Test cases for OWASPCategory enum."""

    def test_owasp_categories_exist(self):
        """Verify all OWASP Top 10 categories exist."""
        categories = [
            OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
            OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
            OWASPCategory.A03_INJECTION,
            OWASPCategory.A04_INSECURE_DESIGN,
            OWASPCategory.A05_SECURITY_MISCONFIGURATION,
            OWASPCategory.A06_VULNERABLE_COMPONENTS,
            OWASPCategory.A07_IDENTIFICATION_FAILURES,
            OWASPCategory.A08_DATA_INTEGRITY_FAILURES,
            OWASPCategory.A09_LOGGING_FAILURES,
            OWASPCategory.A10_SSRF,
        ]
        assert len(categories) == 10

    def test_owasp_category_values(self):
        """Verify OWASP category string values."""
        assert OWASPCategory.A01_BROKEN_ACCESS_CONTROL.value == "A01:2021 - Broken Access Control"
        assert OWASPCategory.A03_INJECTION.value == "A03:2021 - Injection"
        assert OWASPCategory.A10_SSRF.value == "A10:2021 - Server-Side Request Forgery"


class TestAuditReport:
    """Test cases for AuditReport dataclass."""

    def test_audit_report_creation(self):
        """Verify AuditReport can be instantiated."""
        findings = [
            Finding(
                file_path="test.py",
                severity=Severity.HIGH,
                title="Issue 1",
                description="Test",
                remediation="Fix"
            )
        ]
        report = AuditReport(
            findings=findings,
            summary="1 issue found",
            files_scanned=1,
            total_issues=1,
            critical_issues=0,
            high_issues=1,
            medium_issues=0,
            low_issues=0
        )
        assert len(report.findings) == 1
        assert report.summary == "1 issue found"
        assert report.total_issues == 1

    def test_audit_report_empty_findings(self):
        """Verify AuditReport works with empty findings."""
        report = AuditReport(
            findings=[],
            summary="No issues found",
            files_scanned=5,
            total_issues=0,
            critical_issues=0,
            high_issues=0,
            medium_issues=0,
            low_issues=0
        )
        assert len(report.findings) == 0
        assert report.summary == "No issues found"
        assert report.total_issues == 0

    def test_audit_report_multiple_findings(self):
        """Verify AuditReport with multiple findings."""
        findings = [
            Finding(
                file_path="test1.py",
                severity=Severity.CRITICAL,
                title="Issue 1",
                description="Critical",
                remediation="Fix now"
            ),
            Finding(
                file_path="test2.py",
                severity=Severity.LOW,
                title="Issue 2",
                description="Minor",
                remediation="Fix later"
            ),
        ]
        report = AuditReport(
            findings=findings,
            summary="2 issues found",
            files_scanned=2,
            total_issues=2,
            critical_issues=1,
            high_issues=0,
            medium_issues=0,
            low_issues=1
        )
        assert len(report.findings) == 2
        assert report.findings[0].severity == Severity.CRITICAL
        assert report.findings[1].severity == Severity.LOW
        assert report.critical_issues == 1
        assert report.low_issues == 1
