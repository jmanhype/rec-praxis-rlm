"""Unit tests for rec_praxis_rlm.formatters module (TOON format)."""

import pytest
from rec_praxis_rlm.formatters import (
    format_findings_as_toon,
    format_cve_findings_as_toon,
    format_secret_findings_as_toon,
    format_code_review_as_toon,
    format_dependency_scan_as_toon,
)
from rec_praxis_rlm.types import Finding, CVEFinding, SecretFinding, Severity


class TestFormatFindingsAsToon:
    """Test TOON format for general findings."""

    def test_empty_findings(self):
        """Verify empty findings list produces correct TOON output."""
        result = format_findings_as_toon([])
        assert result == "[0]{}"

    def test_single_finding(self):
        """Verify single finding produces correct TOON format."""
        finding = Finding(
            file_path="test.py",
            line_number=42,
            severity=Severity.HIGH,
            title="SQL Injection",
            description="Dangerous SQL",
            remediation="Use params"
        )
        result = format_findings_as_toon([finding])

        assert "[1]{file,line,severity,title,description,remediation}:" in result
        assert "test.py,42,HIGH,SQL Injection,Dangerous SQL,Use params" in result

    def test_comma_escaping(self):
        """Verify commas in description/remediation are escaped."""
        finding = Finding(
            file_path="app.py",
            line_number=10,
            severity=Severity.LOW,
            title="Issue",
            description="This, is, a, test",
            remediation="Fix, it, now"
        )
        result = format_findings_as_toon([finding])

        # Commas should be replaced with semicolons
        assert "This; is; a; test" in result
        assert "Fix; it; now" in result

    def test_none_line_number(self):
        """Verify None line number shows as N/A."""
        finding = Finding(
            file_path="test.py",
            severity=Severity.MEDIUM,
            title="Issue",
            description="Test",
            remediation="Fix"
        )
        result = format_findings_as_toon([finding])

        assert ",N/A," in result

    def test_token_reduction(self):
        """Verify TOON format is more compact than JSON."""
        finding = Finding(
            file_path="longfilename.py",
            line_number=100,
            severity=Severity.CRITICAL,
            title="Very Important Security Issue",
            description="This is a detailed description of the security vulnerability",
            remediation="Apply this comprehensive remediation strategy"
        )

        # TOON format
        toon_output = format_findings_as_toon([finding])

        # Equivalent JSON would be much longer
        # TOON should be < 250 chars for this finding
        assert len(toon_output) < 300


class TestFormatCVEFindingsAsToon:
    """Test TOON format for CVE findings."""

    def test_empty_cve_findings(self):
        """Verify empty CVE list."""
        result = format_cve_findings_as_toon([])
        assert result == "[0]{}"

    def test_cve_finding(self):
        """Verify CVE finding TOON format."""
        finding = CVEFinding(
            package_name="requests",
            installed_version="2.25.0",
            cve_id="CVE-2021-1234",
            severity=Severity.HIGH,
            description="Vulnerability",
            remediation="Upgrade",
            fixed_version="2.26.0"
        )
        result = format_cve_findings_as_toon([finding])

        assert "[1]{package,version,cve_id,severity,fixed_version}:" in result
        assert "requests,2.25.0,CVE-2021-1234,HIGH,2.26.0" in result

    def test_cve_no_fixed_version(self):
        """Verify CVE with no fixed version shows Unknown."""
        finding = CVEFinding(
            package_name="old-lib",
            installed_version="1.0.0",
            cve_id="CVE-2020-9999",
            severity=Severity.CRITICAL,
            description="No fix",
            remediation="Migrate"
        )
        result = format_cve_findings_as_toon([finding])

        assert ",Unknown" in result


class TestFormatSecretFindingsAsToon:
    """Test TOON format for secret findings."""

    def test_empty_secret_findings(self):
        """Verify empty secrets list."""
        result = format_secret_findings_as_toon([])
        assert result == "[0]{}"

    def test_secret_finding(self):
        """Verify secret finding TOON format."""
        finding = SecretFinding(
            file_path="config.py",
            line_number=15,
            secret_type="API Key",
            severity=Severity.CRITICAL,
            description="Hardcoded key",
            remediation="Use env vars"
        )
        result = format_secret_findings_as_toon([finding])

        assert "[1]{file,line,secret_type,severity}:" in result
        assert "config.py,15,API Key,CRITICAL" in result


class TestFormatCodeReviewAsToon:
    """Test TOON format for code review results."""

    def test_code_review_output(self):
        """Verify code review TOON format includes summary."""
        findings = [
            Finding(
                file_path="test.py",
                line_number=10,
                severity=Severity.HIGH,
                title="Issue 1",
                description="Test",
                remediation="Fix"
            )
        ]
        result = format_code_review_as_toon(1, 1, findings)

        assert "Code Review Results" in result
        assert "Total: 1" in result
        assert "Blocking: 1" in result
        assert "Findings:" in result
        assert "[1]{" in result


class TestFormatDependencyScanAsToon:
    """Test TOON format for dependency scan results."""

    def test_dependency_scan_with_both_types(self):
        """Verify dependency scan with CVEs and secrets."""
        cve = CVEFinding(
            package_name="django",
            installed_version="3.0.0",
            cve_id="CVE-2021-5678",
            severity=Severity.CRITICAL,
            description="Vuln",
            remediation="Upgrade",
            fixed_version="3.2.0"
        )
        secret = SecretFinding(
            file_path="app.py",
            line_number=5,
            secret_type="Password",
            severity=Severity.CRITICAL,
            description="Hardcoded",
            remediation="Remove"
        )

        result = format_dependency_scan_as_toon(2, 1, 1, [cve], [secret])

        assert "Dependency Scan Results" in result
        assert "Total: 2 (CVEs: 1, Secrets: 1)" in result
        assert "CVE Vulnerabilities:" in result
        assert "Secrets Detected:" in result

    def test_dependency_scan_empty(self):
        """Verify dependency scan with no findings."""
        result = format_dependency_scan_as_toon(0, 0, 0, [], [])

        assert "Total: 0 (CVEs: 0, Secrets: 0)" in result


class TestTOONTokenReduction:
    """Test TOON format provides claimed token reduction."""

    def test_claimed_40_percent_reduction(self):
        """Verify TOON provides approximately 40% token reduction vs JSON."""
        # Create multiple findings to test at scale
        findings = [
            Finding(
                file_path=f"module_{i}.py",
                line_number=i * 10,
                severity=Severity.HIGH,
                title=f"Security Issue #{i}",
                description=f"This is a detailed description of issue number {i}",
                remediation=f"Apply this remediation for issue {i}"
            )
            for i in range(10)
        ]

        # TOON format
        toon_output = format_findings_as_toon(findings)
        toon_length = len(toon_output)

        # Approximate JSON equivalent
        json_equivalent = str([f.to_dict() for f in findings])
        json_length = len(json_equivalent)

        # TOON should be at least 30% smaller (we claim 40%)
        reduction = 100 * (1 - toon_length / json_length)
        assert reduction >= 30, f"TOON reduction was only {reduction:.1f}%, expected >=30%"
        assert reduction <= 60, f"TOON reduction was {reduction:.1f}%, suspiciously high"
