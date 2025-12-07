"""Production-ready Security Audit Agent for CLI and IDE integration.

OWASP-based security auditing with procedural memory for continuous improvement.
"""

from typing import Dict, List

from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig, RLMContext
from rec_praxis_rlm.types import Finding, Severity, OWASPCategory, AuditReport


class SecurityAuditAgent:
    """Production security audit agent implementing CLI contract."""

    def __init__(self, memory_path: str = ":memory:"):
        """Initialize agent with persistent memory.

        Args:
            memory_path: Path to JSONL file for procedural memory storage
        """
        self.memory_path = memory_path
        self.memory = ProceduralMemory(
            config=MemoryConfig(
                storage_path=memory_path,
                env_weight=0.7,
                goal_weight=0.3,
            )
        )
        self.rlm = RLMContext()

    def generate_audit_report(self, files: Dict[str, str]) -> AuditReport:
        """Generate comprehensive OWASP-based security audit report.

        Args:
            files: Dictionary mapping file paths to file contents

        Returns:
            AuditReport with findings and summary statistics
        """
        findings = []

        for file_path, content in files.items():
            self.rlm.add_document(file_path, content)
            file_findings = self._audit_file(file_path, content)
            findings.extend(file_findings)

        # Calculate statistics
        severity_counts = {s: 0 for s in Severity}
        for f in findings:
            severity_counts[f.severity] += 1

        # Generate summary
        summary = self._generate_summary(findings, len(files))

        return AuditReport(
            findings=findings,
            summary=summary,
            files_scanned=len(files),
            total_issues=len(findings),
            critical_issues=severity_counts[Severity.CRITICAL],
            high_issues=severity_counts[Severity.HIGH],
            medium_issues=severity_counts[Severity.MEDIUM],
            low_issues=severity_counts[Severity.LOW],
        )

    def format_report(self, report: AuditReport) -> str:
        """Format audit report for human-readable output.

        Args:
            report: AuditReport to format

        Returns:
            Formatted string representation
        """
        lines = [
            "\n" + "=" * 70,
            "🔒 SECURITY AUDIT REPORT",
            "=" * 70,
            "",
            f"Files Scanned: {report.files_scanned}",
            f"Total Issues: {report.total_issues}",
            "",
            "Severity Breakdown:",
            f"  🔴 CRITICAL: {report.critical_issues}",
            f"  🟠 HIGH:     {report.high_issues}",
            f"  🟡 MEDIUM:   {report.medium_issues}",
            f"  🟢 LOW:      {report.low_issues}",
            "",
            "=" * 70,
            ""
        ]

        if report.findings:
            lines.append("FINDINGS:\n")
            for i, finding in enumerate(report.findings, 1):
                icon = {
                    Severity.CRITICAL: "🔴",
                    Severity.HIGH: "🟠",
                    Severity.MEDIUM: "🟡",
                    Severity.LOW: "🟢",
                    Severity.INFO: "ℹ️"
                }
                lines.append(f"{i}. {icon[finding.severity]} {finding.severity.name}: {finding.title}")
                lines.append(f"   File: {finding.file_path}:{finding.line_number or 'N/A'}")
                if finding.owasp_category:
                    lines.append(f"   OWASP: {finding.owasp_category.value}")
                if finding.cwe_id:
                    lines.append(f"   CWE: {finding.cwe_id}")
                lines.append(f"   Issue: {finding.description}")
                lines.append(f"   Fix: {finding.remediation}")
                lines.append("")
        else:
            lines.append("✅ No security issues found!\n")
            lines.append("💡 Tip: Template-based detection found no issues.")
            lines.append("   For deeper analysis (hardcoded secrets, SQL injection, etc.):")
            lines.append("   Try: rec-praxis-audit --use-llm <files>\n")

        lines.append("=" * 70)
        lines.append(f"\nSummary: {report.summary}\n")

        return "\n".join(lines)

    def _audit_file(self, file_path: str, content: str) -> List[Finding]:
        """Perform OWASP-based security audit on a single file."""
        findings = []

        # A03:2021 - Injection
        findings.extend(self._check_injection(file_path))

        # A02:2021 - Cryptographic Failures
        findings.extend(self._check_crypto_failures(file_path))

        # A01:2021 - Broken Access Control
        findings.extend(self._check_access_control(file_path))

        # A07:2021 - Identification and Authentication Failures
        findings.extend(self._check_auth_failures(file_path))

        # A05:2021 - Security Misconfiguration
        findings.extend(self._check_misconfig(file_path))

        # A09:2021 - Security Logging Failures
        findings.extend(self._check_logging_failures(file_path))

        return findings

    def _check_injection(self, file_path: str) -> List[Finding]:
        """Check for injection vulnerabilities (OWASP A03)."""
        findings = []

        # SQL Injection
        sql_patterns = [
            r"execute\s*\(\s*f['\"]",
            r"execute\s*\([^)]*\.format\(",
            r"cursor\.execute\([^)]*\+",
        ]
        for pattern in sql_patterns:
            matches = self.rlm.grep(pattern, doc_id=file_path)
            for match in matches:
                findings.append(Finding(
                    file_path=file_path,
                    line_number=match.line_number,
                    severity=Severity.CRITICAL,
                    title="SQL Injection Vulnerability",
                    description="User input concatenated into SQL query",
                    remediation="Use parameterized queries with placeholders",
                    owasp_category=OWASPCategory.A03_INJECTION,
                    cwe_id="CWE-89"
                ))

        # Command Injection
        cmd_injection = self.rlm.grep(r"(os\.system|subprocess\..*shell\s*=\s*True)", doc_id=file_path)
        for match in cmd_injection:
            findings.append(Finding(
                file_path=file_path,
                line_number=match.line_number,
                severity=Severity.HIGH,
                title="Command Injection Risk",
                description="Shell command execution with user input",
                remediation="Use subprocess with shell=False and command list",
                owasp_category=OWASPCategory.A03_INJECTION,
                cwe_id="CWE-78"
            ))

        return findings

    def _check_crypto_failures(self, file_path: str) -> List[Finding]:
        """Check for cryptographic failures (OWASP A02)."""
        findings = []

        # Weak hashing algorithms
        weak_hash = self.rlm.grep(r"hashlib\.(md5|sha1)\(", doc_id=file_path)
        for match in weak_hash:
            findings.append(Finding(
                file_path=file_path,
                line_number=match.line_number,
                severity=Severity.HIGH,
                title="Weak Cryptographic Algorithm",
                description="MD5/SHA1 are cryptographically broken",
                remediation="Use SHA-256+ for data integrity, bcrypt/argon2 for passwords",
                owasp_category=OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
                cwe_id="CWE-327"
            ))

        # Hardcoded secrets
        secrets = self.rlm.grep(r"(password|api_key|secret|token)\s*=\s*['\"][^'\"]{8,}['\"]", doc_id=file_path)
        for match in secrets:
            findings.append(Finding(
                file_path=file_path,
                line_number=match.line_number,
                severity=Severity.CRITICAL,
                title="Hardcoded Secret",
                description="Sensitive credentials in source code",
                remediation="Use environment variables or secure credential management",
                owasp_category=OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
                cwe_id="CWE-798"
            ))

        return findings

    def _check_access_control(self, file_path: str) -> List[Finding]:
        """Check for broken access control (OWASP A01)."""
        findings = []

        # Missing authorization checks
        route_without_auth = self.rlm.grep(r"@app\.route.*\n(?!.*@login_required)(?!.*@auth)", doc_id=file_path)
        if route_without_auth:
            findings.append(Finding(
                file_path=file_path,
                line_number=route_without_auth[0].line_number if route_without_auth else None,
                severity=Severity.MEDIUM,
                title="Potentially Missing Authorization",
                description="Flask route without visible authorization decorator",
                remediation="Ensure routes check user permissions with @login_required or similar",
                owasp_category=OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
                cwe_id="CWE-862"
            ))

        return findings

    def _check_auth_failures(self, file_path: str) -> List[Finding]:
        """Check for authentication failures (OWASP A07)."""
        findings = []

        # Weak password validation
        weak_pass = self.rlm.grep(r"len\(password\)\s*[<>=]+\s*\d+", doc_id=file_path)
        for match in weak_pass:
            findings.append(Finding(
                file_path=file_path,
                line_number=match.line_number,
                severity=Severity.MEDIUM,
                title="Weak Password Validation",
                description="Password validation only checks length",
                remediation="Use comprehensive validation (length, complexity, common passwords)",
                owasp_category=OWASPCategory.A07_IDENTIFICATION_FAILURES,
                cwe_id="CWE-521"
            ))

        return findings

    def _check_misconfig(self, file_path: str) -> List[Finding]:
        """Check for security misconfiguration (OWASP A05)."""
        findings = []

        # Debug mode in production
        debug_enabled = self.rlm.grep(r"debug\s*=\s*True", doc_id=file_path)
        for match in debug_enabled:
            findings.append(Finding(
                file_path=file_path,
                line_number=match.line_number,
                severity=Severity.MEDIUM,
                title="Debug Mode Enabled",
                description="Debug mode should be disabled in production",
                remediation="Set debug=False or use environment-based configuration",
                owasp_category=OWASPCategory.A05_SECURITY_MISCONFIGURATION,
                cwe_id="CWE-489"
            ))

        return findings

    def _check_logging_failures(self, file_path: str) -> List[Finding]:
        """Check for logging and monitoring failures (OWASP A09)."""
        findings = []

        # Bare except blocks hiding errors
        bare_except = self.rlm.grep(r"except\s*:\s*\n\s*pass", doc_id=file_path)
        for match in bare_except:
            findings.append(Finding(
                file_path=file_path,
                line_number=match.line_number,
                severity=Severity.MEDIUM,
                title="Silent Exception Handling",
                description="Exception caught and ignored without logging",
                remediation="Log exceptions: logger.exception('Error message')",
                owasp_category=OWASPCategory.A09_LOGGING_FAILURES,
                cwe_id="CWE-778"
            ))

        return findings

    def _generate_summary(self, findings: List[Finding], files_scanned: int) -> str:
        """Generate human-readable summary of audit results."""
        if not findings:
            return f"Scanned {files_scanned} file(s). No security issues detected."

        critical = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        high = sum(1 for f in findings if f.severity == Severity.HIGH)

        if critical > 0:
            return f"CRITICAL: Found {critical} critical and {high} high severity issues across {files_scanned} file(s). Immediate action required."
        elif high > 0:
            return f"WARNING: Found {high} high severity issues across {files_scanned} file(s). Review and remediate soon."
        else:
            return f"Found {len(findings)} medium/low severity issues across {files_scanned} file(s). Review when possible."
