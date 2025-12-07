#!/usr/bin/env python3
"""Manual test for HTML report generation."""

from rec_praxis_rlm.reporters import generate_html_report
from rec_praxis_rlm.types import Finding, CVEFinding, SecretFinding, Severity, OWASPCategory

# Test data - code review findings
code_findings = [
    Finding(
        file_path="src/auth.py",
        line_number=42,
        column_number=10,
        severity=Severity.CRITICAL,
        title="SQL Injection Vulnerability",
        description="User input concatenated directly into SQL query without sanitization",
        remediation="Use parameterized queries or an ORM like SQLAlchemy",
        owasp_category=OWASPCategory.A03_INJECTION,
        cwe_id="89",
        confidence=0.95
    ),
    Finding(
        file_path="src/config.py",
        line_number=15,
        severity=Severity.HIGH,
        title="Hardcoded API Key",
        description="API key found hardcoded in source code",
        remediation="Move secrets to environment variables or use a secret management service",
        owasp_category=OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
        confidence=1.0
    ),
    Finding(
        file_path="src/utils.py",
        line_number=88,
        severity=Severity.MEDIUM,
        title="Weak Cryptographic Algorithm",
        description="Using MD5 for password hashing - cryptographically broken",
        remediation="Use bcrypt, scrypt, or Argon2 for password hashing",
        owasp_category=OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
        cwe_id="327"
    ),
    Finding(
        file_path="src/api.py",
        line_number=120,
        severity=Severity.LOW,
        title="Missing Input Validation",
        description="No validation on user-provided email address",
        remediation="Add email format validation using regex or email-validator library",
        owasp_category=OWASPCategory.A03_INJECTION
    ),
    Finding(
        file_path="src/logger.py",
        line_number=55,
        severity=Severity.INFO,
        title="Sensitive Data in Logs",
        description="Logging user passwords in plaintext",
        remediation="Remove sensitive data from logs or redact before logging",
        owasp_category=OWASPCategory.A09_LOGGING_FAILURES
    ),
]

# Test data - CVE findings
cve_findings = [
    CVEFinding(
        package_name="requests",
        installed_version="2.25.0",
        severity=Severity.HIGH,
        cve_id="CVE-2023-32681",
        description="Proxy-Authorization header leak in HTTP redirects",
        remediation="Upgrade to requests>=2.31.0",
        fixed_version="2.31.0",
        cvss_score=7.5
    ),
    CVEFinding(
        package_name="pillow",
        installed_version="9.0.0",
        severity=Severity.CRITICAL,
        cve_id="CVE-2023-44271",
        description="Arbitrary code execution via crafted image file",
        remediation="Upgrade to pillow>=10.0.1",
        fixed_version="10.0.1",
        cvss_score=9.8
    ),
]

# Test data - secret findings
secret_findings = [
    SecretFinding(
        file_path="config/settings.py",
        line_number=23,
        secret_type="AWS Access Key",
        severity=Severity.CRITICAL,
        description="Hardcoded AWS access key detected",
        remediation="Move AWS credentials to environment variables or use IAM roles"
    ),
    SecretFinding(
        file_path="tests/test_auth.py",
        line_number=67,
        secret_type="GitHub Token",
        severity=Severity.HIGH,
        description="GitHub personal access token found in test file",
        remediation="Use GitHub Actions secrets or environment variables"
    ),
]

print("=" * 80)
print("TESTING HTML REPORT GENERATION")
print("=" * 80)

# Test 1: Code review report
print("\n1. Generating code review HTML report...")
code_report_path = generate_html_report(
    code_findings,
    output_path="/tmp/code-review-report.html"
)
print(f"✅ Code review report: {code_report_path}")

# Test 2: Security audit report (same as code review)
print("\n2. Generating security audit HTML report...")
audit_report_path = generate_html_report(
    code_findings,
    output_path="/tmp/security-audit-report.html"
)
print(f"✅ Security audit report: {audit_report_path}")

# Test 3: Dependency scan report with CVEs
print("\n3. Generating dependency scan HTML report...")
deps_report_path = generate_html_report(
    [],  # No code findings
    output_path="/tmp/dependency-scan-report.html",
    cve_findings=cve_findings,
    secret_findings=secret_findings
)
print(f"✅ Dependency scan report: {deps_report_path}")

# Test 4: Combined report with everything
print("\n4. Generating combined HTML report...")
combined_report_path = generate_html_report(
    code_findings,
    output_path="/tmp/combined-security-report.html",
    cve_findings=cve_findings,
    secret_findings=secret_findings
)
print(f"✅ Combined report: {combined_report_path}")

print("\n" + "=" * 80)
print("✅ ALL HTML REPORT TESTS PASSED")
print("=" * 80)
print(f"\n📄 Sample HTML reports saved:")
print(f"   {code_report_path}")
print(f"   {audit_report_path}")
print(f"   {deps_report_path}")
print(f"   {combined_report_path}")
print(f"\n💡 Open these files in a browser to view the interactive reports!")
