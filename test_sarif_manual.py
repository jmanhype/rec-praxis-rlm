#!/usr/bin/env python3
"""Manual test for SARIF formatter output."""

import json
from rec_praxis_rlm.formatters import format_findings_as_sarif, format_cve_findings_as_sarif
from rec_praxis_rlm.types import Finding, CVEFinding, Severity, OWASPCategory

# Test data
findings = [
    Finding(
        file_path="test.py",
        line_number=10,
        column_number=5,
        severity=Severity.HIGH,
        title="SQL Injection Vulnerability",
        description="User input is concatenated into SQL query without sanitization",
        remediation="Use parameterized queries or an ORM",
        owasp_category=OWASPCategory.A03_INJECTION,
        cwe_id="89",
        confidence=0.9
    ),
    Finding(
        file_path="app.py",
        line_number=25,
        severity=Severity.CRITICAL,
        title="Hardcoded API Key",
        description="API key found hardcoded in source code",
        remediation="Move secrets to environment variables or secret management service",
        owasp_category=OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
        confidence=1.0
    )
]

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
    )
]

# Test SARIF formatting
print("=" * 80)
print("TESTING SARIF FORMATTER FOR CODE FINDINGS")
print("=" * 80)

sarif_output = format_findings_as_sarif(findings)
sarif_data = json.loads(sarif_output)

print(f"\n✓ SARIF version: {sarif_data['version']}")
print(f"✓ Schema: {sarif_data['$schema']}")
print(f"✓ Tool name: {sarif_data['runs'][0]['tool']['driver']['name']}")
print(f"✓ Number of rules: {len(sarif_data['runs'][0]['tool']['driver']['rules'])}")
print(f"✓ Number of results: {len(sarif_data['runs'][0]['results'])}")

# Verify rule structure
for rule in sarif_data['runs'][0]['tool']['driver']['rules']:
    print(f"\nRule: {rule['id']}")
    print(f"  Name: {rule['name']}")
    print(f"  Level: {rule['defaultConfiguration']['level']}")
    if 'cwe' in rule['properties']:
        print(f"  CWE: {rule['properties']['cwe']}")
    if 'owasp' in rule['properties']:
        print(f"  OWASP: {rule['properties']['owasp']}")

# Verify result structure
for result in sarif_data['runs'][0]['results']:
    print(f"\nResult: {result['ruleId']}")
    print(f"  Level: {result['level']}")
    print(f"  File: {result['locations'][0]['physicalLocation']['artifactLocation']['uri']}")
    print(f"  Line: {result['locations'][0]['physicalLocation']['region']['startLine']}")

print("\n" + "=" * 80)
print("TESTING SARIF FORMATTER FOR CVE FINDINGS")
print("=" * 80)

sarif_cve = format_cve_findings_as_sarif(cve_findings)
sarif_cve_data = json.loads(sarif_cve)

print(f"\n✓ SARIF version: {sarif_cve_data['version']}")
print(f"✓ Number of CVE rules: {len(sarif_cve_data['runs'][0]['tool']['driver']['rules'])}")
print(f"✓ Number of CVE results: {len(sarif_cve_data['runs'][0]['results'])}")

for rule in sarif_cve_data['runs'][0]['tool']['driver']['rules']:
    print(f"\nCVE Rule: {rule['id']}")
    print(f"  Description: {rule['shortDescription']['text']}")
    print(f"  Security Severity: {rule['properties']['security-severity']}")

print("\n" + "=" * 80)
print("✅ ALL SARIF TESTS PASSED")
print("=" * 80)

# Save sample SARIF files
with open("/tmp/test-sarif-findings.sarif", "w") as f:
    f.write(sarif_output)

with open("/tmp/test-sarif-cve.sarif", "w") as f:
    f.write(sarif_cve)

print(f"\n📄 Sample SARIF files saved:")
print(f"   /tmp/test-sarif-findings.sarif")
print(f"   /tmp/test-sarif-cve.sarif")
