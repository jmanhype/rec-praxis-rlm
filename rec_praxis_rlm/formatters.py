"""Output formatters for CLI tools - JSON and TOON formats."""

import json
from typing import Any, Dict, List

from rec_praxis_rlm.types import Finding, CVEFinding, SecretFinding


def format_findings_as_toon(findings: List[Finding]) -> str:
    """Format findings list in TOON format for compact output.

    TOON format example:
    [3]{file,line,severity,title}:
      test.py,10,HIGH,SQL Injection
      app.py,25,CRITICAL,Hardcoded Secret
      utils.py,50,MEDIUM,Weak Crypto

    Provides ~40% token reduction compared to JSON.

    Args:
        findings: List of Finding objects

    Returns:
        TOON-formatted string
    """
    if not findings:
        return "[0]{}"

    # Build header with field names
    fields = ["file", "line", "severity", "title", "description", "remediation"]
    header = f"[{len(findings)}]" + "{" + ",".join(fields) + "}:"

    # Build data rows
    rows = []
    for f in findings:
        row_data = [
            f.file_path,
            str(f.line_number or "N/A"),
            f.severity.name,
            f.title,
            f.description.replace(",", ";"),  # Escape commas
            f.remediation.replace(",", ";")
        ]
        rows.append("  " + ",".join(row_data))

    return header + "\n" + "\n".join(rows)


def format_cve_findings_as_toon(findings: List[CVEFinding]) -> str:
    """Format CVE findings in TOON format.

    Args:
        findings: List of CVEFinding objects

    Returns:
        TOON-formatted string
    """
    if not findings:
        return "[0]{}"

    fields = ["package", "version", "cve_id", "severity", "fixed_version"]
    header = f"[{len(findings)}]" + "{" + ",".join(fields) + "}:"

    rows = []
    for f in findings:
        row_data = [
            f.package_name,
            f.installed_version,
            f.cve_id,
            f.severity.name,
            f.fixed_version or "Unknown"
        ]
        rows.append("  " + ",".join(row_data))

    return header + "\n" + "\n".join(rows)


def format_secret_findings_as_toon(findings: List[SecretFinding]) -> str:
    """Format secret findings in TOON format.

    Args:
        findings: List of SecretFinding objects

    Returns:
        TOON-formatted string
    """
    if not findings:
        return "[0]{}"

    fields = ["file", "line", "secret_type", "severity"]
    header = f"[{len(findings)}]" + "{" + ",".join(fields) + "}:"

    rows = []
    for f in findings:
        row_data = [
            f.file_path,
            str(f.line_number),
            f.secret_type,
            f.severity.name
        ]
        rows.append("  " + ",".join(row_data))

    return header + "\n" + "\n".join(rows)


def format_code_review_as_toon(total_findings: int, blocking_findings: int, findings: List[Finding]) -> str:
    """Format code review results in TOON format.

    Args:
        total_findings: Total number of findings
        blocking_findings: Number of blocking findings
        findings: List of Finding objects

    Returns:
        TOON-formatted string
    """
    output = [
        f"Code Review Results",
        f"Total: {total_findings}",
        f"Blocking: {blocking_findings}",
        "",
        "Findings:",
        format_findings_as_toon(findings)
    ]
    return "\n".join(output)


def format_dependency_scan_as_toon(
    total_findings: int,
    cve_count: int,
    secret_count: int,
    cve_findings: List[CVEFinding],
    secret_findings: List[SecretFinding]
) -> str:
    """Format dependency scan results in TOON format.

    Args:
        total_findings: Total findings count
        cve_count: CVE count
        secret_count: Secret count
        cve_findings: List of CVE findings
        secret_findings: List of secret findings

    Returns:
        TOON-formatted string
    """
    output = [
        f"Dependency Scan Results",
        f"Total: {total_findings} (CVEs: {cve_count}, Secrets: {secret_count})",
        ""
    ]

    if cve_findings:
        output.append("CVE Vulnerabilities:")
        output.append(format_cve_findings_as_toon(cve_findings))
        output.append("")

    if secret_findings:
        output.append("Secrets Detected:")
        output.append(format_secret_findings_as_toon(secret_findings))

    return "\n".join(output)
