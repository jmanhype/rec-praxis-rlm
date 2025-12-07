"""Output formatters for CLI tools - JSON, TOON, and SARIF formats."""

import json
from typing import Any, Dict, List
from datetime import datetime, timezone

from rec_praxis_rlm import __version__
from rec_praxis_rlm.types import Finding, CVEFinding, SecretFinding, Severity


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


def _severity_to_sarif_level(severity: Severity) -> str:
    """Convert Severity enum to SARIF result level.

    SARIF levels: error, warning, note, none
    """
    mapping = {
        Severity.CRITICAL: "error",
        Severity.HIGH: "error",
        Severity.MEDIUM: "warning",
        Severity.LOW: "warning",
        Severity.INFO: "note",
    }
    return mapping.get(severity, "warning")


def format_findings_as_sarif(
    findings: List[Finding],
    tool_name: str = "rec-praxis-rlm",
    tool_version: str = __version__
) -> str:
    """Format findings as SARIF (Static Analysis Results Interchange Format).

    SARIF is the standard format for GitHub Code Scanning and Security tab integration.
    Spec: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html

    Args:
        findings: List of Finding objects
        tool_name: Name of the analysis tool
        tool_version: Version of the analysis tool

    Returns:
        SARIF-formatted JSON string
    """
    rules = {}
    results = []

    for finding in findings:
        # Create unique rule ID from title
        rule_id = finding.title.lower().replace(" ", "-").replace("/", "-")
        if finding.cwe_id:
            rule_id = f"CWE-{finding.cwe_id}"

        # Add rule definition (deduplicated)
        if rule_id not in rules:
            rule_def = {
                "id": rule_id,
                "name": finding.title,
                "shortDescription": {
                    "text": finding.title
                },
                "fullDescription": {
                    "text": finding.description
                },
                "help": {
                    "text": finding.remediation,
                    "markdown": f"## Remediation\n\n{finding.remediation}"
                },
                "defaultConfiguration": {
                    "level": _severity_to_sarif_level(finding.severity)
                },
                "properties": {
                    "tags": ["security"],
                    "precision": "high" if (finding.confidence or 0.8) > 0.7 else "medium"
                }
            }

            if finding.cwe_id:
                rule_def["properties"]["security-severity"] = str(finding.severity.value * 2.5)  # 0-10 scale
                rule_def["properties"]["cwe"] = f"CWE-{finding.cwe_id}"

            if finding.owasp_category:
                rule_def["properties"]["owasp"] = finding.owasp_category.value

            rules[rule_id] = rule_def

        # Add result
        result = {
            "ruleId": rule_id,
            "level": _severity_to_sarif_level(finding.severity),
            "message": {
                "text": finding.description
            },
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": finding.file_path,
                        "uriBaseId": "%SRCROOT%"
                    },
                    "region": {
                        "startLine": finding.line_number or 1,
                        "startColumn": finding.column_number or 1
                    }
                }
            }]
        }

        if finding.confidence:
            result["properties"] = {"confidence": finding.confidence}

        results.append(result)

    # Build SARIF document
    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": tool_name,
                    "version": tool_version,
                    "informationUri": "https://github.com/jmanhype/rec-praxis-rlm",
                    "rules": list(rules.values())
                }
            },
            "results": results,
            "columnKind": "utf16CodeUnits",
            "properties": {
                "analysisTimestamp": datetime.now(timezone.utc).isoformat()
            }
        }]
    }

    return json.dumps(sarif, indent=2)


def format_cve_findings_as_sarif(
    findings: List[CVEFinding],
    tool_name: str = "rec-praxis-rlm-deps",
    tool_version: str = __version__
) -> str:
    """Format CVE findings as SARIF for GitHub Dependabot integration.

    Args:
        findings: List of CVEFinding objects
        tool_name: Name of the analysis tool
        tool_version: Version of the analysis tool

    Returns:
        SARIF-formatted JSON string
    """
    rules = {}
    results = []

    for finding in findings:
        rule_id = finding.cve_id

        # Add rule definition
        if rule_id not in rules:
            rules[rule_id] = {
                "id": rule_id,
                "name": f"Vulnerable dependency: {finding.package_name}",
                "shortDescription": {
                    "text": f"{rule_id} in {finding.package_name} {finding.installed_version}"
                },
                "fullDescription": {
                    "text": finding.description
                },
                "help": {
                    "text": finding.remediation,
                    "markdown": f"## {rule_id}\n\n{finding.description}\n\n### Remediation\n\n{finding.remediation}"
                },
                "defaultConfiguration": {
                    "level": _severity_to_sarif_level(finding.severity)
                },
                "properties": {
                    "tags": ["security", "dependency", "cve"],
                    "precision": "high",
                    "security-severity": str(finding.cvss_score or (finding.severity.value * 2.5))
                }
            }

        # CVE findings don't have file locations, so use package manifest
        result = {
            "ruleId": rule_id,
            "level": _severity_to_sarif_level(finding.severity),
            "message": {
                "text": f"{rule_id} found in {finding.package_name} {finding.installed_version}. Upgrade to {finding.fixed_version or 'latest version'}."
            },
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": "requirements.txt",  # Default, can be customized
                        "uriBaseId": "%SRCROOT%"
                    },
                    "region": {
                        "startLine": 1
                    }
                }
            }]
        }

        results.append(result)

    # Build SARIF document
    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": tool_name,
                    "version": tool_version,
                    "informationUri": "https://github.com/jmanhype/rec-praxis-rlm",
                    "rules": list(rules.values())
                }
            },
            "results": results,
            "columnKind": "utf16CodeUnits"
        }]
    }

    return json.dumps(sarif, indent=2)
