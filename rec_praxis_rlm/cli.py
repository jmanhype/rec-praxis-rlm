"""Command-line interface for rec-praxis-rlm pre-commit hooks and IDE integrations.

This module provides CLI entry points for:
- Pre-commit hooks (code review, security audit, dependency scan)
- IDE integrations (VS Code extension backend)
- CI/CD pipelines (GitHub Actions, GitLab CI)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from rec_praxis_rlm import __version__


def cli_code_review() -> int:
    """Pre-commit hook: Run code review on staged files.

    Returns:
        0 if no HIGH/CRITICAL issues found, 1 otherwise
    """
    parser = argparse.ArgumentParser(description="Run code review on staged files")
    parser.add_argument("files", nargs="+", help="Files to review")
    parser.add_argument("--severity", default="HIGH",
                       choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                       help="Minimum severity to fail on (default: HIGH)")
    parser.add_argument("--json", action="store_true",
                       help="Output JSON for IDE integration")
    parser.add_argument("--format", default="human",
                       choices=["human", "json", "toon", "sarif"],
                       help="Output format (default: human, toon=40%% token reduction, sarif=GitHub Security)")
    parser.add_argument("--memory-dir", default=".rec-praxis-rlm",
                       help="Directory for procedural memory storage")
    args = parser.parse_args()

    # Handle legacy --json flag
    if args.json:
        args.format = "json"

    # Lazy import to avoid loading heavy dependencies unless needed
    try:
        from rec_praxis_rlm.agents import CodeReviewAgent
        from rec_praxis_rlm.types import Severity
    except ImportError as e:
        from rec_praxis_rlm.errors import format_import_error
        print(format_import_error(e, "agents"), file=sys.stderr)
        return 1

    # Initialize agent with persistent memory
    memory_dir = Path(args.memory_dir)
    memory_dir.mkdir(exist_ok=True)
    agent = CodeReviewAgent(memory_path=str(memory_dir / "code_review_memory.jsonl"))

    # Read and review files
    all_findings = []
    for file_path in args.files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
            continue

        findings = agent.review_code({file_path: content})
        all_findings.extend(findings)

    # Filter by severity threshold
    severity_order = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    threshold = severity_order[args.severity]
    blocking_findings = [
        f for f in all_findings
        if severity_order[f.severity.name] >= threshold
    ]

    # Output results
    if args.format == "json":
        output = {
            "total_findings": len(all_findings),
            "blocking_findings": len(blocking_findings),
            "findings": [f.to_dict() for f in all_findings]
        }
        print(json.dumps(output, indent=2))
    elif args.format == "toon":
        from rec_praxis_rlm.formatters import format_code_review_as_toon
        print(format_code_review_as_toon(len(all_findings), len(blocking_findings), all_findings))
    elif args.format == "sarif":
        from rec_praxis_rlm.formatters import format_findings_as_sarif
        print(format_findings_as_sarif(all_findings, tool_name="rec-praxis-review"))
    else:  # human format
        if all_findings:
            print(f"\n🔍 Code Review Results: {len(all_findings)} issue(s) found\n")
            for f in all_findings:
                icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "ℹ️"}
                print(f"{icon[f.severity.name]} {f.severity.name}: {f.title}")
                print(f"   File: {f.file_path}:{f.line_number}")
                print(f"   Issue: {f.description}")
                print(f"   Fix: {f.remediation}\n")
        else:
            print("✅ No issues found")

    return 1 if blocking_findings else 0


def cli_security_audit() -> int:
    """Pre-commit hook: Run security audit on staged files.

    Returns:
        0 if no CRITICAL issues found, 1 otherwise
    """
    parser = argparse.ArgumentParser(description="Run security audit on staged files")
    parser.add_argument("files", nargs="+", help="Files to audit")
    parser.add_argument("--fail-on", default="CRITICAL",
                       choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                       help="Fail on this severity or higher (default: CRITICAL)")
    parser.add_argument("--json", action="store_true",
                       help="Output JSON for IDE integration")
    parser.add_argument("--format", default="human",
                       choices=["human", "json", "toon", "sarif"],
                       help="Output format (default: human, toon=40%% token reduction, sarif=GitHub Security)")
    parser.add_argument("--memory-dir", default=".rec-praxis-rlm",
                       help="Directory for procedural memory storage")
    args = parser.parse_args()

    # Handle legacy --json flag
    if args.json:
        args.format = "json"

    # Lazy import
    try:
        from rec_praxis_rlm.agents import SecurityAuditAgent
        from rec_praxis_rlm.types import Severity
    except ImportError as e:
        from rec_praxis_rlm.errors import format_import_error
        print(format_import_error(e, "agents"), file=sys.stderr)
        return 1

    # Initialize agent
    memory_dir = Path(args.memory_dir)
    memory_dir.mkdir(exist_ok=True)
    agent = SecurityAuditAgent(memory_path=str(memory_dir / "security_audit_memory.jsonl"))

    # Read and audit files
    files_content = {}
    for file_path in args.files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                files_content[file_path] = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)

    report = agent.generate_audit_report(files_content)

    # Filter by fail-on threshold
    severity_order = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    threshold = severity_order[args.fail_on]
    blocking_findings = [
        f for f in report.findings
        if severity_order[f.severity.name] >= threshold
    ]

    # Output results
    if args.format == "json":
        output = {
            "total_findings": len(report.findings),
            "blocking_findings": len(blocking_findings),
            "summary": report.summary,
            "findings": [f.to_dict() for f in report.findings]
        }
        print(json.dumps(output, indent=2))
    elif args.format == "toon":
        from rec_praxis_rlm.formatters import format_findings_as_toon
        print(f"Security Audit Results")
        print(f"Total: {len(report.findings)}")
        print(f"Blocking: {len(blocking_findings)}")
        print(f"\nSummary: {report.summary}\n")
        print("Findings:")
        print(format_findings_as_toon(report.findings))
    elif args.format == "sarif":
        from rec_praxis_rlm.formatters import format_findings_as_sarif
        print(format_findings_as_sarif(report.findings, tool_name="rec-praxis-audit"))
    else:  # human format
        print(agent.format_report(report))

    return 1 if blocking_findings else 0


def cli_dependency_scan() -> int:
    """Pre-commit hook: Scan dependencies and secrets.

    Returns:
        0 if no CRITICAL issues found, 1 otherwise
    """
    parser = argparse.ArgumentParser(description="Scan dependencies and secrets")
    parser.add_argument("--requirements", default="requirements.txt",
                       help="Path to requirements file")
    parser.add_argument("--files", nargs="*", default=[],
                       help="Files to scan for secrets")
    parser.add_argument("--fail-on", default="CRITICAL",
                       choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                       help="Fail on this severity or higher")
    parser.add_argument("--json", action="store_true",
                       help="Output JSON for IDE integration")
    parser.add_argument("--format", default="human",
                       choices=["human", "json", "toon", "sarif"],
                       help="Output format (default: human, toon=40%% token reduction, sarif=GitHub Security)")
    parser.add_argument("--memory-dir", default=".rec-praxis-rlm",
                       help="Directory for procedural memory storage")
    args = parser.parse_args()

    # Handle legacy --json flag
    if args.json:
        args.format = "json"

    # Lazy import
    try:
        from rec_praxis_rlm.agents import DependencyScanAgent
        from rec_praxis_rlm.types import Severity
    except ImportError as e:
        from rec_praxis_rlm.errors import format_import_error
        print(format_import_error(e, "agents"), file=sys.stderr)
        return 1

    # Initialize agent
    memory_dir = Path(args.memory_dir)
    memory_dir.mkdir(exist_ok=True)
    agent = DependencyScanAgent(memory_path=str(memory_dir / "dependency_scan_memory.jsonl"))

    # Scan dependencies
    cve_findings = []
    num_dependencies = 0
    if Path(args.requirements).exists():
        with open(args.requirements, "r", encoding="utf-8") as f:
            requirements_content = f.read()
        cve_findings, dependencies = agent.scan_dependencies(requirements_content)
        num_dependencies = len(dependencies)

    # Scan secrets
    secret_findings = []
    files_content = {}
    for file_path in args.files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                files_content[file_path] = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)

    if files_content:
        secret_findings = agent.scan_secrets(files_content)

    # Generate report
    report = agent.generate_report(
        cve_findings, secret_findings, num_dependencies, len(files_content)
    )

    # Filter by fail-on threshold
    severity_order = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    threshold = severity_order[args.fail_on]

    all_findings = cve_findings + secret_findings
    blocking_findings = [
        f for f in all_findings
        if severity_order[f.severity.name] >= threshold
    ]

    # Output results
    if args.format == "json":
        output = {
            "total_findings": len(all_findings),
            "blocking_findings": len(blocking_findings),
            "cve_count": len(cve_findings),
            "secret_count": len(secret_findings),
            "dependencies_scanned": num_dependencies,
            "files_scanned": len(files_content),
            "findings": [f.to_dict() for f in all_findings]
        }
        print(json.dumps(output, indent=2))
    elif args.format == "toon":
        from rec_praxis_rlm.formatters import format_dependency_scan_as_toon
        print(format_dependency_scan_as_toon(
            len(all_findings), len(cve_findings), len(secret_findings),
            cve_findings, secret_findings
        ))
    elif args.format == "sarif":
        from rec_praxis_rlm.formatters import format_cve_findings_as_sarif
        # For dependency scans, we only output CVE findings in SARIF format
        # Secret findings require file locations which we have but CVE is more important for GitHub Security
        print(format_cve_findings_as_sarif(cve_findings, tool_name="rec-praxis-deps"))
    else:  # human format
        print(report)

    return 1 if blocking_findings else 0


def main() -> int:
    """Main CLI entry point - dispatches to sub-commands."""
    parser = argparse.ArgumentParser(
        description=f"rec-praxis-rlm CLI v{__version__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  rec-praxis-review       - Code review pre-commit hook
  rec-praxis-audit        - Security audit pre-commit hook
  rec-praxis-deps         - Dependency & secret scanning hook
        """
    )
    parser.add_argument("--version", action="version", version=f"rec-praxis-rlm {__version__}")

    # If called with no args, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    parser.parse_args()
    return 0


if __name__ == "__main__":
    sys.exit(main())
