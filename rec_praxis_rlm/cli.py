"""Command-line interface for rec-praxis-rlm pre-commit hooks and IDE integrations.

This module provides CLI entry points for:
- Pre-commit hooks (code review, security audit, dependency scan)
- IDE integrations (VS Code extension backend)
- CI/CD pipelines (GitHub Actions, GitLab CI)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
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
                       choices=["human", "json", "toon", "sarif", "html"],
                       help="Output format (default: human, toon=40%% token reduction, sarif=GitHub Security, html=interactive report)")
    parser.add_argument("--output", type=str,
                       help="Output file path for HTML reports (default: code-review-report.html)")
    parser.add_argument("--memory-dir", default=".rec-praxis-rlm",
                       help="Directory for procedural memory storage")
    parser.add_argument("--mlflow-experiment", type=str,
                       help="MLflow experiment name for metrics tracking (optional)")
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

    # Setup MLflow tracking if requested
    if args.mlflow_experiment:
        try:
            from rec_praxis_rlm.telemetry import setup_mlflow_tracing
            setup_mlflow_tracing(experiment_name=args.mlflow_experiment)
        except ImportError:
            print("Warning: MLflow not installed, metrics tracking disabled", file=sys.stderr)

    # Initialize agent with persistent memory
    memory_dir = Path(args.memory_dir)
    memory_dir.mkdir(exist_ok=True)
    agent = CodeReviewAgent(memory_path=str(memory_dir / "code_review_memory.jsonl"))

    # Track scan start time for metrics
    scan_start = time.time()

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
    elif args.format == "html":
        from rec_praxis_rlm.reporters import generate_html_report
        output_path = args.output or "code-review-report.html"
        report_path = generate_html_report(all_findings, output_path)
        print(f"✅ HTML report generated: {report_path}")
        return 1 if blocking_findings else 0
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
            print("No issues found")

    # Log metrics to MLflow if enabled
    if args.mlflow_experiment:
        try:
            from rec_praxis_rlm.telemetry import log_security_scan_metrics
            import mlflow

            scan_duration = time.time() - scan_start

            with mlflow.start_run(run_name=f"code_review_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"):
                log_security_scan_metrics(
                    findings=all_findings,
                    scan_type="code_review",
                    files_scanned=len(args.files),
                    scan_duration_seconds=scan_duration
                )
        except Exception as e:
            print(f"Warning: Failed to log metrics to MLflow: {e}", file=sys.stderr)

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
                       choices=["human", "json", "toon", "sarif", "html"],
                       help="Output format (default: human, toon=40%% token reduction, sarif=GitHub Security, html=interactive report)")
    parser.add_argument("--output", type=str,
                       help="Output file path for HTML reports (default: security-audit-report.html)")
    parser.add_argument("--memory-dir", default=".rec-praxis-rlm",
                       help="Directory for procedural memory storage")
    parser.add_argument("--mlflow-experiment", type=str,
                       help="MLflow experiment name for metrics tracking (optional)")
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

    # Setup MLflow tracking if requested
    if args.mlflow_experiment:
        try:
            from rec_praxis_rlm.telemetry import setup_mlflow_tracing
            setup_mlflow_tracing(experiment_name=args.mlflow_experiment)
        except ImportError:
            print("Warning: MLflow not installed, metrics tracking disabled", file=sys.stderr)

    # Initialize agent
    memory_dir = Path(args.memory_dir)
    memory_dir.mkdir(exist_ok=True)
    agent = SecurityAuditAgent(memory_path=str(memory_dir / "security_audit_memory.jsonl"))

    # Track scan start time for metrics
    scan_start = time.time()

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
    elif args.format == "html":
        from rec_praxis_rlm.reporters import generate_html_report
        output_path = args.output or "security-audit-report.html"
        report_path = generate_html_report(report.findings, output_path)
        print(f"✅ HTML report generated: {report_path}")
        return 1 if blocking_findings else 0
    else:  # human format
        print(agent.format_report(report))

    # Log metrics to MLflow if enabled
    if args.mlflow_experiment:
        try:
            from rec_praxis_rlm.telemetry import log_security_scan_metrics
            import mlflow

            scan_duration = time.time() - scan_start

            with mlflow.start_run(run_name=f"security_audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"):
                log_security_scan_metrics(
                    findings=report.findings,
                    scan_type="security_audit",
                    files_scanned=len(args.files),
                    scan_duration_seconds=scan_duration
                )
        except Exception as e:
            print(f"Warning: Failed to log metrics to MLflow: {e}", file=sys.stderr)

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
                       choices=["human", "json", "toon", "sarif", "html"],
                       help="Output format (default: human, toon=40%% token reduction, sarif=GitHub Security, html=interactive report)")
    parser.add_argument("--output", type=str,
                       help="Output file path for HTML reports (default: dependency-scan-report.html)")
    parser.add_argument("--memory-dir", default=".rec-praxis-rlm",
                       help="Directory for procedural memory storage")
    parser.add_argument("--mlflow-experiment", type=str,
                       help="MLflow experiment name for metrics tracking (optional)")
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

    # Setup MLflow tracking if requested
    if args.mlflow_experiment:
        try:
            from rec_praxis_rlm.telemetry import setup_mlflow_tracing
            setup_mlflow_tracing(experiment_name=args.mlflow_experiment)
        except ImportError:
            print("Warning: MLflow not installed, metrics tracking disabled", file=sys.stderr)

    # Initialize agent
    memory_dir = Path(args.memory_dir)
    memory_dir.mkdir(exist_ok=True)
    agent = DependencyScanAgent(memory_path=str(memory_dir / "dependency_scan_memory.jsonl"))

    # Track scan start time for metrics
    scan_start = time.time()

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
    elif args.format == "html":
        from rec_praxis_rlm.reporters import generate_html_report
        # For dependency scans, convert CVE/Secret findings to regular findings for HTML report
        # We'll create a pseudo-finding list that combines both
        combined_findings = []
        for cve in cve_findings:
            # Convert CVEFinding to Finding-like dict for HTML template
            combined_findings.append(cve)
        for secret in secret_findings:
            combined_findings.append(secret)
        output_path = args.output or "dependency-scan-report.html"
        report_path = generate_html_report([], output_path, cve_findings=cve_findings, secret_findings=secret_findings)
        print(f"✅ HTML report generated: {report_path}")
        return 1 if blocking_findings else 0
    else:  # human format
        print(report)

    # Log metrics to MLflow if enabled
    if args.mlflow_experiment:
        try:
            from rec_praxis_rlm.telemetry import log_security_scan_metrics
            import mlflow

            scan_duration = time.time() - scan_start

            with mlflow.start_run(run_name=f"dependency_scan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"):
                log_security_scan_metrics(
                    findings=all_findings,
                    scan_type="dependency_scan",
                    files_scanned=num_dependencies + len(files_content),
                    scan_duration_seconds=scan_duration
                )
        except Exception as e:
            print(f"Warning: Failed to log metrics to MLflow: {e}", file=sys.stderr)

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
