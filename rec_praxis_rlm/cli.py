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


def calculate_quality_score(findings: List, total_lines: int = 1000) -> float:
    """Calculate quality score based on findings (0-100 scale).

    Args:
        findings: List of Finding objects
        total_lines: Total lines of code scanned (for normalization)

    Returns:
        Quality score from 0-100 (100 = perfect, 0 = critical issues)
    """
    if not findings:
        return 100.0

    # Severity weights (how many points each severity deducts)
    severity_weights = {
        "CRITICAL": 10.0,
        "HIGH": 5.0,
        "MEDIUM": 2.0,
        "LOW": 0.5,
        "INFO": 0.1
    }

    # Calculate total penalty
    total_penalty = 0.0
    for finding in findings:
        weight = severity_weights.get(finding.severity.name, 1.0)
        total_penalty += weight

    # Normalize by code size (more lines → more tolerant)
    normalized_penalty = (total_penalty / (total_lines / 100))

    # Convert to 0-100 score
    score = max(0.0, 100.0 - normalized_penalty)

    return score


def run_iterative_improvement(
    agent,
    files: List[str],
    severity: str,
    format: str,
    output: Optional[str],
    max_iterations: int,
    target_score: int,
    auto_fix: bool,
    mlflow_experiment: Optional[str],
    memory_dir: Path,
    scan_start: float
) -> int:
    """Run iterative improvement mode with autonomous quality optimization.

    Args:
        agent: CodeReviewAgent instance
        files: List of file paths to review
        severity: Minimum severity threshold
        format: Output format
        output: Output file path
        max_iterations: Maximum iterations to run
        target_score: Target quality score (0-100)
        auto_fix: Whether to suggest fixes
        mlflow_experiment: MLflow experiment name (optional)
        memory_dir: Memory directory path
        scan_start: Scan start timestamp

    Returns:
        0 if target reached, 1 otherwise
    """
    print(f"\n🔄 Iterative Improvement Mode")
    print(f"Target: {target_score}% quality score")
    print(f"Max iterations: {max_iterations}\n")

    # Track progress across iterations
    iteration_history = []
    current_score = 0.0
    severity_order = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    threshold = severity_order[severity]

    # Read files once
    files_content = {}
    total_lines = 0
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                files_content[file_path] = content
                total_lines += len(content.split('\n'))
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{max_iterations}")
        print(f"{'='*60}")

        # Run review
        all_findings = agent.review_code(files_content)

        # Calculate quality score
        current_score = calculate_quality_score(all_findings, total_lines)

        # Filter by severity threshold
        blocking_findings = [
            f for f in all_findings
            if severity_order[f.severity.name] >= threshold
        ]

        # Display results
        print(f"\n📊 Results:")
        print(f"  Quality Score: {current_score:.1f}%")
        print(f"  Total Findings: {len(all_findings)}")
        print(f"  Blocking Findings: {len(blocking_findings)}")

        # Group by severity
        severity_counts = {}
        for f in all_findings:
            severity_counts[f.severity.name] = severity_counts.get(f.severity.name, 0) + 1

        if severity_counts:
            print(f"  Severity Breakdown:")
            for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
                if sev in severity_counts:
                    print(f"    {sev}: {severity_counts[sev]}")

        # Store iteration history
        iteration_history.append({
            "iteration": iteration,
            "score": current_score,
            "total_findings": len(all_findings),
            "blocking_findings": len(blocking_findings),
            "severity_counts": severity_counts
        })

        # Check if target reached
        if current_score >= target_score:
            print(f"\n✅ Target score reached! ({current_score:.1f}% >= {target_score}%)")
            print(f"   Completed in {iteration} iteration(s)")
            break

        # Show improvement suggestions if auto-fix enabled
        if auto_fix and all_findings:
            print(f"\n💡 Suggested Fixes for Next Iteration:")

            # Prioritize CRITICAL and HIGH findings
            priority_findings = [f for f in all_findings if f.severity.name in ("CRITICAL", "HIGH")][:5]

            for idx, finding in enumerate(priority_findings, 1):
                print(f"\n{idx}. {finding.title} ({finding.severity.name})")
                print(f"   File: {finding.file_path}:{finding.line_number}")
                print(f"   Fix: {finding.remediation}")

        # If not last iteration, explain what happens next
        if iteration < max_iterations and current_score < target_score:
            print(f"\n🔄 Continuing to iteration {iteration + 1}...")
            print(f"   Current: {current_score:.1f}% | Target: {target_score}% | Gap: {target_score - current_score:.1f}%")

    # Final summary
    print(f"\n{'='*60}")
    print(f"📈 Improvement Summary")
    print(f"{'='*60}")

    if iteration_history:
        initial_score = iteration_history[0]["score"]
        final_score = iteration_history[-1]["score"]
        improvement = final_score - initial_score

        print(f"Initial Score: {initial_score:.1f}%")
        print(f"Final Score: {final_score:.1f}%")
        print(f"Improvement: {'+' if improvement >= 0 else ''}{improvement:.1f}%")
        print(f"Iterations: {len(iteration_history)}")

        # Show progression
        if len(iteration_history) > 1:
            print(f"\nProgression:")
            for entry in iteration_history:
                bar_length = int(entry["score"] / 2)  # Scale to 50 chars max
                bar = "█" * bar_length
                print(f"  Iter {entry['iteration']}: {bar} {entry['score']:.1f}%")

    # Log to MLflow if enabled
    if mlflow_experiment:
        try:
            from rec_praxis_rlm.telemetry import log_security_scan_metrics
            import mlflow

            scan_duration = time.time() - scan_start

            with mlflow.start_run(run_name=f"iterative_review_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"):
                log_security_scan_metrics(
                    findings=all_findings,
                    scan_type="iterative_code_review",
                    files_scanned=len(files),
                    scan_duration_seconds=scan_duration
                )
                # Log iteration metrics
                mlflow.log_metric("iterations", len(iteration_history))
                mlflow.log_metric("final_score", current_score)
                mlflow.log_metric("target_score", target_score)
                mlflow.log_metric("improvement", improvement if iteration_history else 0)
        except Exception as e:
            print(f"Warning: Failed to log metrics to MLflow: {e}", file=sys.stderr)

    # Output final results in requested format
    if format == "json":
        output_data = {
            "mode": "iterative",
            "iterations": len(iteration_history),
            "final_score": current_score,
            "target_score": target_score,
            "target_reached": current_score >= target_score,
            "total_findings": len(all_findings),
            "blocking_findings": len(blocking_findings),
            "iteration_history": iteration_history,
            "findings": [f.to_dict() for f in all_findings]
        }
        print(f"\n{json.dumps(output_data, indent=2)}")
    elif format == "html":
        from rec_praxis_rlm.reporters import generate_html_report
        output_path = output or "iterative-code-review-report.html"
        report_path = generate_html_report(all_findings, output_path)
        print(f"\n✅ HTML report generated: {report_path}")

    # Return success if target reached
    return 0 if current_score >= target_score else 1


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
    parser.add_argument("--mode", default="standard",
                       choices=["standard", "iterative"],
                       help="Execution mode: standard (single pass) or iterative (autonomous improvement)")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Maximum iterations for iterative mode (default: 5)")
    parser.add_argument("--target-score", type=int, default=95,
                       help="Target quality score for iterative mode (0-100, default: 95)")
    parser.add_argument("--auto-fix", action="store_true",
                       help="Automatically suggest fixes in iterative mode")
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

    # Route to iterative mode if requested
    if args.mode == "iterative":
        return run_iterative_improvement(
            agent, args.files, args.severity, args.format, args.output,
            args.max_iterations, args.target_score, args.auto_fix,
            args.mlflow_experiment, memory_dir, scan_start
        )

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
