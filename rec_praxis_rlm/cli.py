"""Command-line interface for rec-praxis-rlm pre-commit hooks and IDE integrations.

This module provides CLI entry points for:
- Pre-commit hooks (code review, security audit, dependency scan)
- IDE integrations (VS Code extension backend)
- CI/CD pipelines (GitHub Actions, GitLab CI)
"""

# IMPORTANT: Redirect stdout BEFORE any imports to suppress bitsandbytes errors
import sys
_original_stdout = sys.stdout
_original_stderr = sys.stderr
# Temporarily redirect stdout to stderr during all imports
sys.stdout = sys.stderr

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional
from io import StringIO

# Suppress warnings
warnings.filterwarnings("ignore")

from rec_praxis_rlm import __version__
from rec_praxis_rlm.constants import SEVERITY_ORDER_BY_NAME, SEVERITY_ICONS_BY_NAME

# Restore stdout after imports complete
sys.stdout = _original_stdout


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
    threshold = SEVERITY_ORDER_BY_NAME[severity]

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
            if SEVERITY_ORDER_BY_NAME[f.severity.name] >= threshold
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
    parser.add_argument("--use-graph", action="store_true",
                       help="Enable graph-aware analysis via Parseltongue (requires Parseltongue running)")
    parser.add_argument("--parseltongue-url", default="http://localhost:8080",
                       help="Parseltongue HTTP API URL (default: http://localhost:8080)")
    args = parser.parse_args()

    # Handle legacy --json flag
    if args.json:
        args.format = "json"

    # Lazy import to avoid loading heavy dependencies unless needed
    # Redirect stdout to stderr during import to suppress bitsandbytes errors
    _stdout_backup = sys.stdout
    sys.stdout = sys.stderr
    try:
        if args.use_graph:
            from rec_praxis_rlm.agents import GraphAwareCodeReviewAgent as CodeReviewAgent
        else:
            from rec_praxis_rlm.agents import CodeReviewAgent
        from rec_praxis_rlm.types import Severity
    except ImportError as e:
        from rec_praxis_rlm.errors import format_import_error
        print(format_import_error(e, "agents"), file=sys.stderr)
        return 1
    finally:
        sys.stdout = _stdout_backup

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

    if args.use_graph:
        agent = CodeReviewAgent(
            memory_path=str(memory_dir / "code_review_memory.jsonl"),
            parseltongue_url=args.parseltongue_url
        )
        print(f"🔗 Using graph-aware analysis (Parseltongue: {args.parseltongue_url})")
    else:
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
    threshold = SEVERITY_ORDER_BY_NAME[args.severity]
    blocking_findings = [
        f for f in all_findings
        if SEVERITY_ORDER_BY_NAME[f.severity.name] >= threshold
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
                icon = SEVERITY_ICONS_BY_NAME.get(f.severity.name, "")
                print(f"{icon} {f.severity.name}: {f.title}")
                print(f"   File: {f.file_path}:{f.line_number}")
                print(f"   Issue: {f.description}")
                print(f"   Fix: {f.remediation}\n")
        else:
            print("✅ No issues found")
            print("\n💡 Tip: Template-based detection found no issues.")
            print("   For deeper analysis (hardcoded secrets, SQL injection, etc.):")
            print("   Try: rec-praxis-review --use-llm <files>")
            print("   Or: rec-praxis-audit --use-llm <files>")

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
    parser.add_argument("--use-graph", action="store_true",
                       help="Enable graph-aware analysis via Parseltongue (requires Parseltongue running)")
    parser.add_argument("--parseltongue-url", default="http://localhost:8080",
                       help="Parseltongue HTTP API URL (default: http://localhost:8080)")
    args = parser.parse_args()

    # Handle legacy --json flag
    if args.json:
        args.format = "json"

    # Lazy import
    # Redirect stdout to stderr during import to suppress bitsandbytes errors
    _stdout_backup = sys.stdout
    sys.stdout = sys.stderr
    try:
        if args.use_graph:
            from rec_praxis_rlm.agents import GraphAwareCodeReviewAgent as SecurityAuditAgent
        else:
            from rec_praxis_rlm.agents import SecurityAuditAgent
        from rec_praxis_rlm.types import Severity
    except ImportError as e:
        from rec_praxis_rlm.errors import format_import_error
        print(format_import_error(e, "agents"), file=sys.stderr)
        return 1
    finally:
        sys.stdout = _stdout_backup

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

    if args.use_graph:
        agent = SecurityAuditAgent(
            memory_path=str(memory_dir / "security_audit_memory.jsonl"),
            parseltongue_url=args.parseltongue_url
        )
        print(f"🔗 Using graph-aware analysis (Parseltongue: {args.parseltongue_url})")
    else:
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
    threshold = SEVERITY_ORDER_BY_NAME[args.fail_on]
    blocking_findings = [
        f for f in report.findings
        if SEVERITY_ORDER_BY_NAME[f.severity.name] >= threshold
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
    # Redirect stdout to stderr during import to suppress bitsandbytes errors
    _stdout_backup = sys.stdout
    sys.stdout = sys.stderr
    try:
        from rec_praxis_rlm.agents import DependencyScanAgent
        from rec_praxis_rlm.types import Severity
    except ImportError as e:
        from rec_praxis_rlm.errors import format_import_error
        print(format_import_error(e, "agents"), file=sys.stderr)
        return 1
    finally:
        sys.stdout = _stdout_backup

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
    threshold = SEVERITY_ORDER_BY_NAME[args.fail_on]

    all_findings = cve_findings + secret_findings
    blocking_findings = [
        f for f in all_findings
        if SEVERITY_ORDER_BY_NAME[f.severity.name] >= threshold
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


def cli_pr_review() -> int:
    """GitHub PR integration: Post findings as inline review comments.

    Returns:
        0 on success, 1 on failure
    """
    parser = argparse.ArgumentParser(description="Post security findings as GitHub PR comments")
    parser.add_argument("files", nargs="+", help="Files to review")
    parser.add_argument("--pr-number", type=int, required=True,
                       help="Pull request number")
    parser.add_argument("--repo", required=True,
                       help="Repository in owner/repo format")
    parser.add_argument("--github-token",
                       help="GitHub token (defaults to GITHUB_TOKEN env var)")
    parser.add_argument("--severity", default="HIGH",
                       choices=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                       help="Minimum severity to comment on (default: HIGH)")
    parser.add_argument("--memory-dir", default=".rec-praxis-rlm",
                       help="Directory for procedural memory storage")
    parser.add_argument("--commit-sha",
                       help="Commit SHA to comment on (defaults to PR head)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be posted without actually posting")
    args = parser.parse_args()

    # Get GitHub token
    github_token = args.github_token or os.getenv("GITHUB_TOKEN")
    if not github_token and not args.dry_run:
        print("Error: GitHub token required (--github-token or GITHUB_TOKEN env var)", file=sys.stderr)
        return 1

    # Lazy import
    # Redirect stdout to stderr during import to suppress bitsandbytes errors
    _stdout_backup = sys.stdout
    sys.stdout = sys.stderr
    try:
        from rec_praxis_rlm.agents import CodeReviewAgent, SecurityAuditAgent
        from rec_praxis_rlm.types import Severity
    except ImportError as e:
        from rec_praxis_rlm.errors import format_import_error
        print(format_import_error(e, "agents"), file=sys.stderr)
        return 1
    finally:
        sys.stdout = _stdout_backup

    # Initialize agents
    memory_dir = Path(args.memory_dir)
    memory_dir.mkdir(exist_ok=True)
    code_agent = CodeReviewAgent(memory_path=str(memory_dir / "code_review_memory.jsonl"))
    security_agent = SecurityAuditAgent(memory_path=str(memory_dir / "security_audit_memory.jsonl"))

    # Read and review files
    files_content = {}
    for file_path in args.files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                files_content[file_path] = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)

    # Run code review
    code_findings = code_agent.review_code(files_content)

    # Run security audit
    security_report = security_agent.generate_audit_report(files_content)
    all_findings = code_findings + security_report.findings

    # Filter by severity
    threshold = SEVERITY_ORDER_BY_NAME[args.severity]
    filtered_findings = [
        f for f in all_findings
        if SEVERITY_ORDER_BY_NAME[f.severity.name] >= threshold
    ]

    if not filtered_findings:
        print(f"✅ No findings at {args.severity}+ severity")
        return 0

    # Group findings by severity
    severity_counts = {}
    for f in filtered_findings:
        severity_counts[f.severity.name] = severity_counts.get(f.severity.name, 0) + 1

    print(f"\n📊 Found {len(filtered_findings)} issue(s) at {args.severity}+ severity:")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if sev in severity_counts:
            print(f"  {sev}: {severity_counts[sev]}")

    if args.dry_run:
        print("\n🔍 Dry run - would post the following comments:\n")
        for idx, finding in enumerate(filtered_findings[:10], 1):  # Show first 10
            print(f"{idx}. [{finding.severity.name}] {finding.title}")
            print(f"   File: {finding.file_path}:{finding.line_number}")
            print(f"   {finding.description}")
            print(f"   Fix: {finding.remediation}\n")

        if len(filtered_findings) > 10:
            print(f"... and {len(filtered_findings) - 10} more")

        return 0

    # Post to GitHub
    try:
        import requests
    except ImportError:
        print("Error: requests library required for GitHub integration", file=sys.stderr)
        print("Install with: pip install rec-praxis-rlm[github]", file=sys.stderr)
        return 1

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    base_url = f"https://api.github.com/repos/{args.repo}"

    # Get PR details to find commit SHA
    if not args.commit_sha:
        pr_url = f"{base_url}/pulls/{args.pr_number}"
        pr_response = requests.get(pr_url, headers=headers)
        if pr_response.status_code != 200:
            print(f"Error fetching PR: {pr_response.status_code}", file=sys.stderr)
            return 1
        pr_data = pr_response.json()
        commit_sha = pr_data["head"]["sha"]
    else:
        commit_sha = args.commit_sha

    # Post summary comment
    summary_body = f"""## 🔒 rec-praxis-rlm Security Scan Results

**Found {len(filtered_findings)} issue(s) at {args.severity}+ severity**

### Severity Breakdown
"""

    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if sev in severity_counts:
            emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}[sev]
            summary_body += f"- {emoji} **{sev}**: {severity_counts[sev]}\n"

    summary_body += f"\n### Top Issues\n\n"

    for idx, finding in enumerate(filtered_findings[:5], 1):
        emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "ℹ️"}[finding.severity.name]
        summary_body += f"{idx}. {emoji} **{finding.title}** ({finding.severity.name})\n"
        summary_body += f"   - File: `{finding.file_path}:{finding.line_number}`\n"
        summary_body += f"   - {finding.description}\n\n"

    if len(filtered_findings) > 5:
        summary_body += f"\n*...and {len(filtered_findings) - 5} more issue(s)*\n"

    summary_body += f"\n---\n*Powered by [rec-praxis-rlm](https://github.com/jmanhype/rec-praxis-rlm)*"

    # Post summary comment
    comments_url = f"{base_url}/issues/{args.pr_number}/comments"
    comment_response = requests.post(
        comments_url,
        headers=headers,
        json={"body": summary_body}
    )

    if comment_response.status_code not in (200, 201):
        print(f"Warning: Failed to post summary comment: {comment_response.status_code}", file=sys.stderr)
    else:
        print(f"✅ Posted summary comment on PR #{args.pr_number}")

    # Post inline review comments (up to 20 to avoid spam)
    review_comments = []
    for finding in filtered_findings[:20]:
        emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "ℹ️"}[finding.severity.name]
        comment_body = f"{emoji} **{finding.severity.name}: {finding.title}**\n\n"
        comment_body += f"{finding.description}\n\n"
        comment_body += f"**Remediation:**\n{finding.remediation}"

        review_comments.append({
            "path": finding.file_path,
            "line": finding.line_number or 1,
            "body": comment_body
        })

    if review_comments:
        review_url = f"{base_url}/pulls/{args.pr_number}/reviews"
        review_response = requests.post(
            review_url,
            headers=headers,
            json={
                "commit_id": commit_sha,
                "body": f"Found {len(filtered_findings)} security issue(s)",
                "event": "COMMENT",
                "comments": review_comments
            }
        )

        if review_response.status_code not in (200, 201):
            print(f"Warning: Failed to post review comments: {review_response.status_code}", file=sys.stderr)
            print(f"Response: {review_response.text}", file=sys.stderr)
        else:
            print(f"✅ Posted {len(review_comments)} inline review comment(s)")

    return 0


def cli_generate_tests() -> int:
    """Generate pytest tests for uncovered code paths.

    Returns:
        0 on success, 1 on failure
    """
    parser = argparse.ArgumentParser(
        description="Generate pytest tests to increase code coverage"
    )
    parser.add_argument("source_files", nargs="*",
                       help="Source files to generate tests for (optional, scans all if not specified)")
    parser.add_argument("--coverage-file", default=".coverage",
                       help="Path to coverage.py data file (default: .coverage)")
    parser.add_argument("--target-coverage", type=float, default=90.0,
                       help="Target coverage percentage (0-100, default: 90)")
    parser.add_argument("--max-tests", type=int, default=10,
                       help="Maximum number of tests to generate (default: 10)")
    parser.add_argument("--test-dir", default="tests",
                       help="Directory for generated tests (default: tests)")
    parser.add_argument("--memory-dir", default=".rec-praxis-rlm",
                       help="Directory for procedural memory storage")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what tests would be generated without writing files")
    parser.add_argument("--validate", action="store_true",
                       help="Run pytest to validate generated tests")
    parser.add_argument("--format", default="human",
                       choices=["human", "json"],
                       help="Output format (default: human)")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use DSPy LLM for intelligent test generation with assertions (v0.6.0+)")
    parser.add_argument("--lm-model", default="groq/llama-3.3-70b-versatile",
                       help="Language model for DSPy (default: groq/llama-3.3-70b-versatile)")
    parser.add_argument("--analyze-branches", action="store_true", default=True,
                       help="Analyze branch coverage (if/else, try/except) in addition to line coverage (v0.7.0+, default: True)")
    parser.add_argument("--no-analyze-branches", dest="analyze_branches", action="store_false",
                       help="Disable branch coverage analysis (v0.7.0+)")
    parser.add_argument("--use-hypothesis", action="store_true",
                       help="Generate Hypothesis property-based tests with @given decorator (v0.8.0+)")
    parser.add_argument("--language", type=str, default="python",
                       choices=["python", "javascript", "typescript", "go", "rust"],
                       help="Target programming language for test generation (v0.9.0+, default: python)")
    args = parser.parse_args()

    # Lazy import
    # Redirect stdout to stderr during import to suppress bitsandbytes errors
    _stdout_backup = sys.stdout
    sys.stdout = sys.stderr
    try:
        from rec_praxis_rlm.agents import TestGenerationAgent
        from rec_praxis_rlm.agents.test_generation import Language
    except ImportError as e:
        from rec_praxis_rlm.errors import format_import_error
        print(format_import_error(e, "agents"), file=sys.stderr)
        return 1
    finally:
        sys.stdout = _stdout_backup

    # Check for coverage.py
    if not Path(args.coverage_file).exists():
        print(f"Error: Coverage file not found: {args.coverage_file}", file=sys.stderr)
        print("Run pytest with coverage first:", file=sys.stderr)
        print(f"  pytest --cov=your_package --cov-report=term tests/", file=sys.stderr)
        return 1

    # Initialize agent
    memory_dir = Path(args.memory_dir)
    memory_dir.mkdir(exist_ok=True)

    # Prepare agent initialization parameters
    agent_params = {
        "memory_path": str(memory_dir / "test_generation_memory.jsonl"),
        "coverage_data_file": args.coverage_file,
        "test_dir": args.test_dir,
        "analyze_branches": args.analyze_branches,  # v0.7.0: Branch coverage analysis
        "use_hypothesis": args.use_hypothesis,  # v0.8.0: Hypothesis property-based testing
        "target_language": Language[args.language.upper()],  # v0.9.0: Multi-language support
    }

    # Add DSPy parameters if requested
    if args.use_llm:
        agent_params["use_dspy"] = True
        agent_params["lm_model"] = args.lm_model
        print(f"🤖 Using DSPy with model: {args.lm_model}")
        print(f"   Tests will include assertions (not TODO stubs)\n")

    # v0.7.0: Display branch coverage mode
    if args.analyze_branches:
        print(f"📊 Branch coverage analysis: ENABLED (v0.7.0+)")
        print(f"   Analyzing if/else, try/except, match/case branches\n")

    # v0.8.0: Display Hypothesis mode
    if args.use_hypothesis:
        print(f"🔬 Hypothesis property-based testing: ENABLED (v0.8.0+)")
        print(f"   Generating tests with @given decorator and strategies\n")

    # v0.9.0: Display multi-language mode
    if args.language != "python":
        print(f"🌍 Multi-language support: ENABLED (v0.9.0+)")
        print(f"   Target language: {args.language.upper()}")
        print(f"   Generating tests in {args.language} syntax\n")

    try:
        agent = TestGenerationAgent(**agent_params)
    except (ImportError, ValueError, RuntimeError) as e:
        print(f"Error initializing agent: {e}", file=sys.stderr)
        if "dspy" in str(e).lower():
            print("Install DSPy with: pip install dspy-ai", file=sys.stderr)
        if "API_KEY" in str(e):
            print(f"Set the required environment variable for {args.lm_model}", file=sys.stderr)
        return 1

    print(f"🧪 Test Generation Agent v{__version__}")
    print(f"{'='*60}\n")

    # Generate tests
    try:
        generated_tests = agent.generate_tests_for_coverage_gap(
            target_coverage=args.target_coverage,
            max_tests=args.max_tests,
            source_files=args.source_files if args.source_files else None
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during test generation: {e}", file=sys.stderr)
        return 1

    if not generated_tests:
        print("✅ No tests generated - coverage target met or no uncovered regions found")
        return 0

    print(f"\n{'='*60}")
    print(f"📝 Generated {len(generated_tests)} test(s)")
    print(f"{'='*60}\n")

    # Output results
    if args.format == "json":
        output = {
            "tests_generated": len(generated_tests),
            "tests": [t.to_dict() for t in generated_tests]
        }
        print(json.dumps(output, indent=2))
        return 0

    # Human-readable format
    for idx, test in enumerate(generated_tests, 1):
        print(f"{idx}. {test.description}")
        print(f"   Target: {test.target_function} in {test.target_file}")
        print(f"   Test file: {test.test_file_path}")
        print(f"   Estimated coverage gain: {test.estimated_coverage_gain:.1f} lines\n")

        if not args.dry_run:
            # Write test to file
            test_file = Path(test.test_file_path)
            test_file.parent.mkdir(parents=True, exist_ok=True)

            if test_file.exists():
                with open(test_file, 'a') as f:
                    f.write('\n\n' + test.test_code)
                print(f"   ✅ Appended to {test.test_file_path}")
            else:
                with open(test_file, 'w') as f:
                    f.write(test.test_code)
                print(f"   ✅ Created {test.test_file_path}")

            # Validate if requested
            if args.validate:
                success, message = agent.validate_test(test)
                if success:
                    print(f"   ✅ {message}")
                else:
                    print(f"   ❌ {message}")
        else:
            print(f"   [DRY RUN] Would write to {test.test_file_path}")

    if args.dry_run:
        print(f"\n💡 Dry run complete - no files were written")
        print(f"   Run without --dry-run to write tests to disk")

    return 0


def main() -> int:
    """Main CLI entry point - dispatches to sub-commands."""
    # If called with no args, show help
    if len(sys.argv) == 1:
        parser = argparse.ArgumentParser(
            description=f"rec-praxis-rlm CLI v{__version__}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Available commands:
  rec-praxis-review         - Code review pre-commit hook
  rec-praxis-audit          - Security audit pre-commit hook
  rec-praxis-deps           - Dependency & secret scanning hook
  rec-praxis-pr-review      - Post findings as GitHub PR comments
  rec-praxis-generate-tests - Generate pytest tests for uncovered code
            """
        )
        parser.add_argument("--version", action="version", version=f"rec-praxis-rlm {__version__}")
        parser.print_help()
        return 0

    # Check for --version flag
    if len(sys.argv) == 2 and sys.argv[1] == "--version":
        print(f"rec-praxis-rlm {__version__}")
        return 0

    # Dispatch to subcommand based on first argument
    command = sys.argv[1]

    # Remove the command from sys.argv so subcommand parsers work correctly
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    command_map = {
        "rec-praxis-review": cli_code_review,
        "rec-praxis-audit": cli_security_audit,
        "rec-praxis-deps": cli_dependency_scan,
        "rec-praxis-pr-review": cli_pr_review,
        "rec-praxis-generate-tests": cli_generate_tests,
    }

    if command in command_map:
        return command_map[command]()
    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        print(f"Run 'python -m rec_praxis_rlm.cli' for available commands", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
