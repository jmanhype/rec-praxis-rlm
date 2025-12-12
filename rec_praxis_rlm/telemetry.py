"""Telemetry and observability support with MLflow integration."""

from typing import Any, Callable, Optional, List, Dict
from collections import defaultdict, Counter
from datetime import datetime, timezone

try:
    import mlflow
    import mlflow.dspy

    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore[assignment]

# Global hook registry: event_name -> list of callbacks
_hooks: dict[str, list[Callable[[str, dict[str, Any]], None]]] = defaultdict(list)


def setup_mlflow_tracing(
    experiment_name: Optional[str] = None,
    log_traces_from_compile: bool = False,
) -> None:
    """Enable MLflow automatic tracing for DSPy operations.

    Args:
        experiment_name: Optional experiment name for MLflow
        log_traces_from_compile: If True, log traces from optimizer compilation
    """
    if mlflow is None:
        raise ImportError("mlflow not installed. Install with: pip install mlflow")

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    # Enable DSPy automatic logging
    mlflow.dspy.autolog(log_traces=log_traces_from_compile)


def add_telemetry_hook(
    event_name: str,
    callback: Callable[[str, dict[str, Any]], None],
) -> None:
    """Register a custom telemetry hook for an event.

    Args:
        event_name: Name of the event to hook (e.g., 'memory.recall')
        callback: Function to call when event is emitted,
                 signature: callback(event_name: str, data: dict) -> None
    """
    _hooks[event_name].append(callback)


def emit_event(event_name: str, data: dict[str, Any]) -> None:
    """Emit a telemetry event, triggering all registered hooks.

    Args:
        event_name: Name of the event
        data: Event data dictionary
    """
    for callback in _hooks[event_name]:
        try:
            callback(event_name, data)
        except Exception:
            # Don't let telemetry failures crash the application
            pass


def log_security_scan_metrics(
    findings: List[Any],
    scan_type: str = "code_review",
    files_scanned: int = 0,
    scan_duration_seconds: float = 0.0,
    llm_tokens_used: int = 0,
    llm_cost_usd: float = 0.0,
) -> None:
    """Log security scan metrics to MLflow for trend analysis.

    Args:
        findings: List of Finding objects from scan
        scan_type: Type of scan (code_review, security_audit, dependency_scan)
        files_scanned: Number of files analyzed
        scan_duration_seconds: Total scan duration
        llm_tokens_used: Total LLM tokens consumed
        llm_cost_usd: Estimated cost in USD
    """
    if not MLFLOW_AVAILABLE:
        return

    # Count findings by severity
    severity_counts = Counter(f.severity.name for f in findings)

    # Count findings by OWASP category
    owasp_counts: Dict[str, int] = {}
    for f in findings:
        if hasattr(f, "owasp_category") and f.owasp_category:
            category = f.owasp_category.value
            owasp_counts[category] = owasp_counts.get(category, 0) + 1

    # Log scalar metrics
    mlflow.log_metric(f"{scan_type}.total_findings", len(findings))
    mlflow.log_metric(f"{scan_type}.critical_count", severity_counts.get("CRITICAL", 0))
    mlflow.log_metric(f"{scan_type}.high_count", severity_counts.get("HIGH", 0))
    mlflow.log_metric(f"{scan_type}.medium_count", severity_counts.get("MEDIUM", 0))
    mlflow.log_metric(f"{scan_type}.low_count", severity_counts.get("LOW", 0))
    mlflow.log_metric(f"{scan_type}.info_count", severity_counts.get("INFO", 0))

    # Operational metrics
    mlflow.log_metric(f"{scan_type}.files_scanned", files_scanned)
    mlflow.log_metric(f"{scan_type}.scan_duration_seconds", scan_duration_seconds)
    mlflow.log_metric(f"{scan_type}.llm_tokens_used", llm_tokens_used)
    mlflow.log_metric(f"{scan_type}.llm_cost_usd", llm_cost_usd)

    # Derived metrics
    if files_scanned > 0:
        mlflow.log_metric(
            f"{scan_type}.findings_per_file", len(findings) / files_scanned
        )
    if scan_duration_seconds > 0:
        mlflow.log_metric(
            f"{scan_type}.files_per_second", files_scanned / scan_duration_seconds
        )

    # Log OWASP distribution as parameters (for filtering in UI)
    for category, count in owasp_counts.items():
        mlflow.log_param(f"{scan_type}.owasp.{category}", count)

    # Log scan metadata
    mlflow.log_param(f"{scan_type}.timestamp", datetime.now(timezone.utc).isoformat())
    mlflow.log_param(f"{scan_type}.scan_type", scan_type)


def log_remediation_metrics(
    issue_id: str,
    severity: str,
    time_to_fix_hours: float,
    was_reintroduced: bool = False,
) -> None:
    """Log metrics about issue remediation for MTTR tracking.

    Args:
        issue_id: Unique identifier for the security issue
        severity: Severity level (CRITICAL, HIGH, etc.)
        time_to_fix_hours: Time taken to remediate in hours
        was_reintroduced: True if this issue was previously fixed
    """
    if not MLFLOW_AVAILABLE:
        return

    mlflow.log_metric(f"remediation.{severity.lower()}.mttr_hours", time_to_fix_hours)
    mlflow.log_metric("remediation.total_fixed", 1)

    if was_reintroduced:
        mlflow.log_metric("remediation.regressions", 1)

    mlflow.log_param(f"remediation.{issue_id}.severity", severity)
    mlflow.log_param(f"remediation.{issue_id}.fixed_at", datetime.now(timezone.utc).isoformat())


def log_prompt_experiment(
    experiment_name: str,
    prompt_variant: str,
    findings_count: int,
    false_positive_rate: float,
    scan_duration_seconds: float,
) -> None:
    """Log A/B test results for different LLM prompts.

    Args:
        experiment_name: Name of the experiment
        prompt_variant: Identifier for this prompt variant (e.g., "baseline", "variant_a")
        findings_count: Number of findings detected
        false_positive_rate: Estimated false positive rate (0.0 to 1.0)
        scan_duration_seconds: Time taken for scan
    """
    if not MLFLOW_AVAILABLE:
        return

    with mlflow.start_run(run_name=f"{experiment_name}_{prompt_variant}"):
        mlflow.log_param("experiment", experiment_name)
        mlflow.log_param("prompt_variant", prompt_variant)
        mlflow.log_metric("findings_count", findings_count)
        mlflow.log_metric("false_positive_rate", false_positive_rate)
        mlflow.log_metric("scan_duration_seconds", scan_duration_seconds)

        # Quality score: balance between recall and precision
        precision = 1.0 - false_positive_rate
        quality_score = findings_count * precision / scan_duration_seconds if scan_duration_seconds > 0 else 0
        mlflow.log_metric("quality_score", quality_score)


def get_security_posture_summary() -> Dict[str, Any]:
    """Get summary of recent security scans from MLflow.

    Returns:
        Dictionary with security posture metrics
    """
    if not MLFLOW_AVAILABLE:
        return {}

    try:
        # Get recent runs
        experiment = mlflow.get_experiment_by_name(mlflow.get_experiment(mlflow.active_run().info.experiment_id).name)
        if not experiment:
            return {}

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=10
        )

        if runs.empty:
            return {}

        # Aggregate metrics
        latest = runs.iloc[0]
        summary = {
            "total_findings": latest.get("metrics.code_review.total_findings", 0),
            "critical_count": latest.get("metrics.code_review.critical_count", 0),
            "high_count": latest.get("metrics.code_review.high_count", 0),
            "trend": "improving" if len(runs) > 1 and runs.iloc[0]["metrics.code_review.total_findings"] < runs.iloc[1]["metrics.code_review.total_findings"] else "stable",
            "last_scan": latest.get("params.code_review.timestamp", "unknown")
        }

        return summary
    except Exception:
        return {}
