"""Telemetry and observability support with MLflow integration."""

from typing import Any, Callable, Optional
from collections import defaultdict

try:
    import mlflow
    import mlflow.dspy

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    MLFLOW_AVAILABLE = False

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
    if not MLFLOW_AVAILABLE:
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
