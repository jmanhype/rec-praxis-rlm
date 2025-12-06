"""Unit tests for telemetry module."""
import pytest
from unittest.mock import Mock, patch, call

from rec_praxis_rlm.telemetry import setup_mlflow_tracing, add_telemetry_hook, emit_event


class TestSetupMLflowTracing:
    """Tests for MLflow tracing setup."""

    @patch("rec_praxis_rlm.telemetry.mlflow")
    def test_setup_with_defaults(self, mock_mlflow: Mock) -> None:
        """Test MLflow setup with default parameters."""
        setup_mlflow_tracing()
        mock_mlflow.dspy.autolog.assert_called_once()

    @patch("rec_praxis_rlm.telemetry.mlflow")
    def test_setup_with_experiment_name(self, mock_mlflow: Mock) -> None:
        """Test MLflow setup with custom experiment name."""
        setup_mlflow_tracing(experiment_name="test_experiment")
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        mock_mlflow.dspy.autolog.assert_called_once()

    @patch("rec_praxis_rlm.telemetry.mlflow")
    def test_log_traces_from_compile_flag(self, mock_mlflow: Mock) -> None:
        """Test log_traces_from_compile parameter."""
        setup_mlflow_tracing(log_traces_from_compile=True)
        mock_mlflow.dspy.autolog.assert_called_once_with(log_traces=True)


class TestTelemetryHooks:
    """Tests for telemetry hook system."""

    def test_add_telemetry_hook(self) -> None:
        """Test adding a telemetry hook."""
        mock_callback = Mock()
        add_telemetry_hook("test_event", mock_callback)
        # Hook should be registered but not called yet
        mock_callback.assert_not_called()

    def test_emit_event_triggers_hooks(self) -> None:
        """Test that emit_event triggers registered hooks."""
        mock_callback = Mock()
        add_telemetry_hook("memory.recall", mock_callback)

        event_data = {"query": "test", "results": 5}
        emit_event("memory.recall", event_data)

        mock_callback.assert_called_once_with("memory.recall", event_data)

    def test_multiple_hooks_for_same_event(self) -> None:
        """Test multiple hooks can be registered for the same event."""
        mock_callback1 = Mock()
        mock_callback2 = Mock()

        add_telemetry_hook("context.search", mock_callback1)
        add_telemetry_hook("context.search", mock_callback2)

        event_data = {"pattern": "ERROR"}
        emit_event("context.search", event_data)

        mock_callback1.assert_called_once_with("context.search", event_data)
        mock_callback2.assert_called_once_with("context.search", event_data)

    def test_emit_event_silently_handles_callback_exceptions(self) -> None:
        """Test that exceptions in callbacks don't crash emit_event."""
        mock_callback = Mock(side_effect=Exception("Callback failed"))
        add_telemetry_hook("test.event", mock_callback)

        # Should not raise exception
        event_data = {"test": "data"}
        emit_event("test.event", event_data)

        # Callback was called despite exception
        mock_callback.assert_called_once_with("test.event", event_data)

    @patch("rec_praxis_rlm.telemetry.MLFLOW_AVAILABLE", False)
    def test_setup_mlflow_raises_import_error_when_unavailable(self) -> None:
        """Test that setup_mlflow_tracing raises ImportError when MLflow not available."""
        with pytest.raises(ImportError, match="mlflow not installed"):
            setup_mlflow_tracing()
