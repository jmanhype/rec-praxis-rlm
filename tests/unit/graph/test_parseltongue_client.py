"""Unit tests for ParseltongueClient."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from rec_praxis_rlm.graph import ParseltongueClient, CallGraphNode, DataFlowPath


@pytest.fixture
def mock_requests():
    """Mock requests module."""
    with patch("rec_praxis_rlm.graph.parseltongue_client.requests") as mock:
        yield mock


class TestParseltongu eClient:
    """Test ParseltongueClient HTTP client."""

    def test_initialization_success(self, mock_requests):
        """Test client initializes and checks availability."""
        # Mock health check success
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.get.return_value = mock_response

        client = ParseltongueClient("http://localhost:8080")

        assert client.base_url == "http://localhost:8080"
        assert client.timeout == 5
        assert client.is_available() is True
        mock_requests.get.assert_called_once_with(
            "http://localhost:8080/health",
            timeout=2
        )

    def test_initialization_unavailable(self, mock_requests):
        """Test client gracefully handles unavailable service."""
        # Mock health check failure
        mock_requests.get.side_effect = requests.RequestException("Connection refused")

        client = ParseltongueClient("http://localhost:9999")

        assert client.is_available() is False

    def test_get_call_graph_success(self, mock_requests):
        """Test successful call graph query."""
        # Mock health check
        health_response = Mock()
        health_response.status_code = 200

        # Mock call graph response
        call_graph_response = Mock()
        call_graph_response.status_code = 200
        call_graph_response.json.return_value = {
            "function_name": "authenticate",
            "file_path": "auth.py",
            "line_number": 42,
            "callers": ["login", "verify_token"],
            "callees": ["check_password", "create_session"]
        }

        mock_requests.get.side_effect = [health_response, call_graph_response]

        client = ParseltongueClient()
        result = client.get_call_graph("authenticate", "auth.py")

        assert isinstance(result, CallGraphNode)
        assert result.function_name == "authenticate"
        assert result.file_path == "auth.py"
        assert result.line_number == 42
        assert "login" in result.callers
        assert "check_password" in result.callees

    def test_get_call_graph_not_found(self, mock_requests):
        """Test call graph query for non-existent function."""
        health_response = Mock()
        health_response.status_code = 200

        not_found_response = Mock()
        not_found_response.status_code = 404

        mock_requests.get.side_effect = [health_response, not_found_response]

        client = ParseltongueClient()
        result = client.get_call_graph("nonexistent")

        assert result is None

    def test_get_call_graph_unavailable(self, mock_requests):
        """Test call graph query when service unavailable."""
        mock_requests.get.side_effect = requests.RequestException()

        client = ParseltongueClient()
        result = client.get_call_graph("test")

        assert result is None

    def test_get_data_flow_success(self, mock_requests):
        """Test successful data flow query."""
        health_response = Mock()
        health_response.status_code = 200

        data_flow_response = Mock()
        data_flow_response.status_code = 200
        data_flow_response.json.return_value = {
            "paths": [
                {
                    "source": "user_input",
                    "sink": "execute",
                    "path": ["api_handler", "process", "execute_query"],
                    "is_tainted": True
                },
                {
                    "source": "user_input",
                    "sink": "execute",
                    "path": ["api_handler", "sanitize", "execute_query"],
                    "is_tainted": False
                }
            ]
        }

        mock_requests.get.side_effect = [health_response, data_flow_response]

        client = ParseltongueClient()
        results = client.get_data_flow("user_input", "execute")

        assert len(results) == 2
        assert all(isinstance(r, DataFlowPath) for r in results)
        assert results[0].is_tainted is True
        assert results[1].is_tainted is False
        assert "api_handler" in results[0].path

    def test_get_data_flow_empty(self, mock_requests):
        """Test data flow query with no paths found."""
        health_response = Mock()
        health_response.status_code = 200

        empty_response = Mock()
        empty_response.status_code = 200
        empty_response.json.return_value = {"paths": []}

        mock_requests.get.side_effect = [health_response, empty_response]

        client = ParseltongueClient()
        results = client.get_data_flow("source", "sink")

        assert results == []

    def test_get_entry_points_success(self, mock_requests):
        """Test successful entry points query."""
        health_response = Mock()
        health_response.status_code = 200

        entry_points_response = Mock()
        entry_points_response.status_code = 200
        entry_points_response.json.return_value = {
            "entry_points": [
                "/api/users",
                "/api/login",
                "/api/admin/delete"
            ]
        }

        mock_requests.get.side_effect = [health_response, entry_points_response]

        client = ParseltongueClient()
        results = client.get_entry_points(public_only=True)

        assert len(results) == 3
        assert "/api/users" in results
        assert "/api/admin/delete" in results

    def test_find_function_references_success(self, mock_requests):
        """Test successful function references query."""
        health_response = Mock()
        health_response.status_code = 200

        refs_response = Mock()
        refs_response.status_code = 200
        refs_response.json.return_value = {
            "references": [
                {
                    "file_path": "auth.py",
                    "line_number": 42,
                    "context": "result = authenticate(username, password)"
                },
                {
                    "file_path": "admin.py",
                    "line_number": 15,
                    "context": "if authenticate(token):"
                }
            ]
        }

        mock_requests.get.side_effect = [health_response, refs_response]

        client = ParseltongueClient()
        results = client.find_function_references("authenticate")

        assert len(results) == 2
        assert results[0]["file_path"] == "auth.py"
        assert results[0]["line_number"] == 42

    def test_request_timeout(self, mock_requests):
        """Test request timeout handling."""
        health_response = Mock()
        health_response.status_code = 200

        mock_requests.get.side_effect = [
            health_response,
            requests.Timeout("Request timed out")
        ]

        client = ParseltongueClient(timeout=1)
        result = client.get_call_graph("test")

        assert result is None

    def test_invalid_json_response(self, mock_requests):
        """Test handling of invalid JSON in response."""
        health_response = Mock()
        health_response.status_code = 200

        bad_json_response = Mock()
        bad_json_response.status_code = 200
        bad_json_response.json.side_effect = ValueError("Invalid JSON")

        mock_requests.get.side_effect = [health_response, bad_json_response]

        client = ParseltongueClient()
        result = client.get_call_graph("test")

        # Should handle gracefully
        assert result is None or isinstance(result, CallGraphNode)
