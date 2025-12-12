"""Unit tests for CachedParseltongueClient."""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from rec_praxis_rlm.graph import CachedParseltongueClient, CallGraphNode


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_requests():
    """Mock requests module."""
    with patch("rec_praxis_rlm.graph.parseltongue_client.requests") as mock:
        # Mock health check success
        health_response = Mock()
        health_response.status_code = 200
        mock.get.return_value = health_response
        yield mock


class TestCachedClient:
    """Test CachedParseltongueClient caching behavior."""

    def test_cache_hit_memory(self, temp_cache_dir, mock_requests):
        """Test memory cache hit."""
        # Mock first call (cache miss)
        call_graph_response = Mock()
        call_graph_response.status_code = 200
        call_graph_response.json.return_value = {
            "function_name": "test",
            "file_path": "test.py",
            "line_number": 1,
            "callers": [],
            "callees": []
        }

        mock_requests.get.side_effect = [
            Mock(status_code=200),  # Health check
            call_graph_response      # First query
        ]

        client = CachedParseltongueClient(cache_dir=temp_cache_dir)

        # First call - cache miss, queries Parseltongue
        result1 = client.get_call_graph("test")
        assert result1 is not None
        assert result1.function_name == "test"

        # Reset mock
        mock_requests.get.reset_mock()

        # Second call - cache hit, no HTTP request
        result2 = client.get_call_graph("test")
        assert result2 is not None
        assert result2.function_name == "test"

        # Should not have made additional HTTP requests
        mock_requests.get.assert_not_called()

    def test_cache_hit_disk(self, temp_cache_dir, mock_requests):
        """Test disk cache hit after memory cache cleared."""
        call_graph_response = Mock()
        call_graph_response.status_code = 200
        call_graph_response.json.return_value = {
            "function_name": "test",
            "file_path": "test.py",
            "line_number": 1,
            "callers": [],
            "callees": []
        }

        mock_requests.get.side_effect = [
            Mock(status_code=200),  # Health check
            call_graph_response      # First query
        ]

        client1 = CachedParseltongueClient(cache_dir=temp_cache_dir)

        # First call - cache miss
        result1 = client1.get_call_graph("test")
        assert result1 is not None

        # Create new client (new memory cache, but same disk cache)
        client2 = CachedParseltongueClient(cache_dir=temp_cache_dir)

        # Reset mock
        mock_requests.get.reset_mock()
        mock_requests.get.return_value = Mock(status_code=200)  # Health check only

        # Second call - disk cache hit
        result2 = client2.get_call_graph("test")
        assert result2 is not None
        assert result2.function_name == "test"

        # Disk cache hit should avoid any HTTP requests.
        assert mock_requests.get.call_count == 0

    def test_cache_ttl_expiration(self, temp_cache_dir, mock_requests):
        """Test cache TTL expiration."""
        call_graph_response = Mock()
        call_graph_response.status_code = 200
        call_graph_response.json.return_value = {
            "function_name": "test",
            "file_path": "test.py",
            "line_number": 1,
            "callers": [],
            "callees": []
        }

        mock_requests.get.side_effect = [
            Mock(status_code=200),  # Health check
            call_graph_response,     # First query
            call_graph_response,     # Second query (after expiration)
        ]

        # Client with 1 second TTL
        client = CachedParseltongueClient(cache_dir=temp_cache_dir, cache_ttl=1)

        # First call
        result1 = client.get_call_graph("test")
        assert result1 is not None

        # Wait for TTL to expire
        time.sleep(1.1)

        # Second call - cache expired, should query again
        result2 = client.get_call_graph("test")
        assert result2 is not None

        # Should have made 1 health check + 2 call graph queries
        assert mock_requests.get.call_count == 3

    def test_clear_cache(self, temp_cache_dir, mock_requests):
        """Test cache clearing."""
        call_graph_response = Mock()
        call_graph_response.status_code = 200
        call_graph_response.json.return_value = {
            "function_name": "test",
            "file_path": "test.py",
            "line_number": 1,
            "callers": [],
            "callees": []
        }

        mock_requests.get.side_effect = [
            Mock(status_code=200),  # Health check
            call_graph_response,     # First query
            call_graph_response      # After clear
        ]

        client = CachedParseltongueClient(cache_dir=temp_cache_dir)

        # First call - populate cache
        result1 = client.get_call_graph("test")
        assert result1 is not None

        # Clear cache
        client.clear_cache()

        # Reset mock
        mock_requests.get.reset_mock()
        mock_requests.get.side_effect = [
            call_graph_response      # Query after clear
        ]

        # Second call - cache miss after clear
        result2 = client.get_call_graph("test")
        assert result2 is not None

        # Should have made HTTP request again (no extra health check)
        assert mock_requests.get.call_count == 1

    def test_cache_stats(self, temp_cache_dir, mock_requests):
        """Test cache statistics."""
        call_graph_response = Mock()
        call_graph_response.status_code = 200
        call_graph_response.json.return_value = {
            "function_name": "test",
            "file_path": "test.py",
            "line_number": 1,
            "callers": [],
            "callees": []
        }

        mock_requests.get.side_effect = [
            Mock(status_code=200),  # Health check
            call_graph_response,     # Query 1
            call_graph_response,     # Query 2
        ]

        client = CachedParseltongueClient(cache_dir=temp_cache_dir)

        # Populate cache
        client.get_call_graph("test1")
        client.get_call_graph("test2")

        stats = client.get_cache_stats()

        assert stats["memory_entries"] == 2
        assert stats["disk_entries"] == 2
        assert stats["total_size_mb"] > 0

    def test_content_hash_consistency(self, temp_cache_dir):
        """Test content hash is consistent for same input."""
        client = CachedParseltongueClient(cache_dir=temp_cache_dir)

        hash1 = client._get_content_hash(["test", "file.py"])
        hash2 = client._get_content_hash(["test", "file.py"])

        assert hash1 == hash2

    def test_content_hash_uniqueness(self, temp_cache_dir):
        """Test content hash is unique for different inputs."""
        client = CachedParseltongueClient(cache_dir=temp_cache_dir)

        hash1 = client._get_content_hash(["test1", "file.py"])
        hash2 = client._get_content_hash(["test2", "file.py"])

        assert hash1 != hash2

    def test_corrupted_cache_handling(self, temp_cache_dir, mock_requests):
        """Test handling of corrupted disk cache."""
        call_graph_response = Mock()
        call_graph_response.status_code = 200
        call_graph_response.json.return_value = {
            "function_name": "test",
            "file_path": "test.py",
            "line_number": 1,
            "callers": [],
            "callees": []
        }

        mock_requests.get.side_effect = [
            Mock(status_code=200),  # Health check
            call_graph_response      # Query after corrupted cache
        ]

        client = CachedParseltongueClient(cache_dir=temp_cache_dir)

        # Create corrupted cache file
        cache_dir = Path(temp_cache_dir)
        cache_key = client._get_content_hash(["call_graph", "test", "any"])
        cache_file = cache_dir / f"{cache_key}.json"
        cache_file.write_text("{invalid json")

        # Should handle gracefully and query Parseltongue
        result = client.get_call_graph("test")
        assert result is not None

        # Corrupted file should be handled and replaced with valid cache.
        assert cache_file.exists()
        assert cache_file.read_text().strip().startswith("{")
