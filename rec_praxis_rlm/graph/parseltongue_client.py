"""HTTP client for Parseltongue graph query API.

Parseltongue is a Rust-based tool that parses codebases into dependency graphs
using tree-sitter and stores them in CozoDB.

Repository: https://github.com/that-in-rust/parseltongue-dependency-graph-generator
"""

import requests
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CallGraphNode:
    """Represents a function in the call graph."""
    function_name: str
    file_path: str
    line_number: int
    callers: List[str]  # Functions that call this
    callees: List[str]  # Functions this calls


@dataclass
class DataFlowPath:
    """Represents a data flow path from source to sink."""
    source: str
    sink: str
    path: List[str]  # Functions in the flow path
    is_tainted: bool  # Whether data is sanitized


class ParseltongueClient:
    """Client for querying Parseltongue graph API.

    Parseltongue is a Rust-based tool that parses codebases into dependency
    graphs using tree-sitter and stores them in CozoDB.

    Example:
        >>> client = ParseltongueClient("http://localhost:8080")
        >>> if client.is_available():
        ...     call_graph = client.get_call_graph("authenticate")
        ...     print(f"Called by: {call_graph.callers}")
    """

    def __init__(self, base_url: str = "http://localhost:8080", timeout: int = 5):
        """Initialize Parseltongue client.

        Args:
            base_url: Parseltongue HTTP API base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if Parseltongue service is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if Parseltongue is available."""
        return self._available

    def get_call_graph(self, function_name: str, file_path: Optional[str] = None) -> Optional[CallGraphNode]:
        """Get call graph for a function.

        Args:
            function_name: Name of the function to query
            file_path: Optional file path to disambiguate (if multiple functions with same name)

        Returns:
            CallGraphNode with callers and callees, or None if not found

        Example:
            >>> client = ParseltongueClient()
            >>> graph = client.get_call_graph("authenticate", "auth.py")
            >>> print(f"Callers: {graph.callers}")
        """
        if not self.is_available():
            return None

        try:
            params = {"function": function_name}
            if file_path:
                params["file"] = file_path

            response = requests.get(
                f"{self.base_url}/call-graph",
                params=params,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return CallGraphNode(
                    function_name=data["function_name"],
                    file_path=data["file_path"],
                    line_number=data["line_number"],
                    callers=data.get("callers", []),
                    callees=data.get("callees", [])
                )
            return None
        except Exception:
            return None

    def get_data_flow(self, source: str, sink: str, max_depth: int = 10) -> List[DataFlowPath]:
        """Find data flow paths from source to sink.

        Args:
            source: Source function/variable (e.g., "user_input", "request.params")
            sink: Sink function/variable (e.g., "db.execute", "eval")
            max_depth: Maximum depth to search (default: 10)

        Returns:
            List of data flow paths from source to sink

        Example:
            >>> client = ParseltongueClient()
            >>> flows = client.get_data_flow("user_input", "execute")
            >>> for flow in flows:
            ...     if flow.is_tainted:
            ...         print(f"⚠️ Tainted flow: {' → '.join(flow.path)}")
        """
        if not self.is_available():
            return []

        try:
            response = requests.get(
                f"{self.base_url}/data-flow",
                params={"source": source, "sink": sink, "max_depth": max_depth},
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return [
                    DataFlowPath(
                        source=path["source"],
                        sink=path["sink"],
                        path=path["path"],
                        is_tainted=path.get("is_tainted", True)
                    )
                    for path in data.get("paths", [])
                ]
            return []
        except Exception:
            return []

    def get_entry_points(self, public_only: bool = True) -> List[str]:
        """Get all entry points (e.g., API routes, public functions).

        Args:
            public_only: Only return public entry points (default: True)

        Returns:
            List of entry point function names

        Example:
            >>> client = ParseltongueClient()
            >>> entry_points = client.get_entry_points(public_only=True)
            >>> print(f"Attack surface: {len(entry_points)} public endpoints")
        """
        if not self.is_available():
            return []

        try:
            response = requests.get(
                f"{self.base_url}/entry-points",
                params={"public_only": public_only},
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("entry_points", [])
            return []
        except Exception:
            return []

    def find_function_references(self, function_name: str) -> List[Dict[str, str]]:
        """Find all references to a function.

        Args:
            function_name: Function to find references for

        Returns:
            List of {file_path, line_number, context} dicts

        Example:
            >>> client = ParseltongueClient()
            >>> refs = client.find_function_references("authenticate")
            >>> for ref in refs:
            ...     print(f"{ref['file_path']}:{ref['line_number']}")
        """
        if not self.is_available():
            return []

        try:
            response = requests.get(
                f"{self.base_url}/references",
                params={"function": function_name},
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("references", [])
            return []
        except Exception:
            return []
