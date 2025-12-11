"""Unit tests for GraphContextProvider."""

import pytest
import time
from unittest.mock import Mock, MagicMock
from rec_praxis_rlm.graph import GraphContextProvider, CallGraphNode, DataFlowPath
from rec_praxis_rlm.memory import ProceduralMemory, Experience
from rec_praxis_rlm.config import MemoryConfig


@pytest.fixture
def mock_parseltongue():
    """Mock ParseltongueClient."""
    client = Mock()
    client.is_available.return_value = True
    return client


@pytest.fixture
def mock_memory():
    """Mock ProceduralMemory."""
    memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
    return memory


@pytest.fixture
def provider(mock_parseltongue, mock_memory):
    """GraphContextProvider with mocked dependencies."""
    return GraphContextProvider(mock_parseltongue, mock_memory)


class TestGraphContextProvider:
    """Test GraphContextProvider functionality."""

    def test_initialization(self, mock_parseltongue, mock_memory):
        """Test provider initializes correctly."""
        provider = GraphContextProvider(mock_parseltongue, mock_memory)

        assert provider.parseltongue == mock_parseltongue
        assert provider.memory == mock_memory

    def test_get_security_context_with_call_graph(self, provider, mock_parseltongue):
        """Test security context includes call graph."""
        # Mock call graph
        call_graph = CallGraphNode(
            function_name="authenticate",
            file_path="auth.py",
            line_number=42,
            callers=["login", "verify_token"],
            callees=["check_password", "create_session"]
        )
        mock_parseltongue.get_call_graph.return_value = call_graph
        mock_parseltongue.get_data_flow.return_value = []

        context = provider.get_security_context("auth.py", "authenticate")

        assert "Call Graph" in context
        assert "authenticate" in context
        assert "login" in context
        assert "check_password" in context
        assert "auth.py:42" in context

    def test_get_security_context_entry_point(self, provider, mock_parseltongue):
        """Test security context identifies entry points."""
        # Mock entry point (no callers)
        call_graph = CallGraphNode(
            function_name="api_handler",
            file_path="api.py",
            line_number=10,
            callers=[],  # Entry point
            callees=["process_request"]
        )
        mock_parseltongue.get_call_graph.return_value = call_graph
        mock_parseltongue.get_data_flow.return_value = []

        context = provider.get_security_context("api.py", "api_handler")

        assert "Entry point" in context
        assert "no callers" in context

    def test_get_security_context_with_data_flow(self, provider, mock_parseltongue):
        """Test security context includes data flow."""
        mock_parseltongue.get_call_graph.return_value = None

        # Mock data flow (tainted)
        data_flow = DataFlowPath(
            source="user",
            sink="execute",
            path=["api_handler", "process_user", "db.execute"],
            is_tainted=True
        )
        mock_parseltongue.get_data_flow.return_value = [data_flow]

        context = provider.get_security_context("database.py")

        assert "Data Flow" in context
        assert "user → execute" in context
        assert "UNSANITIZED" in context
        assert "api_handler" in context

    def test_get_security_context_sanitized_flow(self, provider, mock_parseltongue):
        """Test security context shows sanitized data flows."""
        mock_parseltongue.get_call_graph.return_value = None

        # Mock data flow (sanitized)
        data_flow = DataFlowPath(
            source="user",
            sink="execute",
            path=["api_handler", "sanitize", "db.execute"],
            is_tainted=False
        )
        mock_parseltongue.get_data_flow.return_value = [data_flow]

        context = provider.get_security_context("database.py")

        assert "Data Flow" in context
        assert "SANITIZED" in context

    def test_get_security_context_with_memory(self, provider, mock_parseltongue, mock_memory):
        """Test security context includes past experiences."""
        mock_parseltongue.get_call_graph.return_value = None
        mock_parseltongue.get_data_flow.return_value = []

        # Add past experiences to memory
        exp1 = Experience(
            env_features=["python", "security", "database.py"],
            goal="detect vulnerabilities",
            action="Found SQL injection in database.py:50",
            result="Fixed with parameterized query",
            success=True,
            timestamp=time.time(),
            metadata={"pattern": "user_input → execute"}
        )
        mock_memory.store(exp1)

        context = provider.get_security_context("database.py")

        assert "Similar Past Findings" in context
        assert "detect vulnerabilities" in context

    def test_get_security_context_empty(self, provider, mock_parseltongue):
        """Test security context returns empty when no context available."""
        mock_parseltongue.get_call_graph.return_value = None
        mock_parseltongue.get_data_flow.return_value = []

        context = provider.get_security_context("test.py")

        assert context == ""

    def test_get_security_context_parseltongue_unavailable(self, provider, mock_parseltongue):
        """Test security context when Parseltongue unavailable."""
        mock_parseltongue.is_available.return_value = False

        context = provider.get_security_context("test.py", "test_func")

        # Should only have memory context (no graph context)
        assert "Call Graph" not in context
        assert "Data Flow" not in context

    def test_get_attack_surface_success(self, provider, mock_parseltongue):
        """Test get attack surface returns entry points."""
        mock_parseltongue.get_entry_points.return_value = [
            "/api/users",
            "/api/login",
            "/api/admin/delete"
        ]

        entry_points = provider.get_attack_surface()

        assert len(entry_points) == 3
        assert "/api/users" in entry_points
        assert "/api/admin/delete" in entry_points

    def test_get_attack_surface_unavailable(self, provider, mock_parseltongue):
        """Test get attack surface when Parseltongue unavailable."""
        mock_parseltongue.is_available.return_value = False

        entry_points = provider.get_attack_surface()

        assert entry_points == []

    def test_analyze_impact_direct_callers(self, provider, mock_parseltongue):
        """Test analyze impact returns direct callers."""
        call_graph = CallGraphNode(
            function_name="authenticate",
            file_path="auth.py",
            line_number=42,
            callers=["login", "verify_token"],
            callees=[]
        )
        mock_parseltongue.get_call_graph.return_value = call_graph

        impact = provider.analyze_impact("authenticate")

        assert "direct_callers" in impact
        assert "login" in impact["direct_callers"]
        assert "verify_token" in impact["direct_callers"]

    def test_analyze_impact_transitive_callers(self, provider, mock_parseltongue):
        """Test analyze impact finds transitive callers."""
        # Root function
        root_graph = CallGraphNode(
            function_name="authenticate",
            file_path="auth.py",
            line_number=42,
            callers=["login"],
            callees=[]
        )

        # Transitive caller
        login_graph = CallGraphNode(
            function_name="login",
            file_path="auth.py",
            line_number=10,
            callers=["api_handler"],
            callees=["authenticate"]
        )

        def mock_get_call_graph(func_name, file_path=None):
            if func_name == "authenticate":
                return root_graph
            elif func_name == "login":
                return login_graph
            return None

        mock_parseltongue.get_call_graph.side_effect = mock_get_call_graph

        impact = provider.analyze_impact("authenticate")

        assert "transitive_callers" in impact
        assert "api_handler" in impact["transitive_callers"]

    def test_analyze_impact_unavailable(self, provider, mock_parseltongue):
        """Test analyze impact when Parseltongue unavailable."""
        mock_parseltongue.is_available.return_value = False

        impact = provider.analyze_impact("test")

        assert impact == {"direct_callers": [], "transitive_callers": []}

    def test_analyze_impact_function_not_found(self, provider, mock_parseltongue):
        """Test analyze impact when function not found."""
        mock_parseltongue.get_call_graph.return_value = None

        impact = provider.analyze_impact("nonexistent")

        assert impact == {"direct_callers": [], "transitive_callers": []}

    def test_get_data_flow_paths_success(self, provider, mock_parseltongue):
        """Test get data flow paths."""
        flow = DataFlowPath(
            source="user_input",
            sink="execute",
            path=["api", "process", "db"],
            is_tainted=True
        )
        mock_parseltongue.get_data_flow.return_value = [flow]

        flows = provider.get_data_flow_paths("user_input", "execute")

        assert len(flows) == 1
        assert flows[0].source == "user_input"
        assert flows[0].sink == "execute"
        assert flows[0].is_tainted is True

    def test_get_data_flow_paths_unavailable(self, provider, mock_parseltongue):
        """Test get data flow paths when Parseltongue unavailable."""
        mock_parseltongue.is_available.return_value = False

        flows = provider.get_data_flow_paths("source", "sink")

        assert flows == []

    def test_find_vulnerability_patterns(self, provider, mock_memory):
        """Test find vulnerability patterns in memory."""
        # Add SQL injection experience
        exp = Experience(
            env_features=["security", "sql_injection"],
            goal="detect sql_injection",
            action="Found SQL injection",
            result="Fixed with parameterized query",
            success=True,
            timestamp=time.time(),
            metadata={"pattern": "user → execute"}
        )
        mock_memory.store(exp)

        patterns = provider.find_vulnerability_patterns("sql_injection")

        assert len(patterns) > 0
        assert patterns[0].goal == "detect sql_injection"

    def test_enrich_finding_with_graph_context(self, provider, mock_parseltongue):
        """Test enrich finding with graph context."""
        call_graph = CallGraphNode(
            function_name="authenticate",
            file_path="auth.py",
            line_number=42,
            callers=["login"],
            callees=["check_password"]
        )
        mock_parseltongue.get_call_graph.return_value = call_graph

        finding = {
            "file_path": "auth.py",
            "function_name": "authenticate",
            "title": "SQL Injection"
        }

        enriched = provider.enrich_finding_with_context(finding)

        assert "graph_context" in enriched
        assert enriched["graph_context"]["callers"] == ["login"]
        assert enriched["graph_context"]["callees"] == ["check_password"]
        assert enriched["graph_context"]["is_entry_point"] is False

    def test_enrich_finding_entry_point(self, provider, mock_parseltongue):
        """Test enrich finding identifies entry points."""
        call_graph = CallGraphNode(
            function_name="api_handler",
            file_path="api.py",
            line_number=10,
            callers=[],  # Entry point
            callees=["process"]
        )
        mock_parseltongue.get_call_graph.return_value = call_graph

        finding = {
            "file_path": "api.py",
            "function_name": "api_handler",
            "title": "Authentication Bypass"
        }

        enriched = provider.enrich_finding_with_context(finding)

        assert enriched["graph_context"]["is_entry_point"] is True

    def test_enrich_finding_with_memory_context(self, provider, mock_memory, mock_parseltongue):
        """Test enrich finding with memory context."""
        mock_parseltongue.get_call_graph.return_value = None

        # Add past SQL injection experience
        exp = Experience(
            env_features=["security", "sql_injection"],
            goal="detect sql_injection",
            action="Found SQL injection",
            result="Fixed",
            success=True,
            timestamp=time.time(),
            metadata={"pattern": "user → execute"}
        )
        mock_memory.store(exp)

        finding = {
            "file_path": "db.py",
            "title": "SQL Injection"
        }

        enriched = provider.enrich_finding_with_context(finding)

        assert "memory_context" in enriched
        assert enriched["memory_context"]["similar_findings_count"] > 0

    def test_enrich_finding_skip_graph(self, provider, mock_parseltongue):
        """Test enrich finding can skip graph context."""
        finding = {
            "file_path": "test.py",
            "function_name": "test"
        }

        enriched = provider.enrich_finding_with_context(finding, include_graph=False)

        assert "graph_context" not in enriched
        mock_parseltongue.get_call_graph.assert_not_called()

    def test_enrich_finding_skip_memory(self, provider, mock_memory):
        """Test enrich finding can skip memory context."""
        finding = {
            "file_path": "test.py",
            "title": "Test Finding"
        }

        enriched = provider.enrich_finding_with_context(finding, include_memory=False)

        assert "memory_context" not in enriched

    def test_transitive_callers_cycle_detection(self, provider, mock_parseltongue):
        """Test transitive callers handles cycles in call graph."""
        # Create circular call graph: A → B → C → A
        graph_a = CallGraphNode("A", "test.py", 1, ["C"], ["B"])
        graph_b = CallGraphNode("B", "test.py", 2, ["A"], ["C"])
        graph_c = CallGraphNode("C", "test.py", 3, ["B"], ["A"])

        def mock_get_call_graph(func_name, file_path=None):
            if func_name == "A":
                return graph_a
            elif func_name == "B":
                return graph_b
            elif func_name == "C":
                return graph_c
            return None

        mock_parseltongue.get_call_graph.side_effect = mock_get_call_graph

        # Should not infinite loop
        impact = provider.analyze_impact("A")

        # Should find some callers but stop at cycle
        assert "transitive_callers" in impact

    def test_transitive_callers_max_depth(self, provider, mock_parseltongue):
        """Test transitive callers respects max depth."""
        # Create deep call chain
        def mock_get_call_graph(func_name, file_path=None):
            if func_name.startswith("func"):
                level = int(func_name[4:])
                return CallGraphNode(
                    func_name,
                    "test.py",
                    level,
                    [f"func{level+1}"] if level < 10 else [],
                    []
                )
            return None

        mock_parseltongue.get_call_graph.side_effect = mock_get_call_graph

        # Should stop at max_depth=5
        transitive = provider._find_transitive_callers(
            "func0",
            ["func1"],
            max_depth=5
        )

        # Should have found func1-func5 (depth 0-4), not func6+
        assert len(transitive) <= 5

    def test_format_call_graph_context_truncation(self, provider):
        """Test call graph context handles many callers/callees."""
        call_graph = CallGraphNode(
            function_name="popular_function",
            file_path="utils.py",
            line_number=100,
            callers=[f"caller_{i}" for i in range(50)],  # Many callers
            callees=[f"callee_{i}" for i in range(30)]   # Many callees
        )

        context = provider._format_call_graph_context(call_graph)

        # Should format without error
        assert "popular_function" in context
        assert "caller_0" in context
        assert "callee_0" in context

    def test_format_memory_context_long_action(self, provider):
        """Test memory context truncates long actions."""
        exp = Experience(
            env_features=["test"],
            goal="test goal",
            action="A" * 500,  # Very long action
            result="test result",
            success=True,
            timestamp=time.time()
        )

        context = provider._format_memory_context([exp])

        # Should truncate action
        assert len(context) < 1000
        assert "..." in context
