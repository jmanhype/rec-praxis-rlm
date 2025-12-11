"""Unit tests for GraphAwareCodeReviewAgent."""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from rec_praxis_rlm.agents.code_review_graph import GraphAwareCodeReviewAgent
from rec_praxis_rlm.graph import CallGraphNode, DataFlowPath
from rec_praxis_rlm.types import Finding, Severity


@pytest.fixture
def mock_parseltongue():
    """Mock ParseltongueClient."""
    with patch("rec_praxis_rlm.agents.code_review_graph.ParseltongueClient") as mock:
        client = mock.return_value
        client.is_available.return_value = True
        client.get_data_flow.return_value = []
        client.get_entry_points.return_value = []
        client.get_call_graph.return_value = None
        yield client


@pytest.fixture
def agent(mock_parseltongue):
    """GraphAwareCodeReviewAgent with mocked Parseltongue."""
    return GraphAwareCodeReviewAgent(memory_path=":memory:")


class TestGraphAwareCodeReviewAgent:
    """Test GraphAwareCodeReviewAgent functionality."""

    def test_initialization_with_available_parseltongue(self, mock_parseltongue):
        """Test agent initializes when Parseltongue is available."""
        agent = GraphAwareCodeReviewAgent(memory_path=":memory:")

        assert agent.parseltongue is not None
        assert agent.graph_context is not None
        assert agent.parseltongue.is_available() is True

    def test_initialization_with_unavailable_parseltongue(self, mock_parseltongue, capsys):
        """Test agent initializes gracefully when Parseltongue unavailable."""
        mock_parseltongue.is_available.return_value = False

        agent = GraphAwareCodeReviewAgent(memory_path=":memory:")

        # Should print warning
        captured = capsys.readouterr()
        assert "Parseltongue not available" in captured.out
        assert "graph analysis disabled" in captured.out

    def test_review_code_combines_pattern_and_graph(self, agent, mock_parseltongue):
        """Test review combines pattern-based and graph-based findings."""
        # Mock data flow vulnerability
        data_flow = DataFlowPath(
            source="user",
            sink="execute",
            path=["api_handler", "process", "db.execute"],
            is_tainted=True
        )
        mock_parseltongue.get_data_flow.return_value = [data_flow]

        # Review code with SQL injection pattern
        code = {
            "db.py": 'cursor.execute(f"SELECT * FROM users WHERE id={user_id}")'
        }

        findings = agent.review_code(code)

        # Should have both pattern-based and graph-based findings
        assert len(findings) > 0
        titles = [f.title for f in findings]
        assert any("SQL Injection" in title for title in titles)

    def test_check_data_flow_vulnerabilities_sql_injection(self, agent, mock_parseltongue):
        """Test detects SQL injection via data flow."""
        data_flow = DataFlowPath(
            source="user",
            sink="execute",
            path=["api_handler", "process_user", "db.execute"],
            is_tainted=True
        )
        mock_parseltongue.get_data_flow.return_value = [data_flow]

        findings = agent._check_data_flow_vulnerabilities()

        assert len(findings) == 1
        assert findings[0].title == "SQL Injection via Data Flow"
        assert findings[0].severity == Severity.CRITICAL
        assert "api_handler → process_user → db.execute" in findings[0].description
        assert "graph" in findings[0].metadata["source"]

    def test_check_data_flow_vulnerabilities_code_injection(self, agent, mock_parseltongue):
        """Test detects code injection via data flow."""
        eval_flow = DataFlowPath(
            source="user",
            sink="eval",
            path=["input_handler", "eval"],
            is_tainted=True
        )
        mock_parseltongue.get_data_flow.return_value = [eval_flow]

        findings = agent._check_data_flow_vulnerabilities()

        assert len(findings) == 1
        assert "Code Injection" in findings[0].title
        assert findings[0].severity == Severity.CRITICAL

    def test_check_data_flow_sanitized_path(self, agent, mock_parseltongue):
        """Test ignores sanitized data flows."""
        sanitized_flow = DataFlowPath(
            source="user",
            sink="execute",
            path=["api_handler", "sanitize", "db.execute"],
            is_tainted=False  # Sanitized
        )
        mock_parseltongue.get_data_flow.return_value = [sanitized_flow]

        findings = agent._check_data_flow_vulnerabilities()

        # Should not flag sanitized flows
        assert len(findings) == 0

    def test_check_authentication_bypass(self, agent, mock_parseltongue):
        """Test detects public endpoints without authentication."""
        # Mock public entry points
        mock_parseltongue.get_entry_points.return_value = ["/api/delete_user", "/api/admin"]

        # Mock call graph showing no auth calls
        call_graph = CallGraphNode(
            function_name="/api/delete_user",
            file_path="api.py",
            line_number=10,
            callers=[],  # Entry point
            callees=["delete_from_db", "log_action"]  # No auth functions
        )
        mock_parseltongue.get_call_graph.return_value = call_graph

        findings = agent._check_authentication_bypass()

        assert len(findings) == 2  # One for each endpoint
        assert findings[0].title == "Potential Authentication Bypass"
        assert findings[0].severity == Severity.HIGH
        assert "/api/delete_user" in findings[0].description

    def test_check_authentication_bypass_with_auth(self, agent, mock_parseltongue):
        """Test does not flag endpoints that call auth functions."""
        mock_parseltongue.get_entry_points.return_value = ["/api/protected"]

        # Mock call graph showing auth check
        call_graph = CallGraphNode(
            function_name="/api/protected",
            file_path="api.py",
            line_number=20,
            callers=[],
            callees=["authenticate", "process_request"]  # Has auth
        )
        mock_parseltongue.get_call_graph.return_value = call_graph

        findings = agent._check_authentication_bypass()

        # Should not flag - endpoint calls authenticate()
        assert len(findings) == 0

    def test_analyze_attack_surface_large(self, agent, mock_parseltongue):
        """Test flags large attack surface."""
        # Mock 25 public entry points
        entry_points = [f"/api/endpoint_{i}" for i in range(25)]
        mock_parseltongue.get_entry_points.return_value = entry_points

        findings = agent._analyze_attack_surface()

        assert len(findings) == 1
        assert findings[0].title == "Large Attack Surface"
        assert findings[0].severity == Severity.MEDIUM
        assert "25 public entry points" in findings[0].description

    def test_analyze_attack_surface_small(self, agent, mock_parseltongue):
        """Test does not flag small attack surface."""
        # Mock 10 entry points (below threshold)
        entry_points = [f"/api/endpoint_{i}" for i in range(10)]
        mock_parseltongue.get_entry_points.return_value = entry_points

        findings = agent._analyze_attack_surface()

        # Should not flag - below threshold of 20
        assert len(findings) == 0

    def test_deduplicate_findings(self, agent):
        """Test deduplicates findings."""
        # Create duplicate findings
        finding1 = Finding(
            file_path="test.py",
            line_number=10,
            severity=Severity.HIGH,
            title="SQL Injection",
            description="Test",
            remediation="Fix"
        )
        finding2 = Finding(
            file_path="test.py",
            line_number=10,
            severity=Severity.HIGH,
            title="SQL Injection",
            description="Different description",
            remediation="Fix"
        )
        finding3 = Finding(
            file_path="test.py",
            line_number=20,
            severity=Severity.HIGH,
            title="SQL Injection",
            description="Test",
            remediation="Fix"
        )

        findings = [finding1, finding2, finding3]
        unique = agent._deduplicate_findings(findings)

        # Should keep first occurrence of (test.py, 10, SQL Injection)
        # and (test.py, 20, SQL Injection)
        assert len(unique) == 2

    def test_deduplicate_findings_without_line_numbers(self, agent):
        """Test deduplicates findings without line numbers."""
        finding1 = Finding(
            file_path="multiple",
            line_number=None,
            severity=Severity.MEDIUM,
            title="Large Attack Surface",
            description="Test 1",
            remediation="Fix"
        )
        finding2 = Finding(
            file_path="multiple",
            line_number=None,
            severity=Severity.MEDIUM,
            title="Large Attack Surface",
            description="Test 2",
            remediation="Fix"
        )

        findings = [finding1, finding2]
        unique = agent._deduplicate_findings(findings)

        # Should deduplicate by (file_path, title) when no line number
        assert len(unique) == 1

    def test_store_graph_pattern(self, agent):
        """Test stores graph patterns in memory."""
        finding = Finding(
            file_path="api.py",
            line_number=10,
            severity=Severity.CRITICAL,
            title="SQL Injection via Data Flow",
            description="Test",
            remediation="Fix",
            metadata={
                "source": "graph",
                "flow_path": ["api_handler", "process", "execute"]
            }
        )

        agent._store_graph_pattern(finding)

        # Should have stored experience in memory
        assert agent.memory.size() == 1
        experiences = agent.memory.recall(
            env_features=["graph_finding"],
            goal="detect sql injection",
            top_k=1
        )
        assert len(experiences) > 0
        assert "api_handler → process → execute" in experiences[0].result

    def test_store_graph_pattern_without_flow_path(self, agent):
        """Test handles findings without flow_path gracefully."""
        finding = Finding(
            file_path="test.py",
            line_number=10,
            severity=Severity.HIGH,
            title="Authentication Bypass",
            description="Test",
            remediation="Fix",
            metadata={"source": "graph"}  # No flow_path
        )

        # Should not crash
        agent._store_graph_pattern(finding)

        # Should not store anything
        assert agent.memory.size() == 0

    def test_get_security_context(self, agent, mock_parseltongue):
        """Test gets security context for a file."""
        call_graph = CallGraphNode(
            function_name="authenticate",
            file_path="auth.py",
            line_number=42,
            callers=["login"],
            callees=["check_password"]
        )
        mock_parseltongue.get_call_graph.return_value = call_graph
        mock_parseltongue.get_data_flow.return_value = []

        context = agent.get_security_context("auth.py", "authenticate")

        assert "Call Graph" in context
        assert "authenticate" in context

    def test_get_attack_surface(self, agent, mock_parseltongue):
        """Test gets attack surface."""
        mock_parseltongue.get_entry_points.return_value = ["/api/users", "/api/admin"]

        surface = agent.get_attack_surface()

        assert len(surface) == 2
        assert "/api/users" in surface

    def test_analyze_impact(self, agent, mock_parseltongue):
        """Test analyzes function impact."""
        call_graph = CallGraphNode(
            function_name="authenticate",
            file_path="auth.py",
            line_number=42,
            callers=["login", "verify"],
            callees=[]
        )
        mock_parseltongue.get_call_graph.return_value = call_graph

        impact = agent.analyze_impact("authenticate")

        assert "direct_callers" in impact
        assert "login" in impact["direct_callers"]

    def test_review_code_parseltongue_unavailable(self, agent, mock_parseltongue):
        """Test review works when Parseltongue unavailable."""
        mock_parseltongue.is_available.return_value = False

        code = {
            "test.py": 'cursor.execute(f"SELECT * FROM users WHERE id={user_id}")'
        }

        findings = agent.review_code(code)

        # Should still get pattern-based findings
        assert len(findings) > 0
        # But no graph-based findings
        assert all(f.metadata.get("source") != "graph" for f in findings if f.metadata)

    def test_multiple_data_flow_sinks(self, agent, mock_parseltongue):
        """Test checks multiple dangerous sinks."""
        # Mock multiple flows for different sinks
        def mock_get_data_flow(source, sink, max_depth=10):
            if sink == "execute":
                return [DataFlowPath(source, sink, ["api", "db"], True)]
            elif sink == "eval":
                return [DataFlowPath(source, sink, ["input", "eval"], True)]
            return []

        mock_parseltongue.get_data_flow.side_effect = mock_get_data_flow

        findings = agent._check_data_flow_vulnerabilities()

        # Should find both SQL injection and code injection
        assert len(findings) == 2
        titles = [f.title for f in findings]
        assert "SQL Injection via Data Flow" in titles
        assert "Code Injection via Data Flow" in titles

    def test_integration_with_parent_patterns(self, agent, mock_parseltongue):
        """Test integration with parent CodeReviewAgent patterns."""
        # Test that parent class patterns still work
        code = {
            "test.py": """
password = "hardcoded123"
cursor.execute(f"SELECT * FROM users WHERE id={user_id}")
eval(user_input)
"""
        }

        findings = agent.review_code(code)

        # Should detect parent class patterns
        titles = [f.title for f in findings]
        assert any("Hardcoded Credentials" in t for t in titles)
        assert any("SQL Injection" in t for t in titles)
        assert any("Dangerous Code Execution" in t for t in titles)

    def test_finding_metadata_includes_source(self, agent, mock_parseltongue):
        """Test graph findings include source metadata."""
        data_flow = DataFlowPath(
            source="user",
            sink="execute",
            path=["handler", "db"],
            is_tainted=True
        )
        mock_parseltongue.get_data_flow.return_value = [data_flow]

        findings = agent._check_data_flow_vulnerabilities()

        assert findings[0].metadata is not None
        assert findings[0].metadata["source"] == "graph"
        assert "flow_path" in findings[0].metadata

    def test_empty_call_graph_handling(self, agent, mock_parseltongue):
        """Test handles None call graph gracefully."""
        mock_parseltongue.get_entry_points.return_value = ["/api/test"]
        mock_parseltongue.get_call_graph.return_value = None

        # Should not crash
        findings = agent._check_authentication_bypass()

        # Should not find issues (no call graph data)
        assert len(findings) == 0

    def test_command_injection_detection(self, agent, mock_parseltongue):
        """Test detects command injection via data flow."""
        cmd_flow = DataFlowPath(
            source="user",
            sink="system",
            path=["api", "execute_command"],
            is_tainted=True
        )
        mock_parseltongue.get_data_flow.return_value = [cmd_flow]

        findings = agent._check_data_flow_vulnerabilities()

        assert len(findings) == 1
        assert "Command Injection" in findings[0].title
        assert findings[0].severity == Severity.CRITICAL
