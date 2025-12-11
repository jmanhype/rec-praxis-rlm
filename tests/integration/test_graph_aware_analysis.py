"""Integration tests for graph-aware security analysis with Parseltongue.

These tests validate the end-to-end workflow of GraphAwareCodeReviewAgent
with mocked Parseltongue responses to simulate graph-based vulnerability detection.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from rec_praxis_rlm.agents.code_review_graph import GraphAwareCodeReviewAgent
from rec_praxis_rlm.graph import CallGraphNode, DataFlowPath
from rec_praxis_rlm.types import Finding, Severity


class TestGraphAwareIntegration:
    """Integration tests for graph-aware analysis workflows."""

    @pytest.fixture
    def mock_parseltongue(self):
        """Mock Parseltongue client for testing."""
        with patch("rec_praxis_rlm.agents.code_review_graph.ParseltongueClient") as mock:
            client = mock.return_value
            client.is_available.return_value = True
            yield client

    def test_cross_function_sql_injection_detection(self, mock_parseltongue):
        """IT-GRAPH-001: Detect SQL injection across multiple functions.

        Scenario: User input flows through api_handler → process_user → db.execute
        Expected: GraphAwareCodeReviewAgent detects cross-function SQL injection
        """
        # Setup: Mock data flow showing tainted path from user input to SQL execute
        sql_data_flow = DataFlowPath(
            source="user",
            sink="execute",
            path=["api_handler", "process_user", "db.execute"],
            is_tainted=True
        )
        mock_parseltongue.get_data_flow.return_value = [sql_data_flow]
        mock_parseltongue.get_entry_points.return_value = []
        mock_parseltongue.get_call_graph.return_value = None

        # Execute: Create agent and review code
        agent = GraphAwareCodeReviewAgent(memory_path=":memory:")

        code = {
            "app.py": """
def api_handler(user_id: str):
    return process_user(user_id)

def process_user(user_id: str):
    return db.execute(user_id)

def execute(query: str):
    cursor.execute(f"SELECT * FROM users WHERE id={query}")
"""
        }

        findings = agent.review_code(code)

        # Verify: Should detect SQL injection via data flow
        sql_findings = [f for f in findings if "SQL Injection" in f.title and "Data Flow" in f.title]
        assert len(sql_findings) > 0, "Should detect cross-function SQL injection"

        sql_finding = sql_findings[0]
        assert sql_finding.severity == Severity.CRITICAL
        assert "api_handler → process_user → db.execute" in sql_finding.description
        assert sql_finding.metadata.get("source") == "graph"
        assert "flow_path" in sql_finding.metadata

    def test_authentication_bypass_detection(self, mock_parseltongue):
        """IT-GRAPH-002: Detect public endpoints without authentication.

        Scenario: /api/delete_user is a public entry point but never calls authenticate()
        Expected: GraphAwareCodeReviewAgent flags authentication bypass
        """
        # Setup: Mock entry points and call graph
        mock_parseltongue.get_entry_points.return_value = ["/api/delete_user"]

        call_graph = CallGraphNode(
            function_name="/api/delete_user",
            file_path="api.py",
            line_number=10,
            callers=[],  # Entry point (no callers)
            callees=["delete_from_db", "log_action"]  # No auth functions
        )
        mock_parseltongue.get_call_graph.return_value = call_graph
        mock_parseltongue.get_data_flow.return_value = []

        # Execute: Create agent and review code
        agent = GraphAwareCodeReviewAgent(memory_path=":memory:")

        code = {
            "api.py": """
@app.route("/api/delete_user")
def delete_user(user_id: str):
    delete_from_db(user_id)
    log_action("deleted", user_id)
"""
        }

        findings = agent.review_code(code)

        # Verify: Should detect authentication bypass
        auth_findings = [f for f in findings if "Authentication Bypass" in f.title]
        assert len(auth_findings) > 0, "Should detect authentication bypass"

        auth_finding = auth_findings[0]
        assert auth_finding.severity == Severity.HIGH
        assert "/api/delete_user" in auth_finding.description
        assert auth_finding.metadata.get("source") == "graph"

    def test_code_injection_via_data_flow(self, mock_parseltongue):
        """IT-GRAPH-003: Detect code injection via eval/exec.

        Scenario: User input flows to eval() function
        Expected: GraphAwareCodeReviewAgent detects code injection vulnerability
        """
        # Setup: Mock data flow to eval
        eval_flow = DataFlowPath(
            source="user",
            sink="eval",
            path=["input_handler", "process_input", "eval"],
            is_tainted=True
        )
        mock_parseltongue.get_data_flow.return_value = [eval_flow]
        mock_parseltongue.get_entry_points.return_value = []
        mock_parseltongue.get_call_graph.return_value = None

        # Execute
        agent = GraphAwareCodeReviewAgent(memory_path=":memory:")

        code = {
            "processor.py": """
def input_handler(user_input: str):
    return process_input(user_input)

def process_input(data: str):
    return eval(data)  # Dangerous!
"""
        }

        findings = agent.review_code(code)

        # Verify: Should detect code injection
        code_injection_findings = [f for f in findings if "Code Injection" in f.title]
        assert len(code_injection_findings) > 0, "Should detect code injection"

        finding = code_injection_findings[0]
        assert finding.severity == Severity.CRITICAL
        assert "eval" in finding.description.lower()

    def test_command_injection_detection(self, mock_parseltongue):
        """IT-GRAPH-004: Detect command injection via os.system.

        Scenario: User input flows to os.system() or subprocess
        Expected: GraphAwareCodeReviewAgent detects command injection
        """
        # Setup: Mock data flow to system command
        cmd_flow = DataFlowPath(
            source="user",
            sink="system",
            path=["api_endpoint", "execute_command"],
            is_tainted=True
        )
        mock_parseltongue.get_data_flow.return_value = [cmd_flow]
        mock_parseltongue.get_entry_points.return_value = []
        mock_parseltongue.get_call_graph.return_value = None

        # Execute
        agent = GraphAwareCodeReviewAgent(memory_path=":memory:")

        code = {
            "admin.py": """
def api_endpoint(filename: str):
    execute_command(filename)

def execute_command(file: str):
    os.system(f"cat {file}")  # Command injection!
"""
        }

        findings = agent.review_code(code)

        # Verify: Should detect command injection
        cmd_findings = [f for f in findings if "Command Injection" in f.title]
        assert len(cmd_findings) > 0, "Should detect command injection"

        finding = cmd_findings[0]
        assert finding.severity == Severity.CRITICAL

    def test_large_attack_surface_analysis(self, mock_parseltongue):
        """IT-GRAPH-005: Detect large attack surface (many entry points).

        Scenario: Application has 25 public entry points
        Expected: GraphAwareCodeReviewAgent flags large attack surface
        """
        # Setup: Mock many entry points
        entry_points = [f"/api/endpoint_{i}" for i in range(25)]
        mock_parseltongue.get_entry_points.return_value = entry_points
        mock_parseltongue.get_call_graph.return_value = None
        mock_parseltongue.get_data_flow.return_value = []

        # Execute
        agent = GraphAwareCodeReviewAgent(memory_path=":memory:")

        code = {"api.py": "# API with 25 endpoints"}
        findings = agent.review_code(code)

        # Verify: Should flag large attack surface
        surface_findings = [f for f in findings if "Large Attack Surface" in f.title]
        assert len(surface_findings) > 0, "Should detect large attack surface"

        finding = surface_findings[0]
        assert finding.severity == Severity.MEDIUM
        assert "25" in finding.description
        assert finding.metadata.get("source") == "graph"

    def test_sanitized_data_flow_not_flagged(self, mock_parseltongue):
        """IT-GRAPH-006: Don't flag sanitized data flows.

        Scenario: Data flows through sanitize() function before reaching SQL
        Expected: No SQL injection finding (flow is not tainted)
        """
        # Setup: Mock sanitized data flow
        safe_flow = DataFlowPath(
            source="user",
            sink="execute",
            path=["api_handler", "sanitize", "db.execute"],
            is_tainted=False  # Sanitized!
        )
        mock_parseltongue.get_data_flow.return_value = [safe_flow]
        mock_parseltongue.get_entry_points.return_value = []
        mock_parseltongue.get_call_graph.return_value = None

        # Execute
        agent = GraphAwareCodeReviewAgent(memory_path=":memory:")

        code = {
            "app.py": """
def api_handler(user_id: str):
    safe_id = sanitize(user_id)
    return db.execute(safe_id)
"""
        }

        findings = agent.review_code(code)

        # Verify: Should NOT flag as SQL injection
        sql_findings = [f for f in findings if "SQL Injection" in f.title and "Data Flow" in f.title]
        assert len(sql_findings) == 0, "Should not flag sanitized data flows"

    def test_endpoint_with_authentication_not_flagged(self, mock_parseltongue):
        """IT-GRAPH-007: Don't flag endpoints that call authentication.

        Scenario: Endpoint calls authenticate() before processing
        Expected: No authentication bypass finding
        """
        # Setup: Mock call graph showing auth check
        call_graph = CallGraphNode(
            function_name="/api/protected",
            file_path="api.py",
            line_number=20,
            callers=[],
            callees=["authenticate", "process_request"]  # Has auth!
        )
        mock_parseltongue.get_entry_points.return_value = ["/api/protected"]
        mock_parseltongue.get_call_graph.return_value = call_graph
        mock_parseltongue.get_data_flow.return_value = []

        # Execute
        agent = GraphAwareCodeReviewAgent(memory_path=":memory:")

        code = {
            "api.py": """
@app.route("/api/protected")
def protected_endpoint():
    authenticate()  # Auth check present
    return process_request()
"""
        }

        findings = agent.review_code(code)

        # Verify: Should NOT flag authentication bypass
        auth_findings = [f for f in findings if "Authentication Bypass" in f.title]
        assert len(auth_findings) == 0, "Should not flag endpoints with authentication"

    def test_graceful_degradation_when_parseltongue_unavailable(self):
        """IT-GRAPH-008: Gracefully handle Parseltongue unavailability.

        Scenario: Parseltongue server is not running
        Expected: Agent falls back to pattern-based detection, no crash
        """
        # Setup: Mock Parseltongue as unavailable
        with patch("rec_praxis_rlm.agents.code_review_graph.ParseltongueClient") as mock:
            client = mock.return_value
            client.is_available.return_value = False

            # Execute
            agent = GraphAwareCodeReviewAgent(memory_path=":memory:")

            code = {
                "app.py": """
cursor.execute(f"SELECT * FROM users WHERE id={user_id}")  # Pattern-based detection
"""
            }

            findings = agent.review_code(code)

            # Verify: Should still find pattern-based issues, no crash
            assert isinstance(findings, list), "Should return findings list"
            # Should detect SQL injection via pattern matching (not graph)
            sql_findings = [f for f in findings if "SQL Injection" in f.title]
            if sql_findings:
                # If detected, should be pattern-based (no graph metadata)
                assert sql_findings[0].metadata.get("source") != "graph"

    def test_deduplication_of_graph_and_pattern_findings(self, mock_parseltongue):
        """IT-GRAPH-009: Deduplicate findings from graph and pattern analysis.

        Scenario: Same SQL injection detected by both graph and pattern analysis
        Expected: Only one finding reported (deduplication works)
        """
        # Setup: Mock data flow that will also trigger pattern detection
        sql_flow = DataFlowPath(
            source="user",
            sink="execute",
            path=["handler", "execute"],
            is_tainted=True
        )
        mock_parseltongue.get_data_flow.return_value = [sql_flow]
        mock_parseltongue.get_entry_points.return_value = []
        mock_parseltongue.get_call_graph.return_value = None

        # Execute
        agent = GraphAwareCodeReviewAgent(memory_path=":memory:")

        code = {
            "app.py": """
def handler(user_id):
    cursor.execute(f"SELECT * FROM users WHERE id={user_id}")  # Both graph + pattern should detect
"""
        }

        findings = agent.review_code(code)

        # Verify: Should have findings, but deduplicated
        sql_findings = [f for f in findings if "SQL" in f.title]

        # Count findings at same location with same title
        unique_locations = set((f.file_path, f.line_number, f.title) for f in sql_findings)

        # Should have deduplication (no exact duplicates)
        assert len(sql_findings) == len(unique_locations), "Should deduplicate findings"

    def test_procedural_memory_stores_graph_patterns(self, mock_parseltongue):
        """IT-GRAPH-010: Store graph-based findings in procedural memory.

        Scenario: Agent detects graph-based vulnerability
        Expected: Finding is stored in procedural memory for future learning
        """
        # Setup
        sql_flow = DataFlowPath(
            source="user",
            sink="execute",
            path=["api", "process", "execute"],
            is_tainted=True
        )
        mock_parseltongue.get_data_flow.return_value = [sql_flow]
        mock_parseltongue.get_entry_points.return_value = []
        mock_parseltongue.get_call_graph.return_value = None

        # Execute
        agent = GraphAwareCodeReviewAgent(memory_path=":memory:")

        code = {"app.py": "cursor.execute(f'SELECT * FROM users WHERE id={user_id}')"}
        findings = agent.review_code(code)

        # Verify: Memory should have stored the graph pattern
        assert agent.memory.size() > 0, "Should store findings in memory"

        # Recall should find graph-related experiences
        experiences = agent.memory.recall(
            env_features=["graph_finding"],
            goal="detect sql injection",
            top_k=5
        )

        assert len(experiences) > 0, "Should recall graph-based patterns"


class TestGraphAwareSecurityContext:
    """Integration tests for graph-based security context."""

    @pytest.fixture
    def mock_parseltongue(self):
        """Mock Parseltongue client."""
        with patch("rec_praxis_rlm.agents.code_review_graph.ParseltongueClient") as mock:
            client = mock.return_value
            client.is_available.return_value = True
            yield client

    def test_get_security_context_for_function(self, mock_parseltongue):
        """IT-GRAPH-011: Get security context for a specific function.

        Scenario: Request security context for authenticate() function
        Expected: Returns call graph and data flow information
        """
        # Setup
        call_graph = CallGraphNode(
            function_name="authenticate",
            file_path="auth.py",
            line_number=42,
            callers=["login", "verify"],
            callees=["check_password", "validate_token"]
        )
        mock_parseltongue.get_call_graph.return_value = call_graph
        mock_parseltongue.get_data_flow.return_value = []

        # Execute
        agent = GraphAwareCodeReviewAgent(memory_path=":memory:")
        context = agent.get_security_context("auth.py", "authenticate")

        # Verify
        assert isinstance(context, str)
        assert "authenticate" in context
        assert "Call Graph" in context or "call graph" in context
        assert len(context) > 0

    def test_get_attack_surface(self, mock_parseltongue):
        """IT-GRAPH-012: Get application attack surface.

        Scenario: Request list of public entry points
        Expected: Returns all API routes and entry points
        """
        # Setup
        entry_points = ["/api/users", "/api/admin", "/api/data"]
        mock_parseltongue.get_entry_points.return_value = entry_points

        # Execute
        agent = GraphAwareCodeReviewAgent(memory_path=":memory:")
        surface = agent.get_attack_surface()

        # Verify
        assert isinstance(surface, list)
        assert len(surface) == 3
        assert "/api/users" in surface
        assert "/api/admin" in surface

    def test_analyze_impact(self, mock_parseltongue):
        """IT-GRAPH-013: Analyze impact of function changes.

        Scenario: Analyze what functions call a critical function
        Expected: Returns list of callers (impact scope)
        """
        # Setup
        call_graph = CallGraphNode(
            function_name="critical_function",
            file_path="core.py",
            line_number=100,
            callers=["api_handler1", "api_handler2", "background_task"],
            callees=[]
        )
        mock_parseltongue.get_call_graph.return_value = call_graph

        # Execute
        agent = GraphAwareCodeReviewAgent(memory_path=":memory:")
        impact = agent.analyze_impact("critical_function")

        # Verify
        assert isinstance(impact, dict)
        assert "direct_callers" in impact
        assert len(impact["direct_callers"]) == 3
        assert "api_handler1" in impact["direct_callers"]
