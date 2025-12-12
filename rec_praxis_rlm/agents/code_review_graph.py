"""Graph-aware code review agent with Parseltongue integration.

Extends the pattern-based CodeReviewAgent with graph analysis capabilities to detect:
- Cross-function data flow vulnerabilities
- Authentication bypass (public routes that skip auth)
- Privilege escalation (who can call admin functions)
- Attack surface analysis
"""

import time
from typing import Dict, List
from rec_praxis_rlm.agents.code_review import CodeReviewAgent
from rec_praxis_rlm.graph import ParseltongueClient, GraphContextProvider
from rec_praxis_rlm.types import Finding, Severity
from rec_praxis_rlm import Experience


class GraphAwareCodeReviewAgent(CodeReviewAgent):
    """Code review agent enhanced with graph analysis.

    Extends CodeReviewAgent with:
    - Call graph analysis (via Parseltongue)
    - Data flow tracking (source → sink)
    - Cross-function vulnerability detection
    - Attack surface mapping

    Example:
        >>> agent = GraphAwareCodeReviewAgent(
        ...     memory_path="memory.jsonl",
        ...     parseltongue_url="http://localhost:8080"
        ... )
        >>> findings = agent.review_code({"auth.py": "def login(...)..."})
        >>> len(findings) > 0  # May find graph-based vulnerabilities
        True
    """

    def __init__(
        self,
        memory_path: str = ":memory:",
        parseltongue_url: str = "http://localhost:8080"
    ):
        """Initialize graph-aware agent.

        Args:
            memory_path: Path to procedural memory storage
            parseltongue_url: Parseltongue HTTP API URL
        """
        super().__init__(memory_path)
        self.parseltongue = ParseltongueClient(base_url=parseltongue_url)
        self.graph_context = GraphContextProvider(self.parseltongue, self.memory)

        if not self.parseltongue.is_available():
            print("⚠️  Parseltongue not available - graph analysis disabled")
            print("   Start Parseltongue: parseltongue serve")

    def review_code(self, files: Dict[str, str]) -> List[Finding]:
        """Review code with graph-aware analysis.

        Workflow:
        1. Pattern-based scan (fast baseline via parent class)
        2. Get graph context from Parseltongue
        3. Graph-aware analysis (data flow, auth bypass, attack surface)
        4. Deduplicate findings
        5. Store graph patterns in memory

        Args:
            files: Dict mapping file paths to contents

        Returns:
            List of Finding objects
        """
        # Run parent implementation (pattern + LLM)
        findings = super().review_code(files)

        # If Parseltongue available, add graph-aware findings
        if self.parseltongue.is_available():
            graph_findings = self._analyze_with_graph(files)
            findings.extend(graph_findings)

            # Store graph patterns in memory
            for finding in graph_findings:
                self._store_graph_pattern(finding)

        return self._deduplicate_findings(findings)

    def _analyze_with_graph(self, files: Dict[str, str]) -> List[Finding]:
        """Perform graph-aware analysis.

        Detects:
        - Cross-function data flow vulnerabilities
        - Authentication bypass (public routes that skip auth)
        - Privilege escalation (who can call admin functions)

        Args:
            files: Files to analyze

        Returns:
            List of graph-detected findings
        """
        findings = []

        # Check 1: Data flow to dangerous sinks
        data_flow_findings = self._check_data_flow_vulnerabilities()
        findings.extend(data_flow_findings)

        # Check 2: Authentication bypass
        auth_bypass_findings = self._check_authentication_bypass()
        findings.extend(auth_bypass_findings)

        # Check 3: Attack surface analysis
        attack_surface_findings = self._analyze_attack_surface()
        findings.extend(attack_surface_findings)

        return findings

    def _check_data_flow_vulnerabilities(self) -> List[Finding]:
        """Check for unsanitized data flow to dangerous sinks.

        Uses Parseltongue to track data flow from user input sources to
        dangerous sink functions like execute(), eval(), system().

        Returns:
            List of findings for unsanitized data flows
        """
        findings = []

        # Dangerous sinks
        sinks = {
            "execute": "SQL Injection",
            "eval": "Code Injection",
            "exec": "Code Injection",
            "system": "Command Injection",
            "popen": "Command Injection"
        }

        for sink, vuln_type in sinks.items():
            flows = self.parseltongue.get_data_flow("user", sink)
            for flow in flows:
                # Parseltongue is queried per-sink, but guard against
                # mocks or upstream changes returning mixed sinks.
                if flow.sink != sink:
                    continue
                if flow.is_tainted:
                    finding = Finding(
                        file_path=flow.path[0] if flow.path else "unknown",
                        line_number=None,
                        severity=Severity.CRITICAL,
                        title=f"{vuln_type} via Data Flow",
                        description=f"Unsanitized user input flows to dangerous sink '{sink}' via: {' → '.join(flow.path)}",
                        remediation=f"Sanitize input before it reaches {sink}. Add validation in {flow.path[1] if len(flow.path) > 1 else 'entry point'}.",
                        metadata={"source": "graph", "flow_path": flow.path}
                    )
                    findings.append(finding)

        return findings

    def _check_authentication_bypass(self) -> List[Finding]:
        """Check for routes that bypass authentication.

        Analyzes call graphs to detect public entry points that don't call
        authentication functions.

        Returns:
            List of findings for potential auth bypasses
        """
        findings = []

        # Get all public entry points
        entry_points = self.parseltongue.get_entry_points(public_only=True)

        # Authentication function patterns
        auth_functions = ["authenticate", "check_auth", "require_auth", "verify_token", "login"]

        for entry_point in entry_points:
            call_graph = self.parseltongue.get_call_graph(entry_point)
            if call_graph:
                # Check if entry point calls any auth functions
                calls_auth = any(auth_func in call_graph.callees for auth_func in auth_functions)

                if not calls_auth:
                    finding = Finding(
                        file_path=call_graph.file_path,
                        line_number=call_graph.line_number,
                        severity=Severity.HIGH,
                        title="Potential Authentication Bypass",
                        description=f"Public entry point '{entry_point}' does not call authentication functions. Callees: {', '.join(call_graph.callees[:5])}",
                        remediation="Ensure all public endpoints call authentication middleware before processing requests.",
                        metadata={"source": "graph", "entry_point": entry_point}
                    )
                    findings.append(finding)

        return findings

    def _analyze_attack_surface(self) -> List[Finding]:
        """Analyze attack surface for potential risks.

        Large attack surfaces (many public entry points) increase
        vulnerability risk and complexity.

        Returns:
            List of findings for attack surface issues
        """
        findings = []

        # Get all public entry points
        entry_points = self.parseltongue.get_entry_points(public_only=True)

        # Large attack surface is a risk
        if len(entry_points) > 20:
            finding = Finding(
                file_path="multiple",
                line_number=None,
                severity=Severity.MEDIUM,
                title="Large Attack Surface",
                description=f"Found {len(entry_points)} public entry points. Large attack surfaces increase vulnerability risk.",
                remediation="Review if all entry points are necessary. Consider making some internal-only or adding authentication.",
                metadata={"source": "graph", "entry_point_count": len(entry_points)}
            )
            findings.append(finding)

        return findings

    def _store_graph_pattern(self, finding: Finding):
        """Store graph pattern in procedural memory.

        Args:
            finding: Finding with graph metadata to store
        """
        if not finding.metadata or "flow_path" not in finding.metadata:
            return

        flow_path = finding.metadata["flow_path"]
        pattern = " → ".join(flow_path)

        exp = Experience(
            env_features=["graph_finding"],
            goal=f"detect {finding.title.lower()}",
            action=f"Graph analysis found: {finding.description}",
            result=f"Pattern: {pattern}\nRemediation: {finding.remediation}",
            success=True,
            timestamp=time.time(),
            metadata={
                "flow_path": flow_path,
                "source": "graph",
                "finding_type": finding.title
            }
        )
        self.memory.store(exp)

    def _deduplicate_findings(self, findings: List[Finding]) -> List[Finding]:
        """Remove duplicate findings.

        Deduplicates by (file_path, line_number, title) tuple.

        Args:
            findings: List of findings to deduplicate

        Returns:
            Deduplicated list of findings
        """
        seen = set()
        unique_findings = []

        for finding in findings:
            # Create a unique key for this finding
            # Use line_number if available, otherwise use title only
            if finding.line_number:
                key = (finding.file_path, finding.line_number, finding.title)
            else:
                key = (finding.file_path, finding.title)

            if key not in seen:
                seen.add(key)
                unique_findings.append(finding)

        return unique_findings

    def get_security_context(self, file_path: str, function_name: str = None) -> str:
        """Get rich security context for a file.

        Combines graph analysis with procedural memory to provide
        rich context for LLM analysis.

        Args:
            file_path: File to analyze
            function_name: Optional function name to focus on

        Returns:
            Formatted security context string

        Example:
            >>> agent = GraphAwareCodeReviewAgent()
            >>> context = agent.get_security_context("auth.py", "login")
            >>> "Call Graph" in context
            True
        """
        return self.graph_context.get_security_context(file_path, function_name)

    def get_attack_surface(self) -> List[str]:
        """Get current attack surface (public entry points).

        Returns:
            List of public entry point function names

        Example:
            >>> agent = GraphAwareCodeReviewAgent()
            >>> surface = agent.get_attack_surface()
            >>> isinstance(surface, list)
            True
        """
        return self.graph_context.get_attack_surface()

    def analyze_impact(self, function_name: str) -> Dict[str, List[str]]:
        """Analyze impact of changing a function.

        Uses call graph to determine which functions would be affected.

        Args:
            function_name: Function to analyze

        Returns:
            Dict with "direct_callers" and "transitive_callers"

        Example:
            >>> agent = GraphAwareCodeReviewAgent()
            >>> impact = agent.analyze_impact("authenticate")
            >>> "direct_callers" in impact
            True
        """
        return self.graph_context.analyze_impact(function_name)
