"""Provides rich graph context for LLM analysis.

This module combines graph analysis from Parseltongue with procedural memory
to create rich security context for LLM-powered code review.
"""

from typing import Dict, List, Optional, Any
from rec_praxis_rlm.graph.parseltongue_client import (
    ParseltongueClient,
    CallGraphNode,
    DataFlowPath
)
from rec_praxis_rlm.memory import ProceduralMemory, Experience


class GraphContextProvider:
    """Combines graph analysis with procedural memory for rich context.

    This class queries Parseltongue for structural context (call graphs,
    data flow) and procedural memory for behavioral context (past findings,
    learned patterns), then combines them into a rich prompt for LLM analysis.

    Example:
        >>> from rec_praxis_rlm.graph import ParseltongueClient
        >>> from rec_praxis_rlm.memory import ProceduralMemory
        >>> from rec_praxis_rlm.config import MemoryConfig
        >>>
        >>> parseltongue = ParseltongueClient("http://localhost:8080")
        >>> memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
        >>> provider = GraphContextProvider(parseltongue, memory)
        >>>
        >>> # Get rich security context for a file
        >>> context = provider.get_security_context("auth.py", "authenticate")
        >>> print(context)  # Rich context with call graph + memory
    """

    def __init__(
        self,
        parseltongue: ParseltongueClient,
        memory: ProceduralMemory
    ):
        """Initialize graph context provider.

        Args:
            parseltongue: Parseltongue client for graph queries
            memory: Procedural memory for past experiences
        """
        self.parseltongue = parseltongue
        self.memory = memory

    def get_security_context(
        self,
        file_path: str,
        function_name: Optional[str] = None
    ) -> str:
        """Get rich security context for LLM analysis.

        Combines:
        1. Call graph (who calls this, what does it call)
        2. Data flow (where does user input flow)
        3. Past experiences (similar vulnerabilities we've seen)
        4. Learned patterns (what patterns caught similar issues)

        Args:
            file_path: File being analyzed
            function_name: Optional function name to focus on

        Returns:
            Rich context string for LLM prompt

        Example:
            >>> context = provider.get_security_context("api.py", "api_handler")
            >>> "Call Graph" in context
            True
            >>> "Data Flow" in context
            True
        """
        context_parts = []

        # Part 1: Call graph context
        if function_name and self.parseltongue.is_available():
            call_graph = self.parseltongue.get_call_graph(function_name, file_path)
            if call_graph:
                context_parts.append(self._format_call_graph_context(call_graph))

        # Part 2: Data flow context
        if self.parseltongue.is_available():
            # Common dangerous sinks
            sinks = ["execute", "eval", "exec", "system", "popen"]
            data_flows = []
            for sink in sinks:
                flows = self.parseltongue.get_data_flow("user", sink)
                data_flows.extend(flows)

            if data_flows:
                context_parts.append(self._format_data_flow_context(data_flows))

        # Part 3: Memory context (past experiences)
        past_experiences = self.memory.recall(
            env_features=["python", "security", file_path],
            goal="detect vulnerabilities",
            top_k=5
        )
        if past_experiences:
            context_parts.append(self._format_memory_context(past_experiences))

        # Combine all context
        if context_parts:
            return "\n\n".join(["## Graph and Memory Context"] + context_parts)
        return ""

    def _format_call_graph_context(self, call_graph: CallGraphNode) -> str:
        """Format call graph into readable context.

        Args:
            call_graph: Call graph node to format

        Returns:
            Formatted call graph context string
        """
        lines = [f"### Call Graph for {call_graph.function_name}"]
        lines.append(f"**Location:** {call_graph.file_path}:{call_graph.line_number}")

        if call_graph.callers:
            lines.append(f"**Called by:** {', '.join(call_graph.callers)}")
        else:
            lines.append("**Called by:** ⚠️ Entry point (no callers - directly accessible)")

        if call_graph.callees:
            lines.append(f"**Calls:** {', '.join(call_graph.callees)}")

        return "\n".join(lines)

    def _format_data_flow_context(self, data_flows: List[DataFlowPath]) -> str:
        """Format data flow paths into readable context.

        Args:
            data_flows: List of data flow paths to format

        Returns:
            Formatted data flow context string
        """
        lines = ["### Data Flow Analysis"]

        for flow in data_flows:
            path_str = " → ".join(flow.path)
            taint_status = "✅ SANITIZED" if not flow.is_tainted else "⚠️ UNSANITIZED"
            lines.append(f"- {flow.source} → {flow.sink}: {path_str} ({taint_status})")

        return "\n".join(lines)

    def _format_memory_context(self, experiences: List[Experience]) -> str:
        """Format past experiences into readable context.

        Args:
            experiences: List of past experiences to format

        Returns:
            Formatted memory context string
        """
        lines = ["### Similar Past Findings"]

        for exp in experiences[:3]:  # Top 3 most relevant
            lines.append(f"- **Goal:** {exp.goal}")

            # Truncate action if too long
            action_preview = exp.action[:200] + "..." if len(exp.action) > 200 else exp.action
            lines.append(f"  **Action:** {action_preview}")

            if exp.metadata and "pattern" in exp.metadata:
                lines.append(f"  **Pattern:** {exp.metadata['pattern']}")

        return "\n".join(lines)

    def get_attack_surface(self) -> List[str]:
        """Get attack surface (public entry points).

        Returns:
            List of public function names that can be reached from outside

        Example:
            >>> entry_points = provider.get_attack_surface()
            >>> "/api/users" in entry_points
            True
        """
        if not self.parseltongue.is_available():
            return []

        return self.parseltongue.get_entry_points(public_only=True)

    def analyze_impact(self, function_name: str) -> Dict[str, List[str]]:
        """Analyze impact of changing a function.

        Uses call graph to determine which functions would be affected if
        the target function is modified or removed.

        Args:
            function_name: Function to analyze

        Returns:
            Dict with "direct_callers" and "transitive_callers"

        Example:
            >>> impact = provider.analyze_impact("authenticate")
            >>> "direct_callers" in impact
            True
            >>> "transitive_callers" in impact
            True
        """
        if not self.parseltongue.is_available():
            return {"direct_callers": [], "transitive_callers": []}

        call_graph = self.parseltongue.get_call_graph(function_name)
        if not call_graph:
            return {"direct_callers": [], "transitive_callers": []}

        # Direct callers
        direct_callers = call_graph.callers

        # Transitive callers (BFS through call graph)
        transitive_callers = self._find_transitive_callers(function_name, direct_callers)

        return {
            "direct_callers": direct_callers,
            "transitive_callers": transitive_callers
        }

    def _find_transitive_callers(
        self,
        root_function: str,
        direct_callers: List[str],
        max_depth: int = 5
    ) -> List[str]:
        """Find transitive callers using BFS.

        Args:
            root_function: Starting function
            direct_callers: Known direct callers
            max_depth: Maximum depth to traverse (default: 5)

        Returns:
            List of transitive caller function names
        """
        transitive = []
        visited = {root_function}  # Track visited to avoid cycles
        queue = [(caller, 0) for caller in direct_callers]  # (function, depth)

        while queue:
            func, depth = queue.pop(0)

            if func in visited or depth >= max_depth:
                continue

            visited.add(func)

            # Get call graph for this function
            call_graph = self.parseltongue.get_call_graph(func)
            if call_graph and call_graph.callers:
                # Add callers to transitive list
                for caller in call_graph.callers:
                    if caller not in visited:
                        transitive.append(caller)
                        queue.append((caller, depth + 1))

        return transitive

    def get_data_flow_paths(
        self,
        source: str,
        sink: str,
        max_depth: int = 10
    ) -> List[DataFlowPath]:
        """Get data flow paths from source to sink.

        Wrapper around Parseltongue's data flow analysis.

        Args:
            source: Source function/variable (e.g., "user_input")
            sink: Sink function/variable (e.g., "execute")
            max_depth: Maximum search depth (default: 10)

        Returns:
            List of data flow paths

        Example:
            >>> flows = provider.get_data_flow_paths("user_input", "execute")
            >>> for flow in flows:
            ...     if flow.is_tainted:
            ...         print(f"Tainted: {' → '.join(flow.path)}")
        """
        if not self.parseltongue.is_available():
            return []

        return self.parseltongue.get_data_flow(source, sink, max_depth)

    def find_vulnerability_patterns(
        self,
        vulnerability_type: str
    ) -> List[Experience]:
        """Find past experiences with similar vulnerability patterns.

        Searches procedural memory for experiences tagged with the given
        vulnerability type.

        Args:
            vulnerability_type: Type of vulnerability (e.g., "sql_injection")

        Returns:
            List of past experiences with similar patterns

        Example:
            >>> patterns = provider.find_vulnerability_patterns("sql_injection")
            >>> len(patterns) > 0
            True
        """
        # Query memory for similar vulnerability experiences
        return self.memory.recall(
            env_features=["security", vulnerability_type],
            goal=f"detect {vulnerability_type}",
            top_k=10
        )

    def enrich_finding_with_context(
        self,
        finding: Dict[str, Any],
        include_graph: bool = True,
        include_memory: bool = True
    ) -> Dict[str, Any]:
        """Enrich a finding with graph and memory context.

        Takes a raw finding dictionary and adds contextual information from
        graph analysis and procedural memory.

        Args:
            finding: Finding dictionary with keys like file_path, line_number, etc.
            include_graph: Include graph context (call graph, data flow)
            include_memory: Include memory context (past similar findings)

        Returns:
            Enriched finding with additional context

        Example:
            >>> raw_finding = {
            ...     "file_path": "auth.py",
            ...     "title": "SQL Injection",
            ...     "function_name": "authenticate"
            ... }
            >>> enriched = provider.enrich_finding_with_context(raw_finding)
            >>> "graph_context" in enriched or "memory_context" in enriched
            True
        """
        enriched = finding.copy()

        # Add graph context
        if include_graph and self.parseltongue.is_available():
            function_name = finding.get("function_name")
            file_path = finding.get("file_path")

            if function_name:
                call_graph = self.parseltongue.get_call_graph(function_name, file_path)
                if call_graph:
                    enriched["graph_context"] = {
                        "callers": call_graph.callers,
                        "callees": call_graph.callees,
                        "is_entry_point": len(call_graph.callers) == 0
                    }

        # Add memory context
        if include_memory:
            vuln_type = finding.get("title", "").lower().replace(" ", "_")
            past_patterns = self.find_vulnerability_patterns(vuln_type)

            if past_patterns:
                enriched["memory_context"] = {
                    "similar_findings_count": len(past_patterns),
                    "patterns": [
                        exp.metadata.get("pattern", "")
                        for exp in past_patterns[:3]
                        if exp.metadata and "pattern" in exp.metadata
                    ]
                }

        return enriched
