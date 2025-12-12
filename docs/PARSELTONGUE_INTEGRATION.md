# Parseltongue Integration for Graph-Aware Security Analysis

## Overview

Integrate [Parseltongue](https://github.com/that-in-rust/parseltongue-dependency-graph-generator) (Rust-based dependency graph generator) with rec-praxis-rlm to enable **graph-aware security analysis** with procedural memory.

**Value Proposition:**
- Detect cross-function vulnerabilities (data flow, authentication bypass, privilege escalation)
- Multi-language support (12 languages via Parseltongue)
- Unique combination: **Graph analysis + Procedural memory + LLM reasoning**

**Competitive Advantage:**
- CodeQL: Has graphs, no learning/memory
- Semgrep: No graphs, no learning
- **rec-praxis-rlm + Parseltongue: Graphs + learning + LLM = best of all worlds**

---

## Architecture

### Integration Model: Client-Server (HTTP API)

**Why HTTP API?**
- ✅ Loose coupling (can upgrade independently)
- ✅ Language agnostic (Rust ↔ Python)
- ✅ Optional dependency (works without Parseltongue)
- ✅ Parseltongue already designed as HTTP service

**Alternatives Rejected:**
- ❌ Embedded (PyO3): Tight coupling, hard to maintain
- ❌ Shared DB: Schema coupling, version conflicts

###Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│ Step 1: Parse Codebase (One-time or on-demand)               │
├──────────────────────────────────────────────────────────────┤
│  $ parseltongue parse /path/to/code                          │
│  → tree-sitter parses 12 languages                           │
│  → Stores entities + dependencies in CozoDB                  │
│  → HTTP API ready at localhost:8080                          │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 2: rec-praxis-rlm Scans File                            │
├──────────────────────────────────────────────────────────────┤
│  $ rec-praxis-review auth.py --use-llm --use-graph           │
│                                                               │
│  1. Pattern scan (fast baseline)                             │
│  2. Query Parseltongue for context:                          │
│     GET /call-graph?function=authenticate                    │
│     GET /data-flow?source=user_input&sink=sql_execute        │
│  3. LLM analysis with graph context                          │
│  4. Store findings + graph metadata in memory                │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 3: Graph-Aware Findings                                 │
├──────────────────────────────────────────────────────────────┤
│  Finding: SQL Injection                                      │
│  File: database.py:42                                        │
│  Severity: CRITICAL                                          │
│  Description: User input flows to SQL query via call chain:  │
│    api_handler() → process_request() → execute_query()       │
│  Graph Context:                                              │
│    - Entry point: /api/users (public endpoint)               │
│    - Sanitization: None found in call chain                  │
│    - Similar patterns: Found in auth.py (closed CVE-2023-X)  │
└──────────────────────────────────────────────────────────────┘
```

---

## Component Design

### Component 1: Parseltongue Client

**File:** `rec_praxis_rlm/graph/parseltongue_client.py`

```python
"""HTTP client for Parseltongue graph query API."""

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

    Repository: https://github.com/that-in-rust/parseltongue-dependency-graph-generator
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
        except requests.RequestException:
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
        except requests.RequestException:
            return None

    def get_data_flow(self, source: str, sink: str, max_depth: int = 10) -> List[DataFlowPath]:
        """Find data flow paths from source to sink.

        Args:
            source: Source function/variable (e.g., "user_input", "request.params")
            sink: Sink function/variable (e.g., "db.execute", "eval")
            max_depth: Maximum depth to search (default: 10)

        Returns:
            List of data flow paths from source to sink
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
        except requests.RequestException:
            return []

    def get_entry_points(self, public_only: bool = True) -> List[str]:
        """Get all entry points (e.g., API routes, public functions).

        Args:
            public_only: Only return public entry points (default: True)

        Returns:
            List of entry point function names
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
        except requests.RequestException:
            return []

    def find_function_references(self, function_name: str) -> List[Dict[str, str]]:
        """Find all references to a function.

        Args:
            function_name: Function to find references for

        Returns:
            List of {file_path, line_number, context} dicts
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
        except requests.RequestException:
            return []
```

---

### Component 2: Graph Context Provider

**File:** `rec_praxis_rlm/graph/context_provider.py`

```python
"""Provides rich graph context for LLM analysis."""

from typing import Dict, List, Optional
from rec_praxis_rlm.graph.parseltongue_client import ParseltongueClient, CallGraphNode, DataFlowPath
from rec_praxis_rlm.memory import ProceduralMemory


class GraphContextProvider:
    """Combines graph analysis with procedural memory for rich context.

    This class queries Parseltongue for structural context (call graphs,
    data flow) and procedural memory for behavioral context (past findings,
    learned patterns), then combines them into a rich prompt for LLM analysis.
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
        """Format call graph into readable context."""
        lines = [f"### Call Graph for {call_graph.function_name}"]

        if call_graph.callers:
            lines.append(f"**Called by:** {', '.join(call_graph.callers)}")
        else:
            lines.append("**Called by:** ⚠️ Entry point (no callers - directly accessible)")

        if call_graph.callees:
            lines.append(f"**Calls:** {', '.join(call_graph.callees)}")

        return "\n".join(lines)

    def _format_data_flow_context(self, data_flows: List[DataFlowPath]) -> str:
        """Format data flow paths into readable context."""
        lines = ["### Data Flow Analysis"]

        for flow in data_flows:
            path_str = " → ".join(flow.path)
            taint_status = "✅ SANITIZED" if not flow.is_tainted else "⚠️ UNSANITIZED"
            lines.append(f"- {flow.source} → {flow.sink}: {path_str} ({taint_status})")

        return "\n".join(lines)

    def _format_memory_context(self, experiences) -> str:
        """Format past experiences into readable context."""
        lines = ["### Similar Past Findings"]

        for exp in experiences[:3]:  # Top 3 most relevant
            lines.append(f"- {exp.goal}: {exp.action[:100]}...")
            if exp.metadata and "pattern" in exp.metadata:
                lines.append(f"  Pattern: {exp.metadata['pattern']}")

        return "\n".join(lines)

    def get_attack_surface(self) -> List[str]:
        """Get attack surface (public entry points).

        Returns:
            List of public function names that can be reached from outside
        """
        if not self.parseltongue.is_available():
            return []

        return self.parseltongue.get_entry_points(public_only=True)

    def analyze_impact(self, function_name: str) -> Dict[str, List[str]]:
        """Analyze impact of changing a function.

        Args:
            function_name: Function to analyze

        Returns:
            Dict with "direct_callers" and "transitive_callers"
        """
        if not self.parseltongue.is_available():
            return {"direct_callers": [], "transitive_callers": []}

        call_graph = self.parseltongue.get_call_graph(function_name)
        if not call_graph:
            return {"direct_callers": [], "transitive_callers": []}

        # TODO: Implement transitive caller detection (BFS on graph)
        return {
            "direct_callers": call_graph.callers,
            "transitive_callers": []  # Placeholder for future implementation
        }
```

---

### Component 3: Graph-Aware LLM Agent

**File:** `rec_praxis_rlm/agents/code_review_graph.py`

```python
"""Graph-aware code review agent with Parseltongue integration."""

from typing import Dict, List, Optional
from rec_praxis_rlm.agents.code_review_llm import CodeReviewAgentLLM
from rec_praxis_rlm.graph.parseltongue_client import ParseltongueClient
from rec_praxis_rlm.graph.context_provider import GraphContextProvider
from rec_praxis_rlm.types import Finding


class GraphAwareCodeReviewAgent(CodeReviewAgentLLM):
    """Code review agent enhanced with graph analysis.

    Extends CodeReviewAgentLLM with:
    - Call graph analysis (via Parseltongue)
    - Data flow tracking (source → sink)
    - Cross-function vulnerability detection
    - Attack surface mapping
    """

    def __init__(
        self,
        memory_path: str,
        planner,
        parseltongue_url: str = "http://localhost:8080"
    ):
        """Initialize graph-aware agent.

        Args:
            memory_path: Path to procedural memory storage
            planner: PraxisRLMPlanner instance for LLM analysis
            parseltongue_url: Parseltongue HTTP API URL
        """
        super().__init__(memory_path, planner)
        self.parseltongue = ParseltongueClient(base_url=parseltongue_url)
        self.graph_context = GraphContextProvider(self.parseltongue, self.memory)

        if not self.parseltongue.is_available():
            print("⚠️  Parseltongue not available - graph analysis disabled")
            print("   Start Parseltongue: parseltongue serve")

    def review_code(self, files: Dict[str, str]) -> List[Finding]:
        """Review code with graph-aware analysis.

        Workflow:
        1. Pattern-based scan (fast baseline)
        2. Get graph context from Parseltongue
        3. LLM analysis with enriched context
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
        """Check for unsanitized data flow to dangerous sinks."""
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
                if flow.is_tainted:
                    finding = Finding(
                        file_path=flow.path[0] if flow.path else "unknown",
                        line_number=None,
                        severity="CRITICAL",
                        title=f"{vuln_type} via Data Flow",
                        description=f"Unsanitized user input flows to dangerous sink '{sink}' via: {' → '.join(flow.path)}",
                        remediation=f"Sanitize input before it reaches {sink}. Add validation in {flow.path[1] if len(flow.path) > 1 else 'entry point'}.",
                        metadata={"source": "graph", "flow_path": flow.path}
                    )
                    findings.append(finding)

        return findings

    def _check_authentication_bypass(self) -> List[Finding]:
        """Check for routes that bypass authentication."""
        findings = []

        # Get all public entry points
        entry_points = self.parseltongue.get_entry_points(public_only=True)

        # Check if they call auth functions
        auth_functions = ["authenticate", "check_auth", "require_auth", "verify_token"]

        for entry_point in entry_points:
            call_graph = self.parseltongue.get_call_graph(entry_point)
            if call_graph:
                calls_auth = any(auth_func in call_graph.callees for auth_func in auth_functions)

                if not calls_auth:
                    finding = Finding(
                        file_path=call_graph.file_path,
                        line_number=call_graph.line_number,
                        severity="HIGH",
                        title="Potential Authentication Bypass",
                        description=f"Public entry point '{entry_point}' does not call authentication functions. Callees: {', '.join(call_graph.callees)}",
                        remediation="Ensure all public endpoints call authentication middleware.",
                        metadata={"source": "graph", "entry_point": entry_point}
                    )
                    findings.append(finding)

        return findings

    def _analyze_attack_surface(self) -> List[Finding]:
        """Analyze attack surface for potential risks."""
        findings = []

        # Get all public entry points
        entry_points = self.parseltongue.get_entry_points(public_only=True)

        # Large attack surface is a risk
        if len(entry_points) > 20:
            finding = Finding(
                file_path="multiple",
                line_number=None,
                severity="MEDIUM",
                title="Large Attack Surface",
                description=f"Found {len(entry_points)} public entry points. Large attack surfaces increase vulnerability risk.",
                remediation="Review if all entry points are necessary. Consider making some internal-only.",
                metadata={"source": "graph", "entry_point_count": len(entry_points)}
            )
            findings.append(finding)

        return findings

    def _store_graph_pattern(self, finding: Finding):
        """Store graph pattern in procedural memory."""
        if not finding.metadata or "flow_path" not in finding.metadata:
            return

        from rec_praxis_rlm import Experience
        import time

        flow_path = finding.metadata["flow_path"]
        pattern = " → ".join(flow_path)

        exp = Experience(
            env_features=["python", "graph_finding", finding.severity.lower()],
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
```

---

## CLI Integration

### Updated CLI Commands

**File:** `rec_praxis_rlm/cli.py`

Add new flag to code review:

```python
def cli_code_review() -> int:
    parser = argparse.ArgumentParser(description="Run code review on staged files")
    # ... existing arguments ...

    # NEW: Graph analysis flag
    parser.add_argument("--use-graph", action="store_true",
                       help="Enable graph-aware analysis via Parseltongue (requires Parseltongue running)")
    parser.add_argument("--parseltongue-url", default="http://localhost:8080",
                       help="Parseltongue HTTP API URL (default: http://localhost:8080)")

    args = parser.parse_args()

    # ... existing code ...

    # NEW: Use graph-aware agent if requested
    if args.use_graph:
        from rec_praxis_rlm.agents.code_review_graph import GraphAwareCodeReviewAgent
        agent = GraphAwareCodeReviewAgent(
            memory_path=str(memory_dir / "code_review_memory.jsonl"),
            planner=planner,  # Only if --use-llm
            parseltongue_url=args.parseltongue_url
        )
        print(f"🔗 Using graph-aware analysis (Parseltongue: {args.parseltongue_url})")
    else:
        # Use existing agent
        agent = CodeReviewAgent(memory_path=str(memory_dir / "code_review_memory.jsonl"))
```

### Example Usage

```bash
# Step 1: Start Parseltongue (one-time setup)
parseltongue parse /path/to/my-project
parseltongue serve  # Starts HTTP API on :8080

# Step 2: Run graph-aware security scan
rec-praxis-review src/**/*.py --use-llm --use-graph

# Example output:
🔗 Using graph-aware analysis (Parseltongue: http://localhost:8080)
🤖 Using DSPy with model: groq/llama-3.3-70b-versatile

🔍 Code Review Results
   Pattern-based: 3 issues found (150ms)
   Graph analysis: 2 cross-function vulnerabilities found (800ms)
   LLM analysis: 2 files (3.2s)
   Deduplication: 1 duplicate removed

🔴 CRITICAL: SQL Injection via Data Flow (database.py:42)
   Source: Graph
   Description: Unsanitized user input flows to dangerous sink 'execute' via:
     api_handler → process_user → load_profile → db.execute
   Fix: Sanitize input in process_user() before passing to load_profile()

🟠 HIGH: Authentication Bypass (api.py:15)
   Source: Graph
   Description: Public entry point '/admin/delete-user' does not call
     authentication functions. Callees: validate_request, execute_delete
   Fix: Ensure all public endpoints call authentication middleware.
```

---

## Performance Optimization

### Caching Strategy

**Problem:** Parseltongue queries add latency (~100-500ms per query)

**Solution:** Multi-layer caching

```python
class CachedParseltongueClient(ParseltongueClient):
    """Parseltongue client with aggressive caching."""

    def __init__(self, base_url: str, cache_ttl: int = 3600):
        super().__init__(base_url)
        self.cache = {}  # In-memory cache
        self.cache_ttl = cache_ttl  # 1 hour default

    def get_call_graph(self, function_name: str, file_path: Optional[str] = None) -> Optional[CallGraphNode]:
        # Check cache first
        cache_key = f"call_graph:{function_name}:{file_path or 'any'}"
        if cache_key in self.cache:
            cached_value, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_value

        # Cache miss - query Parseltongue
        result = super().get_call_graph(function_name, file_path)
        self.cache[cache_key] = (result, time.time())
        return result
```

**Cache Invalidation:**
- On file change (detect via file hash)
- On explicit `--clear-cache` flag
- After TTL expiration (default: 1 hour)

### Async Queries

For multiple graph queries, run in parallel:

```python
import asyncio
import aiohttp

async def get_all_contexts(self, functions: List[str]) -> List[CallGraphNode]:
    """Get call graphs for multiple functions in parallel."""
    async with aiohttp.ClientSession() as session:
        tasks = [self._async_get_call_graph(session, func) for func in functions]
        return await asyncio.gather(*tasks)
```

---

## Testing Strategy

### Unit Tests

**File:** `tests/unit/test_parseltongue_client.py`

```python
def test_parseltongue_client_call_graph():
    """Test call graph query."""
    client = ParseltongueClient("http://localhost:8080")
    call_graph = client.get_call_graph("authenticate")
    assert call_graph.function_name == "authenticate"
    assert "check_password" in call_graph.callees

def test_parseltongue_client_data_flow():
    """Test data flow query."""
    client = ParseltongueClient()
    flows = client.get_data_flow("user_input", "execute")
    assert len(flows) > 0
    assert flows[0].is_tainted == True

def test_parseltongue_unavailable_graceful():
    """Test graceful degradation when Parseltongue unavailable."""
    client = ParseltongueClient("http://nonexistent:9999")
    assert client.is_available() == False
    assert client.get_call_graph("test") is None
```

### Integration Tests

**File:** `tests/integration/test_graph_aware_agent.py`

```python
@pytest.mark.skipif(not parseltongue_running(), reason="Parseltongue not running")
def test_graph_aware_sql_injection_detection():
    """Test graph-aware detection of cross-function SQL injection."""
    code = {
        "api.py": """
def api_handler(request):
    user_id = request.params['id']
    process_user(user_id)
""",
        "database.py": """
def process_user(id):
    load_profile(id)

def load_profile(id):
    db.execute(f"SELECT * FROM users WHERE id={id}")
"""
    }

    agent = GraphAwareCodeReviewAgent(
        memory_path=":memory:",
        planner=mock_planner,
        parseltongue_url="http://localhost:8080"
    )

    findings = agent.review_code(code)

    # Should detect SQL injection via data flow
    sql_findings = [f for f in findings if "SQL Injection" in f.title]
    assert len(sql_findings) > 0
    assert "api_handler → process_user → load_profile" in sql_findings[0].description
```

### Test Fixtures

Create test codebases for graph analysis:

```
tests/fixtures/graph_test_codebase/
├── api.py          # Public routes
├── auth.py         # Authentication logic
├── database.py     # SQL queries
└── utils.py        # Helper functions
```

---

## Success Metrics

### Functional
- ✅ Can query Parseltongue for call graphs, data flow, entry points
- ✅ Detects cross-function vulnerabilities (SQL injection, auth bypass)
- ✅ Graceful degradation when Parseltongue unavailable
- ✅ Graph patterns stored in procedural memory

### Performance
- ✅ Graph query overhead < 200ms per file (with caching)
- ✅ Cache hit rate > 90% on repeat scans
- ✅ Async queries when analyzing multiple functions

### Quality
- ✅ 30-50% more vulnerabilities detected (vs without graph)
- ✅ False positive rate stays < 20%
- ✅ Detects vulnerabilities no other tool can find

---

## Rollout Plan

### Phase 1: Parseltongue Client (Week 1)
- Implement HTTP client for Parseltongue API
- Add caching layer
- Unit tests for all API endpoints

### Phase 2: Graph Context Provider (Week 2)
- Combine graph + memory into rich context
- Format context for LLM prompts
- Integration tests with mock Parseltongue

### Phase 3: Graph-Aware Agent (Week 3)
- Implement data flow analysis
- Authentication bypass detection
- Attack surface analysis

### Phase 4: CLI Integration (Week 4)
- Add `--use-graph` flag to rec-praxis-review
- Add `--use-graph` flag to rec-praxis-audit
- Documentation + examples

---

## Documentation

### README.md

Add section:

```markdown
## Graph-Aware Analysis (Advanced)

For deep security analysis, integrate with [Parseltongue](https://github.com/that-in-rust/parseltongue-dependency-graph-generator):

\`\`\`bash
# Install Parseltongue (Rust)
cargo install parseltongue

# Parse your codebase
parseltongue parse /path/to/project
parseltongue serve

# Run graph-aware security scan
rec-praxis-review src/**/*.py --use-llm --use-graph
\`\`\`

**Detects:**
- Cross-function data flow vulnerabilities
- Authentication bypass (public routes that skip auth)
- Privilege escalation (who can call admin functions)
- Attack surface analysis

**Supports:** 12 languages (Python, JS, TS, Go, Rust, Java, C++, C#, etc.)
```

---

## Related Issues

- beads-rec-praxis-rlm-wdo: Implement LLM-powered code review (prerequisite)
- New: beads-rec-praxis-rlm-graph: Parseltongue integration

---

## References

- [Parseltongue Repository](https://github.com/that-in-rust/parseltongue-dependency-graph-generator)
- [CozoDB Documentation](https://cozodb.org/)
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/)
- [Data Flow Analysis](https://en.wikipedia.org/wiki/Data-flow_analysis)
