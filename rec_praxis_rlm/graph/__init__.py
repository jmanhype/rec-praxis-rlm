"""Graph analysis module for rec-praxis-rlm.

Integrates with Parseltongue (Rust-based dependency graph generator) to enable
graph-aware security analysis.

Repository: https://github.com/that-in-rust/parseltongue-dependency-graph-generator
"""

from rec_praxis_rlm.graph.parseltongue_client import (
    ParseltongueClient,
    CallGraphNode,
    DataFlowPath
)

__all__ = [
    "ParseltongueClient",
    "CallGraphNode",
    "DataFlowPath",
]
