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
from rec_praxis_rlm.graph.cached_client import CachedParseltongu eClient

__all__ = [
    "ParseltongueClient",
    "CachedParseltongu eClient",
    "CallGraphNode",
    "DataFlowPath",
]
