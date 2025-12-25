# Metacognitive Enhancement Specification for RecPraxis RLM

**Version**: 1.0.0-draft
**Date**: 2025-12-25
**Status**: RFC (Request for Comments)
**Authors**: AI-Assisted Design

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Analysis](#2-current-state-analysis)
3. [Proposed Enhancements](#3-proposed-enhancements)
   - 3.1 [Metacognitive Distillation Pipeline](#31-metacognitive-distillation-pipeline)
   - 3.2 [Skill Labeling and Clustering](#32-skill-labeling-and-clustering)
   - 3.3 [Dynamic Behavior Handbook](#33-dynamic-behavior-handbook)
   - 3.4 [Self-Improvement Revision Loops](#34-self-improvement-revision-loops)
   - 3.5 [Subagent Skill Generation](#35-subagent-skill-generation)
4. [Critical Assumption Analysis](#4-critical-assumption-analysis)
5. [Trade-off Evaluation Matrix](#5-trade-off-evaluation-matrix)
6. [Edge Cases and Failure Modes](#6-edge-cases-and-failure-modes)
7. [Context Engineering and Caching](#7-context-engineering-and-caching)
8. [Implementation Roadmap](#8-implementation-roadmap)
9. [Success Metrics and Validation](#9-success-metrics-and-validation)
10. [Self-Critique: Honest Assessment](#self-critique-honest-assessment-of-this-specification)
11. [Appendices](#11-appendices)

---

## 1. Executive Summary

### 1.1 Vision

Transform RecPraxis RLM from an **experience-based memory system** into a **full metacognitive procedural system** that:
- Distills raw experiences into reusable, abstract **behaviors/skills**
- Enables **self-improvement** through reflection loops
- Provides **progressive knowledge disclosure** for token efficiency
- Supports **cross-task transfer** through semantic clustering

### 1.2 Expected Outcomes

| Metric | Current State | Target State | Evidence Basis |
|--------|---------------|--------------|----------------|
| Token usage per planning step | ~2000-3000 | ~1000-1500 (30-50% reduction) | arXiv:2509.13237 claims 46% |
| Cross-task transfer accuracy | N/A | +10-15% improvement | arXiv:2405.12205 exemplar study |
| Memory scalability | ~10k experiences | 100k+ with progressive load | Deepagents-CLI patterns |
| Self-correction rate | Manual only | Automated for 60%+ of failures | 2025 paper's revision loops |

### 1.3 Critical Challenges Identified

1. **Distillation Quality**: LLM-based abstraction is inherently lossy
2. **Behavior Staleness**: Distilled behaviors may become outdated
3. **Context Collapse**: Over-abstraction loses critical nuance
4. **Computational Overhead**: Reflection loops add latency
5. **Evaluation Difficulty**: Hard to measure "behavior quality"

---

## 2. Current State Analysis

### 2.1 Existing Architecture Strengths

```
┌─────────────────────────────────────────────────────────────┐
│                    RecPraxis RLM v0.9.2                     │
├─────────────────────────────────────────────────────────────┤
│  ProceduralMemory                                           │
│  ├─ Experience Storage (JSONL + checksums)                  │
│  ├─ FAISS-accelerated retrieval (10-100x speedup)           │
│  ├─ Hybrid similarity (Jaccard env + cosine goal)           │
│  ├─ Privacy redaction (PII detection)                       │
│  ├─ Concept tagging (semantic extraction)                   │
│  └─ Experience classification (learn/recover/optimize/explore)│
├─────────────────────────────────────────────────────────────┤
│  RLMContext                                                 │
│  ├─ DocumentStore (in-memory with line indexing)            │
│  ├─ DocumentSearcher (ReDoS-protected regex)                │
│  └─ CodeExecutor (sandboxed with SafeExecutor)              │
├─────────────────────────────────────────────────────────────┤
│  PraxisRLMPlanner                                           │
│  ├─ DSPy 3.0 ReAct agent                                    │
│  ├─ Memory/context/exec tool integration                    │
│  └─ MLflow autolog tracing                                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 What Works Well

| Component | Capability | Measured Performance |
|-----------|------------|---------------------|
| FAISS retrieval | 10-100x speedup | Verified in tests |
| Compression | 80-90% token reduction | ObservationCompressor |
| Progressive disclosure | 3-layer retrieval | TokenBudget integration |
| Concept tagging | Automatic extraction | ~15 tech keywords |
| Experience classification | 4 type taxonomy | Keyword-based |

### 2.3 Identified Gaps (Metacognitive)

| Gap | Impact | Current Workaround |
|-----|--------|-------------------|
| No behavior distillation | Raw experiences consume tokens | Manual curation |
| No skill clustering | Poor cross-task transfer | Concept tags (limited) |
| No reflection loops | No self-improvement | Human feedback only |
| No behavior versioning | Stale knowledge persists | Manual cleanup |
| No confidence scoring | Can't prioritize experiences | Success boolean only |

---

## 3. Proposed Enhancements

### 3.1 Metacognitive Distillation Pipeline

#### 3.1.1 Concept

Transform raw experiences into concise, reusable **behaviors**:

```
Experience (verbose)          →  Behavior (concise)
─────────────────────────────────────────────────────
env: ["python", "web_scraping"]     name: "retry_with_backoff"
goal: "Scrape product prices"       instruction: "When scraping
action: "Implemented retry with        flaky endpoints, use
   exponential backoff after           exponential backoff
   3 connection timeouts"              (2^n seconds, max 32s)"
result: "Successfully scraped      preconditions: ["network_io",
   1000 products"                       "external_api"]
success: true                      confidence: 0.85
                                   source_count: 7
```

#### 3.1.2 Proposed Implementation

```python
# New file: rec_praxis_rlm/distillation.py

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class BehaviorConfidence(Enum):
    """Confidence levels based on evidence strength."""
    LOW = 0.3      # 1-2 source experiences
    MEDIUM = 0.6   # 3-5 source experiences
    HIGH = 0.85    # 6+ experiences with consistent outcomes
    VERIFIED = 0.95  # Human-verified or test-validated

@dataclass
class Behavior:
    """A distilled, reusable behavior pattern."""
    name: str                          # Snake_case identifier
    instruction: str                   # Concise action guidance (< 100 words)
    preconditions: List[str]           # When to apply this behavior
    postconditions: List[str]          # Expected outcomes
    confidence: float                  # 0.0-1.0 based on evidence
    source_experience_ids: List[str]   # Traceability
    created_at: float                  # Unix timestamp
    updated_at: float                  # Last refinement
    version: int = 1                   # For evolution tracking
    deprecated: bool = False           # Soft delete
    deprecation_reason: Optional[str] = None
    success_rate: float = 0.0          # Empirical success tracking
    usage_count: int = 0               # How often retrieved
    failure_modes: List[str] = field(default_factory=list)  # Known limitations

class BehaviorDistiller:
    """Distills experiences into reusable behaviors."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        similarity_threshold: float = 0.75,
        min_cluster_size: int = 3,
        max_instruction_tokens: int = 100
    ):
        self.llm = llm_provider
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_instruction_tokens = max_instruction_tokens

    def distill_from_experiences(
        self,
        experiences: List[Experience],
        existing_behaviors: Optional[List[Behavior]] = None
    ) -> List[Behavior]:
        """
        Multi-step distillation pipeline:
        1. Cluster similar experiences
        2. Extract common patterns via LLM
        3. Validate against existing behaviors
        4. Merge or create new behaviors
        """
        pass  # Implementation in section 7

    def _cluster_experiences(
        self,
        experiences: List[Experience]
    ) -> List[List[Experience]]:
        """Group experiences by semantic similarity."""
        pass

    def _extract_behavior(
        self,
        cluster: List[Experience]
    ) -> Optional[Behavior]:
        """LLM-based pattern extraction with structured output."""
        pass

    def _validate_behavior(
        self,
        behavior: Behavior,
        experiences: List[Experience]
    ) -> Tuple[bool, List[str]]:
        """Check behavior against source experiences for consistency."""
        pass
```

#### 3.1.3 Distillation Prompt Template

```
## Behavior Extraction Task

You are analyzing a cluster of similar agent experiences to extract a reusable behavior pattern.

### Source Experiences (N={count})
{formatted_experiences}

### Extraction Requirements
1. Identify the COMMON pattern across all experiences
2. Abstract away task-specific details
3. Preserve critical nuances that affect success/failure
4. Name using snake_case (max 40 chars)
5. Instruction must be actionable and < 100 words

### Output Format (JSON)
```json
{
  "name": "behavior_name",
  "instruction": "Concise, actionable guidance...",
  "preconditions": ["condition_1", "condition_2"],
  "postconditions": ["expected_outcome"],
  "confidence_reasoning": "Why this pattern is reliable...",
  "failure_modes": ["Known limitation 1", "Edge case 2"],
  "abstraction_notes": "What was generalized vs preserved..."
}
```

### Critical Checks
- Does this pattern apply to ALL source experiences? If not, split.
- Is the instruction specific enough to be actionable?
- Are preconditions testable at runtime?
```

#### 3.1.4 Assumptions Being Made

| Assumption | Risk Level | Challenge |
|------------|------------|-----------|
| LLM can identify common patterns | Medium | May overfit to surface similarities |
| 3+ experiences form reliable clusters | Medium | Small clusters may produce fragile behaviors |
| Abstraction preserves actionability | High | Over-generalization loses critical details |
| Behaviors remain valid over time | High | Domain/API changes invalidate behaviors |
| Confidence correlates with reliability | Medium | May not account for edge case coverage |

#### 3.1.5 Trade-off Analysis

| Factor | Keep Raw Experiences | Distill to Behaviors | Hybrid Approach |
|--------|---------------------|---------------------|-----------------|
| Token efficiency | Poor (2-3k/exp) | Excellent (50-100/behavior) | Good (varies) |
| Nuance preservation | Excellent | Poor-Medium | Good |
| Cross-task transfer | Limited | Good | Good |
| Debugging/tracing | Excellent | Poor | Good |
| Maintenance burden | Low | Medium (version mgmt) | Medium |
| Cold-start | Works | Needs experiences first | Works |

**Recommendation**: Hybrid approach where behaviors augment (not replace) experiences.

---

### 3.2 Skill Labeling and Clustering

#### 3.2.1 Concept

Apply hierarchical labeling to experiences and behaviors:

```
Fine-grained skill           →  Coarse category
──────────────────────────────────────────────
extract_prices_via_css           web_data_extraction
parse_json_api_response          api_integration
handle_rate_limit_429            error_handling
retry_with_exponential_backoff   resilience_patterns
```

#### 3.2.2 Proposed Implementation

```python
# Extension to: rec_praxis_rlm/concepts.py

@dataclass
class SkillLabel:
    """Hierarchical skill label."""
    fine_grained: str        # Specific skill name
    category: str            # Broad category
    domain: str              # Domain area (web, data, security, etc.)
    confidence: float        # Labeling confidence

class SkillLabeler:
    """Labels experiences with hierarchical skills."""

    # Pre-defined skill taxonomy (extensible)
    SKILL_TAXONOMY = {
        "web_interaction": {
            "data_extraction": [
                "parse_html_tables",
                "extract_via_css_selectors",
                "handle_pagination",
                "extract_from_json_ld"
            ],
            "navigation": [
                "handle_redirects",
                "manage_sessions",
                "handle_authentication"
            ],
            "resilience": [
                "retry_with_backoff",
                "handle_rate_limits",
                "detect_blocking"
            ]
        },
        "data_processing": {
            "parsing": [
                "parse_json",
                "parse_xml",
                "parse_csv"
            ],
            "transformation": [
                "normalize_data",
                "deduplicate",
                "aggregate"
            ]
        },
        "error_handling": {
            "recovery": [
                "retry_on_failure",
                "fallback_strategy",
                "graceful_degradation"
            ],
            "debugging": [
                "isolate_root_cause",
                "reproduce_issue",
                "trace_data_flow"
            ]
        }
        # ... extensible
    }

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        llm_provider: Optional[LLMProvider] = None,
        use_llm_labeling: bool = False
    ):
        self.embeddings = embedding_provider
        self.llm = llm_provider
        self.use_llm = use_llm_labeling
        self._taxonomy_embeddings = self._embed_taxonomy()

    def label_experience(
        self,
        experience: Experience
    ) -> List[SkillLabel]:
        """Assign skill labels to an experience."""
        if self.use_llm and self.llm:
            return self._llm_label(experience)
        return self._embedding_label(experience)

    def cluster_by_skill(
        self,
        experiences: List[Experience],
        labels: List[List[SkillLabel]]
    ) -> Dict[str, List[Experience]]:
        """Group experiences by their primary skill category."""
        pass
```

#### 3.2.3 Assumptions Being Made

| Assumption | Risk Level | Challenge |
|------------|------------|-----------|
| Pre-defined taxonomy covers most skills | High | New domains require taxonomy expansion |
| Embedding similarity = skill similarity | Medium | Semantic != functional similarity |
| Fine-grained labels are useful | Medium | May be too specific for reuse |
| Categories are mutually exclusive | Low | Many experiences span categories |

#### 3.2.4 Trade-off Analysis

| Approach | Pros | Cons |
|----------|------|------|
| Pre-defined taxonomy | Fast, consistent, interpretable | Limited coverage, maintenance burden |
| LLM-based labeling | Flexible, handles novel skills | Inconsistent, expensive, hallucination risk |
| Embedding clustering | No predefined schema needed | Labels may not be meaningful |
| Hybrid (taxonomy + LLM fallback) | Best coverage | Complexity, two failure modes |

**Recommendation**: Start with pre-defined taxonomy + embedding matching. Add LLM fallback only for unmatched experiences (< 70% confidence).

---

### 3.3 Dynamic Behavior Handbook

#### 3.3.1 Concept

A searchable, versioned repository of distilled behaviors:

```
handbook/
├── behaviors/
│   ├── web_data_extraction/
│   │   ├── retry_with_backoff.json
│   │   ├── handle_pagination.json
│   │   └── _category_meta.json
│   ├── error_handling/
│   │   ├── graceful_degradation.json
│   │   └── root_cause_isolation.json
│   └── _index.json              # Fast lookup
├── deprecated/                   # Archived behaviors
├── pending_review/               # Newly distilled, unverified
└── handbook_meta.json           # Version, stats
```

#### 3.3.2 Proposed Implementation

```python
# New file: rec_praxis_rlm/handbook.py

from pathlib import Path
from typing import Iterator, Optional
import json

@dataclass
class HandbookConfig:
    """Configuration for behavior handbook."""
    storage_path: Path = Path(".rec-praxis-rlm/handbook")
    max_behaviors_per_category: int = 100
    index_rebuild_threshold: int = 50  # Rebuild after N additions
    enable_faiss_index: bool = True
    compression_enabled: bool = False

class BehaviorHandbook:
    """
    Dynamic repository of distilled behaviors with:
    - Progressive disclosure (metadata → full instruction)
    - FAISS-accelerated semantic search
    - Version tracking and deprecation
    - Category-based organization
    """

    def __init__(
        self,
        config: HandbookConfig,
        embedding_provider: Optional[EmbeddingProvider] = None
    ):
        self.config = config
        self.embeddings = embedding_provider
        self._index: Optional[faiss.Index] = None
        self._behavior_cache: Dict[str, Behavior] = {}
        self._metadata_cache: Dict[str, dict] = {}
        self._ensure_directory_structure()

    def add_behavior(
        self,
        behavior: Behavior,
        category: str,
        skip_review: bool = False
    ) -> str:
        """
        Add behavior to handbook.

        Args:
            behavior: The behavior to add
            category: Category path (e.g., "web_data_extraction")
            skip_review: If False, add to pending_review first

        Returns:
            behavior_id: Unique identifier
        """
        pass

    def search(
        self,
        query: str,
        env_context: Optional[List[str]] = None,
        top_k: int = 5,
        min_confidence: float = 0.5,
        progressive: bool = True
    ) -> Iterator[Union[BehaviorMetadata, Behavior]]:
        """
        Search for relevant behaviors.

        Args:
            query: Natural language query or goal description
            env_context: Environmental context for filtering
            top_k: Maximum results
            min_confidence: Minimum behavior confidence
            progressive: If True, yield metadata first, full on demand

        Yields:
            BehaviorMetadata (if progressive) or full Behavior
        """
        pass

    def get_full_behavior(self, behavior_id: str) -> Behavior:
        """Load full behavior details (for progressive disclosure)."""
        pass

    def deprecate_behavior(
        self,
        behavior_id: str,
        reason: str,
        replacement_id: Optional[str] = None
    ) -> None:
        """Mark behavior as deprecated with migration path."""
        pass

    def update_behavior(
        self,
        behavior_id: str,
        updates: Dict[str, Any],
        increment_version: bool = True
    ) -> Behavior:
        """Update behavior with version tracking."""
        pass

    def get_category_summary(self, category: str) -> CategorySummary:
        """Get statistics and top behaviors for a category."""
        pass

    def rebuild_index(self) -> None:
        """Rebuild FAISS index after bulk operations."""
        pass

    def export_for_finetuning(
        self,
        categories: Optional[List[str]] = None,
        min_confidence: float = 0.7,
        format: str = "jsonl"
    ) -> Path:
        """Export behaviors as training data for model fine-tuning."""
        pass
```

#### 3.3.3 Progressive Disclosure Protocol

```
Step 1: Search returns BehaviorMetadata (minimal tokens)
────────────────────────────────────────────────────────
{
  "id": "retry_with_backoff_v3",
  "name": "retry_with_backoff",
  "category": "web_interaction/resilience",
  "summary": "Exponential backoff for transient failures",  # < 20 words
  "confidence": 0.85,
  "usage_count": 47,
  "last_success": "2025-12-20"
}
Token cost: ~50 tokens per behavior

Step 2: Agent requests full behavior (on selection)
────────────────────────────────────────────────────────
{
  "id": "retry_with_backoff_v3",
  "name": "retry_with_backoff",
  "instruction": "When encountering transient failures (HTTP 429, 503,
    connection timeouts), implement exponential backoff: wait 2^n seconds
    where n is attempt number, cap at 32 seconds. After 5 failed attempts,
    escalate to alternative strategy or fail gracefully.",
  "preconditions": ["network_io", "external_api", "transient_error"],
  "postconditions": ["recovered_from_transient", "graceful_failure"],
  "failure_modes": [
    "Does not help with permanent failures (4xx errors)",
    "May cause timeout in time-sensitive operations"
  ],
  "example": "try_with_backoff(requests.get, url, max_attempts=5)"
}
Token cost: ~150-200 tokens per behavior
```

#### 3.3.4 Assumptions Being Made

| Assumption | Risk Level | Challenge |
|------------|------------|-----------|
| File-based storage is sufficient | Low | May need DB for 100k+ behaviors |
| Categories remain stable | Medium | Reorganization requires migration |
| Progressive disclosure saves tokens | Medium | May need multiple round-trips |
| FAISS index fits in memory | Low | 100k behaviors ≈ 500MB |
| Deprecation is sufficient for cleanup | Low | May need hard deletes eventually |

#### 3.3.5 Trade-off Analysis

| Storage Strategy | Pros | Cons |
|------------------|------|------|
| File-based (JSON) | Simple, git-friendly, human-readable | Slow for large datasets, no transactions |
| SQLite | Fast queries, ACID, single file | Less git-friendly, schema migrations |
| PostgreSQL | Scalable, full SQL, concurrent access | Infrastructure overhead, deployment complexity |

**Recommendation**: Start with file-based + FAISS. Migrate to SQLite if handbook exceeds 10k behaviors.

---

### 3.4 Self-Improvement Revision Loops

#### 3.4.1 Concept

Enable agents to revise plans using retrieved behaviors when initial attempts fail:

```
Initial Plan → Execute → Failure Detected
                            ↓
                    Reflect on Failure
                            ↓
                    Retrieve Relevant Behaviors
                            ↓
                    Revise Plan with Behaviors
                            ↓
                    Execute Revised Plan
                            ↓
                    Success? → Store New Experience
                            ↓
                    Still Failed? → Escalate or Iterate
```

#### 3.4.2 Proposed Implementation

```python
# Extension to: rec_praxis_rlm/dspy_agent.py

class RevisionConfig:
    """Configuration for self-improvement loops."""
    max_revisions: int = 3
    min_failure_confidence: float = 0.7  # How certain failure occurred
    behavior_injection_limit: int = 3    # Max behaviors per revision
    enable_trace_logging: bool = True
    escalation_threshold: int = 2        # Revisions before human escalation

class SelfImprovingPlanner(PraxisRLMPlanner):
    """
    Extended planner with behavior-guided revision loops.

    Key differences from base PraxisRLMPlanner:
    1. Catches and classifies execution failures
    2. Retrieves relevant behaviors from handbook
    3. Injects behaviors into revised prompts
    4. Tracks revision history for learning
    """

    def __init__(
        self,
        memory: ProceduralMemory,
        context: RLMContext,
        handbook: BehaviorHandbook,
        revision_config: RevisionConfig = RevisionConfig(),
        **kwargs
    ):
        super().__init__(memory, context, **kwargs)
        self.handbook = handbook
        self.revision_config = revision_config
        self._revision_history: List[RevisionTrace] = []

    def plan_with_revision(
        self,
        goal: str,
        env_features: List[str],
        allow_revisions: bool = True
    ) -> PlanResult:
        """
        Plan with automatic revision on failure.

        Returns:
            PlanResult containing answer, revision_count, and trace
        """
        revision_count = 0
        last_failure = None

        while revision_count <= self.revision_config.max_revisions:
            try:
                # Inject behaviors if this is a revision
                if revision_count > 0 and last_failure:
                    behaviors = self._retrieve_relevant_behaviors(
                        goal, env_features, last_failure
                    )
                    goal = self._inject_behaviors(goal, behaviors)

                result = self.plan(goal, env_features)

                # Validate result (detect semantic failures)
                failure = self._detect_failure(result, goal)
                if failure is None:
                    return PlanResult(
                        answer=result,
                        revision_count=revision_count,
                        trace=self._revision_history
                    )

                last_failure = failure
                revision_count += 1

            except Exception as e:
                last_failure = FailureAnalysis(
                    type="exception",
                    message=str(e),
                    confidence=1.0
                )
                revision_count += 1

        # Max revisions exceeded - escalate
        return self._escalate(goal, env_features, self._revision_history)

    def _retrieve_relevant_behaviors(
        self,
        goal: str,
        env_features: List[str],
        failure: FailureAnalysis
    ) -> List[Behavior]:
        """Find behaviors relevant to the failure context."""
        query = f"Failure: {failure.message}. Goal: {goal}"
        return list(self.handbook.search(
            query=query,
            env_context=env_features,
            top_k=self.revision_config.behavior_injection_limit,
            progressive=False  # Need full behaviors for injection
        ))

    def _inject_behaviors(
        self,
        goal: str,
        behaviors: List[Behavior]
    ) -> str:
        """Augment goal with relevant behaviors."""
        if not behaviors:
            return goal

        behavior_text = "\n".join([
            f"- {b.name}: {b.instruction}"
            for b in behaviors
        ])

        return f"""{goal}

## Relevant Behaviors from Experience
{behavior_text}

Apply these behaviors if applicable to your approach."""

    def _detect_failure(
        self,
        result: str,
        goal: str
    ) -> Optional[FailureAnalysis]:
        """Detect if the result indicates a failure."""
        # Heuristic failure detection
        failure_indicators = [
            "error", "failed", "exception", "could not",
            "unable to", "not found", "timeout", "refused"
        ]

        result_lower = result.lower()
        for indicator in failure_indicators:
            if indicator in result_lower:
                return FailureAnalysis(
                    type="semantic",
                    message=f"Result contains failure indicator: {indicator}",
                    confidence=0.7
                )

        # Could add LLM-based failure detection for higher accuracy
        return None

    def _escalate(
        self,
        goal: str,
        env_features: List[str],
        trace: List[RevisionTrace]
    ) -> PlanResult:
        """Handle max revision exceeded."""
        # Options:
        # 1. Return failure with trace for human review
        # 2. Log for offline analysis
        # 3. Create new experience marked as "needs_review"
        pass

@dataclass
class FailureAnalysis:
    """Structured failure information."""
    type: str              # exception, semantic, timeout, validation
    message: str           # Human-readable description
    confidence: float      # How certain we are this is a failure
    root_cause: Optional[str] = None
    suggested_behaviors: Optional[List[str]] = None

@dataclass
class RevisionTrace:
    """Record of a revision attempt."""
    revision_number: int
    failure: FailureAnalysis
    behaviors_applied: List[str]
    revised_goal: str
    result: str
    success: bool
    timestamp: float
```

#### 3.4.3 Revision Prompt Template

```
## Revision Context

Your previous attempt failed:
- Failure type: {failure.type}
- Details: {failure.message}
- Confidence: {failure.confidence}

## Original Goal
{original_goal}

## Behaviors to Consider
{for behavior in behaviors}
### {behavior.name}
{behavior.instruction}
- Preconditions: {behavior.preconditions}
- Known limitations: {behavior.failure_modes}
{/for}

## Your Task
Revise your approach incorporating relevant behaviors above.
Explain what you're doing differently and why.
```

#### 3.4.4 Assumptions Being Made

| Assumption | Risk Level | Challenge |
|------------|------------|-----------|
| Failures are detectable | High | Semantic failures hard to identify |
| Behaviors are applicable to failures | Medium | May retrieve irrelevant behaviors |
| Revision improves outcomes | Medium | May make things worse |
| 3 revisions is optimal | Low | Domain-dependent |
| Trace logging doesn't bloat memory | Medium | Long traces are expensive |

#### 3.4.5 Trade-off Analysis

| Factor | No Revision | Simple Retry | Behavior-Guided Revision |
|--------|-------------|--------------|-------------------------|
| Success rate | Baseline | +10-20% | +30-40% (claimed) |
| Latency | 1x | 2-3x | 2-3x |
| Token cost | 1x | 2-3x | 2.5-4x |
| Learning | None | None | Captures failure patterns |
| Complexity | Low | Low | High |

**Recommendation**: Enable revision loops but with conservative limits (max 2 revisions) and clear escalation paths.

---

### 3.5 Subagent Skill Generation

#### 3.5.1 Concept

Allow agents to spawn sub-tasks for skill generation and validation:

```
Main Agent                    Skill Generation Subagent
─────────────                 ─────────────────────────
"Complete task X"
       ↓
Execute → Failure
       ↓
"Generate skill for          →  Analyze trajectory
 handling this failure"          ↓
                                 Extract pattern
                                 ↓
                                 Validate skill
                                 ↓
                              ←  Return skill proposal
       ↓
Review and approve
       ↓
Add to handbook
```

#### 3.5.2 Proposed Implementation

```python
# New file: rec_praxis_rlm/subagents.py

from enum import Enum
from typing import Callable, Set

class SubagentPermission(Enum):
    """Permission levels for subagents."""
    READ_MEMORY = "read_memory"
    WRITE_MEMORY = "write_memory"
    EXECUTE_CODE = "execute_code"
    NETWORK_ACCESS = "network_access"
    FILE_ACCESS = "file_access"
    SPAWN_SUBAGENT = "spawn_subagent"

@dataclass
class SubagentConfig:
    """Configuration for skill generation subagent."""
    allowed_permissions: Set[SubagentPermission] = field(default_factory=lambda: {
        SubagentPermission.READ_MEMORY
    })
    max_execution_time: float = 30.0  # seconds
    max_iterations: int = 5
    require_human_approval: bool = True
    sandbox_level: str = "strict"  # strict, moderate, permissive

class SkillGenerationSubagent:
    """
    Subagent specialized for analyzing trajectories and generating skills.

    Runs in a restricted context with limited permissions to ensure safety.
    """

    def __init__(
        self,
        config: SubagentConfig,
        memory: ProceduralMemory,
        distiller: BehaviorDistiller
    ):
        self.config = config
        self.memory = memory
        self.distiller = distiller
        self._validate_permissions()

    def generate_skill_from_trajectory(
        self,
        trajectory: List[TrajectoryStep],
        failure_context: Optional[FailureAnalysis] = None
    ) -> SkillProposal:
        """
        Analyze a trajectory and propose a reusable skill.

        Args:
            trajectory: Sequence of (action, observation, reasoning) steps
            failure_context: If from a failure, what went wrong

        Returns:
            SkillProposal for human review
        """
        # 1. Analyze trajectory for patterns
        patterns = self._extract_patterns(trajectory)

        # 2. Find similar experiences in memory
        similar = self._find_similar_experiences(trajectory)

        # 3. Generate skill proposal
        proposal = self._generate_proposal(patterns, similar, failure_context)

        # 4. Validate proposal
        validation = self._validate_proposal(proposal, trajectory)

        return SkillProposal(
            behavior=proposal,
            validation=validation,
            requires_approval=self.config.require_human_approval,
            source_trajectory=trajectory
        )

    def _validate_permissions(self) -> None:
        """Ensure subagent has only allowed permissions."""
        if SubagentPermission.SPAWN_SUBAGENT in self.config.allowed_permissions:
            raise ValueError("Subagents cannot spawn other subagents")

@dataclass
class TrajectoryStep:
    """Single step in an agent trajectory."""
    action: str
    observation: str
    reasoning: Optional[str]
    timestamp: float
    tool_calls: List[dict]
    tokens_used: int

@dataclass
class SkillProposal:
    """Proposed skill awaiting review."""
    behavior: Behavior
    validation: ValidationResult
    requires_approval: bool
    source_trajectory: List[TrajectoryStep]
    generated_at: float = field(default_factory=time.time)

@dataclass
class ValidationResult:
    """Result of skill validation."""
    is_valid: bool
    confidence: float
    issues: List[str]
    test_results: Optional[List[dict]] = None
```

#### 3.5.3 CLI Integration

```python
# Extension to: rec_praxis_rlm/cli.py

@app.command()
def reflect(
    trajectory_file: Path = typer.Argument(..., help="Path to trajectory JSON"),
    output: Optional[Path] = typer.Option(None, help="Output skill file"),
    auto_approve: bool = typer.Option(False, help="Skip human approval"),
    model: str = typer.Option("openai/gpt-4o-mini", help="LLM for reflection")
):
    """
    Analyze a trajectory and generate a reusable skill.

    Example:
        rec-praxis-reflect trajectory.json --output skill.json
    """
    pass
```

#### 3.5.4 Assumptions Being Made

| Assumption | Risk Level | Challenge |
|------------|------------|-----------|
| Trajectories contain enough info | Medium | May be incomplete or noisy |
| Subagent can generate valid skills | Medium | LLM may hallucinate |
| Human review catches bad skills | Low | Depends on reviewer expertise |
| Sandbox is sufficient | Medium | Novel attack vectors possible |
| Permission model is complete | Medium | May need refinement |

#### 3.5.5 Trade-off Analysis

| Factor | Manual Skill Creation | Fully Automated | Human-in-Loop (Proposed) |
|--------|----------------------|-----------------|--------------------------|
| Quality | High | Low-Medium | Medium-High |
| Speed | Slow | Fast | Medium |
| Coverage | Limited | High | High |
| Risk | Low | High | Low-Medium |
| Scalability | Poor | Excellent | Good |

**Recommendation**: Human-in-the-loop with optional auto-approve for trusted trajectories (success=true, high confidence).

---

## 4. Critical Assumption Analysis

### 4.1 Foundational Assumptions

| # | Assumption | Evidence For | Evidence Against | Risk Mitigation |
|---|------------|--------------|------------------|-----------------|
| 1 | LLMs can reliably extract patterns from experiences | arXiv:2509.13237 shows 46% token savings | Hallucination risk, inconsistency across runs | Validation layer, confidence thresholds |
| 2 | Behaviors remain valid over time | Stable domains (math, logic) | Rapidly changing APIs, libraries | Version tracking, staleness detection |
| 3 | Cross-task transfer works | arXiv:2405.12205 shows 10-15% improvement | Domain gaps, overfitting to task | Skill clustering, domain tagging |
| 4 | Token savings justify complexity | Papers report 30-50% savings | Implementation overhead, latency | Measure actual savings in production |
| 5 | Users want automated distillation | Developer efficiency gains | Loss of control, trust issues | Human-in-the-loop, transparency |

### 4.2 Technical Assumptions

| # | Assumption | Reality Check | Alternative |
|---|------------|---------------|-------------|
| 1 | FAISS index fits in memory | 100k behaviors ≈ 500MB | Use disk-based index (IVF_HNSW) |
| 2 | File-based storage is sufficient | Slow beyond 10k behaviors | Migrate to SQLite |
| 3 | Sentence-transformers embeddings work | May not capture skill semantics | Domain-specific fine-tuning |
| 4 | JSON serialization is fast enough | Fine for < 1k behaviors | MessagePack for larger scale |
| 5 | Single-process is sufficient | Works for CLI usage | Multi-process for server deployment |

### 4.3 Behavioral Assumptions

| # | Assumption | Challenge | Validation Approach |
|---|------------|-----------|---------------------|
| 1 | Agents will use retrieved behaviors | May ignore or misapply | Track behavior usage vs. outcomes |
| 2 | Revision loops converge | May oscillate or diverge | Max iterations + escalation |
| 3 | Failure detection is reliable | False positives/negatives | Calibrate on labeled dataset |
| 4 | Behaviors are composable | Conflicts, ordering issues | Test behavior combinations |
| 5 | Deprecation is honored | Stale behaviors may persist | Active cleanup, TTL |

---

## 5. Trade-off Evaluation Matrix

### 5.1 Overall System Trade-offs

```
                    Token Efficiency
                          ↑
                          │
    Complex              │              Simple
    High Quality ────────┼──────── Lower Quality
                          │
                          │
                    Latency/Cost
                          ↓

Current RecPraxis: Simple, High Quality, Higher Tokens
Proposed Enhanced: Complex, High Quality, Lower Tokens (target)
Risk: Complex, Medium Quality, Variable Tokens (failure mode)
```

### 5.2 Implementation Priority Matrix

| Enhancement | Impact | Complexity | Risk | Priority |
|-------------|--------|------------|------|----------|
| Behavior Distillation | High | High | Medium | P1 |
| Skill Labeling | Medium | Medium | Low | P2 |
| Dynamic Handbook | High | Medium | Low | P1 |
| Revision Loops | High | High | High | P2 |
| Subagent Generation | Medium | High | Medium | P3 |

### 5.3 Decision Framework

```
IF goal is "reduce token usage" AND experiences > 1000:
    → Prioritize Behavior Distillation + Handbook

IF goal is "improve cross-task transfer":
    → Prioritize Skill Labeling + Clustering

IF goal is "reduce human intervention":
    → Prioritize Revision Loops + Subagent Generation

IF goal is "quick win with low risk":
    → Start with Handbook (no distillation, manual behavior creation)
```

---

## 6. Edge Cases and Failure Modes

### 6.1 Distillation Failures

| Edge Case | Description | Detection | Mitigation |
|-----------|-------------|-----------|------------|
| Empty cluster | < min_cluster_size experiences | Count check | Skip distillation |
| Contradictory experiences | Same goal, opposite outcomes | Variance check | Split into separate behaviors |
| Over-generalization | Behavior too abstract to apply | Validation failure | Preserve more specifics |
| Under-generalization | Behavior too specific | Low reuse rate | Merge similar behaviors |
| Hallucinated behavior | LLM generates plausible but wrong | Test against source | Human review gate |

### 6.2 Handbook Failures

| Edge Case | Description | Detection | Mitigation |
|-----------|-------------|-----------|------------|
| Index corruption | FAISS index out of sync | Checksum mismatch | Rebuild from source |
| Category explosion | Too many fine-grained categories | Category count > threshold | Merge categories |
| Stale behaviors | Behaviors reference deprecated APIs | Usage failure rate | Staleness detection |
| Circular deprecation | A deprecates B deprecates A | Graph cycle detection | Validation on deprecation |
| Search timeout | Too many behaviors to search | Latency monitoring | Pagination, caching |

### 6.3 Revision Loop Failures

| Edge Case | Description | Detection | Mitigation |
|-----------|-------------|-----------|------------|
| Infinite loop | Revisions don't converge | Max iterations | Hard limit + escalation |
| Oscillation | Alternating between two failures | Failure hash tracking | Detect and break |
| Behavior misapplication | Wrong behavior applied | Outcome tracking | Confidence thresholds |
| Cost explosion | Too many LLM calls | Token counting | Budget limits |
| Escalation storm | Too many escalations | Rate limiting | Batch escalations |

### 6.4 Subagent Failures

| Edge Case | Description | Detection | Mitigation |
|-----------|-------------|-----------|------------|
| Sandbox escape | Subagent accesses restricted resources | Capability monitoring | Strict sandboxing |
| Resource exhaustion | Subagent runs too long | Timeout | Kill after limit |
| Bad skill injection | Malicious or buggy skill | Validation layer | Human review |
| Permission creep | Subagent requests more permissions | Permission audit | Deny by default |

### 6.5 Data Integrity Failures

| Edge Case | Description | Detection | Mitigation |
|-----------|-------------|-----------|------------|
| Storage corruption | JSONL becomes unreadable | Checksum validation | Line-by-line recovery |
| Version mismatch | Old client, new storage format | Version header | Migration on load |
| Concurrent writes | Multiple processes writing | File locking | fcntl locks |
| Backup loss | No backup before migration | Pre-migration check | Auto-backup |

---

## 7. Context Engineering and Caching

> "As long as attention is quadratic you have no choice but to context engineer" — @irl_danB

### 7.1 The Quadratic Attention Problem

Context management is not optional—it's fundamental. Every token in the context window participates in O(n²) attention computations. This has critical implications for our metacognitive enhancements:

| Context Size | Relative Compute | Implication |
|--------------|------------------|-------------|
| 2k tokens | 1x | Baseline |
| 8k tokens | 16x | Noticeable latency |
| 32k tokens | 256x | Expensive per call |
| 128k tokens | 4096x | Cost-prohibitive for iteration |

**Key Insight**: Distilling 10 experiences (25k tokens) into 3 behaviors (500 tokens) isn't just about storage—it's 2500x less compute per attention layer.

### 7.2 Prompt Caching Considerations

Modern LLM APIs (Anthropic, OpenAI) support prompt caching where repeated prefixes are computed once. Our distillation strategy must be **cache-aware**:

```
┌─────────────────────────────────────────────────────────┐
│ CACHE-FRIENDLY ARCHITECTURE                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Static Base (Cached)          Dynamic Suffix           │
│  ─────────────────────         ──────────────────       │
│  ┌─────────────────────┐       ┌──────────────────┐     │
│  │ System Prompt       │       │ Current Goal     │     │
│  │ Core Behaviors      │       │ Retrieved Context│     │
│  │ Skill Taxonomy      │       │ Recent History   │     │
│  │ Tool Definitions    │       │ User Query       │     │
│  └─────────────────────┘       └──────────────────┘     │
│         ↑                              ↑                │
│    STABLE (cache hit)          VARIABLE (recomputed)    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Cache-Busting Anti-Patterns

| Pattern | Why It Busts Cache | Mitigation |
|---------|-------------------|------------|
| Behaviors in dynamic section | Changes per request | Move to static base |
| Timestamp in prompt | Always unique | Remove or put at end |
| Random behavior ordering | Order changes | Deterministic sort |
| Recursive compaction | Modifies base | Compact to separate layer |

#### Recommended Cache Strategy

```python
class CacheAwarePromptBuilder:
    """Builds prompts that maximize cache utilization."""

    def __init__(self, handbook: BehaviorHandbook):
        self.handbook = handbook
        self._static_base: Optional[str] = None
        self._base_hash: Optional[str] = None

    def build_prompt(
        self,
        goal: str,
        env_features: List[str],
        retrieved_behaviors: List[Behavior]
    ) -> Tuple[str, str]:
        """
        Returns (static_prefix, dynamic_suffix) for cache-aware calling.

        The static prefix should be passed with cache_control markers.
        """
        # Static base: rebuild only when handbook changes
        if self._needs_base_rebuild():
            self._static_base = self._build_static_base()

        # Dynamic suffix: changes per request
        dynamic_suffix = self._build_dynamic_suffix(
            goal, env_features, retrieved_behaviors
        )

        return self._static_base, dynamic_suffix

    def _build_static_base(self) -> str:
        """Build cacheable prefix with core behaviors."""
        core_behaviors = self.handbook.get_core_behaviors(limit=20)
        return f"""## Core Behaviors (Always Available)
{self._format_behaviors(core_behaviors)}

## Available Tools
{self._format_tools()}
"""

    def _build_dynamic_suffix(
        self,
        goal: str,
        env_features: List[str],
        behaviors: List[Behavior]
    ) -> str:
        """Build per-request suffix."""
        return f"""## Current Task
Goal: {goal}
Environment: {', '.join(env_features)}

## Retrieved Behaviors (Task-Specific)
{self._format_behaviors(behaviors)}

## Your Response
"""
```

### 7.3 Recursive Compaction Strategy

When context grows too large, we need compaction. But naive compaction busts the cache. Solution: **layered compaction with stable bases**.

```
Level 0: Raw Experiences (not in context)
    ↓ distill
Level 1: Behaviors (20 core in static base)
    ↓ retrieve
Level 2: Task-Specific Behaviors (3-5 in dynamic suffix)
    ↓ summarize (if still too large)
Level 3: Compressed Summaries (last resort)
```

#### Compaction Hook Interface

Inspired by pi-coding-agent's custom compaction hooks:

```python
class CompactionHook(Protocol):
    """Interface for custom compaction strategies."""

    def should_compact(self, context_tokens: int, budget: int) -> bool:
        """Decide if compaction is needed."""
        ...

    def compact(
        self,
        context: List[Message],
        target_tokens: int
    ) -> List[Message]:
        """Perform compaction, preserving critical information."""
        ...

    def preserves_cache(self) -> bool:
        """Whether this compaction strategy preserves prompt cache."""
        ...

class BehaviorAwareCompactor(CompactionHook):
    """Compactor that summarizes to behaviors, preserving cache."""

    def __init__(self, distiller: BehaviorDistiller):
        self.distiller = distiller

    def compact(
        self,
        context: List[Message],
        target_tokens: int
    ) -> List[Message]:
        # Extract experiences from context
        experiences = self._extract_experiences(context)

        # Distill to behaviors (if enough patterns)
        if len(experiences) >= 3:
            behaviors = self.distiller.distill_from_experiences(experiences)
            # Replace verbose experiences with behavior references
            return self._replace_with_behaviors(context, behaviors)

        # Fallback to summary compaction
        return self._summarize_context(context, target_tokens)

    def preserves_cache(self) -> bool:
        return True  # Only modifies dynamic section
```

### 7.4 Subagent Context Isolation

Multiple subagents sharing context is expensive. Better patterns:

#### Pattern 1: Worktree + Brief Handoff

From @shakermanjonas: "write up a feature brief, create a worktree, launch a terminal with claude code and the brief"

```python
@dataclass
class SubagentBrief:
    """Minimal context handoff to subagent."""
    task_summary: str           # < 200 words
    relevant_files: List[str]   # Paths only, not contents
    behaviors_to_apply: List[str]  # Behavior names/IDs
    success_criteria: List[str]
    parent_session_id: str      # For result reporting

class BriefBasedSubagentLauncher:
    """Launch subagents with minimal context via briefs."""

    def launch(
        self,
        brief: SubagentBrief,
        working_directory: Path
    ) -> SubagentHandle:
        # Write brief to file (subagent reads on startup)
        brief_path = working_directory / ".claude-brief.json"
        brief_path.write_text(brief.to_json())

        # Launch subagent in isolated environment
        # Subagent loads brief + retrieves its own context
        return self._spawn_subprocess(
            working_directory,
            init_command=f"Read .claude-brief.json and begin task"
        )
```

#### Pattern 2: tmux-Based Orchestration

From @ashot: "use tmux to call subagents"

```python
class TmuxSubagentOrchestrator:
    """Orchestrate subagents via tmux sessions."""

    def __init__(self, session_prefix: str = "praxis"):
        self.session_prefix = session_prefix
        self.active_agents: Dict[str, TmuxPane] = {}

    def spawn_agent(
        self,
        agent_id: str,
        task: str,
        handbook: BehaviorHandbook
    ) -> str:
        """Spawn a subagent in a new tmux pane."""
        session_name = f"{self.session_prefix}-{agent_id}"

        # Create isolated session
        subprocess.run([
            "tmux", "new-session", "-d", "-s", session_name
        ])

        # Inject minimal context via environment
        relevant_behaviors = handbook.search(task, top_k=3)
        behavior_ids = ",".join(b.id for b in relevant_behaviors)

        # Launch agent with behavior hints
        subprocess.run([
            "tmux", "send-keys", "-t", session_name,
            f"PRAXIS_BEHAVIORS={behavior_ids} claude-code --task '{task}'",
            "Enter"
        ])

        self.active_agents[agent_id] = session_name
        return session_name

    def collect_result(self, agent_id: str) -> Optional[str]:
        """Collect result from completed subagent."""
        # Read from subagent's output file or shared memory
        pass
```

#### Pattern 3: Skills with Params (Avoid Bash Mishaps)

From @nexsection: "switching to scripts with params in skills so they don't non-deterministically screw up the bashing of their sub-agents"

```python
@dataclass
class ParameterizedSkill:
    """Skill that executes via structured params, not bash."""
    name: str
    description: str
    parameters: Dict[str, ParameterSpec]
    executor: Callable[[Dict[str, Any]], SkillResult]

    def execute(self, params: Dict[str, Any]) -> SkillResult:
        """Execute with validated params—no bash parsing."""
        validated = self._validate_params(params)
        return self.executor(validated)

# Example: Safe subagent invocation
explore_skill = ParameterizedSkill(
    name="explore_codebase",
    description="Spawn explorer subagent for codebase questions",
    parameters={
        "query": ParameterSpec(type=str, required=True),
        "depth": ParameterSpec(type=str, enum=["quick", "medium", "thorough"]),
        "paths": ParameterSpec(type=List[str], default=["."])
    },
    executor=lambda p: spawn_explorer_agent(p["query"], p["depth"], p["paths"])
)
```

### 7.5 Context Budget Allocation

For a 128k context window, recommended allocation:

```
┌─────────────────────────────────────────────────────────────┐
│                    Context Budget (128k)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  System + Tools (cached)     │████████│          8k (6%)   │
│  Core Behaviors (cached)     │████████████│     12k (9%)   │
│  Retrieved Behaviors         │████│              4k (3%)   │
│  Current Files/Context       │████████████████│ 32k (25%)  │
│  Conversation History        │████████████████│ 32k (25%)  │
│  Working Memory (scratch)    │████████████████│ 32k (25%)  │
│  Safety Buffer               │████│              8k (6%)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.6 Assumptions and Risks

| Assumption | Risk | Validation |
|------------|------|------------|
| Prompt caching is available | Medium | May not work with all providers |
| Static base remains stable | Low | Track base changes per session |
| Subagent isolation reduces cost | Medium | Measure actual token savings |
| tmux is available | Low | Fallback to subprocess |
| Brief handoff is sufficient | High | May need more context for complex tasks |

---

## 8. Implementation Roadmap

### 8.1 Phase 1: Foundation (Weeks 1-3)

**Goal**: Establish handbook infrastructure and basic distillation

```
Tasks:
├─ [ ] Create Behavior and BehaviorMetadata dataclasses
├─ [ ] Implement BehaviorHandbook with file-based storage
├─ [ ] Add FAISS indexing for behaviors
├─ [ ] Implement progressive disclosure protocol
├─ [ ] Create CLI command: rec-praxis-handbook (list, search, add, deprecate)
├─ [ ] Add telemetry events for handbook operations
└─ [ ] Unit tests with 90%+ coverage
```

**Deliverables**:
- `rec_praxis_rlm/handbook.py`
- `rec_praxis_rlm/behaviors.py`
- CLI integration
- Tests

**Success Criteria**:
- Handbook can store and retrieve 1000+ behaviors
- Search latency < 100ms for 1000 behaviors
- Progressive disclosure reduces tokens by 50%+

### 8.2 Phase 2: Distillation Pipeline (Weeks 4-6)

**Goal**: Implement LLM-based behavior distillation

```
Tasks:
├─ [ ] Implement BehaviorDistiller class
├─ [ ] Create distillation prompt templates
├─ [ ] Implement experience clustering (embedding-based)
├─ [ ] Add validation layer for distilled behaviors
├─ [ ] Integrate with existing ObservationCompressor
├─ [ ] Create CLI command: rec-praxis-distill
├─ [ ] Add batch distillation for existing experiences
└─ [ ] Integration tests
```

**Deliverables**:
- `rec_praxis_rlm/distillation.py`
- Prompt templates
- CLI integration
- Benchmarks

**Success Criteria**:
- Distillation produces valid behaviors 80%+ of the time
- Distilled behaviors are 10-20x smaller than source experiences
- Validation catches 90%+ of bad behaviors

### 8.3 Phase 3: Skill Labeling (Weeks 7-8)

**Goal**: Add hierarchical skill taxonomy

```
Tasks:
├─ [ ] Define initial skill taxonomy (JSON schema)
├─ [ ] Implement SkillLabeler with embedding matching
├─ [ ] Add LLM fallback for unmatched skills
├─ [ ] Integrate labels with handbook search
├─ [ ] Add category-based handbook organization
└─ [ ] Tests and documentation
```

**Deliverables**:
- `rec_praxis_rlm/skills.py`
- Skill taxonomy configuration
- Enhanced search

**Success Criteria**:
- 80%+ of experiences get accurate labels
- Taxonomy covers 90%+ of common skills
- Label-enhanced search improves precision by 15%+

### 8.4 Phase 4: Revision Loops (Weeks 9-11)

**Goal**: Implement self-improvement with behavior injection

```
Tasks:
├─ [ ] Implement FailureAnalysis and failure detection
├─ [ ] Create SelfImprovingPlanner extending PraxisRLMPlanner
├─ [ ] Implement behavior injection into prompts
├─ [ ] Add revision trace logging
├─ [ ] Implement escalation handling
├─ [ ] Create feedback loop to update behavior confidence
└─ [ ] Integration tests with failure scenarios
```

**Deliverables**:
- `rec_praxis_rlm/revision.py`
- Enhanced planner
- Trace logging

**Success Criteria**:
- Revision loops improve success rate by 20%+
- Token overhead < 50% compared to single attempt
- Escalation rate < 10%

### 8.5 Phase 5: Subagent Generation (Weeks 12-14)

**Goal**: Enable automated skill generation with human review

```
Tasks:
├─ [ ] Implement SubagentConfig and permission model
├─ [ ] Create SkillGenerationSubagent
├─ [ ] Implement trajectory analysis
├─ [ ] Add sandbox execution environment
├─ [ ] Create CLI command: rec-praxis-reflect
├─ [ ] Implement review queue for proposed skills
└─ [ ] Security audit and penetration testing
```

**Deliverables**:
- `rec_praxis_rlm/subagents.py`
- CLI integration
- Security documentation

**Success Criteria**:
- Subagent generates useful skills 60%+ of the time
- Sandbox prevents all unauthorized access
- Human review takes < 2 minutes per skill

### 8.6 Phase 6: Integration and Optimization (Weeks 15-16)

**Goal**: Full system integration and performance optimization

```
Tasks:
├─ [ ] End-to-end integration testing
├─ [ ] Performance benchmarking (tokens, latency, accuracy)
├─ [ ] Documentation and examples
├─ [ ] Migration guide from current version
├─ [ ] Load testing with 100k+ behaviors
└─ [ ] Production readiness review
```

**Deliverables**:
- Benchmark reports
- Documentation
- Migration guide

**Success Criteria**:
- Token usage reduced by 30%+ on benchmark tasks
- No regression in accuracy
- System stable under load

---

## 9. Success Metrics and Validation

### 9.1 Quantitative Metrics

| Metric | Baseline | Target | Measurement Method |
|--------|----------|--------|-------------------|
| Tokens per planning step | 2500 | 1500 | Average across 100 tasks |
| Cross-task transfer accuracy | N/A | +10% | MATH/WebArena benchmarks |
| Behavior retrieval precision | N/A | 0.80 | Manual evaluation on 100 queries |
| Revision loop success rate | N/A | +20% | A/B test on failing tasks |
| Distillation validity rate | N/A | 80% | Validation layer pass rate |
| Memory scalability | 10k exp | 100k | Load test |

### 9.2 Qualitative Metrics

| Metric | Evaluation Method |
|--------|-------------------|
| Behavior quality | Expert review of 50 random behaviors |
| Skill taxonomy coverage | Gap analysis on 100 diverse experiences |
| Handbook usability | Developer survey (N=10) |
| Revision trace clarity | Debugging session evaluation |
| Documentation completeness | Checklist audit |

### 9.3 Validation Benchmarks

**Recommended Benchmarks**:

1. **MATH** (arXiv papers): Mathematical reasoning
   - Measure: Accuracy, token usage, behavior reuse rate

2. **WebArena**: Web navigation and interaction
   - Measure: Task completion, cross-site transfer

3. **SWE-bench**: Software engineering tasks
   - Measure: Patch accuracy, behavior applicability

4. **Custom CodeReview benchmark** (internal):
   - Measure: False positive rate, finding quality

### 9.4 Regression Testing

```python
# Test suite structure
tests/
├── unit/
│   ├── test_handbook.py
│   ├── test_distillation.py
│   ├── test_skills.py
│   ├── test_revision.py
│   └── test_subagents.py
├── integration/
│   ├── test_end_to_end_distillation.py
│   ├── test_revision_loops.py
│   └── test_subagent_skill_generation.py
├── benchmarks/
│   ├── test_token_efficiency.py
│   ├── test_retrieval_latency.py
│   └── test_accuracy_regression.py
└── security/
    ├── test_sandbox_escape.py
    └── test_permission_enforcement.py
```

---

## Self-Critique: Honest Assessment of This Specification

> Applying the Reflexion pattern: What are we getting wrong?

### Identified Flaws

#### Flaw 1: Overengineering Risk (Severity: HIGH)

**Problem**: This spec proposes 5 new systems for a v0.9.2 product:
- BehaviorHandbook
- BehaviorDistiller
- SkillLabeler
- SelfImprovingPlanner
- SkillGenerationSubagent

**Reality check**: The current system already has:
- ObservationCompressor (80-90% token reduction)
- Progressive disclosure (3-layer retrieval)
- Concept tagging (semantic extraction)
- Experience classification (4 types)

**Question we should ask**: Do users actually need behavior distillation, or is compression sufficient?

**Mitigation**: Define an MVP that adds the smallest valuable increment. Recommendation: Start with BehaviorHandbook + manual behavior creation (no distillation).

#### Flaw 2: Unvalidated Token Savings Claims (Severity: HIGH)

**Problem**: We cite "46% token savings" from arXiv:2509.13237, but:
- That paper tested on MATH benchmarks (mathematical reasoning)
- RecPraxis RLM is for code review, security auditing, web scraping
- Domain transfer is uncertain

**Reality check**: Our compression already achieves 80-90% reduction. Distillation may provide marginal additional benefit at high complexity cost.

**Mitigation**: Before implementing, benchmark actual token usage in RecPraxis RLM workflows. Measure the gap between compression and hypothetical distillation.

#### Flaw 3: No Rollback/Degradation Strategy (Severity: HIGH)

**Problem**: The spec assumes distillation and revision loops succeed. What if:
- Distillation produces incorrect behaviors?
- Revision loops make things worse?
- The handbook becomes corrupted?

**Missing**: Graceful degradation paths.

**Mitigation**: Add fallback strategy:
```
IF behavior retrieval fails:
    → Fall back to raw experience retrieval
IF distillation produces low-confidence behavior:
    → Keep as "draft" in pending_review, use raw experiences
IF revision loop exceeds budget:
    → Return to user with explanation, not failure
```

#### Flaw 4: Missing Cost Analysis (Severity: MEDIUM)

**Problem**: Distillation requires LLM calls. The spec doesn't estimate costs.

**Rough estimate**:
- Distilling 100 experiences → ~100 LLM calls
- At ~$0.003/1k tokens (GPT-4o-mini), ~2k tokens/call → ~$0.60
- Distilling 10,000 experiences → ~$60

**Question**: Is this cost acceptable? For some users yes, for others no.

**Mitigation**: Add cost estimates and make distillation opt-in with budget caps.

#### Flaw 5: Provider Lock-in (Severity: MEDIUM)

**Problem**: Cache-aware architecture assumes Anthropic/OpenAI prompt caching. But:
- Local models (Ollama, vLLM) don't have this
- Some cloud providers don't support it
- Caching behavior varies by provider

**Mitigation**: Make cache-aware optimization a pluggable strategy, not hardcoded. Detect provider capabilities at runtime.

#### Flaw 6: Pre-defined Taxonomy Won't Transfer (Severity: MEDIUM)

**Problem**: The skill taxonomy in Section 3.2 is pre-defined for web scraping, data processing, etc. But:
- ML debugging has different skills
- DevOps has different skills
- Security has different skills

**Mitigation**: Make taxonomy extensible/configurable. Or: Drop taxonomy in favor of pure embedding-based clustering (no predefined labels).

#### Flaw 7: 16-Week Roadmap Has No Dependencies (Severity: LOW)

**Problem**: The roadmap lists 6 phases but doesn't show:
- Which phases can be parallelized
- Critical path dependencies
- Resource requirements

**Mitigation**: Add dependency graph and critical path analysis.

### What the MVP Should Actually Be

Based on this critique, the Minimum Viable Enhancement is:

```
MVP Scope (4 weeks instead of 16):
├─ BehaviorHandbook with file-based storage
├─ Manual behavior creation CLI (no distillation)
├─ Progressive disclosure for handbook
├─ Integration with recall() to prioritize behaviors
└─ Telemetry to measure actual usage patterns

NOT in MVP:
├─ Automatic distillation (wait for usage data)
├─ Skill labeling/taxonomy (unvalidated benefit)
├─ Revision loops (high complexity, uncertain ROI)
└─ Subagent generation (cool but not core)
```

### Validation Before Full Implementation

Before building the full spec, we should:

1. **Measure current pain points**: Survey 5-10 users on actual token/cost issues
2. **Prototype handbook**: Build minimal version, measure adoption
3. **A/B test distillation**: Compare distilled vs. compressed experiences
4. **Collect failure modes**: Run distillation on 100 experiences, review quality

### Revised Success Criteria

| Metric | Original Target | Revised Target | Rationale |
|--------|-----------------|----------------|-----------|
| Token reduction | 30-50% | 20% incremental | Already have 80-90% from compression |
| Implementation time | 16 weeks | 4 weeks (MVP) | De-risk with smaller scope |
| Behaviors created | 1000+ | 100 (manual) | Validate usefulness first |
| User adoption | N/A | 50% of active users | New metric: is this even wanted? |

---

## 11. Appendices

### 11.1 Glossary

| Term | Definition |
|------|------------|
| Behavior | Distilled, reusable action pattern extracted from experiences |
| Skill | Hierarchical label describing what capability an experience/behavior demonstrates |
| Handbook | Searchable repository of behaviors organized by category |
| Distillation | Process of extracting abstract behaviors from concrete experiences |
| Progressive Disclosure | Loading minimal metadata first, full details on demand |
| Revision Loop | Iterative improvement process when initial plan fails |
| Trajectory | Sequence of (action, observation, reasoning) steps from agent execution |

### 11.2 Reference Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RecPraxis RLM v2.0                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ ProceduralMemory│    │ BehaviorHandbook│    │   SkillLabeler  │ │
│  │  (Experiences)  │←──→│   (Behaviors)   │←──→│   (Taxonomy)    │ │
│  └────────┬────────┘    └────────┬────────┘    └─────────────────┘ │
│           │                      │                                  │
│           ↓                      ↓                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    BehaviorDistiller                         │   │
│  │  (Cluster → Extract → Validate → Store)                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 SelfImprovingPlanner                         │   │
│  │  (Plan → Execute → Detect Failure → Retrieve Behaviors →    │   │
│  │   Revise → Retry → Escalate)                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              SkillGenerationSubagent                         │   │
│  │  (Analyze Trajectory → Generate Skill → Validate → Review)  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 11.3 Configuration Schema

```python
@dataclass
class MetacognitiveConfig:
    """Master configuration for metacognitive enhancements."""

    # Handbook settings
    handbook_enabled: bool = True
    handbook_path: Path = Path(".rec-praxis-rlm/handbook")
    handbook_max_behaviors: int = 10000

    # Distillation settings
    distillation_enabled: bool = True
    distillation_min_cluster_size: int = 3
    distillation_similarity_threshold: float = 0.75
    distillation_auto_run: bool = False  # Manual trigger by default
    distillation_llm_model: str = "openai/gpt-4o-mini"

    # Skill labeling settings
    skill_labeling_enabled: bool = True
    skill_taxonomy_path: Optional[Path] = None  # Use default if None
    skill_llm_fallback: bool = True

    # Revision loop settings
    revision_enabled: bool = True
    revision_max_iterations: int = 3
    revision_escalation_threshold: int = 2
    revision_behavior_limit: int = 3

    # Subagent settings
    subagent_enabled: bool = False  # Opt-in due to complexity
    subagent_sandbox_level: str = "strict"
    subagent_require_approval: bool = True
    subagent_max_execution_time: float = 30.0

    # Observability
    trace_revisions: bool = True
    trace_distillation: bool = True
    emit_metrics: bool = True
```

### 11.4 Migration Guide (v0.9.x → v2.0)

```markdown
## Migrating to RecPraxis RLM v2.0

### Breaking Changes
1. None - all existing APIs preserved

### New Features (Opt-in)
1. BehaviorHandbook - enable via MetacognitiveConfig
2. Distillation - run `rec-praxis-distill` manually
3. Skill labels - auto-enabled for new experiences
4. Revision loops - use SelfImprovingPlanner instead of PraxisRLMPlanner

### Recommended Migration Path
1. Upgrade package: `pip install rec-praxis-rlm>=2.0`
2. Initialize handbook: `rec-praxis-handbook init`
3. Distill existing experiences: `rec-praxis-distill --all`
4. Review and approve behaviors
5. Enable revision loops in config
```

### 11.5 Security Considerations

| Component | Threat | Mitigation |
|-----------|--------|------------|
| Distillation | Prompt injection via experiences | Sanitize experience text |
| Handbook | Path traversal | Validate paths, use pathlib |
| Subagent | Sandbox escape | Strict capabilities, timeout |
| Behaviors | Code injection | No executable code in behaviors |
| Search | ReDoS | Existing ReDoS protection applies |

### 11.6 Open Questions for Discussion

1. **Behavior Versioning**: Should we support multiple versions of the same behavior, or always replace?

2. **Confidence Decay**: Should behavior confidence decrease over time without usage/validation?

3. **Cross-User Sharing**: Should behaviors be sharable between users/teams? Privacy implications?

4. **Fine-tuning Integration**: How closely should we integrate with DSPy optimization?

5. **Evaluation Dataset**: Should we create a public benchmark for behavior quality?

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0-draft | 2025-12-25 | AI-Assisted | Initial specification |

---

## Feedback and Review

This specification is open for review. Please provide feedback on:
- Completeness of edge case analysis
- Realism of success metrics
- Priority ordering of phases
- Missing security considerations
- Alternative approaches not considered
