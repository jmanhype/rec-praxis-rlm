# Architecture

This document describes the architecture of rec-praxis-rlm, including design decisions, component relationships, and implementation patterns.

## Table of Contents

- [Overview](#overview)
- [Design Principles](#design-principles)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Storage Format](#storage-format)
- [Performance Optimizations](#performance-optimizations)
- [Security Model](#security-model)
- [Testing Strategy](#testing-strategy)

---

## Overview

rec-praxis-rlm provides three core capabilities for autonomous AI agents:

1. **Procedural Memory** - Experience-based learning with hybrid similarity scoring
2. **RLM Context** - Programmatic document inspection and manipulation
3. **Autonomous Planning** - DSPy ReAct agents with integrated tools

The architecture follows SOLID principles with a focus on modularity, testability, and performance.

---

## Design Principles

### 1. Single Responsibility Principle (SRP)

Each module has a focused responsibility:

- **memory.py**: Experience storage and retrieval
- **rlm.py**: Document management and code execution (facade pattern)
- **sandbox.py**: Safe code execution
- **embeddings.py**: Embedding computation and caching
- **dspy_agent.py**: Autonomous planning coordination

### 2. Open/Closed Principle (OCP)

- **EmbeddingProvider**: Abstract base class allows adding new providers without modifying existing code
- **Storage**: Versioned JSONL format supports backward-compatible schema evolution

### 3. Dependency Inversion Principle (DIP)

- ProceduralMemory depends on EmbeddingProvider abstraction, not concrete implementations
- PraxisRLMPlanner depends on ProceduralMemory and RLMContext interfaces

### 4. Fail-Safe Defaults

- Sandbox enabled by default
- ReDoS protection always active
- Strict AST validation for all code

### 5. Performance by Design

- LRU caching for embeddings (10-100x speedup)
- FAISS indexing for large-scale retrieval (100x speedup at 100k+ experiences)
- Lazy loading for storage files
- Append-only writes for crash safety

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  PraxisRLMPlanner (DSPy ReAct)              │
│                  Autonomous decision-making                 │
└───────────────────┬───────────────┬─────────────────────────┘
                    │               │
         ┌──────────▼──────┐   ┌───▼─────────────────┐
         │  Procedural     │   │   RLMContext        │
         │  Memory         │   │   (Facade)          │
         └─────────────────┘   └─────────────────────┘
                │                       │
    ┌───────────┴───────────┐   ┌──────┴──────────┐
    │                       │   │                 │
    ▼                       ▼   ▼                 ▼
┌─────────┐          ┌─────────────┐      ┌──────────────┐
│Embedding│          │   Storage   │      │DocumentStore │
│Provider │          │   Manager   │      │DocumentSearch│
└─────────┘          └─────────────┘      │CodeExecutor  │
    │                       │              └──────────────┘
    ▼                       ▼                      │
┌─────────┐          ┌─────────────┐              ▼
│   LRU   │          │JSONL Storage│      ┌──────────────┐
│  Cache  │          │Append-Only  │      │ SafeExecutor │
└─────────┘          └─────────────┘      │AST Validator │
                            │              └──────────────┘
                            ▼
                    ┌──────────────┐
                    │FAISS Index   │
                    │(optional)    │
                    └──────────────┘
```

---

## Component Details

### Procedural Memory

**Responsibility:** Store and retrieve agent experiences using hybrid similarity scoring

**Key Classes:**
- `ProceduralMemory`: Main memory interface
- `Experience`: Pydantic model for experiences
- `_StorageManager`: Handles JSONL persistence
- `_FAISSIndex`: Optional fast similarity search

**Hybrid Scoring Algorithm:**

```python
similarity = (env_weight * env_similarity) + (goal_weight * goal_similarity)

# Default weights:
# - env_weight = 0.6 (environmental context is more important)
# - goal_weight = 0.4 (goal alignment)
```

**Why Hybrid Scoring?**

Single embedding approaches (combining env + goal into one vector) lose the ability to weight environmental context vs. goal alignment differently. Hybrid scoring allows:
- Prioritizing experiences from similar environments (60%)
- While still considering goal relevance (40%)
- Configurable weights for different use cases

**Performance Characteristics:**

| Operation | Without FAISS | With FAISS | Speedup |
|-----------|---------------|------------|---------|
| Recall (100 exp) | ~2ms | ~2ms | 1x |
| Recall (1,000 exp) | ~20ms | ~3ms | 6.7x |
| Recall (10,000 exp) | ~200ms | ~20ms | 10x |
| Recall (100,000 exp) | ~2000ms | ~20ms | 100x |

---

### RLM Context (Facade Pattern)

**Responsibility:** Coordinate document storage, search, and code execution

**Architecture Decision:** Facade Pattern (Phase 6 refactoring)

Previously, RLMContext was a monolithic class handling all operations. After SRP refactoring:

```
RLMContext (Facade)
├── DocumentStore: Manages document storage/retrieval
├── DocumentSearcher: Handles grep, peek, head, tail
└── CodeExecutor: Manages safe execution
```

**Benefits:**
- Each component has single responsibility
- Better testability (can test components in isolation)
- Easier to extend (add new search algorithms without touching storage)
- Backward compatible (public API unchanged)

**ReDoS Protection:**

The DocumentSearcher validates regex patterns before execution:

```python
# Blocked patterns:
- Nested quantifiers: (a+)+
- Excessive wildcards: .* or .+ more than 3 times
- Pattern length > 500 characters
```

**Why ReDoS Protection Matters:**

Malicious regex patterns can cause exponential backtracking:
```python
# This pattern can hang for hours on 30-character input:
pattern = r"(a+)+"
text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaX"
```

Our protection detects these patterns before execution.

---

### Safe Code Execution

**Responsibility:** Execute Python code in restricted environment

**Security Layers:**

1. **AST Validation** (Pre-execution):
   - Parse code to Abstract Syntax Tree
   - Block prohibited nodes (Import, ImportFrom)
   - Block prohibited names (eval, exec, open, __import__)
   - Block prohibited attributes (__class__, __globals__)

2. **Restricted Builtins** (Runtime):
   - Only safe functions allowed: len, range, sum, max, min, sorted, enumerate, etc.
   - No file I/O, no network access, no introspection

3. **Execution Timeout** (Runtime):
   - Default: 5 seconds
   - Prevents infinite loops

4. **Output Limiting** (Post-execution):
   - Max output: 10,000 characters
   - Prevents memory exhaustion

**Why AST Validation + Restricted Builtins?**

AST validation alone is insufficient - Python has many ways to access dangerous functionality:
```python
# AST allows this (it's just a call), but restricted builtins block it:
getattr(__builtins__, 'open')('secret.txt')
```

Combining both layers provides defense-in-depth.

---

### Embedding System

**Responsibility:** Compute and cache text embeddings

**Architecture:**

```
EmbeddingProvider (ABC)
├── SentenceTransformerEmbedding (local, no API key)
│   └── LRU Cache (10,000 entries)
├── APIEmbedding (OpenAI, requires API key)
│   └── LRU Cache (10,000 entries)
└── TextSimilarityFallback (Jaccard, no embeddings)
```

**LRU Cache Design:**

Cache keys: SHA256 hash of text (prevents cache pollution from similar texts)

```python
cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
```

**Cache Performance:**

For typical agent workflows (repetitive queries):
- Cache hit rate: 60-90%
- Speedup: 10-100x (embedding computation is expensive)
- Memory footprint: ~40MB for 10,000 cached 384-dim embeddings

**Why LRU Eviction?**

Recent embeddings are more likely to be reused (temporal locality). LRU eviction keeps hot embeddings in cache.

---

### DSPy Integration

**Responsibility:** Autonomous planning with ReAct agents

**Tool Architecture:**

```python
PraxisRLMPlanner
├── RecallExperiencesTool(memory)
├── SearchContextTool(contexts)
└── ExecuteCodeTool(contexts)
```

**ReAct Cycle:**

```
1. Thought: "I need to find past experiences with web scraping"
2. Action: recall_experiences(env=["web"], goal="extract data")
3. Observation: [Experience 1, Experience 2, ...]
4. Thought: "These experiences suggest using BeautifulSoup"
5. Action: search_context(pattern="BeautifulSoup", doc_id="docs")
6. Observation: [Match at line 42, ...]
7. Thought: "I have enough information to answer"
8. Answer: "Use BeautifulSoup with CSS selectors..."
```

**MLflow Tracing:**

Automatic instrumentation tracks:
- Tool calls (recall, search, execute)
- LM calls (thoughts, actions)
- Execution times
- Token usage

---

## Data Flow

### Experience Storage Flow

```
1. Agent calls memory.store(experience)
   ↓
2. ProceduralMemory computes embeddings
   - env_embedding = provider.embed(env_features)
   - goal_embedding = provider.embed(goal)
   ↓
3. StorageManager appends to JSONL
   - Atomic write (rename for crash safety)
   - Versioned format (v1.0)
   ↓
4. FAISS index updated (if enabled)
   - Add embeddings to index
   ↓
5. Return to agent
```

### Experience Retrieval Flow

```
1. Agent calls memory.recall(env_features, goal)
   ↓
2. ProceduralMemory computes query embeddings
   - query_env_emb = provider.embed(env_features)
   - query_goal_emb = provider.embed(goal)
   ↓
3. Similarity search
   - Without FAISS: Linear scan, cosine similarity
   - With FAISS: Fast approximate nearest neighbors
   ↓
4. Hybrid scoring
   - score = 0.6 * env_sim + 0.4 * goal_sim
   ↓
5. Filter by threshold and top_k
   ↓
6. Return ranked experiences
```

### Document Search Flow

```
1. Agent calls context.grep(pattern)
   ↓
2. DocumentSearcher validates regex safety
   - Check pattern length < 500
   - Check for nested quantifiers
   - Check for excessive wildcards
   ↓
3. Compile regex
   ↓
4. Search document lines
   - Track line numbers
   - Compute character offsets
   - Extract context (before/after)
   ↓
5. Limit matches (default: 100)
   ↓
6. Emit telemetry event
   ↓
7. Return SearchMatch list
```

---

## Storage Format

### JSONL Format

Each experience is stored as one JSON line:

```json
{"version":"1.0","env_features":["web","python"],"goal":"extract prices","action":"BeautifulSoup + CSS","result":"Extracted 100 prices","success":true,"timestamp":1701619200.0,"env_embedding":[0.1,0.2,...],"goal_embedding":[0.3,0.4,...]}
```

**Why JSONL?**

- Append-only (crash-safe)
- Streamable (can process line-by-line)
- Human-readable (debugging)
- Schema versioning (future migrations)

**Version Migration Path:**

Future schema changes:
```python
# v1.0 → v2.0 migration
if record["version"] == "1.0":
    # Add new field with default
    record["confidence_score"] = 1.0
    record["version"] = "2.0"
```

### FAISS Index Format

Optional binary index for fast similarity search:

```
{storage_path}.faiss
{storage_path}.faiss.pkl  # Metadata (embedding dimension, index type)
```

**Index Type:** IVF (Inverted File Index) with nprobe=10

**Rebuild Strategy:**
- Built on first access if missing
- Rebuilt on embedding model change
- Incremental updates on store()

---

## Performance Optimizations

### 1. Embedding Cache (LRU)

**Impact:** 10-100x speedup for repeated queries

**Implementation:**
```python
self._cache: OrderedDict[str, list[float]] = OrderedDict()

if cache_key in self._cache:
    self._cache.move_to_end(cache_key)  # Mark as recently used
    return self._cache[cache_key]
```

### 2. FAISS Indexing

**Impact:** 100x speedup at 100k+ experiences

**Trade-offs:**
- Memory overhead: ~4 bytes per dimension per vector
- Build time: ~1 second for 10k experiences
- Accuracy: 95-99% recall with nprobe=10

### 3. Lazy Loading

**Impact:** Faster startup for large memory files

**Implementation:**
- Storage loaded on first recall() call
- FAISS index built on demand
- Embeddings computed only when needed

### 4. Batch Operations

**Impact:** 3-5x speedup for bulk operations

**Implementation:**
```python
# Bad: N API calls
for text in texts:
    embed(text)

# Good: 1 API call
embed_batch(texts)
```

### 5. Async Support

**Impact:** Non-blocking operations for web servers

**Implementation:**
```python
# Run CPU-bound embedding in thread pool
loop = asyncio.get_event_loop()
return await loop.run_in_executor(self._async_executor, self.execute, code)
```

---

## Security Model

### Threat Model

**In Scope:**
- Malicious code execution (eval, exec, file I/O)
- ReDoS attacks (exponential regex backtracking)
- Memory exhaustion (infinite loops, large outputs)
- Privilege escalation (introspection, __globals__ access)

**Out of Scope:**
- Side-channel attacks (timing, cache)
- Resource exhaustion from valid operations
- Network-level attacks

### Defense Layers

1. **Input Validation**
   - AST validation for code
   - Regex safety checks for patterns

2. **Runtime Restrictions**
   - Restricted builtins namespace
   - Execution timeout
   - Output size limiting

3. **Audit Trail**
   - All code executions logged with SHA-256 hash
   - Telemetry events for security-relevant operations

### Known Limitations

**Not a VM:** SafeExecutor runs in the same Python process. Determined attackers may find escape vectors.

**Recommended:** For untrusted code, use OS-level sandboxing (Docker, gVisor, Firecracker).

---

## Testing Strategy

### Test Coverage: 99.38% (327 tests)

**Test Pyramid:**

```
      E2E (5%)
     /      \
    /  Integ  \
   /   (15%)   \
  /             \
 /    Unit       \
/     (80%)       \
```

### Unit Tests (265 tests)

**Focus:** Single component behavior

Examples:
- `test_memory.py`: Storage, retrieval, similarity scoring
- `test_rlm.py`: Document operations, search, execution
- `test_sandbox.py`: AST validation, restricted builtins
- `test_embeddings.py`: Caching, batch operations

**Property-Based Testing (14 tests):**

Using Hypothesis library:
```python
@given(st.lists(st.text(), min_size=1))
def test_embed_batch_length_matches_input(texts):
    embeddings = provider.embed_batch(texts)
    assert len(embeddings) == len(texts)
```

### Integration Tests (57 tests)

**Focus:** Component interactions

Examples:
- Memory + Embeddings: End-to-end storage and retrieval
- RLM + Sandbox: Document search with code execution
- DSPy + Memory + RLM: Full agent workflow

### E2E Tests (5 tests)

**Focus:** Full user workflows

Examples:
- Web scraping agent with memory recall
- Log analysis with autonomous planning
- Multi-session memory persistence

### Performance Tests

**Focus:** Benchmarks for SLAs

```python
def test_recall_performance_1k_experiences():
    # Setup: 1000 experiences
    # Assert: recall() < 100ms

def test_search_performance_10mb_document():
    # Setup: 10MB log file
    # Assert: grep() < 500ms
```

### Security Tests

**Focus:** Sandbox escapes and ReDoS

```python
def test_sandbox_blocks_file_io():
    with pytest.raises(ExecutionError):
        executor.execute("open('/etc/passwd')")

def test_redos_protection():
    with pytest.raises(SearchError):
        searcher.grep(r"(a+)+")
```

---

## Design Decisions

### Why JSONL instead of SQLite?

**Pros:**
- Simpler (no DB dependencies)
- Crash-safe (append-only)
- Portable (plain text)
- Version control friendly

**Cons:**
- Slower for complex queries
- No transactions
- Manual indexing

**Decision:** JSONL + FAISS provides good balance for agent memory (<1M experiences).

For >1M experiences, consider migrating to a vector database (Pinecone, Weaviate, Milvus).

---

### Why Pydantic instead of dataclasses?

**Pros:**
- Runtime validation
- JSON serialization built-in
- Better error messages
- Field validators

**Cons:**
- Slightly slower than dataclasses
- Additional dependency

**Decision:** Runtime validation is critical for agent safety (prevent malformed experiences).

---

### Why DSPy instead of LangChain?

**Pros:**
- Declarative signatures
- Automatic optimization (MIPROv2)
- Better observability (MLflow integration)
- Type safety

**Cons:**
- Newer library (less mature)
- Smaller community
- Fewer integrations

**Decision:** DSPy's optimization capabilities align better with autonomous agent use cases.

---

## Future Architecture Considerations

### 1. Distributed Memory

For multi-agent systems:
- Shared memory backend (Redis, PostgreSQL)
- Conflict resolution (last-write-wins, CRDT)
- Partitioning strategies (by agent, by domain)

### 2. Memory Consolidation

For long-running agents:
- Hierarchical memory (working memory → long-term memory)
- Importance weighting (decay old experiences)
- Memory compression (cluster similar experiences)

### 3. Advanced Embeddings

- Multi-modal embeddings (text + images)
- Fine-tuned embeddings (domain-specific)
- Cross-encoder reranking (improve top-k quality)

### 4. Enhanced Security

- WebAssembly sandbox (browser-level isolation)
- Capability-based security (fine-grained permissions)
- Formal verification (prove safety properties)

---

## References

- [DSPy Documentation](https://github.com/stanfordnlp/dspy)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [ReDoS Attacks](https://owasp.org/www-community/attacks/Regular_expression_Denial_of_Service_-_ReDoS)

---

## See Also

- [API Reference](api_reference.md)
- [Examples](../examples/)
- [Contributing Guide](../CONTRIBUTING.md)
