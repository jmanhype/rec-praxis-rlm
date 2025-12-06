# API Reference

Complete API reference for rec-praxis-rlm.

## Table of Contents

- [Memory Module](#memory-module)
- [RLM Module](#rlm-module)
- [Sandbox Module](#sandbox-module)
- [Embeddings Module](#embeddings-module)
- [DSPy Agent Module](#dspy-agent-module)
- [Configuration Module](#configuration-module)
- [Metrics Module](#metrics-module)
- [Telemetry Module](#telemetry-module)
- [Tools Module](#tools-module)
- [Exceptions](#exceptions)

---

## Memory Module

### `ProceduralMemory`

Main class for storing and retrieving agent experiences with hybrid similarity scoring.

#### Constructor

```python
ProceduralMemory(
    config: MemoryConfig,
    embedding_provider: Optional[EmbeddingProvider] = None
)
```

**Parameters:**
- `config` (MemoryConfig): Memory configuration
- `embedding_provider` (Optional[EmbeddingProvider]): Custom embedding provider (defaults to SentenceTransformerEmbedding)

**Example:**
```python
from rec_praxis_rlm import ProceduralMemory, MemoryConfig

memory = ProceduralMemory(MemoryConfig(storage_path="./memory.jsonl"))
```

#### Methods

##### `store(experience: Experience) -> None`

Store a new experience in memory.

**Parameters:**
- `experience` (Experience): Experience to store

**Raises:**
- `StorageError`: If storage write fails

**Example:**
```python
from rec_praxis_rlm import Experience
import time

memory.store(Experience(
    env_features=["web", "python"],
    goal="extract data from website",
    action="Used BeautifulSoup with CSS selectors",
    result="Successfully extracted 100 items",
    success=True,
    timestamp=time.time()
))
```

##### `recall(env_features: list[str], goal: str, top_k: Optional[int] = None) -> list[Experience]`

Retrieve similar experiences using hybrid similarity scoring.

**Parameters:**
- `env_features` (list[str]): Environmental feature tags
- `goal` (str): Goal description
- `top_k` (Optional[int]): Number of results (defaults to config.top_k)

**Returns:**
- `list[Experience]`: Top-k most similar experiences, sorted by similarity score

**Example:**
```python
experiences = memory.recall(
    env_features=["web", "python"],
    goal="scrape product data",
    top_k=5
)

for exp in experiences:
    print(f"Similarity: {exp.similarity_score:.2f}")
    print(f"Action: {exp.action}")
```

##### `arecall(env_features: list[str], goal: str, top_k: Optional[int] = None) -> list[Experience]`

Async version of `recall()` for non-blocking retrieval.

**Parameters:** Same as `recall()`

**Returns:** Same as `recall()`

**Example:**
```python
import asyncio

async def main():
    experiences = await memory.arecall(
        env_features=["web"],
        goal="extract data",
        top_k=3
    )
    print(f"Retrieved {len(experiences)} experiences")

asyncio.run(main())
```

##### `compact(max_size: int, min_similarity: float = 0.0) -> int`

Remove low-value experiences to maintain memory size.

**Parameters:**
- `max_size` (int): Maximum number of experiences to keep
- `min_similarity` (float): Minimum similarity threshold (0.0 to 1.0)

**Returns:**
- `int`: Number of experiences removed

**Example:**
```python
# Keep only top 1000 experiences with similarity >= 0.7
removed = memory.compact(max_size=1000, min_similarity=0.7)
print(f"Removed {removed} low-value experiences")
```

##### `recompute_embeddings(new_provider: EmbeddingProvider) -> None`

Recompute all embeddings using a new embedding provider.

**Parameters:**
- `new_provider` (EmbeddingProvider): New embedding provider

**Raises:**
- `EmbeddingError`: If recomputation fails

**Example:**
```python
from rec_praxis_rlm.embeddings import APIEmbedding

# Switch from local to API embeddings
new_provider = APIEmbedding(api_provider="openai", api_key="sk-...")
memory.recompute_embeddings(new_provider)
```

##### `size() -> int`

Get the number of stored experiences.

**Returns:**
- `int`: Number of experiences

**Example:**
```python
print(f"Memory contains {memory.size()} experiences")
```

---

### `Experience`

Pydantic model representing a stored experience.

#### Fields

- `env_features` (list[str]): Environmental feature tags (e.g., ["web", "python"])
- `goal` (str): Goal description
- `action` (str): Action taken
- `result` (str): Result obtained
- `success` (bool): Whether action succeeded
- `timestamp` (float): Unix timestamp
- `env_embedding` (Optional[list[float]]): Environmental feature embedding (auto-computed)
- `goal_embedding` (Optional[list[float]]): Goal embedding (auto-computed)
- `similarity_score` (float): Similarity score from recall (0.0 to 1.0)

#### Example

```python
from rec_praxis_rlm import Experience
import time

exp = Experience(
    env_features=["database", "postgresql"],
    goal="optimize slow query",
    action="Added index on user_id column",
    result="Query time reduced from 2000ms to 50ms",
    success=True,
    timestamp=time.time()
)
```

---

## RLM Module

### `RLMContext`

Facade for programmatic document inspection and safe code execution.

#### Constructor

```python
RLMContext(config: ReplConfig = ReplConfig())
```

**Parameters:**
- `config` (ReplConfig): REPL configuration

**Example:**
```python
from rec_praxis_rlm import RLMContext, ReplConfig

context = RLMContext(ReplConfig(max_search_matches=100))
```

#### Document Management

##### `add_document(doc_id: str, text: str) -> None`

Add a document to the context.

**Parameters:**
- `doc_id` (str): Document identifier
- `text` (str): Document text content

**Raises:**
- `ValueError`: If doc_id already exists

**Example:**
```python
with open("server.log", "r") as f:
    context.add_document("server_log", f.read())
```

##### `remove_document(doc_id: str) -> None`

Remove a document from context.

**Parameters:**
- `doc_id` (str): Document identifier

**Raises:**
- `DocumentNotFoundError`: If doc_id not found

**Example:**
```python
context.remove_document("server_log")
```

#### Search Operations

##### `grep(pattern: str, doc_id: Optional[str] = None, max_matches: Optional[int] = None) -> list[SearchMatch]`

Search documents for pattern using regex with ReDoS protection.

**Parameters:**
- `pattern` (str): Regular expression pattern
- `doc_id` (Optional[str]): Document ID to search (searches all if None)
- `max_matches` (Optional[int]): Maximum matches (defaults to config.max_search_matches)

**Returns:**
- `list[SearchMatch]`: Search matches with context

**Raises:**
- `SearchError`: If regex is invalid or potentially dangerous

**Example:**
```python
# Search for error patterns
matches = context.grep(r"ERROR.*database", doc_id="server_log")

for match in matches:
    print(f"Line {match.line_number}: {match.match_text}")
    print(f"Context: ...{match.context_before}{match.match_text}{match.context_after}...")
```

##### `peek(doc_id: str, start_char: int, end_char: int) -> str`

Extract a character range from document.

**Parameters:**
- `doc_id` (str): Document identifier
- `start_char` (int): Start character offset
- `end_char` (int): End character offset

**Returns:**
- `str`: Extracted text

**Example:**
```python
# Extract characters 1000-2000
section = context.peek("server_log", 1000, 2000)
```

##### `head(doc_id: str, n_lines: int = 10) -> str`

Get first N lines of document.

**Parameters:**
- `doc_id` (str): Document identifier
- `n_lines` (int): Number of lines to return (default: 10)

**Returns:**
- `str`: First N lines

**Example:**
```python
first_50 = context.head("server_log", n_lines=50)
```

##### `tail(doc_id: str, n_lines: int = 10) -> str`

Get last N lines of document.

**Parameters:**
- `doc_id` (str): Document identifier
- `n_lines` (int): Number of lines to return (default: 10)

**Returns:**
- `str`: Last N lines

**Example:**
```python
recent_logs = context.tail("server_log", n_lines=100)
```

#### Code Execution

##### `safe_exec(code: str, context_vars: Optional[dict[str, Any]] = None) -> ExecutionResult`

Execute code safely in sandboxed environment.

**Parameters:**
- `code` (str): Python code to execute
- `context_vars` (Optional[dict[str, Any]]): Variables to inject

**Returns:**
- `ExecutionResult`: Execution result with output and metadata

**Raises:**
- `ExecutionError`: If code validation fails

**Example:**
```python
result = context.safe_exec(
    code="sum(range(100))",
    context_vars={}
)

if result.success:
    print(f"Output: {result.output}")
    print(f"Execution time: {result.execution_time_seconds:.3f}s")
else:
    print(f"Error: {result.error}")
```

##### `asafe_exec(code: str, context_vars: Optional[dict[str, Any]] = None) -> ExecutionResult`

Async version of `safe_exec()` for non-blocking execution.

**Parameters:** Same as `safe_exec()`

**Returns:** Same as `safe_exec()`

**Example:**
```python
import asyncio

async def main():
    result = await context.asafe_exec("sum(range(1000000))")
    print(result.output)

asyncio.run(main())
```

---

### `SearchMatch`

Pydantic model representing a search match result.

#### Fields

- `doc_id` (str): Document identifier
- `line_number` (int): Line number (1-indexed)
- `match_text` (str): The matched text
- `context_before` (str): Context before the match
- `context_after` (str): Context after the match
- `start_char` (int): Character offset where match starts
- `end_char` (int): Character offset where match ends

---

### `ExecutionResult`

Pydantic model representing code execution result.

#### Fields

- `success` (bool): Whether execution succeeded
- `output` (str): Captured output
- `error` (Optional[str]): Error message if failed
- `execution_time_seconds` (float): Execution time in seconds
- `code_hash` (str): SHA-256 hash of executed code (audit trail)

---

## Sandbox Module

### `SafeExecutor`

Safe code executor with sandboxed environment and AST validation.

#### Constructor

```python
SafeExecutor(config: ReplConfig)
```

**Parameters:**
- `config` (ReplConfig): REPL configuration

#### Methods

##### `execute(code: str, context_vars: Optional[dict[str, Any]] = None) -> _SandboxResult`

Execute code in sandboxed environment.

**Parameters:**
- `code` (str): Python code to execute
- `context_vars` (Optional[dict]): Variables to inject

**Returns:**
- `_SandboxResult`: Internal execution result

**Blocked Operations:**
- All imports (`import`, `from ... import`)
- Dangerous builtins (`eval`, `exec`, `__import__`, `compile`, `open`)
- File system access
- Network access
- Privileged attributes (`__class__`, `__globals__`, `__dict__`)

---

## Embeddings Module

### `EmbeddingProvider`

Abstract base class for embedding providers.

#### Methods

##### `embed(text: str) -> list[float]` (abstract)

Embed a single text string.

##### `embed_batch(texts: list[str]) -> list[list[float]]` (abstract)

Embed multiple texts.

---

### `SentenceTransformerEmbedding`

Local embedding using sentence-transformers library with LRU cache.

#### Constructor

```python
SentenceTransformerEmbedding(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache_size: int = 10000
)
```

**Parameters:**
- `model_name` (str): Sentence-transformers model name
- `cache_size` (int): Maximum cache entries (default: 10,000)

**Example:**
```python
from rec_praxis_rlm.embeddings import SentenceTransformerEmbedding

provider = SentenceTransformerEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
    cache_size=50000
)

embedding = provider.embed("Hello world")
print(f"Embedding dimension: {len(embedding)}")
```

---

### `APIEmbedding`

API-based embedding (OpenAI) with LRU cache.

#### Constructor

```python
APIEmbedding(
    api_provider: str = "openai",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    cache_size: int = 10000
)
```

**Parameters:**
- `api_provider` (str): API provider ("openai")
- `api_key` (Optional[str]): API key
- `model_name` (Optional[str]): Model name (default: "text-embedding-3-small")
- `cache_size` (int): Maximum cache entries (default: 10,000)

**Example:**
```python
from rec_praxis_rlm.embeddings import APIEmbedding

provider = APIEmbedding(
    api_provider="openai",
    api_key="sk-...",
    model_name="text-embedding-3-large"
)

embedding = provider.embed("Hello world")
```

---

### `TextSimilarityFallback`

Text-based similarity fallback using Jaccard similarity (no embeddings).

#### Methods

##### `compute_similarity(text1: str, text2: str) -> float`

Compute Jaccard similarity between two texts.

**Returns:**
- `float`: Similarity score (0.0 to 1.0)

**Example:**
```python
from rec_praxis_rlm.embeddings import TextSimilarityFallback

fallback = TextSimilarityFallback()
score = fallback.compute_similarity("hello world", "hello there")
print(f"Similarity: {score:.2f}")
```

---

## DSPy Agent Module

### `PraxisRLMPlanner`

Autonomous planner using DSPy ReAct agent with integrated tools.

#### Constructor

```python
PraxisRLMPlanner(
    memory: ProceduralMemory,
    config: PlannerConfig
)
```

**Parameters:**
- `memory` (ProceduralMemory): Procedural memory instance
- `config` (PlannerConfig): Planner configuration

**Example:**
```python
from rec_praxis_rlm import PraxisRLMPlanner, ProceduralMemory, PlannerConfig, MemoryConfig

memory = ProceduralMemory(MemoryConfig())
planner = PraxisRLMPlanner(
    memory=memory,
    config=PlannerConfig(lm_model="openai/gpt-4o-mini")
)
```

#### Methods

##### `add_context(context: RLMContext, namespace: str) -> None`

Add an RLM context for document inspection.

**Parameters:**
- `context` (RLMContext): Context instance
- `namespace` (str): Namespace identifier

**Example:**
```python
from rec_praxis_rlm import RLMContext

context = RLMContext()
context.add_document("logs", open("server.log").read())
planner.add_context(context, "server_logs")
```

##### `plan(goal: str, env_features: list[str], max_iters: Optional[int] = None) -> str`

Generate a plan to achieve the goal using ReAct reasoning.

**Parameters:**
- `goal` (str): Goal description
- `env_features` (list[str]): Environmental features
- `max_iters` (Optional[int]): Max iterations (defaults to config.max_iters)

**Returns:**
- `str`: Final answer/plan

**Example:**
```python
answer = planner.plan(
    goal="Analyze server errors and suggest fixes",
    env_features=["production", "high_traffic", "database"]
)
print(answer)
```

---

## Configuration Module

### `MemoryConfig`

Configuration for procedural memory.

#### Fields

- `storage_path` (str): Path to JSONL storage file (use ":memory:" for in-memory)
- `top_k` (int): Number of experiences to retrieve (default: 6)
- `similarity_threshold` (float): Minimum similarity score (default: 0.5)
- `env_weight` (float): Weight for environmental features (default: 0.6)
- `goal_weight` (float): Weight for goal similarity (default: 0.4)
- `require_success` (bool): Only retrieve successful experiences (default: False)
- `embedding_model` (str): Embedding model name (default: "sentence-transformers/all-MiniLM-L6-v2")
- `result_size_limit` (int): Max result size in bytes (default: 50,000)
- `use_faiss` (bool): Enable FAISS indexing (default: False)
- `faiss_nprobe` (int): FAISS search probe count (default: 10)

**Example:**
```python
from rec_praxis_rlm.config import MemoryConfig

config = MemoryConfig(
    storage_path="./agent_memory.jsonl",
    top_k=10,
    similarity_threshold=0.7,
    use_faiss=True
)
```

---

### `ReplConfig`

Configuration for REPL context and safe execution.

#### Fields

- `max_output_chars` (int): Max output capture (default: 10,000)
- `max_search_matches` (int): Max grep results (default: 100)
- `search_context_chars` (int): Context before/after match (default: 200)
- `execution_timeout_seconds` (float): Code execution timeout (default: 5.0)
- `enable_sandbox` (bool): Use sandboxed execution (default: True)
- `log_executions` (bool): Log for audit trail (default: True)
- `allowed_builtins` (list[str]): Allowed built-in functions

**Example:**
```python
from rec_praxis_rlm.config import ReplConfig

config = ReplConfig(
    max_search_matches=200,
    execution_timeout_seconds=10.0,
    allowed_builtins=["len", "range", "sum", "max", "min", "sorted", "enumerate"]
)
```

---

### `PlannerConfig`

Configuration for DSPy autonomous planner.

#### Fields

- `lm_model` (str): Language model (default: "openai/gpt-4o-mini")
- `temperature` (float): Sampling temperature (default: 0.0)
- `max_iters` (int): Max ReAct iterations (default: 10)
- `enable_mlflow_tracing` (bool): MLflow observability (default: True)
- `optimizer` (str): DSPy optimizer (default: "miprov2")
- `optimizer_auto_level` (str): Automation level (default: "medium")

**Example:**
```python
from rec_praxis_rlm.config import PlannerConfig

config = PlannerConfig(
    lm_model="openai/gpt-4o",
    temperature=0.7,
    max_iters=15,
    enable_mlflow_tracing=True
)
```

---

## Metrics Module

### `memory_retrieval_quality`

Metric for evaluating memory retrieval quality.

#### Signature

```python
memory_retrieval_quality(
    example: dict,
    prediction: list[Experience],
    trace: Optional[Any] = None
) -> float
```

**Parameters:**
- `example` (dict): Example with keys: env_features, goal, expected_success_rate
- `prediction` (list[Experience]): Retrieved experiences
- `trace` (Optional[Any]): DSPy trace (unused)

**Returns:**
- `float`: Quality score (0.0 to 1.0)

**Example:**
```python
from rec_praxis_rlm.metrics import memory_retrieval_quality

score = memory_retrieval_quality(
    example={
        "env_features": ["web", "python"],
        "goal": "extract data",
        "expected_success_rate": 0.8
    },
    prediction=retrieved_experiences
)
print(f"Retrieval quality: {score:.2f}")
```

---

### `SemanticF1Score`

Semantic F1 scoring for DSPy optimization.

#### Constructor

```python
SemanticF1Score(relevance_threshold: float = 0.7)
```

**Parameters:**
- `relevance_threshold` (float): Minimum similarity for relevance (default: 0.7)

#### Methods

##### `__call__(example: dict, prediction: list[Experience], trace: Optional[Any] = None) -> float`

Compute semantic F1 score.

**Returns:**
- `float`: F1 score (0.0 to 1.0)

**Example:**
```python
from rec_praxis_rlm.metrics import SemanticF1Score

metric = SemanticF1Score(relevance_threshold=0.75)
score = metric(example, prediction)
print(f"F1 Score: {score:.2f}")
```

---

## Telemetry Module

### `setup_mlflow_tracing`

Setup MLflow tracing for DSPy operations.

#### Signature

```python
setup_mlflow_tracing(experiment_name: str = "dspy-praxis-rlm") -> None
```

**Parameters:**
- `experiment_name` (str): MLflow experiment name

**Example:**
```python
from rec_praxis_rlm.telemetry import setup_mlflow_tracing

setup_mlflow_tracing(experiment_name="my-agent-experiment")

# All DSPy operations are now traced
planner = PraxisRLMPlanner(memory, config)
result = planner.plan(goal="...", env_features=[...])

# View traces: mlflow ui --port 5000
```

---

### `emit_event`

Emit a telemetry event (internal).

#### Signature

```python
emit_event(event_type: str, metadata: dict[str, Any]) -> None
```

---

## Tools Module

DSPy tool implementations for memory recall, context search, and code execution.

### `RecallExperiencesTool`

Tool for recalling past experiences from procedural memory.

---

### `SearchContextTool`

Tool for searching context documents with grep.

---

### `ExecuteCodeTool`

Tool for safe code execution.

---

## Exceptions

### `ProceduralMemoryError`

Base exception for all memory-related errors.

---

### `StorageError`

Storage read/write errors.

---

### `EmbeddingError`

Embedding computation errors.

---

### `DocumentNotFoundError`

Document lookup errors.

---

### `SearchError`

Search operation errors (invalid regex, ReDoS detected).

---

### `ExecutionError`

Code execution validation errors.

---

## Version Information

**Current Version:** 0.1.0

**Minimum Python Version:** 3.10+

**Dependencies:**
- dspy-ai >= 3.0.4
- pydantic >= 2.0
- sentence-transformers >= 2.2 (optional)
- jsonlines >= 3.0
- mlflow >= 3.0
- faiss-cpu or faiss-gpu (optional)
- openai (optional, for API embeddings)

---

## See Also

- [Architecture Documentation](architecture.md)
- [Examples](../examples/)
- [Contributing Guide](../CONTRIBUTING.md)
