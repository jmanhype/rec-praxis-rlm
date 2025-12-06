# rec-praxis-rlm

**Procedural Memory + REPL Context for Autonomous AI Agents**

A Python package that provides persistent procedural memory and safe code execution capabilities for DSPy 3.0 autonomous agents, enabling experience-based learning and programmatic document manipulation.

[![PyPI version](https://img.shields.io/pypi/v/rec-praxis-rlm.svg)](https://pypi.org/project/rec-praxis-rlm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/jmanhype/rec-praxis-rlm/actions/workflows/test.yml/badge.svg)](https://github.com/jmanhype/rec-praxis-rlm/actions/workflows/test.yml)
[![Test Coverage](https://img.shields.io/badge/coverage-99.38%25-brightgreen.svg)](https://github.com/jmanhype/rec-praxis-rlm)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Procedural Memory**: Store and retrieve agent experiences with hybrid similarity scoring (environmental + goal embeddings)
- **FAISS Indexing**: 10-100x faster retrieval at scale (>10k experiences)
- **RLM Context**: Programmatic document inspection (grep, peek, head, tail) with ReDoS protection
- **Safe Code Execution**: Sandboxed Python REPL with AST validation and restricted builtins
- **DSPy 3.0 Integration**: Autonomous planning with ReAct agents and integrated tools
- **MLflow Observability**: Automatic tracing and experiment tracking
- **Production Ready**: 99.38% test coverage, comprehensive error handling, backward-compatible storage versioning

## Requirements

### Core Features (No API Key Required)

The following features work out-of-the-box without any API keys:

- **Procedural Memory**: Uses local `sentence-transformers` for embeddings
- **RLM Context**: Document inspection (grep, peek, head, tail) and safe code execution
- **FAISS Indexing**: Optional performance optimization for large-scale retrieval

### Optional Features (API Key Required)

- **DSPy Autonomous Planning**: Requires OpenAI API key or compatible language model provider
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```

## Quick Start

### Installation

```bash
# Basic installation (works without API key)
pip install rec-praxis-rlm

# With all optional dependencies (FAISS, OpenAI, async support)
pip install rec-praxis-rlm[all]

# Development installation
pip install rec-praxis-rlm[dev]
```

### Example 1: Procedural Memory

```python
from rec_praxis_rlm.memory import ProceduralMemory, Experience
from rec_praxis_rlm.config import MemoryConfig

# Initialize memory
config = MemoryConfig(storage_path="./agent_memory.jsonl")
memory = ProceduralMemory(config)

# Store experiences
memory.store(Experience(
    env_features=["web_scraping", "python", "beautifulsoup"],
    goal="extract product prices from e-commerce site",
    action="Used BeautifulSoup with CSS selectors for price elements",
    result="Successfully extracted 1000 prices with 99% accuracy",
    success=True
))

# Recall similar experiences
experiences = memory.recall(
    env_features=["web_scraping", "python"],
    goal="extract data from website",
    top_k=5
)

for exp in experiences:
    print(f"Similarity: {exp.similarity_score:.2f}")
    print(f"Action: {exp.action}")
    print(f"Result: {exp.result}\n")
```

### Example 2: RLM Context for Document Inspection

```python
from rec_praxis_rlm.rlm import RLMContext
from rec_praxis_rlm.config import ReplConfig

# Initialize context
config = ReplConfig()
context = RLMContext(config)

# Add documents
with open("application.log", "r") as f:
    context.add_document("app_log", f.read())

# Search for patterns
matches = context.grep(r"ERROR.*database", doc_id="app_log")
for match in matches:
    print(f"Line {match.line_number}: {match.match_text}")
    print(f"Context: ...{match.context_before}{match.match_text}{match.context_after}...")

# Extract specific ranges
error_section = context.peek("app_log", start_char=1000, end_char=2000)

# Get first/last N lines
recent_logs = context.tail("app_log", n_lines=50)
```

### Example 3: Safe Code Execution

```python
from rec_praxis_rlm.rlm import RLMContext

context = RLMContext()

# Execute safe code
result = context.safe_exec("""
total = 0
for i in range(10):
    total += i * 2
total
""")

if result.success:
    print(f"Output: {result.output}")
    print(f"Execution time: {result.execution_time_seconds:.3f}s")
else:
    print(f"Error: {result.error}")

# Prohibited operations are blocked
result = context.safe_exec("import os; os.system('rm -rf /')")
# Result: ExecutionError - Import statements not allowed
```

### Example 4: Autonomous Planning with DSPy

```python
from rec_praxis_rlm.dspy_agent import PraxisRLMPlanner
from rec_praxis_rlm.memory import ProceduralMemory
from rec_praxis_rlm.config import PlannerConfig, MemoryConfig

# Initialize memory and planner
memory = ProceduralMemory(MemoryConfig())
planner = PraxisRLMPlanner(
    memory=memory,
    config=PlannerConfig(lm_model="openai/gpt-4o-mini")
)

# Add context for document inspection
from rec_praxis_rlm.rlm import RLMContext
context = RLMContext()
context.add_document("logs", open("server.log").read())
planner.add_context(context, "server_logs")

# Autonomous planning
answer = planner.plan(
    goal="Analyze server errors and suggest fixes",
    env_features=["production", "high_traffic", "database"]
)
print(answer)
```

## Architecture

```
┌─────────────────────────────────────────┐
│     PraxisRLMPlanner (DSPy ReAct)       │
│   Autonomous decision-making layer      │
├─────────────────┬───────────────────────┤
│                 │                       │
│    Tools        │    Tools              │
│                 │                       │
▼                 ▼                       ▼
┌─────────────┐  ┌──────────────┐  ┌─────────────┐
│ Procedural  │  │  RLMContext  │  │   External  │
│   Memory    │  │   (Facade)   │  │    APIs     │
├─────────────┤  ├──────────────┤  └─────────────┘
│ • recall()  │  │ DocumentStore│
│ • store()   │  │ DocSearcher  │
│ • compact() │  │ CodeExecutor │
├─────────────┤  └──────────────┘
│ Embeddings  │
│ ┌─────────┐ │
│ │ Local   │ │  FAISS Index (optional)
│ │ API     │ │  ┌──────────────┐
│ │ Jaccard │◄─┼──┤ 10-100x      │
│ └─────────┘ │  │ faster search│
└─────────────┘  └──────────────┘
       │
       ▼
  Storage (JSONL)
  • Append-only
  • Versioned
  • Crash-safe
```

## Performance

| Operation | Without FAISS | With FAISS | Speedup |
|-----------|---------------|------------|---------|
| Recall (100 exp) | ~2ms | ~2ms | 1x |
| Recall (1,000 exp) | ~20ms | ~3ms | 6.7x |
| Recall (10,000 exp) | ~200ms | ~20ms | 10x |
| Recall (100,000 exp) | ~2000ms | ~20ms | 100x |

| Operation | Performance | Notes |
|-----------|-------------|-------|
| Document grep (10MB) | <500ms | With ReDoS protection |
| Safe code execution | <100ms | Sandboxed environment |
| Memory loading (10k exp) | <1s | With lazy loading |

## Configuration

### Memory Configuration

```python
from rec_praxis_rlm.config import MemoryConfig

config = MemoryConfig(
    storage_path="./memory.jsonl",
    top_k=6,                          # Number of experiences to retrieve
    similarity_threshold=0.5,         # Minimum similarity score
    env_weight=0.6,                   # Weight for environmental features
    goal_weight=0.4,                  # Weight for goal similarity
    require_success=False,            # Only retrieve successful experiences
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    result_size_limit=50000           # Max result size in bytes
)
```

### REPL Configuration

```python
from rec_praxis_rlm.config import ReplConfig

config = ReplConfig(
    max_output_chars=10000,           # Max output capture
    max_search_matches=100,           # Max grep results
    search_context_chars=200,         # Context before/after match
    execution_timeout_seconds=5.0,    # Code execution timeout
    enable_sandbox=True,              # Use sandboxed execution
    log_executions=True,              # Log for audit trail
    allowed_builtins=[                # Allowed built-in functions
        "len", "range", "sum", "max", "min", "sorted", ...
    ]
)
```

### Planner Configuration

```python
from rec_praxis_rlm.config import PlannerConfig

config = PlannerConfig(
    lm_model="openai/gpt-4o-mini",    # Language model
    temperature=0.0,                   # Sampling temperature
    max_iters=10,                      # Max ReAct iterations
    enable_mlflow_tracing=True,        # MLflow observability
    optimizer="miprov2",               # DSPy optimizer
    optimizer_auto_level="medium"      # Automation level
)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rec_praxis_rlm --cov-report=html

# Run specific test suites
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests

# Run performance tests
pytest tests/unit/test_memory.py -k "performance"
```

Current test coverage: **99.38%** (327 passing tests)

## Security

### Sandboxed Code Execution

The `SafeExecutor` provides multiple layers of security:

1. **AST Validation**: Blocks imports, eval, exec, file I/O, network access
2. **Restricted Builtins**: Only safe functions allowed (configurable)
3. **Execution Timeout**: Prevents infinite loops
4. **Output Limiting**: Prevents memory exhaustion
5. **Code Hashing**: Audit trail for all executed code

**Blocked operations**:
- All imports (`import`, `from ... import`)
- Dangerous builtins (`eval`, `exec`, `__import__`, `compile`, `open`)
- File system access
- Network access
- Privileged attributes (`__class__`, `__globals__`, `__dict__`)

### ReDoS Protection

The `DocumentSearcher` validates regex patterns to prevent Regular Expression Denial of Service attacks:

- Pattern length limits (<500 chars)
- Nested quantifier detection (`(a+)+`)
- Excessive wildcard detection (>3 instances of `.*` or `.+`)
- Overlapping alternation warnings

## Advanced Features

### Async Support

```python
import asyncio
from rec_praxis_rlm.memory import ProceduralMemory
from rec_praxis_rlm.rlm import RLMContext

async def main():
    memory = ProceduralMemory(config)
    context = RLMContext(config)

    # Async memory recall
    experiences = await memory.arecall(
        env_features=["python"],
        goal="debug error"
    )

    # Async code execution
    result = await context.asafe_exec("sum(range(1000000))")

asyncio.run(main())
```

### Custom Embedding Providers

```python
from rec_praxis_rlm.embeddings import APIEmbedding
from rec_praxis_rlm.memory import ProceduralMemory

# Use OpenAI embeddings
embedding_provider = APIEmbedding(
    api_provider="openai",
    api_key="sk-...",
    model_name="text-embedding-3-small"
)

memory = ProceduralMemory(
    config,
    embedding_provider=embedding_provider
)
```

### Memory Maintenance

```python
# Compact memory (remove old/low-value experiences)
memory.compact(max_size=1000, min_similarity=0.7)

# Recompute embeddings (after changing embedding model)
new_provider = SentenceTransformerEmbedding("new-model")
memory.recompute_embeddings(new_provider)
```

### Custom Metrics

```python
from rec_praxis_rlm.metrics import memory_retrieval_quality, SemanticF1Score

# Memory retrieval quality metric
score = memory_retrieval_quality(
    example={"env_features": [...], "goal": "...", "expected_success_rate": 0.8},
    prediction=retrieved_experiences
)

# Semantic F1 scoring for DSPy optimization
f1_metric = SemanticF1Score(relevance_threshold=0.7)
score = f1_metric(example, prediction)
```

## MLflow Integration

```python
from rec_praxis_rlm.telemetry import setup_mlflow_tracing

# Enable automatic MLflow tracing
setup_mlflow_tracing(experiment_name="my-agent-experiment")

# All DSPy operations are now traced automatically
planner = PraxisRLMPlanner(memory, config)
result = planner.plan(goal="...", env_features=[...])

# View traces in MLflow UI
# mlflow ui --port 5000
```

## Examples

See the `examples/` directory for complete examples:

- `quickstart.py` - Basic memory and context usage
- `log_analyzer.py` - Log analysis with RLM context
- `web_agent.py` - Web scraping agent with procedural memory
- `optimization.py` - DSPy MIPROv2 optimizer usage

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use rec-praxis-rlm in your research, please cite:

```bibtex
@software{rec_praxis_rlm,
  title = {rec-praxis-rlm: Procedural Memory and REPL Context for Autonomous Agents},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-org/rec-praxis-rlm}
}
```

## Acknowledgments

- Built on [DSPy 3.0](https://github.com/stanfordnlp/dspy) for autonomous agent capabilities
- Uses [sentence-transformers](https://www.sbert.net/) for semantic embeddings
- Integrated with [MLflow](https://mlflow.org/) for experiment tracking
- [FAISS](https://github.com/facebookresearch/faiss) for fast similarity search

## Support

- **Documentation**: [Full API docs](https://github.com/your-org/rec-praxis-rlm#readme)
- **Issues**: [GitHub Issues](https://github.com/your-org/rec-praxis-rlm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/rec-praxis-rlm/discussions)
