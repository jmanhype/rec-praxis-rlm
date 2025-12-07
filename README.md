# rec-praxis-rlm

**Procedural Memory + REPL Context for Autonomous AI Agents**

A Python package that provides persistent procedural memory and safe code execution capabilities for DSPy 3.0 autonomous agents, enabling experience-based learning and programmatic document manipulation.

[![PyPI version](https://img.shields.io/pypi/v/rec-praxis-rlm.svg)](https://pypi.org/project/rec-praxis-rlm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Downloads](https://img.shields.io/pypi/dm/rec-praxis-rlm.svg)](https://pypi.org/project/rec-praxis-rlm/)
[![Tests](https://github.com/jmanhype/rec-praxis-rlm/actions/workflows/test.yml/badge.svg)](https://github.com/jmanhype/rec-praxis-rlm/actions/workflows/test.yml)
[![Test Coverage](https://img.shields.io/badge/coverage-99.38%25-brightgreen.svg)](https://github.com/jmanhype/rec-praxis-rlm)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/jmanhype/rec-praxis-rlm?style=social)](https://github.com/jmanhype/rec-praxis-rlm/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/jmanhype/rec-praxis-rlm?style=social)](https://github.com/jmanhype/rec-praxis-rlm/network/members)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da?logo=discord&logoColor=white)](https://discord.gg/rec-praxis-rlm)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/jmanhype/rec-praxis-rlm/pulls)

## Features

### Core Capabilities
- **Procedural Memory**: Store and retrieve agent experiences with hybrid similarity scoring (environmental + goal embeddings)
- **FAISS Indexing**: 10-100x faster retrieval at scale (>10k experiences)
- **RLM Context**: Programmatic document inspection (grep, peek, head, tail) with ReDoS protection
- **Safe Code Execution**: Sandboxed Python REPL with AST validation and restricted builtins
- **DSPy 3.0 Integration**: Autonomous planning with ReAct agents and integrated tools
- **MLflow Observability**: Automatic tracing and experiment tracking
- **Production Ready**: 99.38% test coverage, comprehensive error handling, backward-compatible storage versioning

### IDE Integrations & Developer Tools (v0.4.0+)
- **Pre-commit Hooks**: Automated code review, security audit, and dependency scanning before git commits
- **VS Code Extension**: Real-time inline diagnostics with procedural memory-powered suggestions
- **GitHub Actions**: CI/CD workflows for automated security scanning on pull requests
- **CLI Tools**: Command-line interface for integration into any development workflow
- **Learning from Fixes**: Agents remember and apply successful code improvements across sessions

## Requirements

### Core Features (No API Key Required)

The following features work out-of-the-box without any API keys:

- **Procedural Memory**: Uses local `sentence-transformers` for embeddings
- **RLM Context**: Document inspection (grep, peek, head, tail) and safe code execution
- **FAISS Indexing**: Optional performance optimization for large-scale retrieval

### Optional Features (API Key Required)

- **DSPy Autonomous Planning**: Requires an API key from one of these providers:
  - **Groq** (recommended - fast and free): `export GROQ_API_KEY="gsk-..."`
  - **OpenAI**: `export OPENAI_API_KEY="sk-..."`
  - **OpenRouter** (access to many models): `export OPENROUTER_API_KEY="sk-or-..."`
  - Any LiteLLM-supported provider

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

# Option 1: Programmatic API key (recommended for Groq)
planner = PraxisRLMPlanner(
    memory=memory,
    config=PlannerConfig(
        lm_model="groq/llama-3.3-70b-versatile",
        api_key="gsk-..."  # Pass key directly
    )
)

# Option 2: Environment variables (works for all providers)
# import os
# os.environ["GROQ_API_KEY"] = "gsk-..."
# planner = PraxisRLMPlanner(
#     memory=memory,
#     config=PlannerConfig(lm_model="groq/llama-3.3-70b-versatile")
# )

# Option 3: OpenAI with programmatic key
# planner = PraxisRLMPlanner(
#     memory=memory,
#     config=PlannerConfig(
#         lm_model="openai/gpt-4o-mini",
#         api_key="sk-..."
#     )
# )

# Option 4: OpenRouter with programmatic key
# planner = PraxisRLMPlanner(
#     memory=memory,
#     config=PlannerConfig(
#         lm_model="openrouter/meta-llama/llama-3.2-3b-instruct:free",
#         api_key="sk-or-..."
#     )
# )

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

## Supported LLM Providers

For DSPy autonomous planning, rec-praxis-rlm supports any LiteLLM-compatible provider:

### Groq (Recommended)
Fast, free API with high rate limits.

```python
import os
os.environ["GROQ_API_KEY"] = "gsk-..."

planner = PraxisRLMPlanner(
    memory=memory,
    config=PlannerConfig(lm_model="groq/llama-3.3-70b-versatile")
)
```

**Available models**: `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, `gemma2-9b-it`

### OpenAI
Industry standard with highest quality models.

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

planner = PraxisRLMPlanner(
    memory=memory,
    config=PlannerConfig(lm_model="openai/gpt-4o-mini")
)
```

**Available models**: `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`

### OpenRouter
Access to 200+ models from multiple providers.

```python
import os
os.environ["OPENROUTER_API_KEY"] = "sk-or-..."

planner = PraxisRLMPlanner(
    memory=memory,
    config=PlannerConfig(lm_model="openrouter/meta-llama/llama-3.2-3b-instruct:free")
)
```

**Available models**: See [OpenRouter models](https://openrouter.ai/models)

### Other Providers
Any LiteLLM-supported provider works: Anthropic, Cohere, Azure, AWS Bedrock, etc.

```python
# Anthropic Claude
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
planner = PraxisRLMPlanner(
    memory=memory,
    config=PlannerConfig(lm_model="anthropic/claude-3-5-sonnet-20241022")
)
```

See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for full list.

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

**Configuration Presets** (v0.4.3+):

Simplify configuration with task-optimized presets:

```python
# Code review preset (precise, successful experiences only)
config = MemoryConfig.for_code_review()

# Security audit preset (broad, includes false positives for learning)
config = MemoryConfig.for_security_audit()

# Web scraping preset (prioritizes site structure)
config = MemoryConfig.for_web_scraping()

# Test generation preset (high precision for test patterns)
config = MemoryConfig.for_testing()
```

**Preset Comparison**:

| Preset | top_k | similarity_threshold | env_weight | goal_weight | require_success | Best For |
|--------|-------|---------------------|------------|-------------|-----------------|----------|
| `for_code_review()` | 4 | 0.7 (high) | 0.3 | 0.7 | True | Precise code quality patterns |
| `for_security_audit()` | 8 | 0.4 (low) | 0.5 | 0.5 | False | Diverse vulnerability detection |
| `for_web_scraping()` | 6 | 0.5 (medium) | 0.7 | 0.3 | True | Site structure similarity |
| `for_testing()` | 5 | 0.75 (very high) | 0.2 | 0.8 | True | Test coverage patterns |

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
    api_key="sk-...",                  # Optional API key (or use env vars)
    temperature=0.0,                   # Sampling temperature
    max_iters=10,                      # Max ReAct iterations
    enable_mlflow_tracing=True,        # MLflow observability
    optimizer="miprov2",               # DSPy optimizer
    optimizer_auto_level="medium",     # Automation level
    use_toon_adapter=False             # Enable TOON format for 40% token reduction (experimental)
)
```

**TOON Format Support (Experimental)**:

Enable TOON (Token-Oriented Object Notation) for ~40% token reduction in DSPy prompts:

```python
# Install TOON support
# pip install rec-praxis-rlm[toon]

config = PlannerConfig(
    lm_model="openai/gpt-4o-mini",
    use_toon_adapter=True  # Enable TOON format
)

planner = PraxisRLMPlanner(memory, config)
# All DSPy interactions now use TOON format for efficiency
```

**Benefits**:
- ~40% reduction in prompt tokens (saves API costs)
- Faster inference (fewer tokens to process)
- Same accuracy as JSON format

**Compatibility**: Requires `dspy-toon>=0.1.0` (install with `pip install rec-praxis-rlm[toon]`)

**Note**: TOON support is experimental in v0.4.1. Future versions (v0.6.0+) will integrate TOON into procedural memory storage for further efficiency gains. See [Issue #1](https://github.com/jmanhype/rec-praxis-rlm/issues/1) for roadmap.

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

## IDE Integrations & Developer Tools

### Pre-commit Hooks

Automatically review code, audit security, and scan dependencies before every commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/jmanhype/rec-praxis-rlm
    rev: v0.4.0
    hooks:
      - id: rec-praxis-review      # Code review (fail on HIGH+)
      - id: rec-praxis-audit        # Security audit (fail on CRITICAL)
      - id: rec-praxis-deps         # Dependency & secret scan
```

Install and run:

```bash
pip install pre-commit rec-praxis-rlm[all]
pre-commit install
git commit -m "feat: add new feature"  # Hooks run automatically
```

### CLI Tools

Use rec-praxis-rlm from the command line:

```bash
# Code review (human-readable format)
rec-praxis-review src/**/*.py --severity=HIGH

# Code review (JSON for IDE integration)
rec-praxis-review src/**/*.py --severity=HIGH --format=json

# Code review (TOON format for 40% token reduction)
rec-praxis-review src/**/*.py --severity=HIGH --format=toon

# Code review (SARIF for GitHub Security tab)
rec-praxis-review src/**/*.py --format=sarif > code-review.sarif

# Security audit
rec-praxis-audit app.py --fail-on=CRITICAL --format=sarif > security-audit.sarif

# Dependency & secret scan
rec-praxis-deps --requirements=requirements.txt --files src/config.py --format=sarif > deps.sarif
```

**Output Formats**:
- **human** (default): Colorful, emoji-rich output for terminal viewing
- **json**: Structured JSON for IDE integration and programmatic parsing
- **toon**: Token-efficient format providing ~40% token reduction (experimental)
- **sarif**: SARIF v2.1.0 format for GitHub Security tab integration (v0.4.3+)

**Features**:
- Configurable severity thresholds
- Persistent procedural memory (learns from past reviews)
- Exit codes for CI/CD pipelines
- TOON format support for cost-effective LLM integration

### VS Code Extension

Install the "rec-praxis-rlm Code Intelligence" extension from the VS Code Marketplace, or build from source:

**Repository**: [github.com/jmanhype/rec-praxis-rlm-vscode](https://github.com/jmanhype/rec-praxis-rlm-vscode)

**Features**:
- **Inline Diagnostics**: See code review and security findings as you type
- **Context Menu**: Right-click to review/audit current file
- **Auto-review on Save**: Real-time feedback (configurable)
- **Dependency Scanning**: Right-click `requirements.txt` to scan for CVEs
- **Procedural Memory Integration**: Learns from past fixes across sessions

**Settings** (F1 → "Preferences: Open Settings (JSON)"):

```json
{
  "rec-praxis-rlm.pythonPath": "python",
  "rec-praxis-rlm.codeReview.severity": "HIGH",
  "rec-praxis-rlm.securityAudit.failOn": "CRITICAL",
  "rec-praxis-rlm.enableDiagnostics": true,
  "rec-praxis-rlm.autoReviewOnSave": false
}
```

**Installation**:
```bash
# From VS Code
# 1. Open Extensions (Ctrl+Shift+X / Cmd+Shift+X)
# 2. Search for "rec-praxis-rlm"
# 3. Click Install

# From source (for developers)
git clone https://github.com/jmanhype/rec-praxis-rlm-vscode.git
cd rec-praxis-rlm-vscode
npm install && npm run compile
npm run package  # Creates .vsix file
# Install .vsix via VS Code: Extensions → ... → Install from VSIX
```

See the [VS Code extension repository](https://github.com/jmanhype/rec-praxis-rlm-vscode) for full documentation.

### GitHub Actions

Automatically scan pull requests for security issues:

```yaml
# .github/workflows/rec-praxis-scan.yml
name: Security Scan

on: [pull_request]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install rec-praxis-rlm[all]
      - run: rec-praxis-review $(git diff --name-only --diff-filter=ACMR origin/main...HEAD | grep '\.py$')
      - run: rec-praxis-audit $(git diff --name-only --diff-filter=ACMR origin/main...HEAD | grep '\.py$')
      - run: rec-praxis-deps --requirements=requirements.txt --fail-on=CRITICAL
```

**Features**:
- Automatic PR comments with findings
- Artifact uploads for review results
- Configurable severity thresholds
- Supports matrix builds (Python 3.10+)
- **Dogfooding**: This repo uses its own tools to scan the `examples/` directory on every push

**Dogfooding Workflow**:

The rec-praxis-rlm project dogfoods its own tools by scanning the `examples/` directory on every push to main:

```yaml
# .github/workflows/rec-praxis-scan.yml (dogfood-examples job)
dogfood-examples:
  name: Dogfood on Examples
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
    - run: pip install -e .[all]  # Install from source
    - run: rec-praxis-review examples/*.py --severity=MEDIUM --json
    - run: rec-praxis-audit examples/*.py --fail-on=HIGH --json
    - run: rec-praxis-deps --requirements=requirements.txt --files examples/*.py
```

This demonstrates:
- **Self-validation**: The tools scan themselves for quality issues
- **Real-world usage**: Shows the tools working on production code
- **Continuous improvement**: Catches regressions in example code
- **Non-blocking**: Uses `continue-on-error: true` to show findings without failing CI

View dogfooding results in the [GitHub Actions artifacts](https://github.com/jmanhype/rec-praxis-rlm/actions).

See `.github/workflows/rec-praxis-scan.yml` for the full workflow implementation.

## Examples

See the `examples/` directory for complete examples:

- `quickstart.py` - Basic memory and context usage
- `log_analyzer.py` - Log analysis with RLM context
- `web_agent.py` - Web scraping agent with procedural memory
- `optimization.py` - DSPy MIPROv2 optimizer usage
- `code_review_agent.py` - Intelligent code review with procedural memory
- `security_audit_agent.py` - OWASP-based security auditing
- `dependency_scan_agent.py` - CVE detection and secret scanning

## Community & Contributing

### Contributing Guidelines

We welcome contributions from the community! Here's how you can help:

**Ways to Contribute:**
- 🐛 Report bugs via [GitHub Issues](https://github.com/jmanhype/rec-praxis-rlm/issues)
- 💡 Propose features or improvements
- 📖 Improve documentation
- 🔧 Submit bug fixes or new features via pull requests
- ⭐ Star the repository to show support
- 💬 Join discussions in [GitHub Discussions](https://github.com/jmanhype/rec-praxis-rlm/discussions)

**Development Setup:**

```bash
# Clone and install in development mode
git clone https://github.com/jmanhype/rec-praxis-rlm.git
cd rec-praxis-rlm
pip install -e .[dev]

# Run tests
pytest --cov=rec_praxis_rlm

# Run linters
ruff check .
black --check .
mypy rec_praxis_rlm

# Run security audit on your changes
bandit -r rec_praxis_rlm
```

**Pull Request Process:**

1. Fork the repository and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass (`pytest`)
4. Run linters (`ruff`, `black`, `mypy`)
5. Update documentation as needed
6. Submit PR with clear description of changes

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Report unacceptable behavior to the maintainers

### Getting Help

- 📚 **Documentation**: [GitHub README](https://github.com/jmanhype/rec-praxis-rlm#readme)
- 💬 **Discord Community**: [Join our Discord](https://discord.gg/rec-praxis-rlm) for real-time chat
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/jmanhype/rec-praxis-rlm/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/jmanhype/rec-praxis-rlm/discussions)
- 📧 **Email**: jmanhype@users.noreply.github.com (for security issues)

### Recognition

Contributors are recognized in our [CONTRIBUTORS.md](CONTRIBUTORS.md) file. Thank you to all who have helped improve rec-praxis-rlm!

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
