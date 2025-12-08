# rec-praxis-rlm

**Procedural Memory + REPL Context for Autonomous AI Agents**

A Python package that provides persistent procedural memory and safe code execution capabilities for DSPy 3.0 autonomous agents, enabling experience-based learning and programmatic document manipulation.

[![PyPI version](https://img.shields.io/pypi/v/rec-praxis-rlm.svg)](https://pypi.org/project/rec-praxis-rlm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Test Coverage](https://img.shields.io/badge/coverage-99.38%25-brightgreen.svg)](https://github.com/jmanhype/rec-praxis-rlm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

### Core Capabilities
- **Procedural Memory**: Store and retrieve agent experiences with hybrid similarity scoring (environmental + goal embeddings)
- **FAISS Indexing**: 10-100x faster retrieval at scale (>10k experiences)
- **RLM Context**: Programmatic document inspection (grep, peek, head, tail) with ReDoS protection
- **Safe Code Execution**: Sandboxed Python REPL with AST validation and restricted builtins
- **DSPy 3.0 Integration**: Autonomous planning with ReAct agents and integrated tools
- **MLflow Observability**: Automatic tracing and experiment tracking
- **Production Ready**: 99.38% test coverage, comprehensive error handling

### Developer Tools (v0.4.0+)
- **CLI Tools**: Code review, security audit, dependency scanning from command line
- **Claude Code Hooks**: Automatic experience capture after every tool use (zero-config)
- **Pre-commit Hooks**: Automated quality checks before every git commit
- **VS Code Extension**: Real-time inline diagnostics with procedural memory
- **GitHub Actions**: CI/CD workflows for automated security scanning
- **HTML Reports**: Interactive security reports with charts and filtering (v0.4.4+)
- **SARIF Support**: GitHub Security tab integration (v0.4.3+)

## Quick Start

```bash
# Install
pip install rec-praxis-rlm

# With FAISS (10-100x faster)
pip install rec-praxis-rlm[all]
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
```

### Example 2: RLM Context for Document Inspection

```python
from rec_praxis_rlm.rlm import RLMContext

context = RLMContext()
context.add_document("app_log", open("application.log").read())

# Search for patterns
matches = context.grep(r"ERROR.*database", doc_id="app_log")
for match in matches:
    print(f"Line {match.line_number}: {match.match_text}")

# Get last N lines
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
else:
    print(f"Error: {result.error}")
```

### Example 4: DSPy Autonomous Planning

```python
from rec_praxis_rlm.dspy_agent import PraxisRLMPlanner
from rec_praxis_rlm.config import PlannerConfig

# Works with Groq (free), OpenAI, or any LiteLLM provider
planner = PraxisRLMPlanner(
    memory=memory,
    config=PlannerConfig(
        lm_model="groq/llama-3.3-70b-versatile",
        api_key="gsk-..."  # or use GROQ_API_KEY env var
    )
)

# Autonomous planning
answer = planner.plan(
    goal="Analyze server errors and suggest fixes",
    env_features=["production", "high_traffic", "database"]
)
```

## CLI Tools

```bash
# Code review (human-readable format)
rec-praxis-review src/**/*.py --severity=HIGH

# Code review (JSON for IDE integration)
rec-praxis-review src/**/*.py --format=json

# Code review (Interactive HTML report)
rec-praxis-review src/**/*.py --format=html --output=security-report.html

# Security audit
rec-praxis-audit app.py --fail-on=CRITICAL

# Dependency & secret scan
rec-praxis-deps --requirements=requirements.txt --files src/**/*.py
```

**Output Formats**:
- **human** (default): Colorful terminal output
- **json**: Structured for IDE integration
- **html**: Interactive reports with charts (v0.4.4+)
- **sarif**: GitHub Security tab integration (v0.4.3+)

## Configuration Presets (v0.4.3+)

Simplify configuration with task-optimized presets:

```python
from rec_praxis_rlm.config import MemoryConfig

# Code review preset (precise, successful experiences only)
config = MemoryConfig.for_code_review()

# Security audit preset (broad, includes false positives for learning)
config = MemoryConfig.for_security_audit()

# Web scraping preset (prioritizes site structure)
config = MemoryConfig.for_web_scraping()

# Test generation preset (high precision for test patterns)
config = MemoryConfig.for_testing()
```

## Claude Code Hooks (Automatic Experience Capture)

**Zero-config automatic learning** - rec-praxis-rlm integrates with Claude Code to automatically capture every tool use as an experience:

```bash
# Already set up in .claude/settings.json!
# Every Bash command, file read/write, grep, etc. is automatically captured
```

**Features**:
- **post_tool_use hook**: Captures tool name, input, output, success status
- **session_start hook**: Shows recent successes/failures at session start
- **Privacy-aware**: Automatically redacts API keys, passwords, emails
- **Local storage**: All data stays in `.claude/memory.jsonl` on your machine

**Example session start:**
```
📚 REC Praxis RLM Context

Memory Statistics:
- Total experiences: 127
- Recent successful patterns: 5
- Recent failures to avoid: 2

Recent Successful Patterns:
1. [optimize] Refactor database query for better performance
   ✓ Query latency reduced from 2s to 50ms

Recent Failures (Learn from these):
1. [recover] Fix test failures in test_privacy.py
   ✗ Still failing - need to adjust pattern length
```

See [.claude/README.md](https://github.com/jmanhype/rec-praxis-rlm/blob/main/.claude/README.md) for full documentation.

## Pre-commit Hooks

Automatically review code before every commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/jmanhype/rec-praxis-rlm
    rev: v0.9.2
    hooks:
      - id: rec-praxis-review      # Code review (fail on HIGH+)
      - id: rec-praxis-audit        # Security audit (fail on CRITICAL)
      - id: rec-praxis-deps         # Dependency & secret scan
```

```bash
pip install pre-commit rec-praxis-rlm[all]
pre-commit install
git commit -m "feat: add new feature"  # Hooks run automatically
```

## GitHub Actions

### Option 1: Use the Official GitHub Action (Recommended)

Zero-config security scanning with the [rec-praxis-action](https://github.com/jmanhype/rec-praxis-action):

```yaml
# .github/workflows/security-scan.yml
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run rec-praxis-rlm Security Scanner
        uses: jmanhype/rec-praxis-action@v1
        with:
          scan-type: 'all'          # review, audit, deps, or all
          severity: 'HIGH'           # Minimum severity to report
          fail-on: 'CRITICAL'        # Fail build on CRITICAL issues
          incremental: 'true'        # Only scan changed files in PRs
```

**Features**:
- Incremental scanning (only changed files)
- Procedural memory across runs
- SARIF output for GitHub Security tab
- Multi-language support (Python + JavaScript/TypeScript)

See [rec-praxis-action](https://github.com/jmanhype/rec-praxis-action) for full documentation.

### Option 2: Manual CLI Integration

```yaml
# .github/workflows/security-scan.yml
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
      - run: rec-praxis-review $(git diff --name-only origin/main...HEAD | grep '\.py$')
      - run: rec-praxis-audit $(git diff --name-only origin/main...HEAD | grep '\.py$')
```

## VS Code Extension

**Installation**:

1. Download [rec-praxis-rlm-vscode-0.4.2.vsix](https://github.com/jmanhype/rec-praxis-rlm/raw/main/vscode-extension/rec-praxis-rlm-vscode-0.4.2.vsix)
2. In VS Code: Extensions → ⋯ → Install from VSIX
3. Select the downloaded `.vsix` file

**Features**:
- Inline diagnostics as you type
- Right-click to review/audit current file
- Auto-review on save (configurable)
- Dependency scanning for `requirements.txt`
- Procedural memory integration (learns from fixes)

**Settings** (F1 → "Preferences: Open Settings (JSON)"):
```json
{
  "rec-praxis-rlm.pythonPath": "python",
  "rec-praxis-rlm.codeReview.severity": "HIGH",
  "rec-praxis-rlm.enableDiagnostics": true,
  "rec-praxis-rlm.autoReviewOnSave": false
}
```

## Interactive HTML Reports (v0.4.4+)

Generate beautiful security reports:

```bash
rec-praxis-review src/**/*.py --format=html --output=security-report.html
```

**Features**:
- Interactive charts (severity distribution, OWASP Top 10)
- Filterable findings table
- Expandable remediation guidance
- Print-to-PDF support
- CVE vulnerability display

## Supported LLM Providers

For DSPy autonomous planning:

### Groq (Recommended - Fast & Free)
```python
config = PlannerConfig(lm_model="groq/llama-3.3-70b-versatile")
```

### OpenAI
```python
config = PlannerConfig(lm_model="openai/gpt-4o-mini")
```

### OpenRouter (200+ models)
```python
config = PlannerConfig(lm_model="openrouter/meta-llama/llama-3.2-3b-instruct:free")
```

See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for full list.

## Performance

| Operation | Without FAISS | With FAISS | Speedup |
|-----------|---------------|------------|---------|
| Recall (1,000 exp) | ~20ms | ~3ms | 6.7x |
| Recall (10,000 exp) | ~200ms | ~20ms | 10x |
| Recall (100,000 exp) | ~2000ms | ~20ms | 100x |

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=rec_praxis_rlm --cov-report=html

# Current coverage: 99.38% (327 passing tests)
```

## Security

### Sandboxed Code Execution

Multiple layers of security:
1. **AST Validation**: Blocks imports, eval, exec, file I/O
2. **Restricted Builtins**: Only safe functions allowed
3. **Execution Timeout**: Prevents infinite loops
4. **Output Limiting**: Prevents memory exhaustion
5. **Code Hashing**: Audit trail for all executed code

### ReDoS Protection

Validates regex patterns to prevent Regular Expression Denial of Service attacks.

## MLflow Integration

```python
from rec_praxis_rlm.telemetry import setup_mlflow_tracing

# Enable automatic tracing
setup_mlflow_tracing(experiment_name="my-agent-experiment")

planner = PraxisRLMPlanner(memory, config)
result = planner.plan(goal="...", env_features=[...])

# View traces: mlflow ui --port 5000
```

## Advanced Features

### Async Support
```python
experiences = await memory.arecall(env_features=["python"], goal="debug error")
result = await context.asafe_exec("sum(range(1000000))")
```

### Custom Embedding Providers
```python
from rec_praxis_rlm.embeddings import APIEmbedding

embedding_provider = APIEmbedding(
    api_provider="openai",
    api_key="sk-...",
    model_name="text-embedding-3-small"
)

memory = ProceduralMemory(config, embedding_provider=embedding_provider)
```

### Memory Maintenance
```python
# Compact memory (remove old/low-value experiences)
memory.compact(max_size=1000, min_similarity=0.7)

# Recompute embeddings after changing model
memory.recompute_embeddings(new_provider)
```

## Examples

See the `examples/` directory:
- `quickstart.py` - Basic memory and context usage
- `log_analyzer.py` - Log analysis with RLM context
- `web_agent.py` - Web scraping agent
- `code_review_agent.py` - Intelligent code review
- `security_audit_agent.py` - OWASP-based auditing

## Documentation

- [Full Documentation](https://github.com/jmanhype/rec-praxis-rlm/tree/main/docs)
- [API Reference](https://github.com/jmanhype/rec-praxis-rlm/blob/main/docs/API.md)
- [Configuration Guide](https://github.com/jmanhype/rec-praxis-rlm/blob/main/docs/CONFIGURATION.md)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone and install
git clone https://github.com/jmanhype/rec-praxis-rlm.git
cd rec-praxis-rlm
pip install -e .[dev]

# Run tests
pytest --cov=rec_praxis_rlm
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [PyPI Package](https://pypi.org/project/rec-praxis-rlm/)
- [GitHub Repository](https://github.com/jmanhype/rec-praxis-rlm)
- [Issue Tracker](https://github.com/jmanhype/rec-praxis-rlm/issues)
- [Changelog](CHANGELOG.md)
