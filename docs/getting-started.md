# Getting Started

Quick start guide for rec-praxis-rlm.

## Installation

### From PyPI (Recommended)

```bash
# Install core package
pip install rec-praxis-rlm

# Install with all optional dependencies
pip install rec-praxis-rlm[all]

# Install specific feature sets
pip install rec-praxis-rlm[faiss]      # FAISS for fast retrieval
pip install rec-praxis-rlm[web]        # Web viewer
pip install rec-praxis-rlm[dspy]       # DSPy agent integration
pip install rec-praxis-rlm[mlflow]     # MLflow telemetry
```

### From Source

```bash
git clone https://github.com/jmanhype/rec-praxis-rlm.git
cd rec-praxis-rlm
pip install -e ".[all]"
```

## Basic Usage

### Procedural Memory

Store and recall agent experiences:

```python
from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig

# Create memory instance
memory = ProceduralMemory(MemoryConfig(storage_path='./memory.jsonl'))

# Store an experience
exp = Experience(
    env_features=['python', 'web_scraping'],
    goal='extract product prices',
    action='Used BeautifulSoup with CSS selectors',
    result='Extracted 1000 prices with 99% accuracy',
    success=True
)
memory.store(exp)

# Recall similar experiences
results = memory.recall(
    env_features=['python', 'web'],
    goal='scrape data',
    top_k=5
)

for exp in results:
    print(f"Goal: {exp.goal}")
    print(f"Action: {exp.action}")
    print(f"Result: {exp.result}")
    print(f"Success: {exp.success}")
    print("---")
```

### RLM Context (Document Inspection)

Manipulate documents programmatically:

```python
from rec_praxis_rlm import RLMContext

# Create context with documents
rlm = RLMContext()
rlm.add_document("server.log", open("server.log").read())

# Grep for patterns
errors = rlm.grep("server.log", "ERROR")
print(f"Found {len(errors)} errors")

# Peek at specific lines
context = rlm.peek("server.log", line=100, context_lines=5)

# Get head/tail
recent_logs = rlm.tail("server.log", n=20)

# Safe code execution
result = rlm.execute("result = sum([1, 2, 3, 4, 5])")
print(result['result'])  # 15
```

### Security Scanning

#### Code Review

```bash
# Human-readable output
rec-praxis-review src/**/*.py

# Filter by severity
rec-praxis-review src/**/*.py --severity=HIGH

# JSON output for tooling
rec-praxis-review src/**/*.py --format=json

# Interactive HTML report
rec-praxis-review src/**/*.py --format=html --output=report.html
```

#### Security Audit

```bash
# Audit with exit code on failures
rec-praxis-audit app.py --fail-on=CRITICAL

# SARIF output for GitHub Security tab
rec-praxis-audit app.py --format=sarif
```

#### Dependency Scanning

```bash
# Scan dependencies and imports
rec-praxis-deps --requirements=requirements.txt --files src/**/*.py

# Check for known CVEs
rec-praxis-deps --requirements=requirements.txt --check-cves

# Detect hardcoded secrets
rec-praxis-deps --files src/**/*.py --check-secrets
```

#### Graph-Aware Analysis (Advanced)

For cross-function vulnerability detection, use `--use-graph` with [Parseltongue](https://github.com/your-org/parseltongue):

```bash
# Start Parseltongue server (one-time setup)
parseltongue serve --port 8080

# Enable graph-aware analysis
rec-praxis-review src/**/*.py --use-graph
rec-praxis-audit app.py --use-graph
```

**Detects:**
- Cross-function SQL injection (data flows through multiple functions)
- Authentication bypass (public endpoints without auth checks)
- Privilege escalation (low-privilege → high-privilege calls)
- Large attack surface (too many public entry points)

**See also:** [Graph-Aware Analysis Documentation](cli-reference.md#graph-aware-analysis-with-parseltongue)

## Configuration Presets

rec-praxis-rlm includes presets for common scenarios:

```python
from rec_praxis_rlm.config import MemoryConfig, PlannerConfig

# Fast in-memory testing
memory = ProceduralMemory(MemoryConfig.in_memory())

# Production with FAISS
memory = ProceduralMemory(MemoryConfig.production())

# Development with detailed logging
memory = ProceduralMemory(MemoryConfig.development())

# Custom configuration
config = MemoryConfig(
    storage_path='./custom.jsonl',
    embedding_provider='sentence-transformers',
    similarity_threshold=0.7,
    enable_privacy=True,
    enable_faiss=True
)
memory = ProceduralMemory(config)
```

## Next Steps

- **[Examples](../examples/README.md)** - Practical code examples
- **[API Reference](api_reference.md)** - Complete API documentation
- **[Architecture](architecture.md)** - System design and internals
- **[CLI Tools](cli-reference.md)** - Command-line tool documentation
- **[Integrations](integrations.md)** - Claude Code, GitHub Actions, VS Code

## Common Issues

### FAISS Installation Issues

If you encounter FAISS installation errors:

```bash
# macOS with Apple Silicon
pip install faiss-cpu

# Linux
pip install faiss-cpu

# Windows
pip install faiss-cpu
```

### Import Errors

If you see `ModuleNotFoundError`:

```bash
# Verify installation
pip list | grep rec-praxis-rlm

# Reinstall with dependencies
pip install --force-reinstall rec-praxis-rlm[all]
```

### Claude Code Hooks Not Working

If automatic experience capture isn't working:

```bash
# Verify hooks are installed
ls -la .claude/hooks/

# Check hook permissions
chmod +x .claude/hooks/*.sh

# Test manually
.claude/hooks/session_start.sh
```

## Getting Help

- **Documentation**: [https://jmanhype.github.io/rec-praxis-rlm/](https://jmanhype.github.io/rec-praxis-rlm/)
- **Issues**: [GitHub Issues](https://github.com/jmanhype/rec-praxis-rlm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jmanhype/rec-praxis-rlm/discussions)
