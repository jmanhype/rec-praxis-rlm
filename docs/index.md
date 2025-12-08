---
layout: default
title: rec-praxis-rlm - Procedural Memory for AI Agents
---

# rec-praxis-rlm

**Procedural Memory + REPL Context for Autonomous AI Agents**

A Python package that gives AI agents persistent memory and integrates security scanning into your development workflow.

[![PyPI version](https://img.shields.io/pypi/v/rec-praxis-rlm.svg)](https://pypi.org/project/rec-praxis-rlm/)
[![Test Coverage](https://img.shields.io/badge/coverage-99.38%25-brightgreen.svg)](https://github.com/jmanhype/rec-praxis-rlm)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Quick Start

```bash
# Install
pip install rec-praxis-rlm[all]

# Use procedural memory
python -c "
from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig

memory = ProceduralMemory(MemoryConfig(storage_path='./memory.jsonl'))
memory.store(Experience(
    env_features=['python', 'web_scraping'],
    goal='extract product prices',
    action='Used BeautifulSoup with CSS selectors',
    result='Extracted 1000 prices with 99% accuracy',
    success=True
))
"

# Security scanning
rec-praxis-review src/**/*.py --severity=HIGH
rec-praxis-audit app.py --fail-on=CRITICAL
```

---

## Features

### Core Capabilities
- **Procedural Memory** - Store and retrieve agent experiences with semantic similarity
- **FAISS Indexing** - 10-100x faster retrieval at scale (>10k experiences)
- **RLM Context** - Programmatic document inspection (grep, peek, head, tail)
- **Safe Code Execution** - Sandboxed Python REPL with AST validation
- **DSPy 3.0 Integration** - Autonomous planning with ReAct agents
- **99.38% Test Coverage** - Production-ready reliability

### Developer Tools
- **Claude Code Hooks** - Automatic experience capture (zero-config)
- **CLI Tools** - Code review, security audit, dependency scanning
- **Pre-commit Hooks** - Automated quality checks before commits
- **GitHub Action** - [rec-praxis-action](https://github.com/jmanhype/rec-praxis-action) for CI/CD
- **VS Code Extension** - Real-time inline diagnostics
- **HTML Reports** - Interactive security reports with charts

---

## Documentation

### Getting Started
- [Installation & Quick Start](https://github.com/jmanhype/rec-praxis-rlm#quick-start)
- [Examples](examples/README.html) - Practical code examples
- [CLI Tools](https://github.com/jmanhype/rec-praxis-rlm#cli-tools)

### Core Features
- [Procedural Memory](api_reference.html#procedural-memory) - Experience storage and recall
- [RLM Context](api_reference.html#rlm-context) - Document manipulation
- [Endless Mode](endless_mode.html) - Long-running autonomous tasks
- [Web Viewer](web_viewer.html) - Interactive memory inspection

### Integrations
- [Claude Code Hooks](https://github.com/jmanhype/rec-praxis-rlm/blob/main/.claude/README.md) - Automatic experience capture
- [GitHub Actions](https://github.com/jmanhype/rec-praxis-action) - CI/CD integration
- [VS Code Extension](https://github.com/jmanhype/rec-praxis-rlm/tree/main/vscode-extension)
- [Pre-commit Hooks](https://github.com/jmanhype/rec-praxis-rlm#pre-commit-hooks)

### Advanced
- [API Reference](api_reference.html) - Complete API documentation
- [Architecture](architecture.html) - System design and data flow
- [Testing](https://github.com/jmanhype/rec-praxis-rlm/blob/main/tests/README.md) - Test structure and coverage

---

## Claude Code Integration

**Zero-config automatic learning** - rec-praxis-rlm integrates with Claude Code to capture every tool use:

```bash
# .claude/settings.json is pre-configured
# Every Bash command, file read/write, grep → automatically stored

# Session start shows what worked (and what failed):
📚 REC Praxis RLM Context

Memory Statistics:
- Total experiences: 127
- Recent successful patterns: 5
- Recent failures to avoid: 2

Recent Successful Patterns:
1. [optimize] Database query optimization
   ✓ Reduced latency from 2s to 50ms
```

See [.claude/README.md](https://github.com/jmanhype/rec-praxis-rlm/blob/main/.claude/README.md) for full documentation.

---

## CLI Tools

### Code Review
```bash
# Human-readable format
rec-praxis-review src/**/*.py --severity=HIGH

# JSON for IDE integration
rec-praxis-review src/**/*.py --format=json

# Interactive HTML report
rec-praxis-review src/**/*.py --format=html --output=report.html

# SARIF for GitHub Security tab
rec-praxis-review src/**/*.py --format=sarif
```

### Security Audit
```bash
rec-praxis-audit app.py --fail-on=CRITICAL
```

### Dependency Scanning
```bash
rec-praxis-deps --requirements=requirements.txt --files src/**/*.py
```

**Output Formats**: human, json, html, sarif

---

## Performance

| Operation | Without FAISS | With FAISS | Speedup |
|-----------|---------------|------------|---------|
| Recall (1,000 exp) | ~20ms | ~3ms | 6.7x |
| Recall (10,000 exp) | ~200ms | ~20ms | 10x |
| Recall (100,000 exp) | ~2000ms | ~20ms | 100x |

---

## Examples

Browse the [examples/](https://github.com/jmanhype/rec-praxis-rlm/tree/main/examples) directory:

- `quickstart.py` - Basic memory and context usage
- `code_review_agent.py` - Intelligent code review with procedural memory
- `security_audit_agent.py` - OWASP-based security auditing
- `log_analyzer.py` - Log analysis with RLM context
- `web_agent.py` - Web scraping agent

---

## Evaluations

Performance benchmarks and analysis:

- [Benchmarks](evaluations/BENCHMARKS.html) - Performance metrics, FAISS speedup
- [Claude-MEM Comparison](evaluations/CLAUDE-MEM-COMPARISON.html) - Comparison with claude-mem
- [Senior Engineering Review](evaluations/SENIOR_ENGINEERING_REVIEW.html) - Code quality assessment

---

## GitHub Action

Use the official [rec-praxis-action](https://github.com/jmanhype/rec-praxis-action):

```yaml
# .github/workflows/security-scan.yml
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: jmanhype/rec-praxis-action@v1
        with:
          scan-type: 'all'
          severity: 'HIGH'
          fail-on: 'CRITICAL'
          incremental: 'true'
```

---

## Supported LLM Providers

For DSPy autonomous planning:

- **Groq** (recommended - fast & free): `groq/llama-3.3-70b-versatile`
- **OpenAI**: `openai/gpt-4o-mini`
- **OpenRouter** (200+ models): `openrouter/meta-llama/llama-3.2-3b-instruct:free`
- Any LiteLLM-supported provider

See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for full list.

---

## Community

### Get Involved
- 🐛 [Report Bugs](https://github.com/jmanhype/rec-praxis-rlm/issues)
- 💡 [Request Features](https://github.com/jmanhype/rec-praxis-rlm/issues/new)
- 💬 [Discussions](https://github.com/jmanhype/rec-praxis-rlm/discussions)
- 🤝 [Contributing](https://github.com/jmanhype/rec-praxis-rlm/blob/main/CONTRIBUTING.md)

### Resources
- **PyPI Package**: [pypi.org/project/rec-praxis-rlm](https://pypi.org/project/rec-praxis-rlm/)
- **GitHub Repository**: [github.com/jmanhype/rec-praxis-rlm](https://github.com/jmanhype/rec-praxis-rlm)
- **GitHub Action**: [github.com/jmanhype/rec-praxis-action](https://github.com/jmanhype/rec-praxis-action)

---

## License

MIT License - See [LICENSE](https://github.com/jmanhype/rec-praxis-rlm/blob/main/LICENSE)

Built with [DSPy 3.0](https://github.com/stanfordnlp/dspy), [sentence-transformers](https://www.sbert.net/), and [MLflow](https://mlflow.org/).

---

<div style="text-align: center; margin-top: 2em;">
  <a href="https://github.com/jmanhype/rec-praxis-rlm" style="font-size: 1.2em; margin-right: 1em;">View on GitHub</a>
  <a href="https://pypi.org/project/rec-praxis-rlm/" style="font-size: 1.2em;">Install from PyPI</a>
</div>
