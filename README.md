# rec-praxis-rlm

**Procedural Memory + REPL Context for Autonomous AI Agents**

A Python package that provides persistent procedural memory and safe code execution capabilities for DSPy 3.0 autonomous agents.

[![PyPI version](https://img.shields.io/pypi/v/rec-praxis-rlm.svg)](https://pypi.org/project/rec-praxis-rlm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Test Coverage](https://img.shields.io/badge/coverage-99.38%25-brightgreen.svg)](https://github.com/jmanhype/rec-praxis-rlm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

```bash
# Install
pip install rec-praxis-rlm

# Basic usage - no API key needed
python -c "
from rec_praxis_rlm.memory import ProceduralMemory, Experience
from rec_praxis_rlm.config import MemoryConfig

memory = ProceduralMemory(MemoryConfig(storage_path='./agent_memory.jsonl'))
memory.store(Experience(
    env_features=['python', 'web_scraping'],
    goal='extract product prices',
    action='Used BeautifulSoup with CSS selectors',
    result='Extracted 1000 prices with 99% accuracy',
    success=True
))

# Recall similar experiences
for exp in memory.recall(env_features=['python'], goal='extract data', top_k=5):
    print(f'{exp.similarity_score:.2f}: {exp.action}')
"
```

## Core Features

- **Procedural Memory**: Store and retrieve agent experiences with semantic similarity
- **RLM Context**: Document inspection (grep, peek, head, tail) with safe code execution
- **DSPy Integration**: Autonomous planning with ReAct agents
- **CLI Tools**: Code review, security audit, dependency scanning
- **99.38% Test Coverage**: Production-ready reliability

## CLI Tools

```bash
# Code review
rec-praxis-review src/**/*.py --severity=HIGH

# Security audit
rec-praxis-audit app.py --fail-on=CRITICAL

# Dependency scan
rec-praxis-deps --requirements=requirements.txt
```

## Documentation

- [Full Documentation](https://github.com/jmanhype/rec-praxis-rlm/tree/main/docs)
- [API Reference](https://github.com/jmanhype/rec-praxis-rlm/blob/main/docs/API.md)
- [Examples](https://github.com/jmanhype/rec-praxis-rlm/tree/main/examples)
- [Configuration Guide](https://github.com/jmanhype/rec-praxis-rlm/blob/main/docs/CONFIGURATION.md)

## Installation Options

```bash
# Basic (no FAISS, works offline)
pip install rec-praxis-rlm

# With FAISS (10-100x faster retrieval)
pip install rec-praxis-rlm[all]

# Development
pip install rec-praxis-rlm[dev]
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [PyPI Package](https://pypi.org/project/rec-praxis-rlm/)
- [GitHub Repository](https://github.com/jmanhype/rec-praxis-rlm)
- [Issue Tracker](https://github.com/jmanhype/rec-praxis-rlm/issues)
- [Changelog](CHANGELOG.md)
