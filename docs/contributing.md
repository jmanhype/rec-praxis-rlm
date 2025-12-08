# Contributing to rec-praxis-rlm

Thank you for your interest in contributing to rec-praxis-rlm!

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to build something useful.

## How to Contribute

### Reporting Bugs

1. Check [existing issues](https://github.com/jmanhype/rec-praxis-rlm/issues)
2. Create new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version, OS, rec-praxis-rlm version
   - Minimal code example

### Suggesting Features

1. Check [discussions](https://github.com/jmanhype/rec-praxis-rlm/discussions)
2. Create feature request with:
   - Use case and motivation
   - Proposed API/interface
   - Alternatives considered
   - Breaking changes (if any)

### Contributing Code

## Development Setup

### Prerequisites

- Python 3.10+ (3.10, 3.11, 3.12, 3.13 supported)
- Git
- pip

### Initial Setup

```bash
# Clone repository
git clone https://github.com/jmanhype/rec-praxis-rlm.git
cd rec-praxis-rlm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[all,dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

The `[dev]` extra includes:

- **Testing**: pytest, pytest-cov, hypothesis
- **Linting**: ruff, mypy
- **Formatting**: black, isort
- **Documentation**: sphinx, mkdocs
- **Pre-commit**: pre-commit hooks

## Project Structure

```
rec-praxis-rlm/
├── rec_praxis_rlm/          # Main package
│   ├── memory.py            # ProceduralMemory core
│   ├── rlm.py               # RLMContext implementation
│   ├── embeddings.py        # Embedding providers
│   ├── agents/              # Specialized agents
│   ├── cli/                 # CLI tools
│   ├── web_viewer.py        # Web UI
│   └── ...
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── fixtures/            # Test fixtures
├── docs/                    # Documentation
├── examples/                # Example scripts
└── pyproject.toml           # Project configuration
```

## Coding Standards

### Style Guide

We follow PEP 8 with these tools:

- **ruff**: Fast Python linter (replaces flake8, isort, pyupgrade)
- **mypy**: Static type checking
- **black**: Code formatting (line length: 100)

### Running Linters

```bash
# Lint with ruff
ruff check .

# Auto-fix issues
ruff check --fix .

# Type check with mypy
mypy rec_praxis_rlm

# Format with black
black rec_praxis_rlm tests
```

### Type Hints

All public APIs must have type hints:

```python
# Good
def recall(
    self,
    env_features: list[str],
    goal: str,
    top_k: int = 5
) -> list[Experience]:
    ...

# Bad
def recall(self, env_features, goal, top_k=5):
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def recall(
    self,
    env_features: list[str],
    goal: str,
    top_k: int = 5
) -> list[Experience]:
    """Recall similar experiences from memory.

    Args:
        env_features: List of environment tags (e.g., ['python', 'web'])
        goal: Goal description to match against
        top_k: Number of results to return (default: 5)

    Returns:
        List of matching Experience objects, sorted by similarity

    Raises:
        ValueError: If top_k is negative or env_features is empty

    Example:
        >>> memory = ProceduralMemory(MemoryConfig())
        >>> results = memory.recall(['python'], 'parse JSON', top_k=3)
        >>> print(len(results))
        3
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rec_praxis_rlm --cov-report=html

# Run specific test file
pytest tests/unit/test_memory.py

# Run specific test
pytest tests/unit/test_memory.py::test_store_and_recall

# Run fast tests only (skip slow integration tests)
pytest -m "not slow"
```

### Writing Tests

#### Unit Tests

Place in `tests/unit/`:

```python
# tests/unit/test_memory.py
import pytest
from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig

def test_store_and_recall():
    """Test basic store and recall functionality."""
    memory = ProceduralMemory(MemoryConfig(storage_path=':memory:'))

    # Store experience
    exp = Experience(
        env_features=['python', 'testing'],
        goal='Write unit test',
        action='Created pytest fixture',
        result='Test passed',
        success=True
    )
    memory.store(exp)

    # Recall
    results = memory.recall(
        env_features=['python'],
        goal='Write test',
        top_k=1
    )

    assert len(results) == 1
    assert results[0].goal == 'Write unit test'
```

#### Integration Tests

Place in `tests/integration/`:

```python
# tests/integration/test_agent_workflow.py
import pytest
from rec_praxis_rlm.agents import CodeReviewAgent

def test_end_to_end_review(tmp_path):
    """Test complete code review workflow."""
    # Create test file
    test_file = tmp_path / "app.py"
    test_file.write_text("password = 'secret123'")

    # Run review
    agent = CodeReviewAgent()
    results = agent.review_file(str(test_file))

    # Verify findings
    assert len(results) > 0
    assert any(f.severity == 'CRITICAL' for f in results)
```

#### Property-Based Tests

Use Hypothesis for property-based testing:

```python
from hypothesis import given, strategies as st

@given(
    env_features=st.lists(st.text(min_size=1), min_size=1),
    goal=st.text(min_size=1),
    top_k=st.integers(min_value=1, max_value=100)
)
def test_recall_always_returns_list(env_features, goal, top_k):
    """Recall should always return a list, regardless of input."""
    memory = ProceduralMemory(MemoryConfig(storage_path=':memory:'))
    results = memory.recall(env_features, goal, top_k)
    assert isinstance(results, list)
```

### Test Coverage Goals

- **Overall**: 95%+ (currently 99.38%)
- **Security-critical code**: 100%
- **Core features**: 95%+
- **CLI tools**: 90%+

Check coverage:

```bash
pytest --cov=rec_praxis_rlm --cov-report=term-missing
```

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write code following style guide
- Add tests for new functionality
- Update documentation
- Run linters and tests locally

### 3. Commit Changes

Use conventional commit format:

```bash
git commit -m "feat: add FAISS indexing support"
git commit -m "fix: handle empty env_features in recall"
git commit -m "docs: update API reference for MemoryConfig"
git commit -m "test: add property-based tests for recall"
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/modifications
- `refactor`: Code restructuring without behavior change
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create PR on GitHub with:

- Clear title and description
- Link to related issue (if any)
- Description of changes
- Breaking changes (if any)
- Testing performed

### 5. Code Review

- Address reviewer feedback
- Push additional commits to same branch
- Keep PR focused (one feature/fix per PR)

### 6. Merge

After approval:
- Squash commits if needed
- Ensure CI passes
- Maintainer will merge

## Release Process

(For maintainers)

### Version Numbering

We use semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist

```bash
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG.md
# 3. Run full test suite
pytest

# 4. Build package
python -m build

# 5. Test package locally
pip install dist/rec_praxis_rlm-*.whl

# 6. Upload to PyPI
python -m twine upload dist/*

# 7. Create GitHub release
gh release create v0.X.Y --title "v0.X.Y" --notes "Release notes..."

# 8. Tag commit
git tag v0.X.Y
git push origin v0.X.Y
```

## Documentation

### Building Docs Locally

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build HTML docs
cd docs
make html

# Open in browser
open _build/html/index.html
```

### Documentation Standards

- Update API reference for new classes/functions
- Add examples for new features
- Update README.md if adding user-facing features
- Add docstrings to all public APIs

## Getting Help

- **Questions**: [GitHub Discussions](https://github.com/jmanhype/rec-praxis-rlm/discussions)
- **Bugs**: [GitHub Issues](https://github.com/jmanhype/rec-praxis-rlm/issues)
- **Security**: Email jmanhype@users.noreply.github.com

## Recognition

Contributors are recognized in:

- CHANGELOG.md for each release
- Contributors section in README.md
- GitHub's contributor graph

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
