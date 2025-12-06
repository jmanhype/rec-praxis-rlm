# Contributing to rec-praxis-rlm

Thank you for your interest in contributing to rec-praxis-rlm! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

---

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

**Expected Behavior:**
- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community

**Unacceptable Behavior:**
- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Other conduct inappropriate for a professional setting

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic understanding of:
  - Procedural memory concepts
  - DSPy framework
  - Semantic embeddings

### Finding Work

Good first issues are tagged with:
- `good-first-issue`: Suitable for newcomers
- `help-wanted`: Need community contributions
- `documentation`: Documentation improvements

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR-USERNAME/rec-praxis-rlm.git
cd rec-praxis-rlm

# Add upstream remote
git remote add upstream https://github.com/your-org/rec-praxis-rlm.git
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import rec_praxis_rlm; print('Installation successful!')"
```

---

## Development Workflow

### 1. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test improvements

### 2. Make Changes

Follow the [Coding Standards](#coding-standards) below.

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rec_praxis_rlm --cov-report=html

# Run specific test file
pytest tests/unit/test_memory.py

# Run specific test
pytest tests/unit/test_memory.py::test_store_and_recall
```

### 4. Lint and Format

```bash
# Run ruff linter
ruff check rec_praxis_rlm/

# Auto-fix linting issues
ruff check --fix rec_praxis_rlm/

# Run black formatter
black rec_praxis_rlm/

# Run all checks together
ruff check --fix rec_praxis_rlm/ && black rec_praxis_rlm/
```

### 5. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: Add FAISS indexing support for memory recall"
```

**Commit Message Format:**

```
<type>: <short summary>

<optional body>

<optional footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `perf`: Performance improvements
- `chore`: Maintenance tasks

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://peps.python.org/pep-0008/) with these conventions:

**Line Length:** 100 characters (not 79)

**Formatting:** Use `black` with default settings

**Linting:** Use `ruff` with project configuration

**Type Hints:** Required for all public APIs

```python
# Good
def recall(
    self, env_features: list[str], goal: str, top_k: Optional[int] = None
) -> list[Experience]:
    """Retrieve similar experiences."""
    pass

# Bad
def recall(self, env_features, goal, top_k=None):
    pass
```

### Docstring Conventions

Use Google-style docstrings:

```python
def store(self, experience: Experience) -> None:
    """Store a new experience in memory.

    Args:
        experience: Experience to store with embeddings

    Raises:
        StorageError: If storage write fails

    Example:
        >>> memory.store(Experience(
        ...     env_features=["web"],
        ...     goal="extract data",
        ...     action="Used BeautifulSoup",
        ...     result="Extracted 100 items",
        ...     success=True
        ... ))
    """
    pass
```

### Code Organization

**SOLID Principles:**
- Single Responsibility: Each class/module has one clear purpose
- Open/Closed: Extend via inheritance/composition, not modification
- Liskov Substitution: Subtypes must be substitutable
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depend on abstractions, not concretions

**Pydantic Models:**
- Use for all data structures
- Add field validators where appropriate
- Include field descriptions

```python
from pydantic import BaseModel, Field, field_validator

class Experience(BaseModel):
    """Agent experience with validation."""

    env_features: list[str] = Field(..., description="Environmental tags")
    goal: str = Field(..., min_length=1, description="Goal description")

    @field_validator("env_features")
    @classmethod
    def env_features_not_empty(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("env_features cannot be empty")
        return v
```

---

## Testing Requirements

### Coverage Requirements

- Minimum coverage: **90%**
- Current coverage: **99.38%**
- All new code must include tests

### Test Structure

```
tests/
├── unit/           # Unit tests (80% of tests)
│   ├── test_memory.py
│   ├── test_rlm.py
│   ├── test_sandbox.py
│   └── test_embeddings.py
├── integration/    # Integration tests (15% of tests)
│   ├── test_memory_integration.py
│   └── test_e2e_agent.py
└── conftest.py     # Shared fixtures
```

### Writing Unit Tests

```python
import pytest
from rec_praxis_rlm import ProceduralMemory, MemoryConfig, Experience

@pytest.fixture
def memory():
    """Provide in-memory procedural memory instance."""
    return ProceduralMemory(MemoryConfig(storage_path=":memory:"))

def test_store_and_recall(memory):
    """Test that stored experiences can be recalled."""
    # Arrange
    experience = Experience(
        env_features=["web"],
        goal="extract data",
        action="Used BeautifulSoup",
        result="Success",
        success=True
    )

    # Act
    memory.store(experience)
    results = memory.recall(env_features=["web"], goal="extract data")

    # Assert
    assert len(results) == 1
    assert results[0].goal == "extract data"
```

---

## Documentation

### API Documentation

- All public functions/classes must have docstrings
- Use Google-style format
- Include examples where helpful

### User Documentation

Located in `docs/`:
- `api_reference.md` - Complete API documentation
- `architecture.md` - Architecture and design decisions

### Example Scripts

Located in `examples/`:
- Keep examples runnable and self-contained
- Add comments explaining key concepts
- Test examples before committing

---

## Submitting Changes

### 1. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 2. Create Pull Request

On GitHub, create a pull request from your fork to `main` branch.

**PR Title Format:**

```
<type>: <short summary>
```

**PR Description Template:**

```markdown
## Description

Brief description of changes.

## Motivation

Why is this change needed?

## Changes

- Change 1
- Change 2

## Testing

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
```

### 3. Code Review

- Address reviewer feedback promptly
- Update PR based on comments
- Keep discussion focused and professional

### 4. Merge

Once approved, maintainers will merge your PR.

---

## Getting Help

- **Questions:** Open a GitHub discussion
- **Bugs:** Open an issue
- **Security:** Email security@example.com (do not open public issues)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to rec-praxis-rlm! 🎉
