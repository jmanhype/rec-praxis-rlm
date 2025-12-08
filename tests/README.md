# Tests

This directory contains the test suite for rec-praxis-rlm.

**Current Coverage: 99.38%** (327 passing tests)

## Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
├── integration/       # Integration tests (slower, multiple components)
├── fixtures/          # Shared test fixtures and data
└── test_*.py          # Special test suites (RAGAS, CLI, ablation)
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run with Coverage
```bash
pytest --cov=rec_praxis_rlm --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Run Specific Test Suites

```bash
# Unit tests only (fast)
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_memory.py

# Specific test function
pytest tests/unit/test_memory.py::test_store_and_recall
```

### Run with Markers

```bash
# Only slow tests
pytest -m slow

# Skip slow tests
pytest -m "not slow"

# Only RAGAS evaluation tests (requires GROQ_API_KEY)
pytest -m ragas
```

## Test Categories

### Unit Tests (`tests/unit/`)

Fast, isolated tests for individual components:

- `test_memory.py` - ProceduralMemory storage, recall, compaction
- `test_rlm.py` - RLMContext document operations, safe execution
- `test_embeddings.py` - Embedding providers (local, API, Jaccard)
- `test_privacy.py` - Privacy redaction patterns
- `test_concepts.py` - Concept tagging and semantic extraction
- `test_endless_mode.py` - EndlessAgent token tracking, compression

### Integration Tests (`tests/integration/`)

Multi-component tests:

- `test_dspy_integration.py` - DSPy + ProceduralMemory integration
- `test_agent_workflow.py` - End-to-end agent workflows
- `test_mlflow_integration.py` - MLflow tracing and metrics

### RAGAS Evaluation Tests (`test_ragas_*.py`)

LLM-based evaluation tests (require API key):

```bash
export GROQ_API_KEY="gsk-..."  # Or OPENAI_API_KEY
pytest -m ragas
```

- `test_ragas_procedural_memory.py` - Memory quality metrics
- `test_ragas_code_review.py` - Code review accuracy
- `test_ragas_security_audit.py` - Security detection quality
- `test_ragas_dependency_scan.py` - Dependency scanning accuracy
- `test_ragas_full_evaluation.py` - Full system evaluation

### CLI Integration Tests (`test_cli_integration.py`)

Tests for command-line tools:
- `rec-praxis-review`
- `rec-praxis-audit`
- `rec-praxis-deps`

### Ablation Study (`test_ablation_study.py`)

Performance comparisons:
- FAISS vs no FAISS
- Different embedding providers
- Similarity threshold tuning

## Writing Tests

### Test Fixtures

Shared fixtures are in `tests/fixtures/`:

```python
# In conftest.py
@pytest.fixture
def memory():
    """Provides a clean ProceduralMemory instance."""
    config = MemoryConfig(storage_path=":memory:")
    return ProceduralMemory(config, use_faiss=False)
```

### Example Test

```python
def test_store_and_recall(memory):
    """Test storing and recalling experiences."""
    # Store experience
    exp = Experience(
        env_features=["python", "testing"],
        goal="Write unit test",
        action="Created pytest fixture",
        result="Test passed",
        success=True
    )
    memory.store(exp)

    # Recall similar experience
    results = memory.recall(
        env_features=["python"],
        goal="Write test",
        top_k=1
    )

    assert len(results) == 1
    assert results[0].goal == "Write unit test"
```

### Test Markers

Available markers (defined in `pyproject.toml`):

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (>1s)
- `@pytest.mark.ragas` - RAGAS evaluation tests (requires API key)

## Continuous Integration

Tests run automatically on:
- Every push to `main`
- Every pull request
- Nightly builds (full test suite including slow tests)

See `.github/workflows/test.yml` for CI configuration.

## Coverage Goals

- **Overall**: 95%+ (currently 99.38%)
- **Security-critical**: 100% (RLS, encryption, auth, privacy redaction)
- **Core features**: 95%+ (memory, RLM, embeddings)
- **CLI tools**: 90%+ (harder to test, more integration-focused)

## Troubleshooting

### Slow Tests

```bash
# Profile slow tests
pytest --durations=10
```

### Failed Tests

```bash
# Show full output
pytest -vv

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf
```

### Coverage Gaps

```bash
# Show missing lines
pytest --cov=rec_praxis_rlm --cov-report=term-missing
```

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure tests pass: `pytest`
3. Check coverage: `pytest --cov`
4. Run linters: `ruff check .` and `mypy rec_praxis_rlm`
5. Update this README if adding new test categories

See [CONTRIBUTING.md](../CONTRIBUTING.md) for full guidelines.
