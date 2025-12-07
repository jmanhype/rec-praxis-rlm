# Changelog

All notable changes to rec-praxis-rlm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Programmatic API key support via `PlannerConfig.api_key` parameter
- Thread-safe model switching using `dspy.context()` in `plan()` method
- Multi-model support: can use different models in same process without conflicts
- Clear "Requirements" section in README documenting what works without API keys
- Comprehensive "Supported LLM Providers" section with examples for Groq, OpenAI, OpenRouter, and others
- Multi-provider examples in DSPy autonomous planning quickstart
- Verified Groq integration (llama-3.3-70b-versatile model)
- Documentation highlighting Groq as recommended provider (fast, free)
- Updated examples to show programmatic API key usage

### Changed
- `PraxisRLMPlanner` now stores LM instance as `self._lm` for context switching
- API keys can now be passed programmatically or via environment variables

### Fixed
- Updated DSPy 3.0 ReAct API compatibility (now requires signature parameter)
- Fixed import path for ReAct (moved from dspy.primitives to dspy directly in DSPy 3.0)

## [0.1.1] - 2025-12-06

### Fixed
- Updated GitHub repository URLs to correct location: `jmanhype/rec-praxis-rlm`
- Fixed author information in package metadata
- Added PyPI link to project URLs

### Added
- GitHub Actions workflows for automated testing (Python 3.10, 3.11, 3.12)
- GitHub Actions workflow for automated PyPI publishing on releases
- Added PyPI version badge to README
- Added test status badge to README

### Changed
- Bumped version to 0.1.1 to reflect corrected metadata

## [0.1.0] - 2025-12-06

### Added
- Initial public release of rec-praxis-rlm
- Procedural memory with hybrid similarity scoring (environmental + goal embeddings)
- FAISS indexing for 10-100x faster retrieval at scale (>10k experiences)
- RLM context for programmatic document inspection (grep, peek, head, tail)
- Safe code execution with sandboxed Python REPL
- ReDoS protection for regex pattern validation
- DSPy 3.0 integration for autonomous planning with ReAct agents
- MLflow observability with automatic tracing
- Async support for non-blocking operations
- LRU caching for embeddings (10-100x speedup)
- Comprehensive documentation (API reference, architecture, examples)
- 99.38% test coverage with 327 passing tests
- Property-based testing with Hypothesis
- Example scripts: quickstart, log analyzer, web agent, optimization

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- AST validation blocks dangerous code patterns (imports, eval, exec, file I/O)
- Restricted builtins prevent privilege escalation
- ReDoS protection prevents exponential regex backtracking
- Execution timeout prevents infinite loops
- Output limiting prevents memory exhaustion
- Code hashing provides audit trail for all executions

---

## [0.1.0] - 2025-01-XX

### Added
- Core procedural memory implementation
  - Experience storage with JSONL persistence
  - Hybrid similarity scoring (60% environmental, 40% goal)
  - Top-k retrieval with configurable threshold
  - Memory compaction for size management
  - Embedding recomputation support

- RLM context capabilities
  - Document storage and management
  - Regex-based grep search with ReDoS protection
  - Character-range extraction (peek)
  - Head/tail line extraction
  - Safe code execution with sandboxing
  - AST validation for security

- Embedding system
  - SentenceTransformerEmbedding (local, no API key)
  - APIEmbedding (OpenAI support)
  - TextSimilarityFallback (Jaccard similarity)
  - LRU caching (10,000 entries by default)
  - Batch embedding support

- DSPy agent integration
  - PraxisRLMPlanner with ReAct agent
  - RecallExperiencesTool for memory access
  - SearchContextTool for document search
  - ExecuteCodeTool for code execution
  - MLflow tracing integration

- FAISS indexing (optional)
  - IVF index for fast similarity search
  - Automatic index building and maintenance
  - 100x speedup for 100k+ experiences

- Configuration system
  - MemoryConfig for procedural memory
  - ReplConfig for REPL context
  - PlannerConfig for DSPy agent
  - Pydantic validation for all configs

- Testing infrastructure
  - 327 unit and integration tests
  - 99.38% code coverage
  - Property-based tests with Hypothesis
  - Performance benchmarks
  - Security tests for sandbox

- Documentation
  - Comprehensive README with quick start examples
  - API reference documentation
  - Architecture documentation with design decisions
  - Example scripts (4 complete examples)
  - Contributing guide
  - Changelog

### Technical Details

**Storage Format:**
- JSONL (one experience per line)
- Versioned format (v1.0) for backward compatibility
- Append-only writes for crash safety

**Performance:**
- Recall from 1,000 experiences: <20ms without FAISS, <3ms with FAISS
- Recall from 10,000 experiences: <200ms without FAISS, <20ms with FAISS
- Document grep (10MB): <500ms with ReDoS protection
- Safe code execution: <100ms in sandboxed environment

**Security:**
- Multi-layer sandbox: AST validation + restricted builtins + timeout
- ReDoS protection: Pattern length limits, nested quantifier detection
- Audit trail: SHA-256 hash of all executed code

**Dependencies:**
- Python 3.10+ required
- Core: dspy-ai >=3.0.4, pydantic >=2.0, jsonlines >=3.0, mlflow >=3.0
- Optional: sentence-transformers >=2.2, faiss-cpu, openai

---

## Version History

### [0.1.0] - 2025-01-XX
Initial release

---

## Release Notes

### 0.1.0 - Initial Release

This is the first public release of rec-praxis-rlm, providing procedural memory and REPL context capabilities for DSPy 3.0 autonomous agents.

**Highlights:**

🧠 **Procedural Memory**
- Store and retrieve agent experiences with hybrid similarity scoring
- 10-100x faster retrieval with FAISS indexing
- Configurable weighting for environmental vs. goal similarity

📄 **RLM Context**
- Programmatic document inspection (grep, peek, head, tail)
- Safe Python REPL with sandboxed execution
- ReDoS protection for regex patterns

🤖 **DSPy Integration**
- Autonomous planning with ReAct agents
- Integrated tools for memory recall, context search, and code execution
- MLflow observability with automatic tracing

🔒 **Production Ready**
- 99.38% test coverage (327 passing tests)
- Comprehensive security layers (AST validation, restricted builtins, ReDoS protection)
- Backward-compatible storage versioning
- Extensive documentation and examples

**Getting Started:**

```bash
pip install rec-praxis-rlm
```

```python
from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig

memory = ProceduralMemory(MemoryConfig())
memory.store(Experience(
    env_features=["web", "python"],
    goal="extract data from website",
    action="Used BeautifulSoup with CSS selectors",
    result="Successfully extracted 100 items",
    success=True
))

experiences = memory.recall(
    env_features=["web"],
    goal="scrape data",
    top_k=5
)
```

See the [README](REC_PRAXIS_RLM_README.md) for complete documentation and examples.

**Contributors:**

Thank you to all contributors who made this release possible!

---

## Upgrade Guide

### 0.0.x → 0.1.0

This is the initial release, so no upgrade guide is needed.

For future releases, this section will provide:
- Breaking changes and migration steps
- New features and how to use them
- Deprecation warnings
- Configuration changes

---

## Support

- **Documentation**: [README](REC_PRAXIS_RLM_README.md) | [API Reference](docs/api_reference.md) | [Architecture](docs/architecture.md)
- **Issues**: [GitHub Issues](https://github.com/your-org/rec-praxis-rlm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/rec-praxis-rlm/discussions)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
