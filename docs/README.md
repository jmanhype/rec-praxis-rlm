# Documentation

Comprehensive documentation for rec-praxis-rlm.

## Quick Links

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Installation, basic usage, configuration |
| [API Reference](api_reference.md) | Complete API documentation for all classes and functions |
| [Architecture](architecture.md) | System architecture, design decisions, and data flow |
| [CLI Reference](cli-reference.md) | Command-line tools documentation |
| [Integrations](integrations.md) | Claude Code, GitHub Actions, VS Code, pre-commit |
| [Contributing](contributing.md) | Development setup, testing, pull request process |

## User Guides

### Getting Started
- [Getting Started Guide](getting-started.md) - Installation, first steps, common issues
- [Main README](../README.md) - Quick start, installation, basic examples
- [Examples](../examples/README.md) - Practical code examples
- [Configuration](api_reference.md#configuration) - MemoryConfig, ReplConfig, PlannerConfig presets

### Advanced Features
- [Endless Mode](endless_mode.md) - Token budget tracking, auto-compression, multi-session workflows
- [Web Viewer](web_viewer.md) - Interactive memory inspection and debugging
- [Claude Code Hooks](../.claude/README.md) - Automatic experience capture integration

### Developer Tools
- [CLI Reference](cli-reference.md) - Complete CLI tools documentation
- [Integrations](integrations.md) - Claude Code, GitHub Actions, VS Code, pre-commit hooks
- [GitHub Actions](https://github.com/jmanhype/rec-praxis-action) - Official GitHub Action
- [VS Code Extension](../vscode-extension/README.md) - IDE integration

## Evaluations

Performance benchmarks and comparative analysis:

| Document | Description |
|----------|-------------|
| [Benchmarks](evaluations/BENCHMARKS.md) | Performance metrics, FAISS speedup, token reduction |
| [Claude-MEM Comparison](evaluations/CLAUDE-MEM-COMPARISON.md) | Comparison with claude-mem project |
| [DX Evaluation](evaluations/DX-EVALUATION.md) | Developer experience analysis |
| [Edge Case Analysis](evaluations/EDGE-CASE-ANALYSIS.md) | Robustness testing, failure modes |
| [Senior Engineering Review](evaluations/SENIOR_ENGINEERING_REVIEW.md) | Code quality and architecture review |

## API Documentation

Full API reference is in [api_reference.md](api_reference.md), organized by module:

### Core Modules
- `rec_praxis_rlm.memory` - ProceduralMemory, Experience, MemoryConfig
- `rec_praxis_rlm.rlm` - RLMContext, DocumentStore, SafeExecutor
- `rec_praxis_rlm.embeddings` - Embedding providers (SentenceTransformer, API, Jaccard)

### Agents & Automation
- `rec_praxis_rlm.dspy_agent` - PraxisRLMPlanner (DSPy integration)
- `rec_praxis_rlm.endless_mode` - EndlessAgent (long-running tasks)
- `rec_praxis_rlm.agents` - Specialized agents (code review, security, dependency)

### Developer Tools
- `rec_praxis_rlm.cli` - Command-line interfaces
- `rec_praxis_rlm.web_viewer` - Web UI for memory inspection
- `rec_praxis_rlm.telemetry` - MLflow integration

### Utilities
- `rec_praxis_rlm.privacy` - Privacy redaction patterns
- `rec_praxis_rlm.concepts` - Concept tagging and semantic extraction
- `rec_praxis_rlm.config` - Configuration classes and presets

## Architecture

See [architecture.md](architecture.md) for:
- System design and component interactions
- Data flow diagrams
- Storage format (JSONL with versioning)
- Security model (sandboxing, privacy)
- Performance optimizations (FAISS, caching)

## Testing

See [../tests/README.md](../tests/README.md) for:
- Test structure (unit, integration, RAGAS)
- Running tests with pytest
- Coverage reports (99.38%)
- Writing new tests

## Contributing

See [contributing.md](contributing.md) for:
- Development setup
- Code style guidelines
- Pull request process
- Testing requirements
- Release process

## Support

- **Issues**: [GitHub Issues](https://github.com/jmanhype/rec-praxis-rlm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jmanhype/rec-praxis-rlm/discussions)
- **Email**: jmanhype@users.noreply.github.com (security issues)

## License

MIT License - see [../LICENSE](../LICENSE) for details.
