# rec-praxis-rlm

Procedural memory and REPL context library for DSPy 3.0 agents. Stores agent experiences as structured records, retrieves them by similarity, and provides sandboxed code execution and document inspection tools.

| | |
|---|---|
| Version | 0.9.2 |
| Python | 3.10+ |
| License | MIT |
| Status | Alpha (PyPI published) |
| Test coverage | 99.38% (claimed) |

## What it does

1. **Procedural memory** -- stores `Experience` records (environment features, goal, action, result, success flag) in JSONL. Retrieves by hybrid similarity over environment + goal embeddings.
2. **FAISS indexing** -- optional drop-in for faster retrieval when the experience store exceeds ~10k records.
3. **RLM context** -- document inspection primitives (`grep`, `peek`, `head`, `tail`) with ReDoS-safe regex validation.
4. **Sandboxed exec** -- runs Python snippets through AST validation and restricted builtins. Designed for trusted agent tool output, not adversarial input.
5. **DSPy planner** -- wraps a ReAct agent with memory recall and RLM tools. Works with any LiteLLM-compatible provider.
6. **CLI tools** -- `rec-praxis-review`, `rec-praxis-audit`, `rec-praxis-deps` for code review, OWASP-based security audit, and dependency/secret scanning.

## Architecture

```
rec_praxis_rlm/
  memory.py            # ProceduralMemory, Experience, FAISS index
  rlm.py               # RLMContext (grep/head/tail/safe_exec)
  dspy_agent.py         # PraxisRLMPlanner (ReAct + memory + tools)
  fact_store.py         # FactStore (semantic memory)
  embeddings.py         # Embedding providers (sentence-transformers, OpenAI API)
  config.py             # MemoryConfig, ReplConfig, PlannerConfig
  cli.py                # CLI entry points
  agents/               # Code review, security audit, dependency scan, test gen
  graph/                # Optional Parseltongue integration for call-graph analysis
  compression.py        # Experience compaction
  telemetry.py          # MLflow tracing hooks
```

## Dependencies

| Package | Role |
|---|---|
| dspy-ai >= 3.0.4 | Agent framework |
| sentence-transformers >= 2.2 | Default embeddings |
| pydantic >= 2.0 | Config/validation |
| jsonlines >= 3.0 | Experience storage |
| mlflow >= 3.0 | Telemetry/tracing |
| tiktoken >= 0.5.0 | Token counting |
| faiss-cpu >= 1.7 | Optional: fast retrieval |

## Install

```bash
pip install rec-praxis-rlm        # core
pip install rec-praxis-rlm[all]   # adds FAISS, openai, aiofiles
```

## Usage

### Store and recall experiences

```python
from rec_praxis_rlm.memory import ProceduralMemory, Experience
from rec_praxis_rlm.config import MemoryConfig

config = MemoryConfig(storage_path="./agent_memory.jsonl")
memory = ProceduralMemory(config)

memory.store(Experience(
    env_features=["web_scraping", "python", "beautifulsoup"],
    goal="extract product prices from e-commerce site",
    action="Used BeautifulSoup with CSS selectors for price elements",
    result="Extracted 1000 prices with 99% accuracy",
    success=True
))

results = memory.recall(
    env_features=["web_scraping", "python"],
    goal="extract data from website",
    top_k=5
)
```

### Document inspection

```python
from rec_praxis_rlm.rlm import RLMContext

context = RLMContext()
context.add_document("app_log", open("application.log").read())

matches = context.grep(r"ERROR.*database", doc_id="app_log")
recent = context.tail("app_log", n_lines=50)
```

### Sandboxed execution

```python
result = context.safe_exec("sum(range(10))")
# result.success -> True, result.output -> "45"
```

The sandbox blocks imports, eval/exec, and file I/O via AST validation. It is not hardened against adversarial input.

### DSPy planner

```python
from rec_praxis_rlm.dspy_agent import PraxisRLMPlanner
from rec_praxis_rlm.config import PlannerConfig

planner = PraxisRLMPlanner(
    memory=memory,
    config=PlannerConfig(
        lm_model="groq/llama-3.3-70b-versatile",
        api_key="gsk-..."
    )
)

answer = planner.plan(
    goal="Analyze server errors and suggest fixes",
    env_features=["production", "high_traffic", "database"]
)
```

### CLI tools

```bash
rec-praxis-review src/**/*.py --severity=HIGH
rec-praxis-review src/**/*.py --format=html --output=report.html
rec-praxis-audit app.py --fail-on=CRITICAL
rec-praxis-deps --requirements=requirements.txt --files src/**/*.py
```

Output formats: `human` (default), `json`, `html`, `sarif`.

Graph-aware analysis (`--use-graph`) requires a running Parseltongue server. Falls back to pattern-based detection without it.

## Configuration presets

```python
MemoryConfig.for_code_review()     # precise, successful experiences only
MemoryConfig.for_security_audit()  # broad, includes false positives
MemoryConfig.for_web_scraping()    # prioritizes site structure
MemoryConfig.for_testing()         # high precision for test patterns
```

## Limitations

- The sandbox is not a security boundary. Do not run untrusted code.
- FAISS index must be rebuilt if the embedding model changes.
- Graph-aware analysis depends on an external Parseltongue server (not included).
- LLM-powered deep analysis (DSPy agents calling review/audit) is not yet integrated into the CLI; planned for v0.10.0.
- The "99.38% coverage" number comes from the test suite but does not cover all optional-dependency code paths.

## Related

- [rec-praxis-action](https://github.com/jmanhype/rec-praxis-action) -- GitHub Action wrapping this package
- [PyPI](https://pypi.org/project/rec-praxis-rlm/)
- [Changelog](CHANGELOG.md)

## License

MIT
