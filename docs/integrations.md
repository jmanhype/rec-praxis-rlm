# Integrations

rec-praxis-rlm integrates with popular development tools and platforms.

## Claude Code

**Zero-config automatic experience capture** - Automatically learn from every tool use in Claude Code sessions.

### Features

- **Automatic Capture**: Every Bash command, file read/write, grep operation → stored as experience
- **Session Context**: Shows recent successes and failures at session start
- **Privacy-Aware**: Automatically redacts API keys, passwords, emails
- **Local Storage**: All data stays in `.claude/memory.jsonl` on your machine

### Setup

The `.claude/hooks/` directory is automatically detected by Claude Code. No configuration required.

```bash
# Verify installation
ls -la .claude/hooks/

# You should see:
# post_tool_use.sh   - Captures tool uses
# session_start.sh   - Shows context at session start
```

### How It Works

1. **post_tool_use hook**: Runs after every Claude Code tool use
   - Captures: tool name, input, output, success/failure
   - Stores as Experience in ProceduralMemory
   - Redacts sensitive data automatically

2. **session_start hook**: Runs at the beginning of each session
   - Shows memory statistics
   - Lists recent successful patterns
   - Warns about recent failures

### Example Output

At session start, you'll see:

```
📚 REC Praxis RLM Context

Memory Statistics:
- Total experiences: 127
- Recent successful patterns: 5
- Recent failures to avoid: 2

Recent Successful Patterns:
1. [optimize] Database query optimization
   ✓ Reduced latency from 2s to 50ms

2. [test] pytest fixture creation
   ✓ Reusable test setup for API tests

Recent Failures to Avoid:
1. [deploy] Docker build without layer caching
   ✗ Build time: 15 minutes (use multi-stage builds)

2. [refactor] Renaming without grep first
   ✗ Missed 12 references, broke imports
```

### Configuration

Customize in `.claude/hooks/config.json` (optional):

```json
{
  "memory_path": ".claude/memory.jsonl",
  "privacy": {
    "enable_redaction": true,
    "custom_patterns": [
      "CUSTOM_SECRET_.*"
    ]
  },
  "filters": {
    "exclude_tools": ["Read"],
    "min_experience_length": 10
  }
}
```

### Accessing Captured Experiences

```python
from rec_praxis_rlm import ProceduralMemory, MemoryConfig

# Load Claude Code experiences
memory = ProceduralMemory(MemoryConfig(storage_path='.claude/memory.jsonl'))

# Query for specific patterns
bash_successes = memory.recall(
    env_features=['bash', 'command'],
    goal='deploy',
    top_k=5
)

for exp in bash_successes:
    if exp.success:
        print(f"Command: {exp.action}")
        print(f"Result: {exp.result}")
```

---

## GitHub Actions

Use the official **rec-praxis-action** for seamless CI/CD integration.

### Repository

[https://github.com/jmanhype/rec-praxis-action](https://github.com/jmanhype/rec-praxis-action)

### Features

- **Incremental Scanning**: Only scan changed files in PRs
- **SARIF Output**: Integrates with GitHub Security tab
- **Procedural Memory**: Learn from past scans across CI runs
- **Multi-language Support**: Python, JavaScript, TypeScript, Go
- **Caching**: Fast subsequent runs with dependency caching

### Basic Setup

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
          scan-type: 'all'          # code-review, security-audit, dependency-scan, all
          severity: 'HIGH'           # CRITICAL, HIGH, MEDIUM, LOW, INFO
          fail-on: 'CRITICAL'        # Fail build on this severity or higher
          incremental: 'true'        # Only scan changed files
```

### Advanced Configuration

```yaml
- uses: jmanhype/rec-praxis-action@v1
  with:
    scan-type: 'all'
    severity: 'HIGH'
    fail-on: 'CRITICAL'
    incremental: 'true'
    output-format: 'sarif'          # human, json, sarif
    config-file: '.rec-praxis.yml'  # Custom config
    memory-persist: 'true'          # Persist procedural memory
    exclude-paths: 'tests/**,docs/**'
    python-version: '3.11'
```

### Outputs

The action provides outputs for custom workflows:

```yaml
- uses: jmanhype/rec-praxis-action@v1
  id: scan

- name: Comment on PR
  if: steps.scan.outputs.findings-count > 0
  uses: actions/github-script@v6
  with:
    script: |
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: `## Security Scan Results\n\nFindings: ${{ steps.scan.outputs.findings-count }}\n\nSee [details](${{ steps.scan.outputs.report-url }})`
      })
```

### Integration with GitHub Security Tab

Enable SARIF upload:

```yaml
- uses: jmanhype/rec-praxis-action@v1
  with:
    output-format: 'sarif'

- name: Upload SARIF
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: rec-praxis-results.sarif
```

---

## VS Code Extension

**Real-time inline diagnostics** powered by rec-praxis-rlm.

### Installation

#### From .vsix File

```bash
# Download from releases
wget https://github.com/jmanhype/rec-praxis-rlm/releases/download/v0.9.2/rec-praxis-rlm-vscode-0.4.2.vsix

# Install in VS Code
code --install-extension rec-praxis-rlm-vscode-0.4.2.vsix
```

#### From Source

```bash
cd vscode-extension
npm install
npm run compile
vsce package
code --install-extension rec-praxis-rlm-vscode-*.vsix
```

### Features

- **Real-time Scanning**: Scan on save, on demand
- **Inline Diagnostics**: Squiggly underlines for issues
- **Quick Fixes**: One-click remediation
- **Problem Panel Integration**: View all findings in Problems panel
- **Severity Icons**: Visual indicators for CRITICAL/HIGH/MEDIUM/LOW

### Configuration

```json
// .vscode/settings.json
{
  "recPraxisRlm.scanOnSave": true,
  "recPraxisRlm.severity": "HIGH",
  "recPraxisRlm.autoFix": false,
  "recPraxisRlm.excludePaths": [
    "node_modules/**",
    "dist/**",
    "test/**"
  ]
}
```

### Commands

- **Rec-Praxis: Scan Current File** - Scan active file
- **Rec-Praxis: Scan Workspace** - Scan entire workspace
- **Rec-Praxis: Clear Diagnostics** - Clear all findings
- **Rec-Praxis: Show Report** - Open HTML report

### Usage

1. Open a Python file
2. Save to trigger automatic scan
3. View diagnostics in Problems panel
4. Click on finding for details
5. Apply quick fix (if available)

---

## Pre-commit Hooks

Integrate with Git pre-commit for automatic quality checks.

### Installation

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml <<EOF
repos:
  - repo: local
    hooks:
      - id: rec-praxis-review
        name: Code Review
        entry: rec-praxis-review
        language: system
        types: [python]
        args: [--severity=HIGH]
        pass_filenames: true

      - id: rec-praxis-audit
        name: Security Audit
        entry: rec-praxis-audit
        language: system
        types: [python]
        args: [--fail-on=CRITICAL]
        pass_filenames: true
EOF

# Install hooks
pre-commit install
```

### Usage

Hooks run automatically on `git commit`:

```bash
# Stage changes
git add app.py

# Commit (triggers hooks)
git commit -m "feat: add new feature"

# Hooks run:
Code Review....................................Passed
Security Audit.................................Failed
- hook id: rec-praxis-audit
- exit code: 1

🔴 CRITICAL: Hardcoded API key (line 45)

# Fix issue, then commit again
```

### Skip Hooks (Emergency Only)

```bash
git commit --no-verify -m "emergency fix"
```

---

## MLflow Integration

Track security scan results and procedural memory metrics with MLflow.

### Setup

```bash
# Install with MLflow support
pip install rec-praxis-rlm[mlflow]

# Start MLflow UI
mlflow ui --port 5000
```

### Usage

```python
from rec_praxis_rlm.agents import CodeReviewAgent
from rec_praxis_rlm.telemetry import MLflowTracker

# Create agent with MLflow tracking
tracker = MLflowTracker(experiment_name="code-review")
agent = CodeReviewAgent(telemetry=tracker)

# Run review (automatically tracked)
results = agent.review_file("app.py")

# Metrics logged to MLflow:
# - findings_count
# - severity_distribution
# - scan_duration
# - memory_recall_accuracy
```

### View Results

Open http://localhost:5000 to see:

- Run history
- Metrics over time
- Parameter comparison
- Artifact storage (reports, SARIF files)

---

## Jupyter Notebooks

Use rec-praxis-rlm in Jupyter for interactive analysis.

### Installation

```bash
pip install rec-praxis-rlm[all] jupyter
```

### Example Notebook

```python
# Cell 1: Setup
from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig
import pandas as pd

memory = ProceduralMemory(MemoryConfig(storage_path='./memory.jsonl'))

# Cell 2: Store experiences
experiences = [
    Experience(
        env_features=['python', 'pandas'],
        goal='merge dataframes',
        action='Used pd.merge with left join',
        result='Successfully merged 1M rows',
        success=True
    ),
    # ... more experiences
]

for exp in experiences:
    memory.store(exp)

# Cell 3: Query and visualize
results = memory.recall(
    env_features=['python'],
    goal='dataframe',
    top_k=10
)

df = pd.DataFrame([
    {
        'goal': exp.goal,
        'success': exp.success,
        'env': ', '.join(exp.env_features)
    }
    for exp in results
])

df.plot.bar(x='goal', y='success')
```

---

## Web Viewer

Interactive web UI for inspecting procedural memory.

### Launch

```bash
# Start web viewer
rec-praxis-web --memory-path=./memory.jsonl --port=8080

# Open browser
open http://localhost:8080
```

### Features

- **Search**: Full-text search across experiences
- **Filters**: By success/failure, environment, date range
- **Visualization**: Timeline view, success rate charts
- **Export**: Export filtered results as JSON/CSV

See [Web Viewer Documentation](web_viewer.md) for details.

---

## API Integrations

### REST API

Expose rec-praxis-rlm as a REST API:

```python
from fastapi import FastAPI
from rec_praxis_rlm import ProceduralMemory, MemoryConfig, Experience

app = FastAPI()
memory = ProceduralMemory(MemoryConfig(storage_path='./api_memory.jsonl'))

@app.post("/experiences")
def store_experience(exp: Experience):
    memory.store(exp)
    return {"status": "stored"}

@app.get("/recall")
def recall(goal: str, env_features: str, top_k: int = 5):
    results = memory.recall(
        env_features=env_features.split(','),
        goal=goal,
        top_k=top_k
    )
    return [exp.dict() for exp in results]
```

### GraphQL

```python
import strawberry
from strawberry.fastapi import GraphQLRouter

@strawberry.type
class ExperienceType:
    goal: str
    action: str
    result: str
    success: bool

@strawberry.type
class Query:
    @strawberry.field
    def recall(self, goal: str, top_k: int = 5) -> list[ExperienceType]:
        results = memory.recall(goal=goal, top_k=top_k)
        return [ExperienceType(**exp.dict()) for exp in results]

schema = strawberry.Schema(query=Query)
app.include_router(GraphQLRouter(schema), prefix="/graphql")
```

---

## Custom Integrations

### Webhook Notifications

Send findings to Slack/Discord:

```python
from rec_praxis_rlm.agents import CodeReviewAgent
import requests

agent = CodeReviewAgent()
results = agent.review_file("app.py")

# Send to Slack
if any(f.severity == "CRITICAL" for f in results):
    requests.post(
        "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        json={
            "text": f"🚨 Critical security finding in app.py\n{results[0].message}"
        }
    )
```

### Database Storage

Store experiences in PostgreSQL:

```python
from rec_praxis_rlm import ProceduralMemory
from sqlalchemy import create_engine, Column, String, Boolean, JSON
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

class ExperienceDB(Base):
    __tablename__ = 'experiences'
    id = Column(String, primary_key=True)
    env_features = Column(JSON)
    goal = Column(String)
    action = Column(String)
    result = Column(String)
    success = Column(Boolean)

engine = create_engine('postgresql://user:pass@localhost/memory')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Custom storage backend
class PostgresMemory(ProceduralMemory):
    def store(self, exp):
        session = Session()
        db_exp = ExperienceDB(**exp.dict())
        session.add(db_exp)
        session.commit()
```

---

## Community Integrations

Integrations built by the community:

- **rec-praxis-docker**: Docker container with pre-configured environment
- **rec-praxis-cli-enhanced**: Enhanced CLI with TUI interface
- **rec-praxis-vscode-pro**: Extended VS Code features

See [Community Integrations](https://github.com/jmanhype/rec-praxis-rlm/wiki/Community-Integrations) for full list.
