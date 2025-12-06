---
layout: default
title: Spec Kit - Specification-Driven Development with Beads
---

# Spec Kit

**Specification-Driven Development with Beads Integration**

A comprehensive workflow system for AI-assisted software development that combines structured specifications with persistent task memory.

---

## The Killer Combo: Spec Kit + Beads 💎

**Spec Kit** provides structure (WHAT/WHY/HOW).
**Beads** provides memory (persistent task graph that survives context limits).

Together they solve:
- ✅ Structured specification-driven development
- ✅ Long-term memory across sessions
- ✅ Dependency tracking that doesn't disappear
- ✅ Work discovery and prioritization

---

## Quick Start

### Installation (5 minutes)

```bash
# Install Spec Kit in your project
cd your-project
curl -sSL https://raw.githubusercontent.com/jmanhype/speckit/main/install.sh | bash

# (Optional) Install Beads for persistent memory
brew tap steveyegge/beads
brew install bd
bd init
```

### Your First Feature

```bash
# In Claude Code
/speckit.specify Add user authentication with email/password

# Follow the workflow
/speckit.plan
/speckit.tasks
/speckit.implement
```

---

## Documentation

### Getting Started
- [Quick Start Guide](https://github.com/jmanhype/speckit/blob/main/QUICKSTART.md) - Get running in 5 minutes
- [Installation Guide](https://github.com/jmanhype/speckit/blob/main/README.md#installation) - Detailed setup instructions

### Workflow
- [Core Workflow](https://github.com/jmanhype/speckit/blob/main/README.md#workflow-commands) - Slash commands and phases
- [Beads Integration](https://github.com/jmanhype/speckit/blob/main/AGENTS.md) - AI agent workflow with persistent memory
- [Task Organization](https://github.com/jmanhype/speckit/blob/main/README.md#task-organization) - Dependency-ordered, priority-based tasks

### Advanced
- [Constitution Template](https://github.com/jmanhype/speckit/blob/main/.specify/memory/constitution.md) - Project principles
- [Custom Templates](https://github.com/jmanhype/speckit/tree/main/.specify/templates) - Customize specs, plans, tasks
- [Contributing](https://github.com/jmanhype/speckit/blob/main/CONTRIBUTING.md) - Join the community

---

## Features

### Specification-Driven Workflow

```
constitution → specify → clarify → plan → checklist → tasks → analyze → implement
```

1. **`/speckit.specify`** - Create feature spec from natural language
2. **`/speckit.clarify`** - Resolve ambiguities (max 3 questions)
3. **`/speckit.plan`** - Generate technical implementation plan
4. **`/speckit.tasks`** - Create dependency-ordered task list
5. **`/speckit.implement`** - Execute implementation

### Quality Gates

- ✅ **Specification Quality** - Technology-agnostic, testable, measurable
- ✅ **Checklist Validation** - Security, accessibility, performance
- ✅ **Consistency Analysis** - Cross-artifact validation
- ✅ **Implementation Gates** - Task completion tracking

### Beads Integration

- 📝 **Persistent Memory** - Task graph survives context limits
- 🔗 **Dependency Tracking** - Complex dependencies with `bd` CLI
- 🔍 **Discovery Tracking** - New work found during implementation
- 🎯 **Smart Prioritization** - `bd ready` shows what's next

---

## Use Cases

### Greenfield Projects

Start new projects with structured specs:

```bash
# Create new project with Spec Kit
./install.sh my-new-project
cd my-new-project

# Define first feature
/speckit.specify Add REST API with authentication
```

### Brownfield Projects

Integrate into existing codebases:

```bash
# Install in current directory
cd existing-project
curl -sSL https://raw.githubusercontent.com/jmanhype/speckit/main/install.sh | bash

# Start documenting features
/speckit.specify Refactor payment processing module
```

### AI-Assisted Development

Work with Claude Code, Copilot, Cursor, or Amp:

- Consistent workflow across sessions
- Context that doesn't disappear
- Discovery tracking during implementation
- Agent instructions in `AGENTS.md`

---

## Architecture

### Directory Structure

```
your-project/
├── .beads/              # Beads task database (persistent memory)
├── .claude/commands/    # Slash commands for Claude Code
├── .specify/            # Templates and automation scripts
├── specs/               # Feature specifications
│   └── 001-feature/
│       ├── spec.md      # WHAT/WHY
│       ├── plan.md      # HOW
│       └── tasks.md     # WHEN (with Beads IDs)
└── AGENTS.md            # AI agent workflow instructions
```

### Workflow Phases

| Phase | Tool | Output | Purpose |
|-------|------|--------|---------|
| **Constitution** | `/speckit.constitution` | `constitution.md` | Project principles |
| **Specify** | `/speckit.specify` | `spec.md` | User requirements (WHAT/WHY) |
| **Clarify** | `/speckit.clarify` | Updated `spec.md` | Resolve ambiguities |
| **Plan** | `/speckit.plan` | `plan.md` | Technical design (HOW) |
| **Checklist** | `/speckit.checklist` | `checklists/*.md` | Quality validation |
| **Tasks** | `/speckit.tasks` | `tasks.md` | Executable steps (WHEN) |
| **Analyze** | `/speckit.analyze` | Validation report | Consistency check |
| **Implement** | `/speckit.implement` | Code | Execute tasks |

---

## Examples

### Task Format

Tasks are organized by user story priority:

```markdown
## Phase 3: User Stories

### P1 - Core Authentication (MVP)
- [ ] (bd-x1y2) T010 [P] [US1] Create User model in src/models/user.py
- [ ] (bd-x3y4) T011 [P] [US1] Create AuthService in src/services/auth.py
- [ ] (bd-x5y6) T012 [US1] Implement login endpoint in src/api/auth.py

### P2 - Social Login
- [ ] (bd-x7y8) T020 [P] [US2] Add OAuth2 provider in src/services/oauth.py
```

**Task Format**: `- [ ] (bd-xxx) [TaskID] [P] [Story] Description with file/path.ext`

- `(bd-xxx)` - Beads issue ID (persistent across sessions)
- `[T###]` - Spec Kit task ID
- `[P]` - Parallelizable (can run concurrently)
- `[US#]` - User story reference
- File path for implementation clarity

### Beads Workflow

```bash
# Check what's ready to work on
bd ready --label feat-001 --json

# Implement task
# (agent implements bd-x1y2)

# Mark complete in both places
# - tasks.md: [ ] → [X]
# - Beads: bd close bd-x1y2

# Repeat until feature is done
```

---

## Community

### Get Involved

- 🐛 [Report Bugs](https://github.com/jmanhype/speckit/issues)
- 💡 [Request Features](https://github.com/jmanhype/speckit/issues/new)
- 💬 [Discussions](https://github.com/jmanhype/speckit/discussions)
- 🤝 [Contributing](https://github.com/jmanhype/speckit/blob/main/CONTRIBUTING.md)

### Resources

- **GitHub Repository**: [jmanhype/speckit](https://github.com/jmanhype/speckit)
- **Beads Project**: [steveyegge/beads](https://github.com/steveyegge/beads)
- **Claude Code**: [claude.com/code](https://claude.com/claude-code)

---

## License

MIT License - See [LICENSE](https://github.com/jmanhype/speckit/blob/main/LICENSE)

Built for use with [Claude Code](https://claude.com/claude-code) by Anthropic.

---

<div style="text-align: center; margin-top: 2em;">
  <a href="https://github.com/jmanhype/speckit" style="font-size: 1.2em;">View on GitHub</a>
</div>
