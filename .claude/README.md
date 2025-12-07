# Claude Code Hook Integration for rec-praxis-rlm

This directory contains Claude Code hooks that enable **automatic experience capture** with zero user intervention.

## Features

### 🎯 Automatic Experience Capture (`post_tool_use.sh`)

After every tool use in Claude Code, this hook:
- Captures the tool name, input, output, and success status
- Stores it as an Experience in `.claude/memory.jsonl`
- Classifies the experience type (learn/recover/optimize/explore)
- Extracts semantic tags for better retrieval

**What gets captured:**
- Bash commands and their output
- File reads, edits, and writes
- Search/grep operations
- Git operations
- All tool uses with their results

### 📚 Context Injection (`session_start.sh`)

At the start of each Claude Code session, this hook:
- Retrieves memory statistics (total experiences, success rate)
- Shows recent successful patterns to reinforce good practices
- Highlights recent failures to avoid repeating mistakes
- Provides continuity across sessions

**Example output:**
```
📚 **REC Praxis RLM Context**

**Memory Statistics:**
- Total experiences: 127
- Recent successful patterns: 5
- Recent failures to avoid: 2

**Recent Successful Patterns:**
1. [optimize] Refactor database query for better performance
   ✓ Query latency reduced from 2s to 50ms

2. [learn] Understand how FAISS indexing works in memory.py
   ✓ Successfully added new similarity search method

**Recent Failures (Learn from these):**
1. [recover] Fix test failures in test_privacy.py
   ✗ Still failing - need to adjust pattern length

💡 **Tip:** Past experiences are automatically retrieved when relevant to your current task.
```

## Setup

### Prerequisites

1. **Install rec-praxis-rlm:**
   ```bash
   pip install -e .
   ```

2. **Claude Code** must be installed and configured

### Configuration

The hooks are configured in `.claude/settings.json`:

```json
{
  "hooks": {
    "post_tool_use": ".claude/hooks/post_tool_use.sh",
    "session_start": ".claude/hooks/session_start.sh"
  }
}
```

### Memory Storage

Experiences are stored in:
```
.claude/memory.jsonl
```

This file grows over time as you work. You can:
- **Review it** to see what's been captured
- **Compact it** using `ProceduralMemory.compact(keep_recent_n=1000)`
- **Delete it** to start fresh (it will be recreated automatically)

## How It Works

### Experience Capture Flow

```
┌─────────────────┐
│ Claude Code     │
│ executes tool   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ post_tool_use.sh hook   │
│ - Reads env vars        │
│ - Extracts goal/result  │
│ - Creates Experience    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ ProceduralMemory.store()│
│ - Privacy redaction     │
│ - Concept tagging       │
│ - Type classification   │
│ - Persistent storage    │
└─────────────────────────┘
```

### Context Injection Flow

```
┌─────────────────┐
│ New Claude Code │
│ session starts  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ session_start.sh hook   │
│ - Loads memory.jsonl    │
│ - Retrieves stats       │
│ - Formats context       │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Claude sees context     │
│ - Recent successes      │
│ - Recent failures       │
│ - Memory statistics     │
└─────────────────────────┘
```

## Privacy & Security

### What's Captured

The hooks capture:
- Tool names and descriptions
- File paths (relative to working directory)
- Command outputs (truncated to 1000 chars)
- Success/failure status

### What's NOT Captured

The hooks automatically redact:
- API keys (OpenAI, Anthropic, etc.)
- Passwords and tokens
- Private IP addresses
- Email addresses
- Credit card numbers

See `rec_praxis_rlm/privacy.py` for full list of redaction patterns.

### Data Location

All data stays **local** on your machine:
- `.claude/memory.jsonl` (in your project directory)
- No cloud sync
- No external uploads
- You control the data

## Advanced Usage

### Manual Experience Retrieval

You can query your experience memory programmatically:

```python
from rec_praxis_rlm import ProceduralMemory, MemoryConfig

# Load your session memory
config = MemoryConfig(storage_path=".claude/memory.jsonl", embedding_model="")
memory = ProceduralMemory(config=config, use_faiss=False)

# Recall experiences similar to a goal
experiences = memory.recall(
    env_features=["python", "testing"],
    goal="Fix failing pytest tests",
    top_k=5
)

for exp in experiences:
    print(f"[{exp.experience_type}] {exp.goal}")
    print(f"  Action: {exp.action[:100]}")
    print(f"  Result: {exp.result[:100]}")
    print()
```

### Compacting Memory

Over time, your memory file will grow. To keep only recent experiences:

```python
from rec_praxis_rlm import ProceduralMemory, MemoryConfig

config = MemoryConfig(storage_path=".claude/memory.jsonl", embedding_model="")
memory = ProceduralMemory(config=config, use_faiss=False)

# Keep only last 1000 experiences
removed = memory.compact(keep_recent_n=1000)
print(f"Removed {removed} old experiences")
```

### Disabling Hooks

To temporarily disable hooks:

1. **Rename settings.json:**
   ```bash
   mv .claude/settings.json .claude/settings.json.disabled
   ```

2. **Or comment out hooks:**
   ```json
   {
     "hooks": {
       // "post_tool_use": ".claude/hooks/post_tool_use.sh",
       // "session_start": ".claude/hooks/session_start.sh"
     }
   }
   ```

## Troubleshooting

### Hooks Not Running

1. **Check hook permissions:**
   ```bash
   ls -l .claude/hooks/
   ```
   Should show `-rwxr-xr-x` (executable)

2. **Check Python availability:**
   ```bash
   which python3
   ```

3. **Check rec-praxis-rlm installation:**
   ```bash
   python3 -c "import rec_praxis_rlm; print(rec_praxis_rlm.__version__)"
   ```

### Memory File Growing Too Large

```bash
# Check size
du -h .claude/memory.jsonl

# Compact to last 500 experiences
python3 -c "
from rec_praxis_rlm import ProceduralMemory, MemoryConfig
m = ProceduralMemory(config=MemoryConfig(storage_path='.claude/memory.jsonl', embedding_model=''), use_faiss=False)
print(f'Removed {m.compact(keep_recent_n=500)} experiences')
"
```

### Hook Errors in Logs

Check Claude Code logs for errors:
```bash
# Hook errors appear in stderr
# Check your Claude Code log file location
```

## Examples

### Example 1: Learning from Failed Tests

**Session 1:**
```
❯ Run pytest tests/
✗ 5 tests failed

[post_tool_use captures failure]
```

**Session 2 (next day):**
```
📚 **REC Praxis RLM Context**

**Recent Failures (Learn from these):**
1. [recover] Run pytest tests in test_privacy.py
   ✗ 5 tests failed - pattern too strict

💡 **Tip:** Check test_privacy.py failures before running tests again
```

### Example 2: Remembering Optimization Patterns

**Session 1:**
```
❯ Optimize database query performance
✓ Added index, latency reduced 95%

[post_tool_use captures success]
```

**Session 2:**
```
📚 **REC Praxis RLM Context**

**Recent Successful Patterns:**
1. [optimize] Optimize database query performance
   ✓ Added index on user_id column, latency reduced from 2s to 50ms

💡 Similar optimization approaches may work for other slow queries
```

## Contributing

To improve the hooks:

1. **Test changes:**
   ```bash
   # Test post_tool_use hook
   TOOL_NAME="Bash" TOOL_INPUT='{"command":"echo test"}' \
   TOOL_OUTPUT="test" TOOL_SUCCESS="true" \
   WORKING_DIR="$(pwd)" \
   .claude/hooks/post_tool_use.sh

   # Test session_start hook
   WORKING_DIR="$(pwd)" SESSION_ID="test-123" \
   .claude/hooks/session_start.sh
   ```

2. **Submit changes:**
   - Open an issue describing the improvement
   - Submit a PR with tests
   - Update this README

## License

MIT License - Same as rec-praxis-rlm
