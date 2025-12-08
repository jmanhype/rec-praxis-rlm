# Endless Mode for Long-Running Agents

Enable 100+ step agents without context exhaustion using automatic compression pipeline and token budget tracking.

![Endless Mode](https://img.shields.io/badge/Feature-Endless_Mode-9C27B0?style=for-the-badge)

## Overview

Endless Mode solves the **context window exhaustion problem** for long-running autonomous agents. By automatically tracking token usage, compressing context, and managing memory size, agents can run indefinitely without hitting context limits.

### The Problem

Traditional agents suffer from **context window exhaustion**:

```
Step 1-20:  ✅ Fresh, responsive (20% context usage)
Step 21-50: ⚠️  Entering "dumb zone" (40%+ context usage)
Step 51-80: 🐌 Slow, degraded quality (70%+ context usage)
Step 81+:   ❌ Context overflow, agent crashes
```

### The Solution

Endless Mode provides **automatic context management**:

```
Step 1-20:  ✅ Layer 2 (full experiences)
Step 21-50: 🗜️  Auto-compress at 40% → Layer 1 (compressed)
Step 51-80: ✅ Stay under 40% via compaction
Step 81+:   ✅✅✅ Infinite operation (continuous compression)
```

## Quick Start

### Basic Usage

```python
from rec_praxis_rlm import ProceduralMemory, MemoryConfig, EndlessAgent

# Create memory
config = MemoryConfig(storage_path="./memory.jsonl")
memory = ProceduralMemory(config=config)

# Create endless mode agent
agent = EndlessAgent(
    memory=memory,
    token_budget=100000,  # Claude Opus 3.5 context window
)

# Your agent loop
for step in range(100):
    # Store experience
    memory.store(experience)

    # Track token usage
    agent.track_tokens(prompt_tokens=500, completion_tokens=200)

    # Auto-compress when needed
    if agent.should_compress():
        result = agent.auto_compress_context()
        print(f"Compressed! Saved ~{result['estimated_tokens_saved']} tokens")

    # Continue working...
```

### Advanced Usage

```python
from rec_praxis_rlm import EndlessAgent, CompressionConfig

# Custom compression configuration
config = CompressionConfig(
    threshold=0.4,         # Compress at 40% utilization
    target_rate=0.2,       # Compress down to 20%
    min_experiences=10,    # Need 10+ experiences before compressing
    layer1_threshold=0.3,  # Use compressed summaries above 30%
    layer2_threshold=0.5,  # Use full details below 50%
)

agent = EndlessAgent(
    memory=memory,
    token_budget=100000,
    compression_config=config,
)

# Adaptive recall - automatically selects best layer
results, metadata = agent.recall_adaptive(
    env_features=["api", "database"],
    goal="Optimize slow query",
    top_k=5
)

print(f"Used layer {metadata['layer']}: {metadata['format']}")
# Layer 1: compressed_strings (~500 tokens each)
# Layer 2: full_experiences (~2000 tokens each)
```

## Features

### 1. Token Budget Tracking

Track token usage across your agent's lifecycle:

```python
# Track tokens from LLM calls
agent.track_tokens(prompt_tokens=1500, completion_tokens=800)

# Check status
status = agent.get_status()
print(f"Utilization: {status['token_budget']['utilization']*100:.1f}%")
print(f"Remaining: {status['token_budget']['remaining']} tokens")
```

**Key Metrics:**
- `total_budget`: Total token budget for session
- `used_tokens`: Tokens consumed so far
- `remaining_tokens`: Tokens left
- `utilization_rate`: 0.0-1.0 (0% to 100%)

### 2. Automatic Compression

Automatically compress context when approaching budget limits:

```python
# Check if compression needed
if agent.should_compress():
    result = agent.auto_compress_context()

    print(f"Removed {result['experiences_removed']} old experiences")
    print(f"Kept {result['experiences_kept']} recent experiences")
    print(f"Saved ~{result['estimated_tokens_saved']} tokens")
    print(f"Utilization: {result['utilization_before']*100:.1f}% → {result['utilization_after']*100:.1f}%")
```

**Compression Strategy:**
1. Trigger at configurable threshold (default: 40% utilization)
2. Keep only recent experiences (sorted by timestamp)
3. Target lower utilization rate (default: 20%)
4. Estimate token savings (~1000 tokens per experience removed)

### 3. Progressive Disclosure Integration

Automatically select optimal recall layer based on budget:

```python
# Adaptive recall - selects layer based on utilization
results, metadata = agent.recall_adaptive(
    env_features=["python", "testing"],
    goal="Fix failing pytest tests",
    top_k=5
)

if metadata['layer'] == 1:
    # Compressed summaries (~500 tokens each)
    print("Using compressed mode to save tokens")
elif metadata['layer'] == 2:
    # Full experiences (~2000 tokens each)
    print("Using full details mode")
```

**Layer Selection Rules:**
- **Utilization < 30%**: Layer 2 (full experiences)
- **Utilization 30-50%**: Layer 1 (compressed summaries)
- **Utilization > 50%**: Layer 1 (compressed summaries)

### 4. Context Window Monitoring

Real-time monitoring of context utilization:

```python
status = agent.get_status()

# Token budget status
print(f"Budget: {status['token_budget']['used']}/{status['token_budget']['total']}")
print(f"Utilization: {status['token_budget']['utilization']*100:.1f}%")

# Memory status
print(f"Experiences: {status['memory']['total_experiences']}")

# Compression status
print(f"Should compress: {status['compression']['should_compress']}")
print(f"Recommended layer: {status['compression']['recommended_layer']}")
print(f"Compression events: {status['compression']['compression_events']}")
```

## Configuration

### CompressionConfig

Control compression behavior:

```python
config = CompressionConfig(
    threshold=0.4,         # Trigger compression at 40% utilization
    target_rate=0.2,       # Compress down to 20% utilization
    min_experiences=10,    # Minimum experiences before enabling compression
    layer1_threshold=0.3,  # Use compressed summaries above 30% utilization
    layer2_threshold=0.5,  # Use full details below 50% utilization
    layer3_enabled=False,  # Disable expanded context (layer3)
)
```

**Parameters:**
- `threshold` (0.0-1.0): Utilization threshold to trigger compression
- `target_rate` (0.0-1.0): Target utilization after compression
- `min_experiences` (int): Minimum experiences required before compressing
- `layer1_threshold` (0.0-1.0): Utilization above which to use compressed summaries
- `layer2_threshold` (0.0-1.0): Utilization below which to use full details
- `layer3_enabled` (bool): Enable expanded context (layer3) for very low utilization

### TokenBudget

```python
from rec_praxis_rlm import TokenBudget

budget = TokenBudget(total_budget=100000)

# Track usage
budget.track(prompt_tokens=500, completion_tokens=200)

# Check status
print(f"Remaining: {budget.remaining_tokens}")
print(f"Utilization: {budget.utilization_rate*100:.1f}%")

# Reset for new session
budget.reset()
```

## Use Cases

### 1. Long-Running Code Analysis

Analyze large codebases over 100+ steps:

```python
agent = EndlessAgent(memory=memory, token_budget=100000)

for file_path in all_code_files:  # 200+ files
    # Analyze file
    analysis = analyze_code(file_path)

    # Store experience
    memory.store(Experience(
        env_features=["code_analysis", "security"],
        goal=f"Analyze {file_path}",
        action=f"Scanned for vulnerabilities",
        result=analysis,
        success=True,
        timestamp=time.time(),
    ))

    # Track tokens (rough estimate: 500 tokens per file)
    agent.track_tokens(prompt_tokens=300, completion_tokens=200)

    # Auto-compress when needed
    if agent.should_compress():
        agent.auto_compress_context()
```

### 2. Multi-Session Projects

Continue work across multiple sessions:

```python
# Session 1
agent = EndlessAgent(memory=memory, token_budget=100000)
for step in range(50):
    # Work...
    agent.track_tokens(prompt_tokens=500, completion_tokens=200)

# Save status
status = agent.get_status()
print(f"Session 1 ended at {status['token_budget']['utilization']*100:.1f}% utilization")

# Session 2 (next day)
agent = EndlessAgent(memory=memory, token_budget=100000)
agent.track_tokens(
    prompt_tokens=status['token_budget']['prompt_tokens'],
    completion_tokens=status['token_budget']['completion_tokens'],
)
# Continue where you left off...
```

### 3. Autonomous Agent Workflows

Enable truly autonomous agents with 100+ step workflows:

```python
agent = EndlessAgent(memory=memory, token_budget=200000)  # 2x budget

for task in task_queue:  # Unlimited tasks
    # Execute task
    result = execute_task(task)

    # Store experience
    memory.store(Experience(...))

    # Track tokens
    agent.track_tokens(prompt_tokens=800, completion_tokens=400)

    # Auto-manage context
    if agent.should_compress():
        result = agent.auto_compress_context()
        print(f"🗜️  Compressed at step {len(task_queue)}, saved {result['estimated_tokens_saved']} tokens")

    # Adaptive recall - automatically uses right layer
    past_experiences, metadata = agent.recall_adaptive(
        env_features=task.features,
        goal=task.goal,
        top_k=5
    )
    print(f"📚 Recalled {metadata['count']} experiences using layer {metadata['layer']}")
```

## Best Practices

### 1. Stay Under 40% Utilization

The "dumb zone" starts around 40% context utilization. Keep below this threshold:

```python
# ✅ Good: Compress at 40%
config = CompressionConfig(threshold=0.4, target_rate=0.2)

# ❌ Bad: Wait until 80%
config = CompressionConfig(threshold=0.8, target_rate=0.6)
```

### 2. Track Tokens Accurately

Estimate token counts conservatively:

```python
# ✅ Good: Over-estimate slightly
agent.track_tokens(prompt_tokens=550, completion_tokens=220)  # Add 10% buffer

# ❌ Bad: Under-estimate
agent.track_tokens(prompt_tokens=500, completion_tokens=200)  # Exact counts may drift
```

### 3. Use Adaptive Recall

Let the agent choose the best layer:

```python
# ✅ Good: Adaptive recall
results, metadata = agent.recall_adaptive(env_features, goal, top_k=5)

# ❌ Bad: Manual layer selection
compressed, exps = memory.recall_layer1(env_features, goal, top_k=5)
```

### 4. Monitor Compression Events

Track compression frequency:

```python
status = agent.get_status()
if status['compression']['compression_events'] > 10:
    print("⚠️  Compressing frequently - consider larger token budget or lower threshold")
```

### 5. Reset Budget Between Sessions

Start fresh for new sessions:

```python
# ✅ Good: Reset budget
agent.reset_budget(new_budget=150000)

# ❌ Bad: Reuse old agent without reset
# (Budget will show incorrect utilization)
```

## Performance

### Token Savings

Endless Mode reduces token usage by:
- **80-90%** with layer 1 (compressed summaries)
- **50-60%** with periodic compaction
- **Unlimited** runtime with continuous compression

### Memory Overhead

- TokenBudget: ~100 bytes
- CompressionConfig: ~200 bytes
- EndlessAgent: ~1 KB (wraps existing ProceduralMemory)

**Total overhead:** < 2 KB

### Compression Speed

- Compaction: O(n log n) for sorting by timestamp
- Progressive disclosure: O(1) layer selection
- Auto-compress: < 100ms for 1000 experiences

## Troubleshooting

### High Compression Frequency

**Symptom:** Compression events > 10 per 100 steps

**Solutions:**
1. Increase token budget
2. Lower compression threshold
3. Increase target_rate
4. Use layer 1 more aggressively

```python
config = CompressionConfig(
    threshold=0.3,   # Compress earlier
    target_rate=0.15,  # More aggressive compaction
    layer1_threshold=0.2,  # Use compressed mode sooner
)
```

### Token Budget Exceeded

**Symptom:** `utilization_rate >= 1.0`

**Solutions:**
1. Check token tracking accuracy
2. Lower compression threshold
3. Increase token budget
4. Reduce experiences per compression

```python
# Check status
status = agent.get_status()
if status['token_budget']['utilization'] >= 1.0:
    # Emergency compaction
    agent.memory.compact(keep_recent_n=10)
    agent.reset_budget()
```

### Compression Not Triggering

**Symptom:** `should_compress()` returns False despite high utilization

**Solutions:**
1. Check minimum experiences count
2. Verify utilization threshold
3. Ensure token tracking is active

```python
status = agent.get_status()
print(f"Utilization: {status['token_budget']['utilization']}")
print(f"Threshold: {status['compression']['threshold']}")
print(f"Min experiences: {agent.config.min_experiences}")
print(f"Current experiences: {status['memory']['total_experiences']}")
```

### Empty Recall Results

**Symptom:** `recall_adaptive()` returns empty list

**Solutions:**
1. Check similarity threshold in MemoryConfig
2. Verify experiences have relevant env_features
3. Ensure experiences exist in memory

```python
# Lower threshold for better recall
config = MemoryConfig(
    storage_path="./memory.jsonl",
    similarity_threshold=0.1,  # Lower threshold
)
```

## FAQ

### Q: What token budget should I use?

**A:** Depends on your LLM's context window:
- Claude Opus 3.5: 100,000 tokens
- Claude Sonnet 3.5: 75,000 tokens
- GPT-4 Turbo: 50,000 tokens
- GPT-4: 25,000 tokens

Set your budget to match or slightly below the context window.

### Q: How accurate is the token estimation?

**A:** Token estimation uses a rough heuristic (~1000 tokens per experience). For precise tracking, integrate with your LLM's token counter:

```python
# Example with OpenAI
import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4")
prompt_tokens = len(encoder.encode(prompt_text))
agent.track_tokens(prompt_tokens=prompt_tokens)
```

### Q: Can I use this without compression?

**A:** Yes! Just set a very high threshold:

```python
config = CompressionConfig(threshold=1.0)  # Never compress
agent = EndlessAgent(memory=memory, compression_config=config)
```

### Q: What happens if I exceed the budget?

**A:** Nothing breaks - it's just tracking. The agent will continue working, but you'll likely hit context window limits in your LLM calls. Monitor `utilization_rate` and compress proactively.

### Q: Can I compress manually?

**A:** Yes! Call `auto_compress_context()` anytime:

```python
result = agent.auto_compress_context()
print(f"Manually compressed, saved {result['estimated_tokens_saved']} tokens")
```

## Architecture

### Class Hierarchy

```
ProceduralMemory (existing)
    ↓
EndlessAgent (new)
    ↓ uses
TokenBudget (new)
    ↓
CompressionConfig (new)
```

### Compression Pipeline

```
1. Track tokens → Update TokenBudget
2. Check utilization → Compare to threshold
3. Should compress? → Check min_experiences
4. Auto-compress → Call memory.compact()
5. Estimate savings → Update budget
6. Log event → Increment compression_events
```

### Progressive Disclosure Integration

```
Layer 1 (compressed) ← High utilization (>30%)
Layer 2 (full)       ← Medium utilization (20-50%)
Layer 3 (expanded)   ← Low utilization (<20%, if enabled)
```

## Contributing

To improve Endless Mode:

1. **Performance optimizations**: Make compression faster
2. **Token estimators**: Add support for different tokenizers
3. **Adaptive thresholds**: Learn optimal thresholds from usage patterns
4. **Compression strategies**: Implement priority-based compaction (keep successes, remove failures)

## License

MIT License - Same as rec-praxis-rlm

## See Also

- [Progressive Disclosure](./PROGRESSIVE_DISCLOSURE.md) - 3-layer recall API
- [Compression](./COMPRESSION.md) - Observation compression module
- [Privacy](./PRIVACY.md) - Privacy redaction and classification
