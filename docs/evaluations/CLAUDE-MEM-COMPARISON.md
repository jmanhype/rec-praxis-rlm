# claude-mem vs rec-praxis-rlm: Comparative Analysis

**Analysis Date**: 2025-12-07
**claude-mem Repository**: https://github.com/thedotmack/claude-mem
**Purpose**: Identify integration opportunities and architectural insights

---

## Executive Summary

**claude-mem** and **rec-praxis-rlm** solve complementary problems in the AI agent memory space:

| Aspect | claude-mem | rec-praxis-rlm |
|--------|-----------|----------------|
| **Primary Goal** | Extend Claude Code context window via compression | Provide procedural memory for autonomous agents |
| **Integration** | Tightly coupled to Claude Code lifecycle hooks | Standalone library for any Python application |
| **Memory Type** | Observations (compressed summaries of sessions) | Experiences (env features + goal + action + result) |
| **Retrieval** | Semantic search + FTS5 full-text search | Hybrid Jaccard + cosine similarity |
| **Storage** | SQLite + Chroma vector DB | JSONL + optional FAISS index |
| **Context Injection** | Automatic via hooks | Manual via `recall()` API |
| **Token Optimization** | **95% reduction**, ~20x more tool uses | Not a primary goal (focuses on quality) |

**Key Insight**: claude-mem excels at **automatic context compression** for Claude Code, while rec-praxis-rlm excels at **structured procedural learning** for general AI agents.

---

## Architectural Comparison

### 1. Data Model

#### claude-mem: Observation-Centric
```python
Observation {
    session_id: str
    timestamp: datetime
    type: "decision" | "bugfix" | "feature" | "refactor"
    summary: str  # ~500 tokens, AI-compressed
    concept_tags: List[str]
    file_references: List[str]
    privacy_tags: List[str]  # <private>, <claude-mem-context>
}
```

**Strengths**:
- Automatic categorization (decision, bugfix, feature, refactor)
- AI-powered compression (full tool output → 500 tokens)
- Privacy-aware tagging system
- File-centric organization

**Use Case**: "Show me all bugfixes related to authentication in the last week"

#### rec-praxis-rlm: Experience-Centric
```python
Experience {
    env_features: List[str]  # ["chrome", "button_exists", "modal_open"]
    goal: str  # "Click the submit button"
    action: str  # "browser.click('#submit-btn')"
    result: str  # "Success: Form submitted"
    success: bool
    timestamp: float
    embedding: List[float]  # 384-dim goal embedding
    cost: Optional[float]
    metadata: Dict
}
```

**Strengths**:
- Explicit success/failure tracking
- Environmental context for situational matching
- Pre-computed embeddings for fast similarity
- Cost tracking for optimization

**Use Case**: "Given similar button layout, what action successfully clicked submit before?"

### 2. Retrieval Strategy

#### claude-mem: Progressive Disclosure (3 Layers)
```
Layer 1: Indexed Observations
├─ Shows observation summaries (~500 tokens each)
├─ Token cost displayed per result
└─ User decides which to expand

Layer 2: Detailed Narrative
├─ Full session narrative
├─ Code snippets and decisions
└─ Expanded context on demand

Layer 3: Full Source
├─ Complete tool outputs
├─ Full transcripts
└─ Deep investigation
```

**Token Budget**: Start with 10 observations × 500 tokens = **5,000 tokens**, expand selectively.

#### rec-praxis-rlm: Hybrid Similarity Ranking
```python
def recall(env_features: List[str], goal: str, top_k: int = 6):
    # 1. Compute goal embedding
    goal_emb = embed(goal)

    # 2. Hybrid score = env_weight * Jaccard(env) + goal_weight * Cosine(goal)
    for exp in experiences:
        env_sim = jaccard(exp.env_features, env_features)
        goal_sim = cosine(exp.embedding, goal_emb)
        score = 0.6 * env_sim + 0.4 * goal_sim

    # 3. Return top-k by score
    return sorted_experiences[:top_k]
```

**Token Budget**: Return top-6 experiences (configurable), user gets all or none.

**Comparison**:
- claude-mem: **User-controlled expansion** → Better token efficiency
- rec-praxis-rlm: **Automatic top-k** → Simpler API, less control

### 3. Context Injection

#### claude-mem: Automatic Hook-Based
```yaml
# .claude/settings.json hooks
hooks:
  session_start:
    command: python mem_worker.py inject_context
    # Automatically injects recent observations

  post_tool_use:
    command: python mem_worker.py capture_observation
    # Captures tool output, compresses to observation
```

**Strengths**:
- Zero user intervention
- Seamless Claude Code integration
- Automatic observation creation

**Weaknesses**:
- Tightly coupled to Claude Code
- 60-90s latency per tool use (Endless Mode)
- Requires background worker service

#### rec-praxis-rlm: Explicit API
```python
# Manual integration
memory = ProceduralMemory()

# Store experience
memory.store(Experience(
    env_features=["chrome", "login_page"],
    goal="Log in user",
    action="fill_credentials()",
    result="Success",
    success=True,
    timestamp=time.time()
))

# Recall similar experiences
similar = memory.recall(
    env_features=["chrome", "login_page"],
    goal="Log in user",
    top_k=6
)
```

**Strengths**:
- Explicit control over what's stored
- No background services required
- Works in any Python application

**Weaknesses**:
- Requires manual integration
- No automatic compression
- User must decide when to store/recall

### 4. Storage Backend

#### claude-mem: SQLite + Chroma
```sql
-- SQLite schema
CREATE VIRTUAL TABLE observations USING fts5(
    session_id,
    summary,
    concept_tags,
    file_refs
);

-- Chroma vector DB for embeddings
chroma_client.create_collection("observations")
```

**Strengths**:
- FTS5 for fast full-text search
- Chroma for hybrid semantic search
- Rich query capabilities (concept, file, type, timeline)

**Storage Size**: ~500 tokens per observation × 1,000 observations = **~500KB** of text

#### rec-praxis-rlm: JSONL + FAISS
```jsonl
{"__version__": "1.0"}
{"env_features": [...], "goal": "...", "action": "...", "result": "...", "success": true, "timestamp": 1234.56, "embedding": [0.1, ...]}
{"env_features": [...], "goal": "...", "action": "...", "result": "...", "success": true, "timestamp": 1234.57, "embedding": [0.2, ...]}
```

**Strengths**:
- Human-readable JSONL format
- Crash-safe append-only writes
- FAISS for fast similarity search (optional)
- No external dependencies (FAISS optional)

**Storage Size**: ~1-2KB per experience × 1,000 experiences = **~1-2MB** (full tool outputs)

**Comparison**:
- claude-mem: **Compressed storage** (~500KB for 1K obs), requires SQLite + Chroma
- rec-praxis-rlm: **Full-fidelity storage** (~1-2MB for 1K exp), simpler dependencies

---

## Key Innovations from claude-mem

### 1. ✨ Observation Compression (95% token reduction)

**How it works**:
```
Tool Output (10,000 tokens)
    ↓ AI-powered summarization
Observation Summary (500 tokens)
    ↓ 95% reduction
Context Injection Cost: 500 tokens instead of 10,000
```

**Applicability to rec-praxis-rlm**: **HIGH**

We could add observation compression to reduce memory recall cost:

```python
class CompressedExperience(BaseModel):
    """Compressed version of Experience for token efficiency."""
    env_features: List[str]
    goal: str
    action_summary: str  # Compressed from full action
    result_summary: str  # Compressed from full result
    success: bool
    timestamp: float
    embedding: List[float]

    @classmethod
    def from_experience(cls, exp: Experience) -> "CompressedExperience":
        """Compress experience using LLM summarization."""
        # Use DSPy to compress action + result to ~100 tokens
        action_summary = summarize(exp.action, max_tokens=50)
        result_summary = summarize(exp.result, max_tokens=50)

        return cls(
            env_features=exp.env_features,
            goal=exp.goal,
            action_summary=action_summary,
            result_summary=result_summary,
            success=exp.success,
            timestamp=exp.timestamp,
            embedding=exp.embedding
        )
```

**Benefits**:
- Reduce recall token cost by 80-90%
- Enable returning top-20 instead of top-6
- Better context window utilization

**Drawbacks**:
- Adds LLM cost for compression
- Potential information loss
- Requires LLM API access

### 2. ✨ Progressive Disclosure (Layered Retrieval)

**How it works**:
```
User Query: "Show authentication bugs"
    ↓
Layer 1: 10 observation summaries (5,000 tokens)
    │
    ├─ User selects observation #3
    ↓
Layer 2: Full narrative for #3 (2,000 tokens)
    │
    ├─ User requests source code
    ↓
Layer 3: Complete tool outputs (10,000 tokens)
```

**Applicability to rec-praxis-rlm**: **MEDIUM**

We could implement a similar pattern:

```python
class LayeredRecall:
    def recall_layer1(self, env_features: List[str], goal: str, top_k: int = 20):
        """Return compressed summaries (50 tokens each)."""
        experiences = self.memory.recall(env_features, goal, top_k)
        return [CompressedExperience.from_experience(exp) for exp in experiences]

    def recall_layer2(self, experience_id: str):
        """Return full experience details."""
        return self.memory.get_by_timestamp(experience_id)

    def recall_layer3(self, experience_id: str):
        """Return full tool outputs and context."""
        exp = self.memory.get_by_timestamp(experience_id)
        return {
            "experience": exp,
            "related_facts": self.fact_store.query(exp.goal),
            "similar_experiences": self.memory.recall(exp.env_features, exp.goal, top_k=5)
        }
```

**Benefits**:
- User controls token cost
- Better for large memory stores (10K+ experiences)
- Aligns with human information-seeking behavior

**Drawbacks**:
- Requires interactive UI (not just API)
- More complex API surface
- May not fit all use cases (autonomous agents need automatic)

### 3. ✨ Privacy Tagging System

**How it works**:
```xml
<!-- User marks sensitive content -->
<private>
API_KEY=sk-proj-abc123
DATABASE_PASSWORD=secret123
</private>

<!-- System marks recursive context -->
<claude-mem-context>
Previous observation summary: ...
</claude-mem-context>
```

**Edge processing** ensures private content never reaches storage.

**Applicability to rec-praxis-rlm**: **HIGH**

We could add privacy filtering:

```python
class Experience(BaseModel):
    # ... existing fields ...
    privacy_level: Literal["public", "private", "pii"] = "public"
    redacted_fields: List[str] = Field(default_factory=list)

    def sanitize(self) -> "Experience":
        """Remove sensitive data before storage."""
        sanitized = self.model_copy()

        if self.privacy_level == "private":
            # Redact action and result
            sanitized.action = "[REDACTED]"
            sanitized.result = "[REDACTED]"

        elif self.privacy_level == "pii":
            # Redact specific fields
            for field in self.redacted_fields:
                if hasattr(sanitized, field):
                    setattr(sanitized, field, "[REDACTED]")

        return sanitized

# Usage
memory.store(Experience(
    env_features=["api_call"],
    goal="Authenticate user",
    action="requests.post('/auth', json={'api_key': 'sk-abc123'})",  # Sensitive!
    result="200 OK",
    success=True,
    timestamp=time.time(),
    privacy_level="private"  # Redacts action/result
))
```

**Benefits**:
- Prevents accidental leakage of secrets
- Compliance with data protection regulations
- User control over sensitive data

**Drawbacks**:
- Reduces utility of redacted experiences
- Requires user to manually tag privacy

### 4. ✨ Concept Tagging and File References

**How it works**:
```python
Observation {
    summary: "Fixed authentication bug in user login",
    concept_tags: ["authentication", "security", "login"],  # AI-extracted
    file_references: ["src/auth/login.py", "tests/test_auth.py"],
    type: "bugfix"
}
```

**Search capabilities**:
- "Show all authentication-related observations"
- "What changed in login.py recently?"
- "Show all bugfixes from this week"

**Applicability to rec-praxis-rlm**: **MEDIUM-HIGH**

We could extend Experience with semantic tags:

```python
class Experience(BaseModel):
    # ... existing fields ...
    concept_tags: List[str] = Field(default_factory=list)
    file_references: List[str] = Field(default_factory=list)
    experience_type: Literal["learn", "recover", "optimize", "explore"] = "learn"

    @classmethod
    def with_auto_tags(cls, goal: str, action: str, result: str, **kwargs) -> "Experience":
        """Auto-extract concept tags and file references."""
        # Use DSPy to extract concepts
        concepts = extract_concepts(goal + " " + action + " " + result)

        # Extract file paths with regex
        file_pattern = r'["\']([^"\']+\.(py|js|java|go|rs))["\']'
        files = re.findall(file_pattern, action + " " + result)

        return cls(
            goal=goal,
            action=action,
            result=result,
            concept_tags=concepts,
            file_references=[f[0] for f in files],
            **kwargs
        )
```

**Benefits**:
- Rich semantic search ("Show all database optimization experiences")
- File-based filtering ("What did we learn about auth.py?")
- Better organization for large memory stores

**Drawbacks**:
- Requires LLM for concept extraction
- Tag quality depends on LLM capability
- May introduce noise

### 5. ✨ Endless Mode (Beta) - Context Window Extension

**How it works**:
```
Traditional Agent Loop:
    User → Agent → Tool → Output (append to context)
    ↓
    Context grows: O(N²) - each tool use adds to cumulative context
    ↓
    After ~50 tool uses: Context window exhausted

Endless Mode:
    User → Agent → Tool → Output → Compress to Observation (500 tokens)
    ↓
    Context grows: O(N) - fixed 500 tokens per tool use
    ↓
    After ~1000 tool uses: Still within context window ✓
```

**Performance**:
- **95% token reduction** per tool use
- **~20x more tool uses** before exhaustion
- **60-90s latency** per tool use (compression overhead)

**Applicability to rec-praxis-rlm**: **LOW-MEDIUM**

This is highly specific to Claude Code's context window problem. However, we could offer a similar pattern for autonomous agents:

```python
class MemoryEfficientAgent:
    def __init__(self):
        self.memory = ProceduralMemory()
        self.compressor = ObservationCompressor()  # LLM-based

    def execute_task(self, task: str):
        """Execute task with automatic memory compression."""
        # Recall relevant experiences (compressed)
        context = self.memory.recall_layer1(
            env_features=self.get_env(),
            goal=task,
            top_k=20  # 20 × 50 tokens = 1,000 tokens
        )

        # Execute task
        result = self.agent.run(task, context)

        # Compress result to observation
        observation = self.compressor.compress(result)  # 10,000 → 500 tokens

        # Store compressed observation
        self.memory.store(CompressedExperience.from_dict(observation))

        return result
```

**Benefits**:
- Enables long-running agents (100+ steps)
- Reduces LLM API costs (fewer tokens)
- Better scalability

**Drawbacks**:
- Compression latency (60-90s per step)
- Information loss in compression
- Requires powerful LLM for compression

---

## Integration Opportunities

### Option 1: rec-praxis-rlm as Backend for claude-mem

**Proposal**: Use rec-praxis-rlm's procedural memory engine inside claude-mem.

**Benefits**:
- claude-mem gets FAISS acceleration
- rec-praxis-rlm gets tested in production (Claude Code)
- Shared development effort

**Implementation**:
```python
# In claude-mem's observation storage layer
from rec_praxis_rlm import ProceduralMemory, Experience

class ObservationStore:
    def __init__(self):
        self.memory = ProceduralMemory(config=MemoryConfig(
            storage_path="./claude_mem_observations.jsonl",
            top_k=20,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        ))

    def add_observation(self, obs: Observation):
        """Store observation as experience."""
        self.memory.store(Experience(
            env_features=obs.concept_tags + obs.file_references,
            goal=obs.summary,
            action=obs.type,  # "bugfix", "feature", etc.
            result=obs.summary,  # Use summary as result
            success=True,
            timestamp=obs.timestamp.timestamp(),
            metadata={
                "session_id": obs.session_id,
                "privacy_tags": obs.privacy_tags
            }
        ))

    def search_observations(self, query: str, concept_tags: List[str]):
        """Search using rec-praxis-rlm's hybrid similarity."""
        results = self.memory.recall(
            env_features=concept_tags,
            goal=query,
            top_k=20
        )
        return [self.to_observation(exp) for exp in results]
```

**Challenges**:
- Impedance mismatch (Observation vs Experience data models)
- claude-mem already has SQLite + Chroma integration
- May not justify migration effort

### Option 2: Add claude-mem's Compression to rec-praxis-rlm

**Proposal**: Add observation compression as optional feature in rec-praxis-rlm.

**Benefits**:
- Reduce token cost for agents
- Enable "Endless Mode" for autonomous agents
- Backward compatible (optional feature)

**Implementation**:
```python
# New module: rec_praxis_rlm/compression.py
class ObservationCompressor:
    def __init__(self, lm_model: str = "openai/gpt-4o-mini"):
        self.lm = dspy.LM(lm_model)

    def compress(self, experience: Experience, max_tokens: int = 500) -> CompressedExperience:
        """Compress experience to observation summary."""
        prompt = f"""
        Compress this experience into a {max_tokens}-token summary:

        Environment: {', '.join(experience.env_features)}
        Goal: {experience.goal}
        Action: {experience.action}
        Result: {experience.result}
        Success: {experience.success}

        Extract:
        1. Key concepts (tags)
        2. Files involved
        3. Outcome summary
        4. Lessons learned
        """

        response = self.lm(prompt)

        return CompressedExperience(
            env_features=experience.env_features,
            goal=experience.goal,
            action_summary=response.action_summary,
            result_summary=response.result_summary,
            success=experience.success,
            timestamp=experience.timestamp,
            embedding=experience.embedding
        )

# Usage
memory = ProceduralMemory()
compressor = ObservationCompressor()

# Store full experience
experience = Experience(...)
memory.store(experience)

# Also store compressed version for token efficiency
compressed = compressor.compress(experience)
memory.store_compressed(compressed)
```

**Challenges**:
- Adds LLM dependency for compression
- Increases storage cost (store both full + compressed)
- Complexity in API

### Option 3: Standalone Integration via Hooks

**Proposal**: rec-praxis-rlm provides hooks for Claude Code, complementing claude-mem.

**Benefits**:
- rec-praxis-rlm focuses on procedural memory
- claude-mem focuses on observation compression
- Users can use both (complementary)

**Implementation**:
```json
// .claude/settings.json
{
  "hooks": {
    "session_start": {
      "command": "python -m rec_praxis_rlm.hooks session_start"
    },
    "post_tool_use": {
      "command": "python -m rec_praxis_rlm.hooks post_tool_use"
    }
  }
}
```

```python
# rec_praxis_rlm/hooks.py
def post_tool_use_hook(tool_name: str, tool_input: str, tool_output: str):
    """Capture tool use as experience."""
    memory = ProceduralMemory()

    experience = Experience(
        env_features=[tool_name],
        goal=tool_input,
        action=f"execute_{tool_name}",
        result=tool_output,
        success=True,  # Infer from output
        timestamp=time.time()
    )

    memory.store(experience)
```

**Challenges**:
- Overlap with claude-mem functionality
- User confusion (which to use?)
- May not be necessary

---

## Recommendations for rec-praxis-rlm

### High Priority (v0.10.0)

1. **Add Observation Compression (Optional)**
   - New module: `rec_praxis_rlm/compression.py`
   - Compress experiences to ~500 tokens using LLM
   - Opt-in feature (doesn't break existing API)
   - Reduces recall token cost by 80-90%

2. **Add Privacy Tagging**
   - Extend Experience with `privacy_level` field
   - Auto-redaction for "private" and "pii" levels
   - Prevent accidental storage of secrets
   - Aligns with data protection best practices

3. **Add Concept Tagging**
   - Extract semantic tags from goal/action/result
   - Enable rich semantic search
   - File reference extraction
   - Improves organization for large memory stores

### Medium Priority (v0.11.0)

4. **Implement Progressive Disclosure**
   - Add `recall_layer1()` (compressed summaries)
   - Add `recall_layer2()` (full experience)
   - Add `recall_layer3()` (related context)
   - User-controlled token budget

5. **Add Experience Type Classification**
   - Auto-classify experiences: learn, recover, optimize, explore
   - Enable type-based filtering
   - Improve recall relevance

6. **Build Web UI for Memory Viewer**
   - Visualize experiences timeline
   - Search and filter interface
   - Concept tag cloud
   - File reference graph

### Low Priority (v0.12.0+)

7. **Endless Mode for Agents**
   - Automatic compression pipeline
   - Token budget tracking
   - Long-running agent support
   - Requires compression feature first

---

## Conclusion

**claude-mem** provides excellent inspiration for improving rec-praxis-rlm's token efficiency and user experience. The key innovations are:

1. ✅ **Observation Compression** - Highly applicable, major value
2. ✅ **Privacy Tagging** - Highly applicable, important for production
3. ✅ **Concept Tagging** - Highly applicable, improves search
4. ⚠️ **Progressive Disclosure** - Medium applicability, requires UI
5. ⚠️ **Endless Mode** - Low-medium applicability, niche use case

**Next Steps**:
1. Implement observation compression as opt-in feature (v0.10.0)
2. Add privacy tagging system (v0.10.0)
3. Add concept extraction (v0.10.0)
4. Evaluate progressive disclosure for web UI (v0.11.0)

These enhancements will make rec-praxis-rlm more token-efficient, privacy-aware, and production-ready while maintaining its core strength: structured procedural learning for autonomous agents.

---

**References**:
- claude-mem: https://github.com/thedotmack/claude-mem
- rec-praxis-rlm: https://github.com/jmanhype/rec-praxis-rlm
- FAISS: https://github.com/facebookresearch/faiss
- Chroma: https://www.trychroma.com/
