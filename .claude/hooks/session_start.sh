#!/bin/bash
# Session-start hook for rec-praxis-rlm context injection
#
# This hook is called at the start of each Claude Code session.
# It retrieves relevant past experiences and injects them as context.
#
# Environment variables available:
# - WORKING_DIR: Current working directory
# - SESSION_ID: Unique session identifier

# Only inject context if we're in the rec-praxis-rlm project
if [[ ! "$WORKING_DIR" =~ rec-praxis-rlm ]]; then
  exit 0
fi

# Skip if Python is not available
if ! command -v python3 &> /dev/null; then
  exit 0
fi

# Create a Python script to retrieve and format context
python3 << 'PYTHON_SCRIPT'
import os
import sys
from pathlib import Path

# Add rec_praxis_rlm to path
sys.path.insert(0, str(Path(os.environ.get("WORKING_DIR", ".")).resolve()))

try:
    from rec_praxis_rlm import ProceduralMemory, MemoryConfig

    working_dir = os.environ.get("WORKING_DIR", ".")
    memory_path = os.path.join(working_dir, ".claude", "memory.jsonl")

    # Check if memory file exists
    if not os.path.exists(memory_path):
        # No memory yet, provide basic context
        print("\n📚 **REC Praxis RLM Context**")
        print("\nThis is your first session with automatic experience capture enabled.")
        print("Your tool uses and results will be automatically stored in `.claude/memory.jsonl`.")
        print("Future sessions will retrieve relevant past experiences to help you avoid repeating mistakes.\n")
        exit(0)

    # Initialize memory
    config = MemoryConfig(
        storage_path=memory_path,
        embedding_model="",  # Disable embeddings for speed
        similarity_threshold=0.3,  # Lower threshold for session start
    )
    memory = ProceduralMemory(config=config, use_faiss=False)

    # Get memory statistics
    total_experiences = memory.size()

    if total_experiences == 0:
        print("\n📚 **REC Praxis RLM Context**")
        print("\nMemory initialized but no experiences captured yet.\n")
        exit(0)

    # Recall recent successful experiences
    recent_successes = [exp for exp in memory.experiences if exp.success][-5:]

    # Recall recent failures (for learning)
    recent_failures = [exp for exp in memory.experiences if not exp.success][-3:]

    # Format context
    print("\n📚 **REC Praxis RLM Context**")
    print(f"\n**Memory Statistics:**")
    print(f"- Total experiences: {total_experiences}")
    print(f"- Recent successful patterns: {len(recent_successes)}")
    print(f"- Recent failures to avoid: {len(recent_failures)}")

    if recent_successes:
        print("\n**Recent Successful Patterns:**")
        for i, exp in enumerate(recent_successes[-3:], 1):  # Show last 3
            exp_type = exp.experience_type if hasattr(exp, 'experience_type') else 'unknown'
            print(f"{i}. [{exp_type}] {exp.goal[:80]}")
            print(f"   ✓ {exp.result[:100]}")

    if recent_failures:
        print("\n**Recent Failures (Learn from these):**")
        for i, exp in enumerate(recent_failures[-2:], 1):  # Show last 2
            exp_type = exp.experience_type if hasattr(exp, 'experience_type') else 'unknown'
            print(f"{i}. [{exp_type}] {exp.goal[:80]}")
            print(f"   ✗ {exp.result[:100]}")

    print("\n💡 **Tip:** Past experiences are automatically retrieved when relevant to your current task.\n")

except ImportError:
    # rec_praxis_rlm not installed, skip
    pass
except Exception as e:
    # Log error to stderr but don't fail the hook
    print(f"Error retrieving context: {e}", file=sys.stderr)

PYTHON_SCRIPT

exit 0
