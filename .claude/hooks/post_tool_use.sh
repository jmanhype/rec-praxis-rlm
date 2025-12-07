#!/bin/bash
# Post-tool-use hook for automatic experience capture in rec-praxis-rlm
#
# This hook is called after each tool use in Claude Code.
# It captures the tool call as an experience in procedural memory.
#
# Environment variables available:
# - TOOL_NAME: Name of the tool that was used
# - TOOL_INPUT: JSON string of tool input
# - TOOL_OUTPUT: Tool output/result
# - TOOL_SUCCESS: "true" or "false"
# - WORKING_DIR: Current working directory

# Only capture if we're in the rec-praxis-rlm project
if [[ ! "$WORKING_DIR" =~ rec-praxis-rlm ]]; then
  exit 0
fi

# Skip if Python is not available
if ! command -v python3 &> /dev/null; then
  exit 0
fi

# Create a Python script to capture the experience
python3 << 'PYTHON_SCRIPT'
import os
import json
import sys
import time
from pathlib import Path

# Add rec_praxis_rlm to path
sys.path.insert(0, str(Path(os.environ.get("WORKING_DIR", ".")).resolve()))

try:
    from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig

    # Get environment variables
    tool_name = os.environ.get("TOOL_NAME", "unknown")
    tool_input = os.environ.get("TOOL_INPUT", "{}")
    tool_output = os.environ.get("TOOL_OUTPUT", "")
    tool_success = os.environ.get("TOOL_SUCCESS", "false") == "true"
    working_dir = os.environ.get("WORKING_DIR", ".")

    # Parse tool input
    try:
        tool_input_obj = json.loads(tool_input)
        # Extract meaningful description from input
        if "description" in tool_input_obj:
            goal = tool_input_obj["description"]
        elif "command" in tool_input_obj:
            goal = f"Execute command: {tool_input_obj['command'][:100]}"
        elif "file_path" in tool_input_obj:
            goal = f"Operate on file: {tool_input_obj['file_path']}"
        else:
            goal = f"Use {tool_name} tool"
    except:
        goal = f"Use {tool_name} tool"

    # Initialize memory (append to .claude/memory.jsonl)
    config = MemoryConfig(
        storage_path=os.path.join(working_dir, ".claude", "memory.jsonl"),
        embedding_model="",  # Disable embeddings for speed
    )
    memory = ProceduralMemory(config=config, use_faiss=False)

    # Create experience
    experience = Experience(
        env_features=["claude_code", tool_name, "automated_capture"],
        goal=goal,
        action=f"{tool_name}: {tool_input[:500]}",  # Truncate long inputs
        result=tool_output[:1000] if tool_output else "No output",  # Truncate long outputs
        success=tool_success,
        timestamp=time.time(),
        metadata={
            "tool_name": tool_name,
            "working_dir": working_dir,
            "capture_source": "claude_code_hook",
        }
    )

    # Store experience
    memory.store(experience)

except ImportError:
    # rec_praxis_rlm not installed, skip
    pass
except Exception as e:
    # Log error to stderr but don't fail the hook
    print(f"Error capturing experience: {e}", file=sys.stderr)

PYTHON_SCRIPT

exit 0
