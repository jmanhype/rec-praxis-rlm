#!/usr/bin/env python3
"""Test DSPy autonomous planning with Groq API."""

import os
import time
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig
from rec_praxis_rlm.dspy_agent import PraxisRLMPlanner
from rec_praxis_rlm.config import PlannerConfig
from rec_praxis_rlm.rlm import RLMContext

print("=" * 60)
print("Testing DSPy Autonomous Planning with Groq")
print("=" * 60)

# Initialize memory with some experiences
memory = ProceduralMemory(MemoryConfig(storage_path=":memory:"))
memory.store(Experience(
    env_features=["python", "debugging"],
    goal="find error in code",
    action="Used print statements to trace execution",
    result="Found off-by-one error in loop",
    success=True,
    timestamp=time.time()
))
memory.store(Experience(
    env_features=["python", "optimization"],
    goal="speed up slow function",
    action="Replaced list comprehension with generator",
    result="Reduced memory usage by 80%",
    success=True,
    timestamp=time.time()
))

# Initialize RLM context with sample document
context = RLMContext()
context.add_document("sample_log", """
2025-12-06 10:00:00 INFO Starting application
2025-12-06 10:00:01 INFO Database connected
2025-12-06 10:00:05 ERROR Connection timeout on port 5432
2025-12-06 10:00:06 INFO Retrying connection...
2025-12-06 10:00:10 ERROR Max retries exceeded
""")

# Test with Groq API - llama-3.3-70b-versatile (latest model)
print("\n1. Testing with Groq (llama-3.3-70b-versatile)...")
try:
    # Set Groq API key in environment for DSPy
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

    planner = PraxisRLMPlanner(
        memory=memory,
        config=PlannerConfig(
            lm_model="groq/llama-3.3-70b-versatile",  # Latest Groq model
            temperature=0.0,
            max_iters=3,  # Limit iterations for testing
            enable_mlflow_tracing=False  # Disable for quick test
        )
    )
    planner.add_context(context, "server_logs")

    result = planner.plan(
        goal="Analyze the error logs and suggest a fix",
        env_features=["python", "debugging", "database"]
    )

    print(f"✅ Groq planning succeeded!")
    print(f"Result: {result[:200]}...")  # First 200 chars

except Exception as e:
    print(f"❌ Groq test failed: {e}")
    import traceback
    traceback.print_exc()

# Test with OpenRouter
print("\n2. Testing with OpenRouter (meta-llama/llama-3.2-3b-instruct:free)...")
try:
    # Set OpenRouter API key in environment for DSPy
    os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")

    planner_or = PraxisRLMPlanner(
        memory=memory,
        config=PlannerConfig(
            lm_model="openrouter/meta-llama/llama-3.2-3b-instruct:free",
            temperature=0.0,
            max_iters=3,
            enable_mlflow_tracing=False
        )
    )
    planner_or.add_context(context, "server_logs")

    result = planner_or.plan(
        goal="Summarize the main issue in the logs",
        env_features=["debugging"]
    )

    print(f"✅ OpenRouter planning succeeded!")
    print(f"Result: {result[:200]}...")

except Exception as e:
    print(f"❌ OpenRouter test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing complete!")
print("=" * 60)
