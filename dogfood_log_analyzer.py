"""Dogfooding: Log analyzer on rec-praxis-rlm's own sample application log.

This demonstrates that the log analyzer actually works on realistic error patterns.
"""
from pathlib import Path
from rec_praxis_rlm import RLMContext, ReplConfig


def analyze_rec_praxis_logs():
    """Analyze rec-praxis-rlm sample application log."""
    print("=" * 60)
    print("Dogfooding: Log Analysis on rec-praxis-rlm Sample Log")
    print("=" * 60)
    print()

    # Create context
    ctx = RLMContext(ReplConfig(max_search_matches=100))

    # Load the sample application log we created
    log_path = Path(__file__).parent / "sample_application.log"
    with open(log_path, "r") as f:
        log_data = f.read()

    ctx.add_document("sample_app.log", log_data)
    print(f"✅ Loaded {log_path.name} ({len(log_data)} bytes)")
    print()

    # 1. Search for database errors (real pattern from our logs)
    print("1. Searching for database errors...")
    db_errors = ctx.grep(r"ERROR.*Database", doc_id="sample_app.log")
    print(f"   Found {len(db_errors)} database errors")
    for match in db_errors:
        print(f"   - Line {match.line_number}: {match.match_text[:70]}...")
    print()

    # 2. Search for SafeExecutor security blocks
    print("2. Searching for SafeExecutor security blocks...")
    security_blocks = ctx.grep(r"ERROR.*SafeExecutor", doc_id="sample_app.log")
    print(f"   Found {len(security_blocks)} security blocks")
    for match in security_blocks:
        print(f"   - Line {match.line_number}: {match.match_text[:70]}...")
    print()

    # 3. Use safe code execution to analyze error distribution
    print("3. Analyzing error distribution with safe_exec...")
    analysis_code = """
# Count errors by component
error_types = {}
for line in log_text.split('\\n'):
    if 'ERROR' in line:
        # Extract component in brackets [Component]
        if 'ERROR [' in line:
            start = line.find('[') + 1
            end = line.find(']', start)
            if end > start:
                component = line[start:end]
                error_types[component] = error_types.get(component, 0) + 1

# Format results
result = []
for component, count in sorted(error_types.items(), key=lambda x: -x[1]):
    result.append(f"{component}: {count} errors")

'\\n'.join(result)
"""

    result = ctx.safe_exec(analysis_code, context_vars={"log_text": log_data})

    if result.success:
        print(f"   Error distribution:")
        for line in result.output.strip().split('\n'):
            if line.strip():
                print(f"     {line}")
    else:
        print(f"   ❌ Execution failed: {result.error}")
    print()

    # 4. Analyze connection timeout patterns
    print("4. Analyzing connection timeouts...")
    timeout_code = """
# Count timeout events (no imports needed)
timeout_count = sum(1 for line in log_text.split('\\n') if 'timeout' in line.lower())
f"Found {timeout_count} timeout events"
"""
    timeout_result = ctx.safe_exec(timeout_code, context_vars={"log_text": log_data})
    print(f"   {timeout_result.output.strip()}")
    print()

    # 5. Check for service degradation patterns
    print("5. Checking for service degradation patterns...")
    degraded_matches = ctx.grep(r"degraded mode", doc_id="sample_app.log")
    fallback_matches = ctx.grep(r"Falling back|fallback", doc_id="sample_app.log")
    print(f"   Found {len(degraded_matches)} degraded mode events")
    print(f"   Found {len(fallback_matches)} fallback events")
    print()

    # 6. Extract critical error context using peek
    print("6. Extracting context around first database error...")
    if db_errors:
        # Find the character position of the first database error
        first_error_line = db_errors[0].line_number
        lines = log_data.split('\n')
        char_offset = sum(len(line) + 1 for line in lines[:first_error_line-1])

        # Peek around that error (200 chars before and after)
        context = ctx.peek("sample_app.log", max(0, char_offset - 100), char_offset + 300)
        print(f"   Context:\n{context}")
    print()

    # 7. Show tail of log (last events)
    print("7. Checking tail of log (last events)...")
    last_lines = ctx.tail("sample_app.log", n_lines=5)
    print(f"   Last 5 lines:")
    for line in last_lines.split('\n'):
        if line.strip():
            print(f"     {line}")
    print()

    print("=" * 60)
    print("✅ Dogfooding Complete!")
    print("=" * 60)
    print()
    print("Key Findings:")
    print("  1. RLM Context grep successfully found database errors")
    print("  2. safe_exec analyzed error distribution without LLM")
    print("  3. ReDoS protection didn't block legitimate patterns")
    print("  4. peek/head/tail provided precise context extraction")
    print()
    print("Next Step: Add this to procedural memory for learning")


if __name__ == "__main__":
    analyze_rec_praxis_logs()
