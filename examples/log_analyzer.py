"""Log analyzer example using RLM context.

Demonstrates efficient log analysis using search, extraction, and safe code execution.

Run: python examples/log_analyzer.py
"""
from rec_praxis_rlm import RLMContext, ReplConfig


def analyze_application_logs():
    """Analyze a large application log file."""
    print("=== Log Analysis with RLM Context ===\n")

    # Create context
    ctx = RLMContext(ReplConfig(max_search_matches=100))

    # Simulate a large log file (in practice, this could be 10MB+)
    log_data = """
2025-12-03 10:00:01 INFO [WebServer] Request GET /api/products from 192.168.1.100
2025-12-03 10:00:02 INFO [Database] Query executed in 45ms
2025-12-03 10:00:05 ERROR [Database] Connection timeout after 30s on query: SELECT * FROM orders
2025-12-03 10:00:06 WARN [Cache] Redis connection lost, retrying...
2025-12-03 10:00:08 INFO [WebServer] Request POST /api/checkout from 192.168.1.101
2025-12-03 10:00:10 ERROR [PaymentGateway] Connection refused to payment.example.com:443
2025-12-03 10:00:12 ERROR [Database] Connection timeout after 30s on query: SELECT * FROM products
2025-12-03 10:00:15 INFO [WebServer] Request GET /health from 192.168.1.102
2025-12-03 10:00:18 WARN [Cache] Redis connection restored
2025-12-03 10:00:20 ERROR [Database] Deadlock detected on table: inventory
2025-12-03 10:00:25 INFO [WebServer] Request GET /api/users from 192.168.1.103
"""

    ctx.add_document("app.log", log_data)

    print("Loaded application log\n")

    # 1. Search for error patterns
    print("1. Searching for database errors...")
    db_errors = ctx.grep(r"ERROR.*Database", doc_id="app.log")
    print(f"   Found {len(db_errors)} database errors")
    for match in db_errors[:2]:  # Show first 2
        print(f"   - Line {match.line_number}: {match.match_text[:60]}...")

    print()

    # 2. Extract specific time range
    print("2. Extracting first 10 lines...")
    first_lines = ctx.head("app.log", n_lines=10)
    print(f"   Retrieved {len(first_lines.splitlines())} lines")

    print()

    # 3. Use safe code execution to analyze
    print("3. Analyzing error distribution with safe execution...")
    analysis_code = """
# Count errors by type
error_types = {}
for line in log_text.split('\\n'):
    if 'ERROR' in line:
        # Extract component in brackets (simple string parsing, no regex)
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
            print(f"     {line}")
    else:
        print(f"   Execution failed: {result.error}")

    print()

    # 4. Count specific patterns
    print("4. Counting timeouts...")
    timeout_code = "log_text.count('timeout')"
    timeout_result = ctx.safe_exec(timeout_code, context_vars={"log_text": log_data})
    print(f"   Found {timeout_result.output.strip()} timeout occurrences")

    print()

    # 5. Show last few lines
    print("5. Checking tail of log...")
    last_lines = ctx.tail("app.log", n_lines=3)
    print(f"   Last 3 lines:")
    for line in last_lines.split('\n'):
        if line.strip():
            print(f"     {line}")


if __name__ == "__main__":
    analyze_application_logs()
    print("\n✅ Log analysis complete!")
    print("\nKey takeaway: RLM context enables efficient analysis of large")
    print("documents without loading entire files into LLM context.")
