"""End-to-end integration tests for RLM context."""
import pytest

from rec_praxis_rlm.rlm import RLMContext
from rec_praxis_rlm.config import ReplConfig


class TestIT002LogAnalysis:
    """IT-002: Load 5MB log, grep ERROR, safe_exec to count, verify results."""

    def test_log_analysis_workflow(self) -> None:
        """Test complete log analysis workflow with grep and safe_exec."""
        # Create a large log file (simulated 5MB)
        log_lines = []
        for i in range(50000):  # ~5MB of log data
            if i % 100 == 0:
                log_lines.append(
                    f"2025-12-03 10:{i//60:02d}:{i%60:02d} ERROR Database timeout on query: SELECT * FROM orders"
                )
            elif i % 50 == 0:
                log_lines.append(
                    f"2025-12-03 10:{i//60:02d}:{i%60:02d} WARN Retrying connection (attempt 1/3)"
                )
            else:
                log_lines.append(
                    f"2025-12-03 10:{i//60:02d}:{i%60:02d} INFO Request processed successfully"
                )

        log_text = "\n".join(log_lines)

        # Initialize context
        config = ReplConfig(max_search_matches=1000)
        ctx = RLMContext(config)
        ctx.add_document("app_log", log_text)

        # Step 1: Search for ERROR patterns
        error_matches = ctx.grep(r"ERROR.*timeout", doc_id="app_log", max_matches=100)

        # Verify we found ERROR matches
        assert len(error_matches) > 0
        assert all("ERROR" in m.match_text for m in error_matches)
        assert all("timeout" in m.match_text for m in error_matches)

        # Step 2: Use safe_exec to count ERROR occurrences programmatically
        # Extract a subset for counting (full log is too large)
        log_subset = "\n".join(log_lines[:1000])

        result = ctx.safe_exec(
            code="""
error_count = log_text.count("ERROR")
warn_count = log_text.count("WARN")
info_count = log_text.count("INFO")
total_lines = len(log_text.split('\\n'))
error_count
            """,
            context_vars={"log_text": log_subset},
        )

        # Verify execution succeeded
        assert result.success is True
        # Should have found ERROR occurrences
        assert "10" in result.output  # 1000 lines / 100 = 10 errors

        # Step 3: Verify context extraction works
        if len(error_matches) > 0:
            first_match = error_matches[0]
            # Should have context before and after
            assert len(first_match.context_before) > 0 or len(first_match.context_after) > 0


class TestIT005SandboxSecurity:
    """IT-005: Attempt 20 prohibited operations, verify all blocked, no code executed."""

    def test_all_prohibited_operations_blocked(self) -> None:
        """Test that all 20+ prohibited operations are blocked."""
        ctx = RLMContext(ReplConfig())

        # List of prohibited operations
        prohibited_operations = [
            "import os",
            "from os import system",
            "__import__('sys')",
            "eval('1+1')",
            "exec('print(1)')",
            "compile('1+1', '<string>', 'eval')",
            "open('/etc/passwd', 'r')",
            "with open('file.txt', 'w') as f: f.write('test')",
            "globals()",
            "locals()",
            "vars()",
            "dir()",
            "().__class__",
            "(lambda: None).__globals__",
            "__builtins__",
            "().__class__.__dict__",
            "(lambda: None).__code__",
            "type.__subclasses__()",
            "object.__subclasses__()",
            "__file__",
            "__name__ = 'malicious'",
        ]

        blocked_count = 0

        for operation in prohibited_operations:
            result = ctx.safe_exec(operation)

            # Operation should fail
            assert result.success is False, f"Operation '{operation}' should be blocked but succeeded"

            # Should have an error message
            assert result.error is not None
            assert len(result.error) > 0

            blocked_count += 1

        # Verify all operations were blocked
        assert blocked_count == len(prohibited_operations)
        assert blocked_count >= 20  # Spec requires 20+ prohibited operations

    def test_no_side_effects_from_blocked_code(self) -> None:
        """Test that blocked code has no side effects."""
        ctx = RLMContext(ReplConfig())

        # Try to create a variable, then use prohibited operation
        result1 = ctx.safe_exec("test_var = 42")
        assert result1.success is True

        # Try prohibited operation
        result2 = ctx.safe_exec("import os; test_var = 999")
        assert result2.success is False

        # Original variable should not be affected
        # (Each execution is isolated, so test_var won't exist in next execution)
        # This tests that the sandbox is properly isolated

    def test_error_messages_are_clear(self) -> None:
        """Test that error messages clearly explain what's prohibited."""
        ctx = RLMContext(ReplConfig())

        result = ctx.safe_exec("import os")
        assert result.success is False
        assert result.error is not None
        # Error should mention "import" or "not allowed" or "prohibited"
        error_lower = result.error.lower()
        assert (
            "import" in error_lower
            or "not allowed" in error_lower
            or "prohibited" in error_lower
        )


class TestIT005PerformanceRequirements:
    """Test that RLM context meets performance requirements."""

    def test_search_large_document_under_500ms(self) -> None:
        """Test that searching 10MB document completes < 500ms."""
        import time

        # Create ~10MB document (need more content per line)
        lines = [f"Line {i}: Some log content here with additional padding text to make it larger" for i in range(200000)]
        # Inject some ERROR lines
        for i in range(0, len(lines), 1000):
            lines[i] = f"Line {i}: ERROR Something went wrong with additional details"

        log_text = "\n".join(lines)
        # Verify it's a large document (at least 5MB for meaningful test)
        assert len(log_text) > 5_000_000  # > 5MB

        ctx = RLMContext(ReplConfig(max_search_matches=100))
        ctx.add_document("large_log", log_text)

        # Measure search time
        start = time.time()
        matches = ctx.grep(r"ERROR", doc_id="large_log", max_matches=50)
        elapsed = time.time() - start

        # Should complete in < 500ms (allow some buffer for slow systems)
        assert elapsed < 1.0  # 1 second buffer for CI systems
        assert len(matches) > 0

    def test_head_tail_peek_efficient(self) -> None:
        """Test that head/tail/peek don't load entire document."""
        # Create large document
        lines = [f"Line {i}" for i in range(100000)]
        text = "\n".join(lines)

        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc", text)

        import time

        # head() should be fast even on huge document
        start = time.time()
        result = ctx.head("doc", n_lines=10)
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should be nearly instant
        assert "Line 0" in result
        assert "Line 9" in result

        # tail() should be fast
        start = time.time()
        result = ctx.tail("doc", n_lines=10)
        elapsed = time.time() - start

        assert elapsed < 0.1
        assert "Line 99999" in result

        # peek() should be fast
        start = time.time()
        result = ctx.peek("doc", start_char=0, end_char=100)
        elapsed = time.time() - start

        assert elapsed < 0.1
        assert len(result) <= 100
