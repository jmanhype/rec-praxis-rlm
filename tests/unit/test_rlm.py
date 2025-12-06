"""Unit tests for RLM context module."""
import pytest
from pydantic import ValidationError

from rec_praxis_rlm.rlm import SearchMatch, ExecutionResult, RLMContext
from rec_praxis_rlm.config import ReplConfig


class TestSearchMatch:
    """Tests for SearchMatch Pydantic model."""

    def test_valid_search_match(self) -> None:
        """Test creating a valid search match."""
        match = SearchMatch(
            doc_id="test_doc",
            line_number=10,
            match_text="ERROR: Database timeout",
            context_before="2025-12-03 10:15:23 INFO Server started",
            context_after="2025-12-03 10:15:25 WARN Retrying connection",
            start_char=1000,
            end_char=1025,
        )
        assert match.doc_id == "test_doc"
        assert match.line_number == 10
        assert match.match_text == "ERROR: Database timeout"
        assert match.context_before == "2025-12-03 10:15:23 INFO Server started"
        assert match.context_after == "2025-12-03 10:15:25 WARN Retrying connection"
        assert match.start_char == 1000
        assert match.end_char == 1025

    def test_line_number_must_be_positive(self) -> None:
        """Test that line_number must be >= 1."""
        # Valid: line 1
        match = SearchMatch(
            doc_id="test",
            line_number=1,
            match_text="test",
            context_before="",
            context_after="",
            start_char=0,
            end_char=10,
        )
        assert match.line_number == 1

        # Invalid: line 0
        with pytest.raises(ValidationError):
            SearchMatch(
                doc_id="test",
                line_number=0,
                match_text="test",
                context_before="",
                context_after="",
                start_char=0,
                end_char=10,
            )

    def test_start_char_less_than_end_char(self) -> None:
        """Test that start_char < end_char."""
        # Valid: start < end
        match = SearchMatch(
            doc_id="test",
            line_number=1,
            match_text="test",
            context_before="",
            context_after="",
            start_char=10,
            end_char=20,
        )
        assert match.start_char < match.end_char

        # Invalid: start >= end
        with pytest.raises(ValidationError):
            SearchMatch(
                doc_id="test",
                line_number=1,
                match_text="test",
                context_before="",
                context_after="",
                start_char=20,
                end_char=10,
            )

    def test_optional_context_fields(self) -> None:
        """Test that context_before and context_after can be empty."""
        match = SearchMatch(
            doc_id="test",
            line_number=1,
            match_text="test",
            context_before="",
            context_after="",
            start_char=0,
            end_char=10,
        )
        assert match.context_before == ""
        assert match.context_after == ""


class TestExecutionResult:
    """Tests for ExecutionResult Pydantic model."""

    def test_successful_execution(self) -> None:
        """Test execution result for successful code."""
        result = ExecutionResult(
            success=True,
            output="42",
            error=None,
            execution_time_seconds=0.123,
            code_hash="abc123def456",
        )
        assert result.success is True
        assert result.output == "42"
        assert result.error is None
        assert result.execution_time_seconds == 0.123
        assert result.code_hash == "abc123def456"

    def test_failed_execution(self) -> None:
        """Test execution result for failed code."""
        result = ExecutionResult(
            success=False,
            output="",
            error="ZeroDivisionError: division by zero",
            execution_time_seconds=0.05,
            code_hash="xyz789",
        )
        assert result.success is False
        assert result.output == ""
        assert result.error == "ZeroDivisionError: division by zero"

    def test_execution_time_must_be_non_negative(self) -> None:
        """Test that execution_time_seconds must be >= 0."""
        # Valid: 0 seconds
        result = ExecutionResult(
            success=True,
            output="test",
            error=None,
            execution_time_seconds=0.0,
            code_hash="test",
        )
        assert result.execution_time_seconds == 0.0

        # Invalid: negative time
        with pytest.raises(ValidationError):
            ExecutionResult(
                success=True,
                output="test",
                error=None,
                execution_time_seconds=-0.1,
                code_hash="test",
            )

    def test_code_hash_required(self) -> None:
        """Test that code_hash is required for audit trail."""
        result = ExecutionResult(
            success=True,
            output="test",
            error=None,
            execution_time_seconds=0.1,
            code_hash="sha256:abcd1234",
        )
        assert result.code_hash == "sha256:abcd1234"


class TestRLMContextDocumentManagement:
    """Tests for RLMContext document management."""

    def test_add_document(self) -> None:
        """Test adding a document to context."""
        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc1", "Line 1\nLine 2\nLine 3")

        # Document should be stored
        assert ctx.documents.has("doc1")

    def test_remove_document(self) -> None:
        """Test removing a document from context."""
        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc1", "test content")
        ctx.remove_document("doc1")

        # Document should be removed
        assert not ctx.documents.has("doc1")

    def test_remove_nonexistent_document_raises_error(self) -> None:
        """Test removing nonexistent document raises DocumentNotFoundError."""
        from rec_praxis_rlm.exceptions import DocumentNotFoundError
        ctx = RLMContext(ReplConfig())

        with pytest.raises(DocumentNotFoundError, match="not found"):
            ctx.remove_document("nonexistent")

    def test_duplicate_doc_id_rejection(self) -> None:
        """Test that duplicate doc_id raises error."""
        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc1", "content 1")

        # Adding same doc_id should raise error
        with pytest.raises(ValueError, match="already exists"):
            ctx.add_document("doc1", "content 2")

    def test_document_indexing(self) -> None:
        """Test that documents are indexed by line and char offsets."""
        ctx = RLMContext(ReplConfig())
        text = "Line 1\nLine 2\nLine 3"
        ctx.add_document("doc1", text)

        # Should have line and char indices computed
        doc = ctx.documents.get("doc1")
        assert hasattr(doc, "line_starts")


class TestRLMContextGrep:
    """Tests for RLMContext.grep() method."""

    def test_grep_regex_pattern(self) -> None:
        """Test grep with regex pattern."""
        ctx = RLMContext(ReplConfig())
        log_text = """2025-12-03 10:15:23 INFO Server started
2025-12-03 10:16:12 ERROR Database timeout
2025-12-03 10:16:20 ERROR Database timeout
2025-12-03 10:17:00 INFO User logout"""
        ctx.add_document("log", log_text)

        matches = ctx.grep(r"ERROR.*timeout", doc_id="log")
        assert len(matches) == 2
        assert all("ERROR" in m.match_text for m in matches)

    def test_grep_case_sensitivity(self) -> None:
        """Test grep case sensitivity."""
        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc", "Error\nerror\nERROR")

        # Case-sensitive by default
        matches = ctx.grep(r"ERROR", doc_id="doc")
        assert len(matches) == 1

    def test_grep_max_matches_limit(self) -> None:
        """Test that grep respects max_matches limit."""
        ctx = RLMContext(ReplConfig(max_search_matches=2))
        ctx.add_document("doc", "test\ntest\ntest\ntest\ntest")

        matches = ctx.grep(r"test", doc_id="doc", max_matches=2)
        assert len(matches) == 2

    def test_grep_context_chars(self) -> None:
        """Test that grep includes context before/after match."""
        ctx = RLMContext(ReplConfig(search_context_chars=10))
        ctx.add_document("doc", "prefix ERROR suffix more text")

        matches = ctx.grep(r"ERROR", doc_id="doc")
        assert len(matches) == 1
        # Should have context before and after
        assert len(matches[0].context_before) > 0
        assert len(matches[0].context_after) > 0

    def test_grep_invalid_regex_raises_search_error(self) -> None:
        """Test grep with invalid regex pattern raises SearchError."""
        from rec_praxis_rlm.exceptions import SearchError
        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc", "test content")

        with pytest.raises(SearchError, match="Invalid regex"):
            ctx.grep(r"[invalid(regex", doc_id="doc")

    def test_grep_nonexistent_doc_raises_error(self) -> None:
        """Test grep on nonexistent document raises DocumentNotFoundError."""
        from rec_praxis_rlm.exceptions import DocumentNotFoundError
        ctx = RLMContext(ReplConfig())

        with pytest.raises(DocumentNotFoundError, match="not found"):
            ctx.grep(r"test", doc_id="nonexistent")

    def test_grep_redos_protection_pattern_too_long(self) -> None:
        """Test grep rejects extremely long regex patterns."""
        from rec_praxis_rlm.exceptions import SearchError
        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc", "test content")

        # Pattern longer than 500 chars
        long_pattern = "a" * 501

        with pytest.raises(SearchError, match="too long"):
            ctx.grep(long_pattern, doc_id="doc")

    def test_grep_redos_protection_nested_quantifiers(self) -> None:
        """Test grep rejects nested quantifiers that cause ReDoS."""
        from rec_praxis_rlm.exceptions import SearchError
        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc", "test content")

        # Classic ReDoS pattern: (a+)+
        with pytest.raises(SearchError, match="nested quantifiers"):
            ctx.grep(r"(a+)+", doc_id="doc")

        # Another variant: (a*)*
        with pytest.raises(SearchError, match="nested quantifiers"):
            ctx.grep(r"(a*)*", doc_id="doc")

    def test_grep_redos_protection_excessive_wildcards(self) -> None:
        """Test grep rejects excessive wildcard quantifiers."""
        from rec_praxis_rlm.exceptions import SearchError
        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc", "test content")

        # More than 3 wildcard quantifiers
        with pytest.raises(SearchError, match="too many wildcard quantifiers"):
            ctx.grep(r".*.*.*.*", doc_id="doc")

    def test_grep_allows_safe_patterns(self) -> None:
        """Test grep allows safe regex patterns."""
        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc", "The quick brown fox jumps")

        # These should all work fine
        ctx.grep(r"\w+", doc_id="doc")  # Simple word match
        ctx.grep(r"^The", doc_id="doc")  # Anchor
        ctx.grep(r"[a-z]+", doc_id="doc")  # Character class
        ctx.grep(r"(quick|brown)", doc_id="doc")  # Simple alternation

    def test_grep_warns_on_alternation_with_quantifier(self) -> None:
        """Test grep logs warning for alternation+quantifier patterns."""
        import logging
        from unittest.mock import patch

        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc", "ababa test content")

        # Pattern with alternation+quantifier: (ab|a)+
        with patch.object(logging.getLogger('rec_praxis_rlm.rlm'), 'warning') as mock_warning:
            ctx.grep(r"(ab|a)+", doc_id="doc")
            # Should have logged a warning
            mock_warning.assert_called_once()
            assert "alternation" in mock_warning.call_args[0][0].lower()


class TestRLMContextPeek:
    """Tests for RLMContext.peek() method."""

    def test_peek_extracts_char_range(self) -> None:
        """Test peek extracts correct character range."""
        ctx = RLMContext(ReplConfig())
        text = "0123456789ABCDEFGHIJ"
        ctx.add_document("doc", text)

        # Extract chars 5-10
        result = ctx.peek("doc", start_char=5, end_char=10)
        assert result == "56789"

    def test_peek_out_of_bounds_handling(self) -> None:
        """Test peek handles out-of-bounds ranges."""
        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc", "0123456789")

        # End beyond document length
        result = ctx.peek("doc", start_char=5, end_char=1000)
        # Should return from start to end of document
        assert result == "56789"

    def test_peek_doc_not_found_error(self) -> None:
        """Test peek raises error for missing document."""
        ctx = RLMContext(ReplConfig())

        with pytest.raises(Exception):  # DocumentNotFoundError
            ctx.peek("nonexistent", start_char=0, end_char=10)


class TestRLMContextHeadTail:
    """Tests for RLMContext.head() and tail() methods."""

    def test_head_first_n_lines(self) -> None:
        """Test head returns first N lines."""
        ctx = RLMContext(ReplConfig())
        text = "\n".join([f"Line {i}" for i in range(1, 11)])
        ctx.add_document("doc", text)

        result = ctx.head("doc", n_lines=3)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == "Line 1"
        assert lines[2] == "Line 3"

    def test_tail_last_n_lines(self) -> None:
        """Test tail returns last N lines."""
        ctx = RLMContext(ReplConfig())
        text = "\n".join([f"Line {i}" for i in range(1, 11)])
        ctx.add_document("doc", text)

        result = ctx.tail("doc", n_lines=3)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert lines[0] == "Line 8"
        assert lines[2] == "Line 10"

    def test_head_n_lines_greater_than_document(self) -> None:
        """Test head when n_lines > document length."""
        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc", "Line 1\nLine 2")

        result = ctx.head("doc", n_lines=10)
        lines = result.strip().split("\n")
        # Should return all lines (2)
        assert len(lines) == 2

    def test_head_nonexistent_doc_raises_error(self) -> None:
        """Test head on nonexistent document raises DocumentNotFoundError."""
        from rec_praxis_rlm.exceptions import DocumentNotFoundError
        ctx = RLMContext(ReplConfig())

        with pytest.raises(DocumentNotFoundError, match="not found"):
            ctx.head("nonexistent")

    def test_tail_nonexistent_doc_raises_error(self) -> None:
        """Test tail on nonexistent document raises DocumentNotFoundError."""
        from rec_praxis_rlm.exceptions import DocumentNotFoundError
        ctx = RLMContext(ReplConfig())

        with pytest.raises(DocumentNotFoundError, match="not found"):
            ctx.tail("nonexistent")

    def test_tail_with_zero_lines(self) -> None:
        """Test tail with n_lines=0 returns empty string."""
        ctx = RLMContext(ReplConfig())
        ctx.add_document("doc", "Line 1\nLine 2\nLine 3")

        result = ctx.tail("doc", n_lines=0)
        assert result == ""


class TestRLMContextSafeExec:
    """Tests for RLMContext.safe_exec() method (basic tests, security in test_sandbox.py)."""

    def test_safe_exec_simple_code(self) -> None:
        """Test safe_exec with simple allowed code."""
        ctx = RLMContext(ReplConfig())

        result = ctx.safe_exec("sum([1, 2, 3, 4, 5])")
        assert result.success is True
        assert "15" in result.output

    def test_safe_exec_with_context_vars(self) -> None:
        """Test safe_exec with injected context variables."""
        ctx = RLMContext(ReplConfig())

        result = ctx.safe_exec(
            "x + y",
            context_vars={"x": 10, "y": 20}
        )
        assert result.success is True
        assert "30" in result.output

    def test_safe_exec_captures_output(self) -> None:
        """Test that safe_exec captures print output."""
        ctx = RLMContext(ReplConfig())

        result = ctx.safe_exec("print('hello world')")
        assert result.success is True
        assert "hello world" in result.output

    def test_safe_exec_execution_time_recorded(self) -> None:
        """Test that execution time is recorded."""
        ctx = RLMContext(ReplConfig())

        result = ctx.safe_exec("sum(range(100))")
        assert result.execution_time_seconds >= 0.0

    def test_safe_exec_code_hash_computed(self) -> None:
        """Test that code hash is computed for audit trail."""
        ctx = RLMContext(ReplConfig())

        result = ctx.safe_exec("1 + 1")
        assert result.code_hash is not None
        assert len(result.code_hash) > 0

    def test_safe_exec_exception_handling(self) -> None:
        """Test that safe_exec handles exceptions during execution."""
        from unittest.mock import Mock, patch
        ctx = RLMContext(ReplConfig())

        # Mock executor.execute to raise an exception
        with patch.object(ctx.executor._executor, 'execute', side_effect=RuntimeError("Mock error")):
            result = ctx.safe_exec("1 + 1")
            assert result.success is False
            assert "Mock error" in result.error
            assert result.execution_time_seconds >= 0.0

    def test_asafe_exec_uses_threadpool(self) -> None:
        """Test that asafe_exec runs in thread pool without blocking event loop."""
        import asyncio
        ctx = RLMContext(ReplConfig())

        async def test_async() -> None:
            result = await ctx.asafe_exec("2 + 3")
            assert result.success is True
            assert "5" in result.output

        asyncio.run(test_async())

    def test_asafe_exec_concurrent_execution(self) -> None:
        """Test that multiple asafe_exec() calls can run concurrently."""
        import asyncio
        ctx = RLMContext(ReplConfig())

        async def run_concurrent_execs():
            # Run 5 concurrent executions
            tasks = [
                ctx.asafe_exec(f"{i} * 2")
                for i in range(1, 6)
            ]
            results = await asyncio.gather(*tasks)
            return results

        # Execute concurrent calls
        results = asyncio.run(run_concurrent_execs())

        # All 5 should complete successfully
        assert len(results) == 5
        assert all(r.success for r in results)
        # Verify outputs
        outputs = [r.output.strip() for r in results]
        assert "2" in outputs[0]
        assert "10" in outputs[4]

    def test_asafe_exec_passes_context_vars_correctly(self) -> None:
        """Test that asafe_exec correctly passes context variables."""
        import asyncio
        ctx = RLMContext(ReplConfig())

        async def test_with_vars():
            result = await ctx.asafe_exec(
                "x + y * z",
                context_vars={"x": 10, "y": 5, "z": 3}
            )
            return result

        result = asyncio.run(test_with_vars())
        assert result.success is True
        assert "25" in result.output  # 10 + 5 * 3 = 10 + 15 = 25

    def test_asafe_exec_handles_exceptions(self) -> None:
        """Test that asafe_exec handles exceptions in async context."""
        import asyncio
        from unittest.mock import patch
        ctx = RLMContext(ReplConfig())

        async def test_exception():
            with patch.object(ctx.executor._executor, 'execute', side_effect=RuntimeError("Async error")):
                result = await ctx.asafe_exec("1 + 1")
                return result

        result = asyncio.run(test_exception())
        assert result.success is False
        assert "Async error" in result.error
