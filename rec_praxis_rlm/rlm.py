"""RLM context for programmatic document inspection and safe code execution.

This module provides a facade pattern (RLMContext) that coordinates three focused components:
- DocumentStore: Manages document storage and retrieval
- DocumentSearcher: Handles search operations (grep, peek, head, tail)
- CodeExecutor: Manages safe code execution in sandboxed environment

This separation of concerns (SRP) improves testability and maintainability while keeping
the public API unchanged for backward compatibility.
"""

import asyncio
import hashlib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any

from pydantic import BaseModel, Field, field_validator

from rec_praxis_rlm.config import ReplConfig
from rec_praxis_rlm.exceptions import DocumentNotFoundError, SearchError
from rec_praxis_rlm.sandbox import SafeExecutor
from rec_praxis_rlm.telemetry import emit_event

logger = logging.getLogger(__name__)


class SearchMatch(BaseModel):
    """A search match result from grep operation.

    Attributes:
        doc_id: Document identifier
        line_number: Line number where match occurred (1-indexed)
        match_text: The matched text
        context_before: Context before the match
        context_after: Context after the match
        start_char: Character offset where match starts
        end_char: Character offset where match ends
    """

    doc_id: str = Field(..., description="Document identifier")
    line_number: int = Field(..., ge=1, description="Line number (1-indexed)")
    match_text: str = Field(..., description="The matched text")
    context_before: str = Field(default="", description="Context before match")
    context_after: str = Field(default="", description="Context after match")
    start_char: int = Field(..., ge=0, description="Character offset where match starts")
    end_char: int = Field(..., ge=0, description="Character offset where match ends")

    @field_validator("end_char")
    @classmethod
    def end_char_must_be_greater_than_start(cls, v: int, info: Any) -> int:
        """Validate that end_char > start_char."""
        if "start_char" in info.data and v <= info.data["start_char"]:
            raise ValueError("end_char must be greater than start_char")
        return v


class ExecutionResult(BaseModel):
    """Result from safe code execution.

    Attributes:
        success: Whether execution succeeded
        output: Captured output from execution
        error: Error message if execution failed
        execution_time_seconds: Time taken to execute code
        code_hash: SHA-256 hash of executed code for audit trail
    """

    success: bool = Field(..., description="Whether execution succeeded")
    output: str = Field(..., description="Captured output")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_seconds: float = Field(..., ge=0.0, description="Execution time in seconds")
    code_hash: str = Field(..., description="SHA-256 hash of code")


class _Document:
    """Internal document representation with indices."""

    def __init__(self, doc_id: str, text: str):
        self.doc_id = doc_id
        self.text = text
        self.lines = text.split("\n")

        # Build line start character offsets
        self.line_starts: list[int] = [0]
        pos = 0
        for line in self.lines[:-1]:
            pos += len(line) + 1  # +1 for newline
            self.line_starts.append(pos)


# ============================================================================
# Component Classes (SRP Refactoring)
# ============================================================================


class DocumentStore:
    """Manages document storage and retrieval.

    Focused responsibility: Store and retrieve documents with metadata indexing.
    """

    def __init__(self) -> None:
        """Initialize empty document store."""
        self._documents: dict[str, _Document] = {}

    def add(self, doc_id: str, text: str) -> None:
        """Add a document to the store.

        Args:
            doc_id: Document identifier
            text: Document text content

        Raises:
            ValueError: If doc_id already exists
        """
        if doc_id in self._documents:
            raise ValueError(f"Document '{doc_id}' already exists")

        self._documents[doc_id] = _Document(doc_id, text)
        logger.info(
            f"Added document '{doc_id}' ({len(text)} chars, {len(text.split(chr(10)))} lines)"
        )

    def remove(self, doc_id: str) -> None:
        """Remove a document from store.

        Args:
            doc_id: Document identifier

        Raises:
            DocumentNotFoundError: If doc_id not found
        """
        if doc_id not in self._documents:
            raise DocumentNotFoundError(f"Document '{doc_id}' not found")

        del self._documents[doc_id]
        logger.info(f"Removed document '{doc_id}'")

    def get(self, doc_id: str) -> _Document:
        """Get a document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document object

        Raises:
            DocumentNotFoundError: If doc_id not found
        """
        if doc_id not in self._documents:
            raise DocumentNotFoundError(f"Document '{doc_id}' not found")
        return self._documents[doc_id]

    def get_all(self) -> list[_Document]:
        """Get all documents.

        Returns:
            List of all documents
        """
        return list(self._documents.values())

    def has(self, doc_id: str) -> bool:
        """Check if document exists.

        Args:
            doc_id: Document identifier

        Returns:
            True if document exists
        """
        return doc_id in self._documents


class DocumentSearcher:
    """Handles document search operations with ReDoS protection.

    Focused responsibility: Search documents using various methods (grep, peek, head, tail).
    """

    def __init__(self, document_store: DocumentStore, config: ReplConfig) -> None:
        """Initialize document searcher.

        Args:
            document_store: Document store to search
            config: REPL configuration
        """
        self._store = document_store
        self._config = config

    def _validate_regex_safety(self, pattern: str) -> None:
        """Validate regex pattern safety to prevent ReDoS attacks.

        Args:
            pattern: Regular expression pattern to validate

        Raises:
            SearchError: If pattern is potentially unsafe
        """
        # Length check: prevent extremely long patterns
        MAX_PATTERN_LENGTH = 500
        if len(pattern) > MAX_PATTERN_LENGTH:
            raise SearchError(
                f"Regex pattern too long ({len(pattern)} chars, max {MAX_PATTERN_LENGTH})"
            )

        # Detect common ReDoS patterns
        # Pattern 1: Nested quantifiers like (a+)+ or (a*)*
        if re.search(r"\([^)]*[+*]\)\s*[+*]", pattern):
            raise SearchError(
                "Potentially dangerous regex: nested quantifiers detected (e.g., (a+)+)"
            )

        # Pattern 2: Overlapping alternations like (a|a)* or (ab|a)*
        if re.search(r"\([^)]*\|[^)]*\)\s*[*+]", pattern):
            # This is a heuristic - some patterns are safe, but we're conservative
            logger.warning(f"Potentially slow regex pattern with alternation+quantifier: {pattern}")

        # Pattern 3: Excessive backtracking with .+ or .*
        consecutive_wildcards = pattern.count(".*") + pattern.count(".+")
        if consecutive_wildcards > 3:
            msg = (
                f"Potentially dangerous regex: "
                f"too many wildcard quantifiers ({consecutive_wildcards})"
            )
            raise SearchError(msg)

    def grep(
        self,
        pattern: str,
        doc_id: Optional[str] = None,
        max_matches: Optional[int] = None,
    ) -> list[SearchMatch]:
        """Search documents for pattern using regex.

        Args:
            pattern: Regular expression pattern
            doc_id: Optional document ID to search (searches all if None)
            max_matches: Maximum matches to return (defaults to config.max_search_matches)

        Returns:
            List of search matches

        Raises:
            SearchError: If regex compilation or search fails or pattern is unsafe
        """
        if max_matches is None:
            max_matches = self._config.max_search_matches

        # ReDoS protection: validate pattern complexity
        self._validate_regex_safety(pattern)

        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise SearchError(f"Invalid regex pattern: {e}")

        matches: list[SearchMatch] = []
        docs_to_search = [self._store.get(doc_id)] if doc_id else self._store.get_all()

        for doc in docs_to_search:
            for line_num, line in enumerate(doc.lines, start=1):
                if len(matches) >= max_matches:
                    break

                match = regex.search(line)
                if match:
                    # Compute character offsets
                    line_start = doc.line_starts[line_num - 1]
                    start_char = line_start + match.start()
                    end_char = line_start + match.end()

                    # Extract context
                    context_chars = self._config.search_context_chars
                    context_before = doc.text[max(0, start_char - context_chars) : start_char]
                    context_after = doc.text[end_char : end_char + context_chars]

                    matches.append(
                        SearchMatch(
                            doc_id=doc.doc_id,
                            line_number=line_num,
                            match_text=match.group(0),
                            context_before=context_before,
                            context_after=context_after,
                            start_char=start_char,
                            end_char=end_char,
                        )
                    )

            if len(matches) >= max_matches:
                break

        # Emit telemetry
        emit_event(
            "context.search",
            {
                "pattern": pattern,
                "doc_id": doc_id,
                "matches_found": len(matches),
            },
        )

        return matches

    def peek(self, doc_id: str, start_char: int, end_char: int) -> str:
        """Extract a character range from document.

        Args:
            doc_id: Document identifier
            start_char: Start character offset
            end_char: End character offset

        Returns:
            Extracted text

        Raises:
            DocumentNotFoundError: If doc_id not found
        """
        doc = self._store.get(doc_id)

        # Clamp to document bounds
        start_char = max(0, start_char)
        end_char = min(len(doc.text), end_char)

        return doc.text[start_char:end_char]

    def head(self, doc_id: str, n_lines: int = 10) -> str:
        """Get first N lines of document.

        Args:
            doc_id: Document identifier
            n_lines: Number of lines to return

        Returns:
            First N lines as string

        Raises:
            DocumentNotFoundError: If doc_id not found
        """
        doc = self._store.get(doc_id)
        lines = doc.lines[:n_lines]
        return "\n".join(lines)

    def tail(self, doc_id: str, n_lines: int = 10) -> str:
        """Get last N lines of document.

        Args:
            doc_id: Document identifier
            n_lines: Number of lines to return

        Returns:
            Last N lines as string

        Raises:
            DocumentNotFoundError: If doc_id not found
        """
        doc = self._store.get(doc_id)
        lines = doc.lines[-n_lines:] if n_lines > 0 else []
        return "\n".join(lines)


class CodeExecutor:
    """Manages safe code execution in sandboxed environment.

    Focused responsibility: Execute code safely with telemetry and error handling.
    """

    def __init__(self, config: ReplConfig) -> None:
        """Initialize code executor.

        Args:
            config: REPL configuration
        """
        self._config = config
        self._executor = SafeExecutor(config)
        self._async_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="executor-async"
        )

    def execute(self, code: str, context_vars: Optional[dict[str, Any]] = None) -> ExecutionResult:
        """Execute code safely in sandboxed environment.

        Args:
            code: Python code to execute
            context_vars: Variables to inject into execution context

        Returns:
            Execution result with output, error, and metadata

        Raises:
            ExecutionError: If code validation fails
        """
        if context_vars is None:
            context_vars = {}

        # Compute code hash for audit trail
        code_hash = hashlib.sha256(code.encode()).hexdigest()

        # Execute in sandbox
        start_time = time.time()
        try:
            result = self._executor.execute(code, context_vars)
            execution_time = time.time() - start_time

            # Emit telemetry
            emit_event(
                "context.exec",
                {
                    "code_hash": code_hash,
                    "success": result.success,
                    "execution_time": execution_time,
                },
            )

            return ExecutionResult(
                success=result.success,
                output=result.output,
                error=result.error,
                execution_time_seconds=execution_time,
                code_hash=code_hash,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Code execution failed: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time_seconds=execution_time,
                code_hash=code_hash,
            )

    async def aexecute(
        self, code: str, context_vars: Optional[dict[str, Any]] = None
    ) -> ExecutionResult:
        """Async version of execute for non-blocking execution.

        Uses ThreadPoolExecutor to run the synchronous execute() in a thread pool,
        preventing blocking of the event loop.

        Args:
            code: Python code to execute
            context_vars: Variables to inject

        Returns:
            Execution result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._async_executor, self.execute, code, context_vars)


# ============================================================================
# Facade Pattern (Backward Compatibility)
# ============================================================================


class RLMContext:
    """RLM context for programmatic document inspection and safe code execution.

    This class acts as a facade, coordinating three focused components:
    - DocumentStore: Manages documents
    - DocumentSearcher: Handles search operations
    - CodeExecutor: Executes code safely

    The public API remains unchanged for backward compatibility.
    """

    def __init__(self, config: ReplConfig = ReplConfig()) -> None:
        """Initialize RLM context.

        Args:
            config: REPL configuration
        """
        self.config = config

        # Initialize focused components (SRP refactoring)
        self.documents = DocumentStore()
        self.searcher = DocumentSearcher(self.documents, config)
        self.executor = CodeExecutor(config)

    # ========================================================================
    # Document Management (delegates to DocumentStore)
    # ========================================================================

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the context.

        Args:
            doc_id: Document identifier
            text: Document text content

        Raises:
            ValueError: If doc_id already exists
        """
        self.documents.add(doc_id, text)

    def remove_document(self, doc_id: str) -> None:
        """Remove a document from context.

        Args:
            doc_id: Document identifier

        Raises:
            DocumentNotFoundError: If doc_id not found
        """
        self.documents.remove(doc_id)

    # ========================================================================
    # Search Operations (delegates to DocumentSearcher)
    # ========================================================================

    def grep(
        self,
        pattern: str,
        doc_id: Optional[str] = None,
        max_matches: Optional[int] = None,
    ) -> list[SearchMatch]:
        """Search documents for pattern using regex.

        Args:
            pattern: Regular expression pattern
            doc_id: Optional document ID to search (searches all if None)
            max_matches: Maximum matches to return (defaults to config.max_search_matches)

        Returns:
            List of search matches

        Raises:
            SearchError: If regex compilation or search fails or pattern is unsafe
        """
        return self.searcher.grep(pattern, doc_id, max_matches)

    def peek(self, doc_id: str, start_char: int, end_char: int) -> str:
        """Extract a character range from document.

        Args:
            doc_id: Document identifier
            start_char: Start character offset
            end_char: End character offset

        Returns:
            Extracted text

        Raises:
            DocumentNotFoundError: If doc_id not found
        """
        return self.searcher.peek(doc_id, start_char, end_char)

    def head(self, doc_id: str, n_lines: int = 10) -> str:
        """Get first N lines of document.

        Args:
            doc_id: Document identifier
            n_lines: Number of lines to return

        Returns:
            First N lines as string

        Raises:
            DocumentNotFoundError: If doc_id not found
        """
        return self.searcher.head(doc_id, n_lines)

    def tail(self, doc_id: str, n_lines: int = 10) -> str:
        """Get last N lines of document.

        Args:
            doc_id: Document identifier
            n_lines: Number of lines to return

        Returns:
            Last N lines as string

        Raises:
            DocumentNotFoundError: If doc_id not found
        """
        return self.searcher.tail(doc_id, n_lines)

    # ========================================================================
    # Code Execution (delegates to CodeExecutor)
    # ========================================================================

    def safe_exec(
        self, code: str, context_vars: Optional[dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute code safely in sandboxed environment.

        Args:
            code: Python code to execute
            context_vars: Variables to inject into execution context

        Returns:
            Execution result with output, error, and metadata

        Raises:
            ExecutionError: If code validation fails
        """
        return self.executor.execute(code, context_vars)

    async def asafe_exec(
        self, code: str, context_vars: Optional[dict[str, Any]] = None
    ) -> ExecutionResult:
        """Async version of safe_exec for non-blocking execution.

        Args:
            code: Python code to execute
            context_vars: Variables to inject

        Returns:
            Execution result
        """
        return await self.executor.aexecute(code, context_vars)
