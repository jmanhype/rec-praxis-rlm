"""Safe code execution sandbox with validation and restricted environment."""

import ast
import io
import logging
from typing import Any, Optional
from contextlib import redirect_stdout, redirect_stderr

from pydantic import BaseModel

from rec_praxis_rlm.config import ReplConfig
from rec_praxis_rlm.exceptions import ExecutionError

logger = logging.getLogger(__name__)


class _SandboxResult(BaseModel):
    """Internal sandbox execution result."""

    success: bool
    output: str
    error: Optional[str] = None


# Prohibited AST node types and patterns
PROHIBITED_NODES = {
    ast.Import,  # import statements
    ast.ImportFrom,  # from ... import statements
}

PROHIBITED_NAMES = {
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "globals",
    "locals",
    "vars",
    "dir",
    "__builtins__",
    "__file__",
    "__name__",
    "type",
    "object",
}

PROHIBITED_ATTRIBUTES = {
    "__class__",
    "__globals__",
    "__dict__",
    "__code__",
    "__subclasses__",
}


class _CodeValidator(ast.NodeVisitor):
    """AST visitor to validate code for prohibited patterns."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """Block import statements."""
        self.errors.append("Import statements are not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Block from...import statements."""
        self.errors.append("From-import statements are not allowed")

    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for prohibited functions."""
        # Check for prohibited function names
        if isinstance(node.func, ast.Name):
            if node.func.id in PROHIBITED_NAMES:
                self.errors.append(f"Function '{node.func.id}' is not allowed")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check attribute access for prohibited attributes."""
        if node.attr in PROHIBITED_ATTRIBUTES:
            self.errors.append(f"Attribute '{node.attr}' is not allowed")

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Check name usage for prohibited names."""
        if node.id in PROHIBITED_NAMES:
            # These names are always prohibited (both read and write)
            self.errors.append(f"Use of '{node.id}' is not allowed")

        self.generic_visit(node)


def _validate_code(code: str) -> None:
    """Validate code for prohibited patterns.

    Args:
        code: Python code to validate

    Raises:
        ExecutionError: If code contains prohibited patterns
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ExecutionError(f"Syntax error: {e}")

    validator = _CodeValidator()
    validator.visit(tree)

    if validator.errors:
        error_msg = "; ".join(validator.errors)
        raise ExecutionError(f"Code validation failed: {error_msg}")


class SafeExecutor:
    """Safe code executor with sandboxed environment.

    Executes Python code in a restricted namespace with prohibited
    operations blocked via AST validation and restricted builtins.
    """

    def __init__(self, config: ReplConfig) -> None:
        """Initialize safe executor.

        Args:
            config: REPL configuration
        """
        self.config = config

        # Build restricted builtins dictionary
        # Note: __builtins__ can be either a dict or module depending on context
        builtins_dict = (
            __builtins__
            if isinstance(__builtins__, dict)  # type: ignore[has-type]
            else __builtins__.__dict__  # type: ignore[attr-defined]
        )
        self._safe_builtins = {
            name: builtins_dict[name] for name in config.allowed_builtins if name in builtins_dict
        }

        # Add essential builtins that are always safe
        self._safe_builtins.update(
            {
                "True": True,
                "False": False,
                "None": None,
                "print": print,
                "range": range,
            }
        )

    def execute(self, code: str, context_vars: Optional[dict[str, Any]] = None) -> _SandboxResult:
        """Execute code in sandboxed environment.

        Args:
            code: Python code to execute
            context_vars: Variables to inject into execution context

        Returns:
            Execution result with output and error information
        """
        if context_vars is None:
            context_vars = {}

        # Validate code
        try:
            _validate_code(code)
        except ExecutionError as e:
            logger.warning(f"Code validation failed: {e}")
            return _SandboxResult(success=False, output="", error=str(e))

        # Build restricted namespace
        namespace: dict[str, Any] = {
            "__builtins__": self._safe_builtins,
        }
        namespace.update(context_vars)

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Execute code with output redirection
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Try to evaluate as expression first (for single expressions)
                try:
                    result = eval(compile(code, "<sandbox>", "eval"), namespace)
                    if result is not None:
                        print(result)
                except SyntaxError:
                    # Not a simple expression, execute as statements
                    compiled = compile(code, "<sandbox>", "exec")
                    exec(compiled, namespace)

                    # Try to get the value of the last expression if it's a simple name
                    # This handles cases like:
                    # total = 0
                    # for i in range(5): total += i
                    # total  <- this should print the value
                    code_lines = code.strip().split("\n")
                    last_line = code_lines[-1].strip()

                    if last_line and not any(
                        last_line.startswith(kw)
                        for kw in ["for", "while", "if", "def", "class", "import", "from"]
                    ):  # pragma: no branch
                        try:
                            last_value = eval(compile(last_line, "<sandbox>", "eval"), namespace)
                            if last_value is not None:
                                print(last_value)
                        except Exception:  # noqa: S110
                            # Last line wasn't an expression, that's fine
                            pass

            # Get captured output
            output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            if stderr_output:
                output += "\n" + stderr_output

            # Limit output size
            if len(output) > self.config.max_output_chars:
                output = output[: self.config.max_output_chars] + "\n... (output truncated)"

            return _SandboxResult(success=True, output=output, error=None)

        except Exception as e:
            logger.warning(f"Code execution failed: {e}")
            error_msg = f"{type(e).__name__}: {str(e)}"
            return _SandboxResult(success=False, output="", error=error_msg)
        finally:
            stdout_capture.close()
            stderr_capture.close()
