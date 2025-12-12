"""Safe code execution sandbox with validation and restricted environment."""

import ast
import io
import logging
import multiprocessing as mp
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Optional

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


def _execute_code_worker(
    code: str,
    context_vars: dict[str, Any],
    allowed_builtins: list[str],
    max_output_chars: int,
) -> dict[str, Any]:
    """Execute validated code in a restricted namespace and capture output.

    This function is top-level so it can be used as a multiprocessing target.
    """
    builtins_dict = (
        __builtins__  # type: ignore[has-type]
        if isinstance(__builtins__, dict)
        else __builtins__.__dict__  # type: ignore[attr-defined]
    )
    safe_builtins = {name: builtins_dict[name] for name in allowed_builtins if name in builtins_dict}
    safe_builtins.update(
        {
            "True": True,
            "False": False,
            "None": None,
            "print": print,
            "range": range,
        }
    )

    namespace: dict[str, Any] = {"__builtins__": safe_builtins}
    namespace.update(context_vars)

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                result = eval(compile(code, "<sandbox>", "eval"), namespace)
                if result is not None:
                    print(result)
            except SyntaxError:
                compiled = compile(code, "<sandbox>", "exec")
                exec(compiled, namespace)

                code_lines = code.strip().split("\n")
                last_line = code_lines[-1].strip() if code_lines else ""
                if last_line and not any(
                    last_line.startswith(kw)
                    for kw in ["for", "while", "if", "def", "class", "import", "from"]
                ):
                    try:
                        last_value = eval(compile(last_line, "<sandbox>", "eval"), namespace)
                        if last_value is not None:
                            print(last_value)
                    except Exception:
                        pass

        output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        if stderr_output:
            output += "\n" + stderr_output

        if len(output) > max_output_chars:
            output = output[:max_output_chars] + "\n... (output truncated)"

        return {"success": True, "output": output, "error": None}
    except Exception as e:
        return {"success": False, "output": "", "error": f"{type(e).__name__}: {str(e)}"}
    finally:
        stdout_capture.close()
        stderr_capture.close()


def _subprocess_target(
    conn: Any,
    code: str,
    context_vars: dict[str, Any],
    allowed_builtins: list[str],
    max_output_chars: int,
) -> None:
    """Multiprocessing target to run sandboxed code and send back result."""
    try:
        result = _execute_code_worker(code, context_vars, allowed_builtins, max_output_chars)
        conn.send(result)
    except Exception as e:
        conn.send({"success": False, "output": "", "error": f"{type(e).__name__}: {str(e)}"})
    finally:
        conn.close()


class SafeExecutor:
    """Safe code executor with sandboxed environment.

    Executes Python code in a restricted namespace with prohibited
    operations blocked via AST validation and restricted builtins.
    This is designed for trusted helper snippets, not adversarial sandboxing.
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

        # Fast path: run in-process if sandboxing disabled.
        if not self.config.enable_sandbox:
            result = _execute_code_worker(
                code, context_vars, self.config.allowed_builtins, self.config.max_output_chars
            )
            return _SandboxResult(**result)

        timeout = self.config.execution_timeout_seconds
        if timeout <= 0:
            result = _execute_code_worker(
                code, context_vars, self.config.allowed_builtins, self.config.max_output_chars
            )
            return _SandboxResult(**result)

        parent_conn = None
        child_conn = None
        try:
            # Run in a subprocess so we can enforce timeouts safely.
            start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
            ctx = mp.get_context(start_method)
            parent_conn, child_conn = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=_subprocess_target,
                args=(
                    child_conn,
                    code,
                    context_vars,
                    self.config.allowed_builtins,
                    self.config.max_output_chars,
                ),
            )
            proc.start()
            child_conn.close()

            proc.join(timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join()
                return _SandboxResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {timeout}s",
                )

            if parent_conn.poll():
                result = parent_conn.recv()
            else:
                result = {"success": False, "output": "", "error": "Execution failed with no result"}

            return _SandboxResult(**result)
        except Exception as e:
            logger.warning(
                f"Subprocess sandbox failed ({type(e).__name__}: {e}); "
                "falling back to in-process execution."
            )
            result = _execute_code_worker(
                code, context_vars, self.config.allowed_builtins, self.config.max_output_chars
            )
            return _SandboxResult(**result)
        finally:
            if parent_conn is not None:
                try:
                    parent_conn.close()
                except Exception:
                    pass
