"""Enhanced error messages with actionable guidance for CLI tools."""

import sys
from typing import Optional


class CLIError(Exception):
    """Base class for CLI errors with enhanced messaging."""

    def __init__(self, message: str, hint: Optional[str] = None, docs_url: Optional[str] = None):
        self.message = message
        self.hint = hint
        self.docs_url = docs_url
        super().__init__(message)

    def print_error(self) -> None:
        """Print formatted error message to stderr."""
        print(f"\n❌ Error: {self.message}", file=sys.stderr)
        if self.hint:
            print(f"💡 Hint: {self.hint}", file=sys.stderr)
        if self.docs_url:
            print(f"📖 Docs: {self.docs_url}", file=sys.stderr)
        print("", file=sys.stderr)


class APIKeyError(CLIError):
    """Error related to missing or invalid API keys."""

    def __init__(self, provider: str):
        super().__init__(
            message=f"{provider} API key not found",
            hint=f"Set {provider.upper()}_API_KEY environment variable",
            docs_url="https://github.com/jmanhype/rec-praxis-rlm#configuration"
        )


class DependencyError(CLIError):
    """Error related to missing dependencies."""

    def __init__(self, package: str, extra: Optional[str] = None):
        install_cmd = f"pip install rec-praxis-rlm[{extra}]" if extra else f"pip install {package}"
        super().__init__(
            message=f"Required package '{package}' not found",
            hint=f"Install with: {install_cmd}",
            docs_url="https://github.com/jmanhype/rec-praxis-rlm#installation"
        )


class FileNotFoundError(CLIError):
    """Error related to missing files."""

    def __init__(self, file_path: str, expected_location: Optional[str] = None):
        hint_msg = f"Expected at: {expected_location}" if expected_location else "Check file path and try again"
        super().__init__(
            message=f"File not found: {file_path}",
            hint=hint_msg,
            docs_url="https://github.com/jmanhype/rec-praxis-rlm#usage"
        )


class ConfigurationError(CLIError):
    """Error related to invalid configuration."""

    def __init__(self, config_issue: str, suggestion: str):
        super().__init__(
            message=f"Invalid configuration: {config_issue}",
            hint=suggestion,
            docs_url="https://github.com/jmanhype/rec-praxis-rlm#configuration"
        )


def format_import_error(import_error: ImportError, module_name: str) -> str:
    """Format ImportError with actionable guidance.

    Args:
        import_error: The original ImportError
        module_name: Name of the module that failed to import

    Returns:
        Formatted error message with installation instructions
    """
    # Map common modules to installation extras
    extras_map = {
        "sentence_transformers": "all",
        "faiss": "all",
        "openai": "all",
        "dspy": "all",
    }

    extra = extras_map.get(module_name, "all")

    return f"""
❌ Failed to import {module_name}: {import_error}

💡 Install with: pip install rec-praxis-rlm[{extra}]

📖 Docs: https://github.com/jmanhype/rec-praxis-rlm#installation
"""


def format_api_error(status_code: int, provider: str) -> str:
    """Format API error with actionable guidance.

    Args:
        status_code: HTTP status code
        provider: API provider name (e.g., "OpenAI", "Groq")

    Returns:
        Formatted error message with troubleshooting steps
    """
    if status_code == 401:
        return f"""
❌ {provider} API authentication failed (401 Unauthorized)

💡 Check your API key:
   - Verify {provider.upper()}_API_KEY environment variable is set
   - Ensure the API key is valid and not expired
   - Check for typos or extra whitespace

📖 Docs: https://github.com/jmanhype/rec-praxis-rlm#api-keys
"""
    elif status_code == 429:
        return f"""
❌ {provider} API rate limit exceeded (429 Too Many Requests)

💡 Troubleshooting:
   - Wait a few minutes before retrying
   - Check your API usage quota at {provider.lower()}.com
   - Consider upgrading your API plan
   - Reduce --max-iters to make fewer API calls

📖 Docs: https://github.com/jmanhype/rec-praxis-rlm#rate-limits
"""
    elif status_code == 500:
        return f"""
❌ {provider} API server error (500 Internal Server Error)

💡 This is a temporary issue on {provider}'s side:
   - Wait a few minutes and retry
   - Check {provider} status page for known issues
   - Try a different model if available

📖 Status: https://status.{provider.lower()}.com
"""
    else:
        return f"""
❌ {provider} API error (HTTP {status_code})

💡 Check:
   - Your API key is valid
   - Your account has sufficient credits
   - Network connection is stable

📖 Docs: https://github.com/jmanhype/rec-praxis-rlm#troubleshooting
"""


def format_memory_error(memory_path: str, error: Exception) -> str:
    """Format memory storage error with actionable guidance.

    Args:
        memory_path: Path to memory file
        error: The original exception

    Returns:
        Formatted error message with troubleshooting steps
    """
    return f"""
❌ Failed to access memory storage: {memory_path}

Error details: {error}

💡 Troubleshooting:
   - Check if directory exists and is writable
   - Ensure disk space is available
   - Verify file is not corrupted (try deleting and recreating)
   - Use --memory-dir to specify different location

📖 Docs: https://github.com/jmanhype/rec-praxis-rlm#procedural-memory
"""
