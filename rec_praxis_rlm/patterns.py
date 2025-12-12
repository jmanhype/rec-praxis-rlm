"""Shared regex patterns for security and quality agents.

These patterns are intended to be language-agnostic heuristics used by multiple
agents (code review, security audit, dependency scan). Keep them consistent
across agents to reduce drift.
"""

from __future__ import annotations

from typing import Final

# SQL injection heuristics (Python DB-API style).
SQL_INJECTION_PATTERNS: Final[list[tuple[str, str]]] = [
    (r"execute\s*\(\s*f['\"]", "f-string in SQL execute()"),
    (r"execute\s*\(\s*['\"].*%s", "String formatting in SQL execute()"),
    (r"execute\s*\([^)]*\.format\(", ".format() in SQL execute()"),
    (r"cursor\.execute\([^)]*\+", "String concatenation in SQL execute()"),
]

# Hardcoded credential heuristics.
HARDCODED_CREDENTIAL_PATTERNS: Final[list[tuple[str, str]]] = [
    (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password"),
    (r"api_key\s*=\s*['\"](?!.*\$\{)(?!.*os\.getenv)[^'\"]{10,}['\"]", "Hardcoded API key"),
    (r"secret\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret"),
    (r"token\s*=\s*['\"](?!.*\$\{)(?!.*os\.getenv)[^'\"]{10,}['\"]", "Hardcoded token"),
]

# Weak hashing algorithms.
WEAK_HASH_REGEX: Final[str] = r"hashlib\.(md5|sha1)\("

# Bare except blocks.
BARE_EXCEPT_REGEX: Final[str] = r"^\s*except\s*:\s*$"

# Excessive debug prints.
PRINT_REGEX: Final[str] = r"^\s*print\s*\("

# Dangerous eval/exec usage.
DANGEROUS_EXEC_REGEX: Final[str] = r"\b(eval|exec)\s*\("

# subprocess shell=True (command injection risk).
SHELL_TRUE_REGEX: Final[str] = r"subprocess\.(call|run|Popen)\([^)]*shell\s*=\s*True"

# Generic command injection via shell execution.
COMMAND_INJECTION_REGEX: Final[str] = r"(os\.system|subprocess\..*shell\s*=\s*True)"

