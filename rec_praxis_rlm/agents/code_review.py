"""Production-ready Code Review Agent for CLI and IDE integration.

This agent implements the interface contract expected by rec-praxis-rlm CLI tools.
It uses procedural memory to learn from past code reviews and provides consistent
findings across sessions.
"""

import re
import io
import tokenize
import time
from pathlib import Path
from typing import Dict, List

from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig, RLMContext
from rec_praxis_rlm.patterns import (
    SQL_INJECTION_PATTERNS,
    HARDCODED_CREDENTIAL_PATTERNS,
    WEAK_HASH_REGEX,
    BARE_EXCEPT_REGEX,
    PRINT_REGEX,
    DANGEROUS_EXEC_REGEX,
    SHELL_TRUE_REGEX,
)
from rec_praxis_rlm.types import Finding, Severity


class CodeReviewAgent:
    """Production code review agent with persistent procedural memory.

    This implementation matches the CLI contract:
    - Constructor takes memory_path parameter
    - review_code() takes dict[str, str] and returns list[Finding]
    - All findings have required fields for JSON serialization
    """

    def __init__(self, memory_path: str = ":memory:"):
        """Initialize agent with persistent memory.

        Args:
            memory_path: Path to JSONL file for procedural memory storage.
                        Use ":memory:" for in-memory (testing only).
        """
        self.memory_path = memory_path
        self.memory = ProceduralMemory(
            config=MemoryConfig(
                storage_path=memory_path,
                env_weight=0.6,
                goal_weight=0.4,
            )
        )
        self.rlm = RLMContext()

    def review_code(self, files: Dict[str, str]) -> List[Finding]:
        """Review code files and return structured findings.

        Args:
            files: Dictionary mapping file paths to file contents

        Returns:
            List of Finding objects matching CLI contract
        """
        # Reset context per run to avoid duplicate doc IDs across calls.
        self.rlm = RLMContext()
        all_findings = []

        for file_path, content in files.items():
            # Add to RLM context for pattern matching
            self.rlm.add_document(file_path, content)

            # Run pattern-based checks
            findings = self._check_patterns(file_path, content)
            all_findings.extend(findings)

            # Store successful reviews in memory for future reference
            if findings:
                self._store_review_experience(file_path, findings)

        return all_findings

    def _check_patterns(self, file_path: str, content: str) -> List[Finding]:
        """Run pattern-based security and quality checks.

        Args:
            file_path: Path to the file being reviewed
            content: File content

        Returns:
            List of findings detected by pattern matching
        """
        findings = []

        # 1. SQL Injection patterns
        for pattern, desc in SQL_INJECTION_PATTERNS:
            matches = self.rlm.grep(pattern, doc_id=file_path)
            if matches:
                for match in matches:
                    findings.append(Finding(
                        file_path=file_path,
                        line_number=match.line_number,
                        severity=Severity.HIGH,
                        title="SQL Injection Risk",
                        description=f"Potential SQL injection: {desc}",
                        remediation="Use parameterized queries instead. Example: cursor.execute('SELECT * FROM users WHERE id=?', (user_id,))"
                    ))

        # 2. Hardcoded credentials
        for pattern, desc in HARDCODED_CREDENTIAL_PATTERNS:
            matches = self.rlm.grep(pattern, doc_id=file_path)
            if matches:
                for match in matches:
                    findings.append(Finding(
                        file_path=file_path,
                        line_number=match.line_number,
                        severity=Severity.CRITICAL,
                        title="Hardcoded Credentials",
                        description=f"{desc} found in source code",
                        remediation="Use environment variables: os.getenv('API_KEY') or configuration files excluded from version control"
                    ))

        # 3. Weak cryptography
        weak_crypto = self.rlm.grep(WEAK_HASH_REGEX, doc_id=file_path)
        if weak_crypto:
            normalized_path = file_path.replace("\\", "/")
            is_test_file = normalized_path.startswith("tests/") or "/tests/" in normalized_path
            severity = Severity.MEDIUM if is_test_file else Severity.HIGH
            for match in weak_crypto:
                findings.append(
                    Finding(
                        file_path=file_path,
                        line_number=match.line_number,
                        severity=severity,
                        title="Weak Cryptography",
                        description="MD5/SHA1 used for hashing (deprecated for security)",
                        remediation="Use bcrypt for passwords: bcrypt.hashpw(password, bcrypt.gensalt()) or SHA-256+ for data integrity",
                    )
                )

        # 4. Bare except blocks
        bare_except = self.rlm.grep(BARE_EXCEPT_REGEX, doc_id=file_path)
        if bare_except:
            for match in bare_except:
                findings.append(Finding(
                    file_path=file_path,
                    line_number=match.line_number,
                    severity=Severity.MEDIUM,
                    title="Overly Broad Exception Handling",
                    description="Bare except: block catches all exceptions including system exits",
                    remediation="Use specific exception types: except (ValueError, KeyError) as e:"
                ))

        # 5. Debug print statements (only flag if many)
        print_statements = self.rlm.grep(PRINT_REGEX, doc_id=file_path)
        if len(print_statements) > 5:  # Only flag if excessive
            # Group by line number to avoid duplicates
            unique_lines = set(m.line_number for m in print_statements)
            if len(unique_lines) > 5:
                findings.append(Finding(
                    file_path=file_path,
                    line_number=list(unique_lines)[0],
                    severity=Severity.LOW,
                    title="Excessive Debug Print Statements",
                    description=f"Found {len(unique_lines)} print() statements - consider using logging",
                    remediation="Use logging module: logger.info('message') instead of print()"
                ))

        # 6. Eval/exec usage (filter out false positives in strings/comments)
        dangerous_funcs = self.rlm.grep(DANGEROUS_EXEC_REGEX, doc_id=file_path)
        if dangerous_funcs:
            normalized_path = file_path.replace("\\", "/")
            # Allowlisted internal trusted sandbox implementation.
            if normalized_path == "rec_praxis_rlm/sandbox.py":
                dangerous_funcs = []

        if dangerous_funcs:
            lines = content.split("\n")

            # Build a set of line numbers that are inside string literals/docstrings.
            string_lines: set[int] = set()
            try:
                for tok in tokenize.generate_tokens(io.StringIO(content).readline):
                    if tok.type == tokenize.STRING:
                        start_line = tok.start[0]
                        end_line = tok.end[0]
                        string_lines.update(range(start_line, end_line + 1))
            except tokenize.TokenError:
                # If tokenization fails, fall back to heuristic checks below.
                pass

            for match in dangerous_funcs:
                if match.line_number is None:
                    continue

                # Ignore matches inside strings/docstrings.
                if match.line_number in string_lines:
                    continue

                if match.line_number <= len(lines):
                    full_line = lines[match.line_number - 1]
                    stripped = full_line.strip()

                    # Skip comment lines
                    if stripped.startswith("#"):
                        continue

                    # Also skip if line contains string assignment keywords
                    if any(keyword in full_line for keyword in ["description=", "remediation=", "title=", "help="]):
                        continue

                findings.append(
                    Finding(
                        file_path=file_path,
                        line_number=match.line_number,
                        severity=Severity.CRITICAL,
                        title="Dangerous Code Execution",
                        description="eval() or exec() enables arbitrary code execution",
                        remediation="Avoid eval/exec. Use safer alternatives like ast.literal_eval() for data or explicit function dispatch",
                    )
                )

        # 7. Shell injection via subprocess
        shell_injection = self.rlm.grep(SHELL_TRUE_REGEX, doc_id=file_path)
        if shell_injection:
            for match in shell_injection:
                findings.append(Finding(
                    file_path=file_path,
                    line_number=match.line_number,
                    severity=Severity.HIGH,
                    title="Shell Injection Risk",
                    description="subprocess with shell=True enables command injection",
                    remediation="Use shell=False and pass command as list: subprocess.run(['ls', '-la'])"
                ))

        return findings

    def _store_review_experience(self, file_path: str, findings: List[Finding]):
        """Store review results in procedural memory for future learning.

        Args:
            file_path: Path to reviewed file
            findings: Findings detected in this review
        """
        # Only store if memory is persistent (not :memory:)
        if self.memory_path == ":memory:":
            return

        # Group findings by severity
        severity_counts = {}
        for f in findings:
            severity_counts[f.severity.name] = severity_counts.get(f.severity.name, 0) + 1

        # Store high-value experiences (CRITICAL or HIGH findings)
        critical_findings = [f for f in findings if f.severity in (Severity.CRITICAL, Severity.HIGH)]
        if critical_findings:
            for finding in critical_findings[:3]:  # Store up to 3 most important
                exp = Experience(
                    env_features=["python", "code_review", finding.severity.name.lower()],
                    goal=f"detect {finding.title.lower()}",
                    action=f"Found: {finding.description}",
                    result=f"Remediation: {finding.remediation}",
                    success=True,
                    timestamp=time.time()
                )
                self.memory.store(exp)
