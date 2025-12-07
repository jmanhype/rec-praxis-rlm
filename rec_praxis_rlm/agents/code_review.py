"""Production-ready Code Review Agent for CLI and IDE integration.

This agent implements the interface contract expected by rec-praxis-rlm CLI tools.
It uses procedural memory to learn from past code reviews and provides consistent
findings across sessions.
"""

import re
import time
from pathlib import Path
from typing import Dict, List

from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig, RLMContext
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
        sql_patterns = [
            (r"execute\s*\(\s*f['\"]", "f-string in SQL execute()"),
            (r"execute\s*\(\s*['\"].*%s", "String formatting in SQL execute()"),
            (r"execute\s*\([^)]*\.format\(", ".format() in SQL execute()"),
            (r"cursor\.execute\([^)]*\+", "String concatenation in SQL execute()"),
        ]

        for pattern, desc in sql_patterns:
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
        cred_patterns = [
            (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password"),
            (r"api_key\s*=\s*['\"](?!.*\$\{)(?!.*os\.getenv)[^'\"]{10,}['\"]", "Hardcoded API key"),
            (r"secret\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret"),
            (r"token\s*=\s*['\"](?!.*\$\{)(?!.*os\.getenv)[^'\"]{10,}['\"]", "Hardcoded token"),
        ]

        for pattern, desc in cred_patterns:
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
        weak_crypto = self.rlm.grep(r"hashlib\.(md5|sha1)\(", doc_id=file_path)
        if weak_crypto:
            for match in weak_crypto:
                findings.append(Finding(
                    file_path=file_path,
                    line_number=match.line_number,
                    severity=Severity.HIGH,
                    title="Weak Cryptography",
                    description="MD5/SHA1 used for hashing (deprecated for security)",
                    remediation="Use bcrypt for passwords: bcrypt.hashpw(password, bcrypt.gensalt()) or SHA-256+ for data integrity"
                ))

        # 4. Bare except blocks
        bare_except = self.rlm.grep(r"^\s*except\s*:\s*$", doc_id=file_path)
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
        print_statements = self.rlm.grep(r"^\s*print\s*\(", doc_id=file_path)
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
        dangerous_funcs = self.rlm.grep(r"\b(eval|exec)\s*\(", doc_id=file_path)
        if dangerous_funcs:
            # Get full content to check line context
            lines = content.split('\n')

            for match in dangerous_funcs:
                # Skip if it's in a comment line
                if match.line_number <= len(lines):
                    full_line = lines[match.line_number - 1]
                    stripped = full_line.strip()

                    # Skip comment lines
                    if stripped.startswith('#'):
                        continue

                    # Skip if eval/exec appears after a quote on the same line
                    # This catches: description = "The eval() function..."
                    match_pos = full_line.find('eval(') if 'eval(' in full_line else full_line.find('exec(')
                    if match_pos > 0:
                        # Check if there's an opening quote before the match
                        before_match = full_line[:match_pos]
                        # Count quotes before the match
                        double_quotes = before_match.count('"')
                        single_quotes = before_match.count("'")
                        # If odd number of quotes, we're inside a string literal
                        if double_quotes % 2 == 1 or single_quotes % 2 == 1:
                            continue

                    # Also skip if line contains string assignment keywords
                    if any(keyword in full_line for keyword in ['description=', 'remediation=', 'title=', 'help=']):
                        continue

                findings.append(Finding(
                    file_path=file_path,
                    line_number=match.line_number,
                    severity=Severity.CRITICAL,
                    title="Dangerous Code Execution",
                    description="eval() or exec() enables arbitrary code execution",
                    remediation="Avoid eval/exec. Use safer alternatives like ast.literal_eval() for data or explicit function dispatch"
                ))

        # 7. Shell injection via subprocess
        shell_injection = self.rlm.grep(r"subprocess\.(call|run|Popen)\([^)]*shell\s*=\s*True", doc_id=file_path)
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
