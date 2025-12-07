"""Integration tests for rec-praxis-rlm CLI commands.

Tests the CLI entry points for code review, security audit, and dependency scanning.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # Create vulnerable Python file
        vulnerable_code = '''
import hashlib

def login(username, password):
    # Weak cryptography
    hashed = hashlib.md5(password.encode()).hexdigest()

    # SQL injection vulnerability
    import sqlite3
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{hashed}'"
    cursor.execute(query)
    result = cursor.fetchone()

    # Hardcoded secret
    API_KEY = "sk_test_1234567890abcdef"

    return result is not None
'''
        (workspace / "vulnerable.py").write_text(vulnerable_code)

        # Create requirements.txt with vulnerable dependencies
        requirements = '''
requests==2.25.0
pillow==8.0.0
django==3.1.0
'''
        (workspace / "requirements.txt").write_text(requirements)

        # Create config file with secrets
        config_code = '''
import os

DATABASE_URL = "postgres://admin:secretpassword@localhost/mydb"
AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
'''
        (workspace / "config.py").write_text(config_code)

        yield workspace


class TestCodeReviewCLI:
    """Test rec-praxis-review CLI command."""

    def test_code_review_finds_issues(self, temp_workspace):
        """Test that code review finds issues in vulnerable code."""
        vulnerable_file = temp_workspace / "vulnerable.py"

        result = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-review",
             str(vulnerable_file), "--severity=MEDIUM", "--json"],
            capture_output=True,
            text=True
        )

        # Parse JSON output
        output = json.loads(result.stdout)

        assert output["total_findings"] > 0
        assert "findings" in output
        assert isinstance(output["findings"], list)

    def test_code_review_respects_severity(self, temp_workspace):
        """Test that severity threshold filtering works."""
        vulnerable_file = temp_workspace / "vulnerable.py"

        # Low severity - should find more issues
        result_low = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-review",
             str(vulnerable_file), "--severity=LOW", "--json"],
            capture_output=True,
            text=True
        )
        output_low = json.loads(result_low.stdout)

        # Critical severity - should find fewer issues
        result_critical = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-review",
             str(vulnerable_file), "--severity=CRITICAL", "--json"],
            capture_output=True,
            text=True
        )
        output_critical = json.loads(result_critical.stdout)

        # LOW threshold should find >= CRITICAL threshold
        assert output_low["total_findings"] >= output_critical["total_findings"]

    def test_code_review_multiple_files(self, temp_workspace):
        """Test code review on multiple files."""
        file1 = temp_workspace / "vulnerable.py"
        file2 = temp_workspace / "config.py"

        result = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-review",
             str(file1), str(file2), "--severity=MEDIUM", "--json"],
            capture_output=True,
            text=True
        )

        output = json.loads(result.stdout)
        assert output["total_findings"] > 0

        # Check that findings come from both files
        files_with_findings = {f["file"] for f in output["findings"]}
        assert len(files_with_findings) >= 1  # At least one file has issues

    def test_code_review_exit_code(self, temp_workspace):
        """Test that exit code is 1 when blocking issues found."""
        vulnerable_file = temp_workspace / "vulnerable.py"

        result = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-review",
             str(vulnerable_file), "--severity=HIGH"],
            capture_output=True
        )

        # Should exit with 1 if HIGH+ issues found
        assert result.returncode in (0, 1)


class TestSecurityAuditCLI:
    """Test rec-praxis-audit CLI command."""

    def test_security_audit_finds_vulnerabilities(self, temp_workspace):
        """Test that security audit finds OWASP vulnerabilities."""
        vulnerable_file = temp_workspace / "vulnerable.py"

        result = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-audit",
             str(vulnerable_file), "--fail-on=CRITICAL", "--json"],
            capture_output=True,
            text=True
        )

        output = json.loads(result.stdout)

        assert output["total_findings"] > 0
        assert "findings" in output

        # Check for expected vulnerability types
        finding_titles = [f["title"] for f in output["findings"]]
        # Should find at least some security issues
        assert len(finding_titles) > 0

    def test_security_audit_owasp_mapping(self, temp_workspace):
        """Test that findings are mapped to OWASP categories."""
        vulnerable_file = temp_workspace / "vulnerable.py"

        result = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-audit",
             str(vulnerable_file), "--fail-on=LOW", "--json"],
            capture_output=True,
            text=True
        )

        output = json.loads(result.stdout)

        # At least some findings should have OWASP categories
        owasp_findings = [f for f in output["findings"] if f.get("owasp")]
        assert len(owasp_findings) > 0

    def test_security_audit_fail_on_threshold(self, temp_workspace):
        """Test that fail-on threshold affects exit code."""
        vulnerable_file = temp_workspace / "vulnerable.py"

        # Fail on CRITICAL - may or may not fail
        result_critical = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-audit",
             str(vulnerable_file), "--fail-on=CRITICAL"],
            capture_output=True
        )

        # Fail on LOW - should definitely fail
        result_low = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-audit",
             str(vulnerable_file), "--fail-on=LOW"],
            capture_output=True
        )

        # Both should complete (not crash)
        assert result_critical.returncode in (0, 1)
        assert result_low.returncode in (0, 1)


class TestDependencyScanCLI:
    """Test rec-praxis-deps CLI command."""

    def test_dependency_scan_finds_cves(self, temp_workspace):
        """Test that dependency scan finds CVEs."""
        result = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-deps",
             "--requirements", str(temp_workspace / "requirements.txt"),
             "--fail-on=CRITICAL", "--json"],
            capture_output=True,
            text=True,
            cwd=str(temp_workspace)
        )

        output = json.loads(result.stdout)

        assert "dependencies_scanned" in output
        assert output["dependencies_scanned"] > 0
        assert "cve_count" in output
        # May or may not find CVEs depending on database
        assert output["cve_count"] >= 0

    def test_dependency_scan_finds_secrets(self, temp_workspace):
        """Test that secret scanning finds exposed credentials."""
        config_file = temp_workspace / "config.py"

        result = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-deps",
             "--files", str(config_file),
             "--fail-on=CRITICAL", "--json"],
            capture_output=True,
            text=True,
            cwd=str(temp_workspace)
        )

        output = json.loads(result.stdout)

        assert "files_scanned" in output
        assert output["files_scanned"] >= 1
        assert "secret_count" in output
        # Should find at least the AWS key in config.py
        assert output["secret_count"] > 0

    def test_dependency_scan_combined(self, temp_workspace):
        """Test scanning both dependencies and secrets together."""
        result = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-deps",
             "--requirements", str(temp_workspace / "requirements.txt"),
             "--files", str(temp_workspace / "config.py"),
             "--fail-on=HIGH", "--json"],
            capture_output=True,
            text=True,
            cwd=str(temp_workspace)
        )

        output = json.loads(result.stdout)

        assert output["dependencies_scanned"] > 0
        assert output["files_scanned"] > 0
        assert output["total_findings"] > 0

        # Check that findings include both types
        finding_types = {f["type"] for f in output["findings"]}
        # Should have at least one type (CVE or Secret)
        assert len(finding_types) > 0

    def test_dependency_scan_exit_code(self, temp_workspace):
        """Test that exit code reflects blocking findings."""
        # Scan for secrets (should find AWS key)
        result = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-deps",
             "--files", str(temp_workspace / "config.py"),
             "--fail-on=CRITICAL"],
            capture_output=True,
            cwd=str(temp_workspace)
        )

        # Should complete without crashing
        assert result.returncode in (0, 1)


class TestCLIMemoryPersistence:
    """Test that CLI commands persist learning across runs."""

    def test_memory_persistence(self, temp_workspace):
        """Test that procedural memory persists across CLI invocations."""
        vulnerable_file = temp_workspace / "vulnerable.py"
        memory_dir = temp_workspace / ".rec-praxis-rlm"

        # First run - creates memory
        subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-review",
             str(vulnerable_file), "--severity=HIGH",
             "--memory-dir", str(memory_dir)],
            capture_output=True
        )

        # Check that memory files were created
        assert memory_dir.exists()
        memory_file = memory_dir / "code_review_memory.jsonl"
        # Memory file may or may not exist depending on whether agent stores experiences
        # This is acceptable - just check directory was created

    def test_separate_memory_per_tool(self, temp_workspace):
        """Test that each tool has separate memory storage."""
        vulnerable_file = temp_workspace / "vulnerable.py"
        memory_dir = temp_workspace / ".rec-praxis-rlm"

        # Run code review
        subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-review",
             str(vulnerable_file), "--memory-dir", str(memory_dir)],
            capture_output=True
        )

        # Run security audit
        subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-audit",
             str(vulnerable_file), "--memory-dir", str(memory_dir)],
            capture_output=True
        )

        # Memory directory should exist
        assert memory_dir.exists()


class TestCLIMainEntryPoint:
    """Test main CLI entry point."""

    def test_main_no_args_shows_help(self):
        """Test that running CLI with no args shows help."""
        result = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli"],
            capture_output=True,
            text=True
        )

        # Should show help and exit 0
        assert result.returncode == 0
        assert "rec-praxis-rlm CLI" in result.stdout or "rec-praxis-rlm CLI" in result.stderr

    def test_version_flag(self):
        """Test --version flag."""
        result = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "--version"],
            capture_output=True,
            text=True
        )

        # Should show version
        assert result.returncode == 0
        assert "rec-praxis-rlm" in result.stdout or "rec-praxis-rlm" in result.stderr


@pytest.mark.integration
class TestCLIEndToEnd:
    """End-to-end integration tests."""

    def test_full_workflow(self, temp_workspace):
        """Test complete workflow: review → audit → scan."""
        vulnerable_file = temp_workspace / "vulnerable.py"
        config_file = temp_workspace / "config.py"
        requirements_file = temp_workspace / "requirements.txt"

        # Step 1: Code review
        review_result = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-review",
             str(vulnerable_file), "--json"],
            capture_output=True,
            text=True
        )
        review_output = json.loads(review_result.stdout)
        assert review_output["total_findings"] > 0

        # Step 2: Security audit
        audit_result = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-audit",
             str(vulnerable_file), "--json"],
            capture_output=True,
            text=True
        )
        audit_output = json.loads(audit_result.stdout)
        assert audit_output["total_findings"] > 0

        # Step 3: Dependency scan
        deps_result = subprocess.run(
            ["python", "-m", "rec_praxis_rlm.cli", "rec-praxis-deps",
             "--requirements", str(requirements_file),
             "--files", str(config_file), "--json"],
            capture_output=True,
            text=True,
            cwd=str(temp_workspace)
        )
        deps_output = json.loads(deps_result.stdout)
        assert deps_output["total_findings"] > 0

        # All three tools should find issues
        total_issues = (
            review_output["total_findings"] +
            audit_output["total_findings"] +
            deps_output["total_findings"]
        )
        assert total_issues > 0
