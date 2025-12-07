"""Integration tests for CLI ↔ Agent contract.

These tests ensure that the agents implement the exact interface expected by the CLI tools.
This prevents the contract mismatch that was discovered in v0.4.0.
"""

import subprocess
import json
import tempfile
from pathlib import Path

import pytest


class TestCodeReviewAgentContract:
    """Test that CodeReviewAgent implements the CLI contract."""

    def test_agent_has_required_interface(self):
        """Verify CodeReviewAgent has the required methods and constructor signature."""
        from rec_praxis_rlm.agents import CodeReviewAgent
        from rec_praxis_rlm.types import Finding, Severity

        # Test constructor accepts memory_path
        agent = CodeReviewAgent(memory_path=":memory:")

        # Test review_code method exists and accepts dict[str, str]
        test_code = {"test.py": "print('hello')"}
        result = agent.review_code(test_code)

        # Test return type is list
        assert isinstance(result, list)

        # If findings exist, test they match Finding contract
        for finding in result:
            assert isinstance(finding, Finding)
            assert hasattr(finding, 'file_path')
            assert hasattr(finding, 'severity')
            assert isinstance(finding.severity, Severity)
            assert hasattr(finding, 'title')
            assert hasattr(finding, 'description')
            assert hasattr(finding, 'remediation')
            assert hasattr(finding, 'line_number')

    def test_cli_code_review_e2e(self):
        """Test rec-praxis-review command end-to-end."""
        # Create a test file with a known issue
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("password = 'hardcoded123'\n")
            test_file = f.name

        try:
            # Run CLI command
            result = subprocess.run(
                ["rec-praxis-review", test_file, "--severity=HIGH", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Parse JSON output (ignore stderr warnings)
            # Extract JSON by finding first { and last }
            stdout = result.stdout
            start = stdout.find('{')
            end = stdout.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = stdout[start:end]
                output = json.loads(json_str)
            else:
                raise ValueError(f"No JSON found in output: {stdout}")

            # Verify structure
            assert "total_findings" in output
            assert "blocking_findings" in output
            assert "findings" in output
            assert isinstance(output["findings"], list)

            # Should find hardcoded password
            assert output["total_findings"] >= 1

            # Verify findings have required fields
            for finding in output["findings"]:
                assert "file" in finding
                assert "severity" in finding
                assert "title" in finding
                assert "description" in finding
                assert "remediation" in finding

        finally:
            Path(test_file).unlink(missing_ok=True)

    def test_cli_handles_no_findings(self):
        """Test CLI handles files with no findings gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Clean code\n")
            test_file = f.name

        try:
            result = subprocess.run(
                ["rec-praxis-review", test_file, "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )

            stdout = result.stdout
            start = stdout.find('{')
            end = stdout.rfind('}') + 1
            json_str = stdout[start:end] if start >= 0 and end > start else "{}"
            output = json.loads(json_str)
            assert output["total_findings"] == 0
            assert output["blocking_findings"] == 0
            assert output["findings"] == []

        finally:
            Path(test_file).unlink(missing_ok=True)


class TestSecurityAuditAgentContract:
    """Test that SecurityAuditAgent implements the CLI contract."""

    def test_agent_has_required_interface(self):
        """Verify SecurityAuditAgent has the required methods."""
        from rec_praxis_rlm.agents import SecurityAuditAgent
        from rec_praxis_rlm.types import AuditReport

        agent = SecurityAuditAgent(memory_path=":memory:")

        # Test generate_audit_report
        test_files = {"test.py": "print('hello')"}
        report = agent.generate_audit_report(test_files)

        # Verify report structure
        assert isinstance(report, AuditReport)
        assert hasattr(report, 'findings')
        assert hasattr(report, 'summary')
        assert hasattr(report, 'total_issues')

    def test_cli_security_audit_e2e(self):
        """Test rec-praxis-audit command end-to-end."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("cursor.execute(f'SELECT * FROM users WHERE id={user_id}')\n")
            test_file = f.name

        try:
            result = subprocess.run(
                ["rec-praxis-audit", test_file, "--fail-on=HIGH", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )

            stdout = result.stdout
            start = stdout.find('{')
            end = stdout.rfind('}') + 1
            json_str = stdout[start:end]
            output = json.loads(json_str)

            assert "total_findings" in output
            assert "summary" in output
            assert "findings" in output

            # Should find SQL injection
            assert output["total_findings"] >= 1

        finally:
            Path(test_file).unlink(missing_ok=True)


class TestDependencyScanAgentContract:
    """Test that DependencyScanAgent implements the CLI contract."""

    def test_agent_has_required_interface(self):
        """Verify DependencyScanAgent has the required methods."""
        from rec_praxis_rlm.agents import DependencyScanAgent

        agent = DependencyScanAgent(memory_path=":memory:")

        # Test scan_dependencies
        requirements = "urllib3==1.26.4\n"
        cve_findings, dependencies = agent.scan_dependencies(requirements)
        assert isinstance(cve_findings, list)
        assert isinstance(dependencies, list)

        # Test scan_secrets
        files = {"test.py": "api_key = 'sk-1234567890abcdefghijklmnopqrstuvwxyz'\n"}
        secret_findings = agent.scan_secrets(files)
        assert isinstance(secret_findings, list)

    def test_cli_dependency_scan_e2e(self):
        """Test rec-praxis-deps command end-to-end."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("urllib3==1.26.4\n")
            requirements_file = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("token = 'ghp_1234567890abcdefghijklmnopqrstuvwxyz'\n")
            test_file = f.name

        try:
            result = subprocess.run(
                ["rec-praxis-deps", "--requirements", requirements_file, "--files", test_file, "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )

            stdout = result.stdout
            start = stdout.find('{')
            end = stdout.rfind('}') + 1
            json_str = stdout[start:end]
            output = json.loads(json_str)

            assert "total_findings" in output
            assert "cve_count" in output
            assert "secret_count" in output
            assert "findings" in output

            # Should find at least the CVE or secret
            assert output["total_findings"] >= 1

        finally:
            Path(requirements_file).unlink(missing_ok=True)
            Path(test_file).unlink(missing_ok=True)


class TestTypeContract:
    """Test that types module provides required classes."""

    def test_severity_enum_exists(self):
        """Verify Severity enum has all required values."""
        from rec_praxis_rlm.types import Severity

        assert hasattr(Severity, 'INFO')
        assert hasattr(Severity, 'LOW')
        assert hasattr(Severity, 'MEDIUM')
        assert hasattr(Severity, 'HIGH')
        assert hasattr(Severity, 'CRITICAL')

    def test_finding_dataclass_exists(self):
        """Verify Finding dataclass has required fields."""
        from rec_praxis_rlm.types import Finding, Severity

        finding = Finding(
            file_path="test.py",
            line_number=10,
            severity=Severity.HIGH,
            title="Test Issue",
            description="Test description",
            remediation="Fix it"
        )

        assert finding.file_path == "test.py"
        assert finding.line_number == 10
        assert finding.severity == Severity.HIGH

        # Test to_dict method
        d = finding.to_dict()
        assert isinstance(d, dict)
        assert d["file"] == "test.py"
        assert d["severity"] == "HIGH"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
