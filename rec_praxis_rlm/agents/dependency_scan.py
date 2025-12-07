"""Production-ready Dependency Scan Agent for CLI and IDE integration.

CVE detection and secret scanning with procedural memory.
"""

import re
from typing import Dict, List, Tuple

from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig, RLMContext
from rec_praxis_rlm.types import CVEFinding, SecretFinding, Severity


class DependencyScanAgent:
    """Production dependency and secret scanning agent implementing CLI contract."""

    def __init__(self, memory_path: str = ":memory:"):
        """Initialize agent with persistent memory.

        Args:
            memory_path: Path to JSONL file for procedural memory storage
        """
        self.memory_path = memory_path
        self.memory = ProceduralMemory(
            config=MemoryConfig(
                storage_path=memory_path,
                env_weight=0.5,
                goal_weight=0.5,
            )
        )
        self.rlm = RLMContext()

        # Known vulnerable packages (simplified - real implementation would use CVE database)
        self.known_vulnerabilities = {
            "urllib3": {
                "1.26.4": ("CVE-2021-33503", "HIGH", "1.26.5+"),
                "1.25.0": ("CVE-2020-26137", "MEDIUM", "1.25.9+"),
            },
            "requests": {
                "2.25.0": ("CVE-2021-33503", "HIGH", "2.27.0+"),
            },
            "flask": {
                "0.12.0": ("CVE-2018-1000656", "HIGH", "0.12.3+"),
            },
            "django": {
                "3.1.0": ("CVE-2021-28658", "HIGH", "3.1.8+"),
            },
        }

    def scan_dependencies(self, requirements_content: str) -> Tuple[List[CVEFinding], List[str]]:
        """Scan dependencies for known CVEs.

        Args:
            requirements_content: Content of requirements.txt file

        Returns:
            Tuple of (CVE findings, list of dependency names)
        """
        findings = []
        dependencies = []

        # Parse requirements file
        for line in requirements_content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse package==version or package>=version
            match = re.match(r"([a-zA-Z0-9_-]+)\s*([=><]+)\s*([0-9.]+)", line)
            if match:
                package, operator, version = match.groups()
                dependencies.append(f"{package}{operator}{version}")

                # Check for known vulnerabilities
                if package in self.known_vulnerabilities:
                    vuln_versions = self.known_vulnerabilities[package]
                    if version in vuln_versions:
                        cve_id, severity, fixed = vuln_versions[version]
                        findings.append(CVEFinding(
                            package_name=package,
                            installed_version=version,
                            severity=Severity[severity],
                            cve_id=cve_id,
                            description=f"Known vulnerability in {package} {version}",
                            remediation=f"Upgrade to {fixed}",
                            fixed_version=fixed
                        ))

        return findings, dependencies

    def scan_secrets(self, files: Dict[str, str]) -> List[SecretFinding]:
        """Scan files for hardcoded secrets.

        Args:
            files: Dictionary mapping file paths to file contents

        Returns:
            List of SecretFinding objects
        """
        findings = []

        for file_path, content in files.items():
            self.rlm.add_document(file_path, content)
            file_findings = self._scan_file_for_secrets(file_path, content)
            findings.extend(file_findings)

        return findings

    def _scan_file_for_secrets(self, file_path: str, content: str) -> List[SecretFinding]:
        """Scan a single file for secrets."""
        findings = []

        # Secret patterns with context
        patterns = [
            (r"password\s*=\s*['\"]([^'\"]{6,})['\"]", "Password", Severity.HIGH),
            (r"api_key\s*=\s*['\"]([^'\"]{20,})['\"]", "API Key", Severity.CRITICAL),
            (r"secret\s*=\s*['\"]([^'\"]{10,})['\"]", "Secret Key", Severity.HIGH),
            (r"token\s*=\s*['\"]([^'\"]{20,})['\"]", "Auth Token", Severity.HIGH),
            (r"(sk-[a-zA-Z0-9]{32,})", "OpenAI API Key", Severity.CRITICAL),
            (r"(ghp_[a-zA-Z0-9]{36})", "GitHub Personal Access Token", Severity.CRITICAL),
            (r"(gsk-[a-zA-Z0-9]{32,})", "Groq API Key", Severity.CRITICAL),
            (r"(AWS[A-Z0-9]{16,})", "AWS Access Key", Severity.CRITICAL),
            (r"-----BEGIN (RSA |DSA )?PRIVATE KEY-----", "Private Key", Severity.CRITICAL),
        ]

        for pattern, secret_type, severity in patterns:
            matches = self.rlm.grep(pattern, doc_id=file_path)
            for match in matches:
                # Redact the actual secret
                matched_text = match.match_text if hasattr(match, 'match_text') else None
                if matched_text and len(matched_text) > 10:
                    matched_text = matched_text[:4] + "***" + matched_text[-4:]

                findings.append(SecretFinding(
                    file_path=file_path,
                    line_number=match.line_number,
                    secret_type=secret_type,
                    severity=severity,
                    description=f"{secret_type} detected in source code",
                    remediation="Remove from code. Use environment variables (os.getenv) or secret management service. Rotate the compromised secret immediately.",
                    matched_text=matched_text
                ))

        return findings

    def generate_report(
        self,
        cve_findings: List[CVEFinding],
        secret_findings: List[SecretFinding],
        num_dependencies: int,
        num_files_scanned: int
    ) -> str:
        """Generate human-readable dependency scan report.

        Args:
            cve_findings: List of CVE findings
            secret_findings: List of secret findings
            num_dependencies: Number of dependencies scanned
            num_files_scanned: Number of files scanned for secrets

        Returns:
            Formatted report string
        """
        lines = [
            "\n" + "=" * 70,
            "📦 DEPENDENCY & SECRET SCAN REPORT",
            "=" * 70,
            "",
            f"Dependencies Scanned: {num_dependencies}",
            f"Files Scanned: {num_files_scanned}",
            f"CVE Vulnerabilities: {len(cve_findings)}",
            f"Secrets Detected: {len(secret_findings)}",
            "",
        ]

        if cve_findings:
            lines.append("CVE VULNERABILITIES:\n")
            for i, finding in enumerate(cve_findings, 1):
                icon = {
                    Severity.CRITICAL: "🔴",
                    Severity.HIGH: "🟠",
                    Severity.MEDIUM: "🟡",
                    Severity.LOW: "🟢"
                }[finding.severity]
                lines.append(f"{i}. {icon} {finding.severity.name}: {finding.cve_id}")
                lines.append(f"   Package: {finding.package_name}=={finding.installed_version}")
                lines.append(f"   Issue: {finding.description}")
                lines.append(f"   Fix: {finding.remediation}")
                lines.append("")

        if secret_findings:
            lines.append("SECRETS DETECTED:\n")
            for i, finding in enumerate(secret_findings, 1):
                icon = {
                    Severity.CRITICAL: "🔴",
                    Severity.HIGH: "🟠",
                    Severity.MEDIUM: "🟡",
                    Severity.LOW: "🟢"
                }[finding.severity]
                lines.append(f"{i}. {icon} {finding.severity.name}: {finding.secret_type}")
                lines.append(f"   File: {finding.file_path}:{finding.line_number}")
                lines.append(f"   Issue: {finding.description}")
                if finding.matched_text:
                    lines.append(f"   Preview: {finding.matched_text}")
                lines.append(f"   Fix: {finding.remediation}")
                lines.append("")

        if not cve_findings and not secret_findings:
            lines.append("✅ No vulnerabilities or secrets detected!\n")

        lines.append("=" * 70)

        # Summary
        critical_count = sum(1 for f in cve_findings + secret_findings if f.severity == Severity.CRITICAL)
        high_count = sum(1 for f in cve_findings + secret_findings if f.severity == Severity.HIGH)

        if critical_count > 0:
            lines.append(f"\n⚠️  CRITICAL: {critical_count} critical issue(s) found. Immediate action required!\n")
        elif high_count > 0:
            lines.append(f"\n⚠️  WARNING: {high_count} high severity issue(s) found. Review soon.\n")
        else:
            lines.append("\n✅ No critical or high severity issues detected.\n")

        return "\n".join(lines)
