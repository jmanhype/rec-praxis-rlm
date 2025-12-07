"""
Dependency Scan Agent - Multi-Modal Memory Demo

This example demonstrates Week 5 capabilities:
1. CVE detection in Python dependencies
2. Secret scanning (API keys, tokens, credentials)
3. Multi-modal memory: Learn from past dependency upgrades
4. Suggest upgrade paths based on successful past migrations

Architecture:
- Procedural Memory: 10 past dependency upgrades and secret incidents
- Semantic Memory: CVE facts and security knowledge
- RLM Context: File parsing and pattern matching

Run this example:
    python examples/dependency_scan_agent.py
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig, RLMContext, FactStore


class Severity(Enum):
    """CVE and secret severity levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class FindingType(Enum):
    """Type of security finding."""

    CVE = "CVE"
    SECRET = "SECRET"
    OUTDATED_DEPENDENCY = "OUTDATED"


@dataclass
class Dependency:
    """Parsed dependency from requirements file."""

    name: str
    version: str
    line_number: Optional[int] = None


@dataclass
class CVEFinding:
    """CVE vulnerability finding."""

    cve_id: str
    package: str
    vulnerable_version: str
    severity: Severity
    cvss_score: float
    description: str
    fixed_version: str
    remediation: str
    references: List[str] = field(default_factory=list)


@dataclass
class SecretFinding:
    """Exposed secret finding."""

    secret_type: str
    file_path: str
    line_number: Optional[int]
    severity: Severity
    description: str
    remediation: str
    entropy: Optional[float] = None


@dataclass
class DependencyScanReport:
    """Comprehensive dependency and secret scan report."""

    scan_date: str
    cve_findings: List[CVEFinding]
    secret_findings: List[SecretFinding]
    dependencies_scanned: int
    files_scanned: int
    summary: Dict[str, int]  # severity -> count


class DependencyScanAgent:
    """
    Multi-modal dependency and secret scanning agent.

    Learns from past dependency upgrades and secret incidents to provide
    context-aware remediation guidance.
    """

    def __init__(self):
        """Initialize agent with multi-modal memory."""
        # Procedural memory for past upgrades and incidents
        self.memory = ProceduralMemory(
            config=MemoryConfig(
                storage_path=":memory:",
                similarity_threshold=0.3,
                env_weight=0.6,
                goal_weight=0.4,
            )
        )

        # RLM context for file parsing and pattern matching
        self.rlm = RLMContext()

        # Semantic memory for CVE facts and security knowledge
        self.fact_store = FactStore(storage_path=":memory:")

        # Populate memory with past experiences
        self._populate_upgrade_history()
        self._load_security_knowledge()

        # Secret detection patterns
        self._secret_patterns = self._build_secret_patterns()

    def _populate_upgrade_history(self) -> None:
        """Populate memory with 10 past dependency upgrades and secret incidents."""
        import time

        # Use current time minus days as Unix timestamps
        now = time.time()
        days = 24 * 3600

        past_experiences = [
            # Dependency upgrades
            Experience(
                env_features=["python", "dependencies", "security", "cve", "requests"],
                goal="upgrade vulnerable requests library",
                action="Upgraded requests from 2.25.0 to 2.31.0 to fix CVE-2023-32681 (proxy credential leakage)",
                result="Successfully patched. All 150 tests pass. No breaking changes. Critical vulnerability eliminated.",
                success=True,
                timestamp=now - (5 * days),  # 5 days ago
            ),
            Experience(
                env_features=["python", "dependencies", "django", "breaking-change"],
                goal="upgrade Django with major version change",
                action="Migrated from Django 2.2 to 3.2 LTS. Updated URL patterns from url() to path(), fixed deprecated imports.",
                result="Migration took 2 days. 15 files changed. All tests pass. 6 CVEs patched.",
                success=True,
                timestamp=now - (20 * days),  # 20 days ago
            ),
            Experience(
                env_features=["python", "dependencies", "pillow", "cve"],
                goal="fix Pillow image processing vulnerability",
                action="Upgraded Pillow from 8.0.0 to 10.1.0 to fix CVE-2022-45198 (arbitrary code execution)",
                result="Patched successfully. Image processing still works. No API changes needed.",
                success=True,
                timestamp=now - (15 * days),  # 15 days ago
            ),
            Experience(
                env_features=["python", "dependencies", "sqlalchemy", "minor-upgrade"],
                goal="upgrade SQLAlchemy for security patch",
                action="Upgraded SQLAlchemy from 1.4.25 to 1.4.46 (patch version)",
                result="Smooth upgrade. No code changes required. Security issue resolved.",
                success=True,
                timestamp=now - (25 * days),  # 25 days ago
            ),
            Experience(
                env_features=["python", "dependencies", "cryptography", "breaking-change"],
                goal="upgrade cryptography library with breaking changes",
                action="Upgraded cryptography from 3.4.8 to 41.0.0. Updated deprecated backend parameter usage.",
                result="Breaking changes in 38.0.0 required code updates. All encryption still works.",
                success=True,
                timestamp=now - (40 * days),  # 40 days ago
            ),
            # Secret incidents
            Experience(
                env_features=["security", "secrets", "aws", "credentials"],
                goal="fix exposed AWS credentials",
                action="Rotated AWS access keys after accidental commit. Moved credentials to environment variables.",
                result="Credentials rotated within 1 hour. Added .env to .gitignore. Implemented pre-commit hook.",
                success=True,
                timestamp=now - (2 * days),  # 2 days ago
            ),
            Experience(
                env_features=["security", "secrets", "github", "token"],
                goal="remediate leaked GitHub token",
                action="Revoked GitHub personal access token found in config.py. Generated new token with minimal scopes.",
                result="Token revoked immediately. Moved to GitHub Actions secrets. No unauthorized access detected.",
                success=True,
                timestamp=now - (7 * days),  # 7 days ago
            ),
            Experience(
                env_features=["security", "secrets", "database", "password"],
                goal="fix hardcoded database password",
                action="Moved database credentials from code to .env file. Updated deployment to use environment variables.",
                result="Database access still works. Credentials no longer in git history. Audit trail clean.",
                success=True,
                timestamp=now - (17 * days),  # 17 days ago
            ),
            Experience(
                env_features=["security", "secrets", "api-key", "stripe"],
                goal="secure Stripe API keys",
                action="Replaced hardcoded Stripe keys with environment variables. Separated test/prod keys.",
                result="Payment processing works. Test keys in .env.test, prod keys in secure vault.",
                success=True,
                timestamp=now - (30 * days),  # 30 days ago
            ),
            Experience(
                env_features=["security", "secrets", "ssh", "private-key"],
                goal="remove committed SSH private key",
                action="Removed SSH private key from git history using BFG Repo-Cleaner. Generated new keypair.",
                result="Private key no longer in history. New key deployed to servers. Access maintained.",
                success=True,
                timestamp=now - (45 * days),  # 45 days ago
            ),
        ]

        for exp in past_experiences:
            self.memory.store(exp)

    def _load_security_knowledge(self) -> None:
        """Load CVE and security facts into semantic memory."""
        security_facts = """
        CVE = Common Vulnerabilities and Exposures
        CVSS = Common Vulnerability Scoring System (0-10 scale)
        NVD = National Vulnerability Database
        CVE-2023-32681 = Requests library proxy credential leakage (CVSS 7.5)
        CVE-2022-45198 = Pillow arbitrary code execution via crafted image
        CVE-2019-12781 = Django incorrect HTTP detection in admin
        CRITICAL = CVSS score 9.0-10.0
        HIGH = CVSS score 7.0-8.9
        MEDIUM = CVSS score 4.0-6.9
        LOW = CVSS score 0.1-3.9
        """
        self.fact_store.extract_facts(security_facts, source_id="cve_knowledge")

    def _build_secret_patterns(self) -> Dict[str, Tuple[str, Severity]]:
        """
        Build regex patterns for secret detection.

        Returns:
            Dict mapping secret type to (pattern, severity)
        """
        return {
            "AWS Access Key": (r"AKIA[0-9A-Z]{16}", Severity.CRITICAL),
            "GitHub Token": (r"ghp_[a-zA-Z0-9]{36}", Severity.HIGH),
            "Generic API Key": (r"api[_-]?key['\"\s:=]+['\"]([a-zA-Z0-9_\-]{20,})['\"]", Severity.HIGH),
            "Private Key": (r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----", Severity.CRITICAL),
            "Database URL with Password": (
                r"(postgres|mysql|mongodb)://[^:]+:([^@]+)@",
                Severity.HIGH,
            ),
            "Generic Password": (r"password['\"\s:=]+['\"]([^'\"]{8,})['\"]", Severity.MEDIUM),
            "JWT Token": (r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+", Severity.HIGH),
            "Slack Token": (r"xox[baprs]-[0-9a-zA-Z]{10,}", Severity.HIGH),
        }

    def parse_requirements(self, requirements_content: str) -> List[Dependency]:
        """
        Parse requirements.txt file.

        Args:
            requirements_content: Content of requirements.txt

        Returns:
            List of parsed dependencies
        """
        dependencies = []
        lines = requirements_content.strip().split("\n")

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Parse package==version format
            match = re.match(r"^([a-zA-Z0-9_\-]+)==([0-9\.]+)", line)
            if match:
                name, version = match.groups()
                dependencies.append(Dependency(name=name, version=version, line_number=line_num))

        return dependencies

    def check_cve(self, package: str, version: str) -> List[CVEFinding]:
        """
        Check if package version has known CVEs.

        This is a simplified implementation using hardcoded CVE data.
        In production, this would query NVD API, GitHub Security Advisories, or PyUp Safety DB.

        Args:
            package: Package name
            version: Package version

        Returns:
            List of CVE findings
        """
        # Hardcoded CVE database (simplified for demo)
        cve_database = {
            "requests": {
                "2.25.0": [
                    CVEFinding(
                        cve_id="CVE-2023-32681",
                        package="requests",
                        vulnerable_version="2.25.0",
                        severity=Severity.CRITICAL,
                        cvss_score=7.5,
                        description="Proxy credential leakage when using HTTP proxies. Credentials may be sent to incorrect destinations.",
                        fixed_version="2.31.0",
                        remediation="Upgrade to requests>=2.31.0",
                        references=[
                            "https://nvd.nist.gov/vuln/detail/CVE-2023-32681",
                            "https://github.com/advisories/GHSA-j8r2-6x86-q33q",
                        ],
                    )
                ],
            },
            "Django": {
                "2.0.0": [
                    CVEFinding(
                        cve_id="CVE-2019-12781",
                        package="Django",
                        vulnerable_version="2.0.0",
                        severity=Severity.HIGH,
                        cvss_score=7.8,
                        description="Incorrect HTTP detection allows attackers to bypass certain security restrictions.",
                        fixed_version="3.2.0",
                        remediation="Upgrade to Django>=3.2 LTS (requires migration for URL patterns)",
                        references=["https://nvd.nist.gov/vuln/detail/CVE-2019-12781"],
                    ),
                    CVEFinding(
                        cve_id="CVE-2019-14234",
                        package="Django",
                        vulnerable_version="2.0.0",
                        severity=Severity.HIGH,
                        cvss_score=7.5,
                        description="SQL injection in key transforms for JSONField/HStoreField.",
                        fixed_version="3.2.0",
                        remediation="Upgrade to Django>=3.2 LTS",
                        references=["https://nvd.nist.gov/vuln/detail/CVE-2019-14234"],
                    ),
                ],
            },
            "pillow": {
                "8.0.0": [
                    CVEFinding(
                        cve_id="CVE-2022-45198",
                        package="pillow",
                        vulnerable_version="8.0.0",
                        severity=Severity.MEDIUM,
                        cvss_score=5.5,
                        description="Arbitrary code execution via crafted image file.",
                        fixed_version="10.1.0",
                        remediation="Upgrade to pillow>=10.1.0",
                        references=["https://nvd.nist.gov/vuln/detail/CVE-2022-45198"],
                    )
                ],
            },
        }

        package_lower = package.lower()
        if package_lower in cve_database and version in cve_database[package_lower]:
            return cve_database[package_lower][version]

        return []

    def scan_dependencies(self, requirements_content: str) -> Tuple[List[CVEFinding], List[Dependency]]:
        """
        Scan dependencies for known CVEs.

        Args:
            requirements_content: Content of requirements.txt

        Returns:
            Tuple of (CVE findings, all dependencies)
        """
        dependencies = self.parse_requirements(requirements_content)
        cve_findings = []

        for dep in dependencies:
            findings = self.check_cve(dep.name, dep.version)
            cve_findings.extend(findings)

        return cve_findings, dependencies

    def calculate_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of text (bits per character).

        High entropy strings (>4.5) are likely secrets.

        Args:
            text: String to analyze

        Returns:
            Entropy in bits per character
        """
        if not text:
            return 0.0

        import math
        from collections import Counter

        counter = Counter(text)
        length = len(text)
        entropy = -sum((count / length) * math.log2(count / length) for count in counter.values())

        return entropy

    def scan_secrets(self, files: Dict[str, str]) -> List[SecretFinding]:
        """
        Scan files for exposed secrets.

        Args:
            files: Dict mapping file path to file content

        Returns:
            List of secret findings
        """
        findings = []

        for file_path, content in files.items():
            # Add to RLM context for pattern matching
            self.rlm.add_document(file_path, content)

            # Scan with each secret pattern
            for secret_type, (pattern, severity) in self._secret_patterns.items():
                matches = self.rlm.grep(pattern, doc_id=file_path)

                for match in matches:
                    # Calculate entropy if it's a credential value
                    entropy = None
                    if "password" in secret_type.lower() or "key" in secret_type.lower():
                        # Extract the matched value for entropy analysis
                        regex_match = re.search(pattern, match.match_text, re.IGNORECASE)
                        if regex_match:
                            if regex_match.groups():
                                value = regex_match.group(1)
                            else:
                                value = regex_match.group(0)
                            entropy = self.calculate_entropy(value)

                    # Skip if entropy is too low (likely false positive)
                    if entropy is not None and entropy < 3.0:
                        continue

                    # Recall past secret incidents for remediation guidance
                    past_incidents = self.memory.recall(
                        env_features=["security", "secrets", secret_type.lower().replace(" ", "-")],
                        goal=f"fix exposed {secret_type.lower()}",
                        top_k=2,
                    )

                    if past_incidents:
                        remediation = past_incidents[0].action
                    else:
                        remediation = f"Move {secret_type} to environment variable or secret manager"

                    findings.append(
                        SecretFinding(
                            secret_type=secret_type,
                            file_path=file_path,
                            line_number=match.line_number,
                            severity=severity,
                            description=f"Found {secret_type} in source code",
                            remediation=remediation,
                            entropy=entropy,
                        )
                    )

        return findings

    def generate_report(
        self,
        cve_findings: List[CVEFinding],
        secret_findings: List[SecretFinding],
        dependencies_scanned: int,
        files_scanned: int,
    ) -> DependencyScanReport:
        """
        Generate comprehensive scan report.

        Args:
            cve_findings: List of CVE findings
            secret_findings: List of secret findings
            dependencies_scanned: Number of dependencies scanned
            files_scanned: Number of files scanned

        Returns:
            Structured scan report
        """
        # Calculate summary statistics
        summary = {
            "CRITICAL": sum(
                1 for f in cve_findings if f.severity == Severity.CRITICAL
            )
            + sum(1 for f in secret_findings if f.severity == Severity.CRITICAL),
            "HIGH": sum(1 for f in cve_findings if f.severity == Severity.HIGH)
            + sum(1 for f in secret_findings if f.severity == Severity.HIGH),
            "MEDIUM": sum(1 for f in cve_findings if f.severity == Severity.MEDIUM)
            + sum(1 for f in secret_findings if f.severity == Severity.MEDIUM),
            "LOW": sum(1 for f in cve_findings if f.severity == Severity.LOW)
            + sum(1 for f in secret_findings if f.severity == Severity.LOW),
        }

        return DependencyScanReport(
            scan_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cve_findings=cve_findings,
            secret_findings=secret_findings,
            dependencies_scanned=dependencies_scanned,
            files_scanned=files_scanned,
            summary=summary,
        )

    def print_report(self, report: DependencyScanReport) -> None:
        """Print formatted scan report."""
        print("=" * 70)
        print("Dependency & Secret Scan Report")
        print("=" * 70)
        print(f"Scan Date: {report.scan_date}")
        print(f"Dependencies Scanned: {report.dependencies_scanned}")
        print(f"Files Scanned: {report.files_scanned}")
        print()

        print("📊 Summary:")
        for severity, count in report.summary.items():
            if count > 0:
                print(f"   {severity}: {count}")
        print()

        # CVE findings
        if report.cve_findings:
            print(f"🚨 CVE Findings ({len(report.cve_findings)} total):")
            print()
            for i, finding in enumerate(report.cve_findings, 1):
                print(f"{i}. [{finding.severity.value}] {finding.cve_id} - {finding.package} {finding.vulnerable_version}")
                print(f"   CVSS Score: {finding.cvss_score}")
                print(f"   Description: {finding.description}")
                print(f"   Remediation: {finding.remediation}")

                # Recall past upgrade experience for context
                past_upgrades = self.memory.recall(
                    env_features=["python", "dependencies", finding.package.lower()],
                    goal=f"upgrade {finding.package.lower()}",
                    top_k=1,
                )
                if past_upgrades:
                    print(f"   💡 Past Experience: \"{past_upgrades[0].action[:80]}...\"")

                if finding.references:
                    print(f"   References: {finding.references[0]}")
                print()

        # Secret findings
        if report.secret_findings:
            print(f"🔒 Secret Findings ({len(report.secret_findings)} total):")
            print()
            for i, finding in enumerate(report.secret_findings, 1):
                print(f"{i}. [{finding.severity.value}] {finding.secret_type}")
                print(f"   File: {finding.file_path}:{finding.line_number}")
                print(f"   Description: {finding.description}")
                if finding.entropy:
                    print(f"   Entropy: {finding.entropy:.2f} bits/char")
                print(f"   Remediation: {finding.remediation}")
                print()

        if not report.cve_findings and not report.secret_findings:
            print("✅ No vulnerabilities or secrets found!")
            print()

        print("=" * 70)


def main():
    """Run dependency and secret scan demo."""
    print("\n" + "=" * 70)
    print("Dependency Scan Agent - Multi-Modal Memory Demo")
    print("=" * 70)
    print()

    # Initialize agent
    agent = DependencyScanAgent()

    print("📚 Memory initialized:")
    print("   - 10 past dependency upgrades and secret incidents")
    print("   - 10 CVE/security facts")
    print()

    # Sample vulnerable requirements.txt
    requirements_txt = """
# Core dependencies
requests==2.25.0
Django==2.0.0
pillow==8.0.0
sqlalchemy==1.4.46
flask==2.3.0
"""

    # Sample config file with exposed secrets
    config_py = """
# Configuration file
import os

# AWS Credentials
AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# GitHub Token
GITHUB_TOKEN = "ghp_1234567890abcdefghijklmnopqrstuv1234"

# Database
DATABASE_URL = "postgres://admin:SuperSecret123@localhost:5432/mydb"

# API Keys (example - obviously fake)
STRIPE_API_KEY = "stripe_key_abc123def456ghi789jkl012"
"""

    print("=" * 70)
    print("🔍 Scanning Dependencies (requirements.txt)")
    print("=" * 70)
    print()

    # Scan dependencies
    cve_findings, dependencies = agent.scan_dependencies(requirements_txt)

    print("=" * 70)
    print("🔍 Scanning Secrets (config.py)")
    print("=" * 70)
    print()

    # Scan secrets
    files = {"config.py": config_py}
    secret_findings = agent.scan_secrets(files)

    # Generate and print report
    report = agent.generate_report(
        cve_findings=cve_findings,
        secret_findings=secret_findings,
        dependencies_scanned=len(dependencies),
        files_scanned=len(files),
    )

    agent.print_report(report)

    # Summary statistics
    print()
    print("=" * 70)
    print("Scan Statistics")
    print("=" * 70)
    print()
    print("✅ Agent successfully:")
    print(f"   - Scanned {report.dependencies_scanned} dependencies")
    print(f"   - Found {len(report.cve_findings)} CVE vulnerabilities")
    print(f"   - Detected {len(report.secret_findings)} exposed secrets")
    print(f"   - Provided remediation from {len(agent.memory.recall([], '', top_k=100))} past experiences")
    print()
    print("💡 Multi-Modal Memory Benefits:")
    print("   - Procedural: Learned from 10 past upgrades/incidents")
    print("   - Semantic: 10 CVE/security facts")
    print("   - RLM Context: Pattern matching across files")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
