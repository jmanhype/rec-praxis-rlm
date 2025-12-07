"""Security Audit Agent - Week 4 Advanced Example.

This example demonstrates comprehensive security auditing using multi-modal memory:
1. Learn from past security fixes and vulnerabilities
2. Detect OWASP Top 10 vulnerabilities
3. Generate structured audit reports with remediation guidance
4. Map findings to compliance requirements

Usage:
    python examples/security_audit_agent.py
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig, RLMContext, FactStore


class Severity(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class OWASPCategory(Enum):
    """OWASP Top 10 categories."""
    A01_BROKEN_ACCESS_CONTROL = "A01:2021-Broken Access Control"
    A02_CRYPTOGRAPHIC_FAILURES = "A02:2021-Cryptographic Failures"
    A03_INJECTION = "A03:2021-Injection"
    A04_INSECURE_DESIGN = "A04:2021-Insecure Design"
    A05_SECURITY_MISCONFIGURATION = "A05:2021-Security Misconfiguration"
    A06_VULNERABLE_COMPONENTS = "A06:2021-Vulnerable and Outdated Components"
    A07_IDENTIFICATION_AUTH_FAILURES = "A07:2021-Identification and Authentication Failures"
    A08_SOFTWARE_DATA_INTEGRITY = "A08:2021-Software and Data Integrity Failures"
    A09_SECURITY_LOGGING = "A09:2021-Security Logging and Monitoring Failures"
    A10_SSRF = "A10:2021-Server-Side Request Forgery"


@dataclass
class Finding:
    """Security vulnerability finding."""
    title: str
    severity: Severity
    owasp_category: OWASPCategory
    file_path: str
    line_number: Optional[int]
    description: str
    remediation: str
    cwe_id: Optional[str] = None
    references: List[str] = field(default_factory=list)


@dataclass
class AuditReport:
    """Comprehensive security audit report."""
    application_name: str
    audit_date: str
    findings: List[Finding]
    summary: Dict[str, int]  # severity -> count
    owasp_coverage: Dict[str, int]  # category -> count
    compliance_notes: List[str]

    def __str__(self):
        """Generate human-readable report."""
        report = []
        report.append("=" * 70)
        report.append(f"Security Audit Report: {self.application_name}")
        report.append(f"Date: {self.audit_date}")
        report.append("=" * 70)

        report.append(f"\n📊 Summary:")
        for severity, count in sorted(self.summary.items(), reverse=True):
            report.append(f"   {severity}: {count}")

        report.append(f"\n🔍 OWASP Top 10 Coverage:")
        for category, count in self.owasp_coverage.items():
            if count > 0:
                report.append(f"   {category}: {count} finding(s)")

        report.append(f"\n🚨 Findings ({len(self.findings)} total):\n")
        for i, finding in enumerate(self.findings, 1):
            report.append(f"{i}. [{finding.severity.value}] {finding.title}")
            report.append(f"   File: {finding.file_path}")
            if finding.line_number:
                report.append(f"   Line: {finding.line_number}")
            report.append(f"   OWASP: {finding.owasp_category.value}")
            if finding.cwe_id:
                report.append(f"   CWE: {finding.cwe_id}")
            report.append(f"   Description: {finding.description}")
            report.append(f"   Remediation: {finding.remediation}")
            if finding.references:
                report.append(f"   References: {', '.join(finding.references)}")
            report.append("")

        if self.compliance_notes:
            report.append(f"\n📋 Compliance Notes:")
            for note in self.compliance_notes:
                report.append(f"   - {note}")

        report.append("\n" + "=" * 70)
        return "\n".join(report)


class SecurityAuditAgent:
    """Comprehensive security audit agent with multi-modal memory."""

    def __init__(self):
        """Initialize agent with memory and security knowledge."""
        self.memory = ProceduralMemory(
            config=MemoryConfig(
                storage_path=":memory:",
                env_weight=0.6,
                goal_weight=0.4,
            )
        )
        self.rlm = RLMContext()
        self.fact_store = FactStore(storage_path=":memory:")

        # Populate with past security fixes
        self._populate_security_history()

        # Load OWASP/security knowledge
        self._load_security_knowledge()

    def _populate_security_history(self):
        """Add past security fix experiences to memory."""
        base_time = time.time() - (180 * 24 * 3600)  # 180 days ago

        experiences = [
            # SQL Injection fixes
            Experience(
                env_features=["python", "flask", "database", "security", "sql-injection"],
                goal="fix SQL injection vulnerability in user search",
                action="Replaced string concatenation with parameterized query using cursor.execute(sql, params)",
                result="Vulnerability eliminated. Penetration test passed. No performance impact.",
                success=True,
                timestamp=base_time
            ),

            # Authentication vulnerabilities
            Experience(
                env_features=["python", "authentication", "security", "session"],
                goal="fix session fixation vulnerability",
                action="Implemented session regeneration after login using session.regenerate_id()",
                result="Session fixation attack prevented. OWASP compliance achieved.",
                success=True,
                timestamp=base_time + (10 * 86400)
            ),
            Experience(
                env_features=["python", "jwt", "authentication", "security"],
                goal="fix JWT token expiration issues",
                action="Added exp claim validation and 15-minute token expiration",
                result="Token hijacking window reduced from indefinite to 15 minutes.",
                success=True,
                timestamp=base_time + (20 * 86400)
            ),

            # XSS vulnerabilities
            Experience(
                env_features=["python", "flask", "xss", "security"],
                goal="prevent XSS in user-generated content",
                action="Enabled auto-escaping in Jinja2 templates and added Content-Security-Policy header",
                result="XSS attacks blocked. CSP violations logged for monitoring.",
                success=True,
                timestamp=base_time + (30 * 86400)
            ),

            # CSRF protection
            Experience(
                env_features=["python", "flask", "csrf", "security"],
                goal="add CSRF protection to state-changing endpoints",
                action="Implemented Flask-WTF CSRF tokens for all POST/PUT/DELETE requests",
                result="CSRF attacks prevented. User experience unchanged (hidden tokens).",
                success=True,
                timestamp=base_time + (40 * 86400)
            ),

            # Insecure deserialization
            Experience(
                env_features=["python", "pickle", "security", "deserialization"],
                goal="fix insecure pickle deserialization",
                action="Replaced pickle with JSON for session storage. Added input validation.",
                result="Remote code execution vulnerability eliminated. No compatibility issues.",
                success=True,
                timestamp=base_time + (50 * 86400)
            ),

            # Path traversal
            Experience(
                env_features=["python", "flask", "path-traversal", "security"],
                goal="prevent directory traversal in file downloads",
                action="Added os.path.abspath() and path prefix validation before serving files",
                result="Path traversal blocked. Users can only access intended directory.",
                success=True,
                timestamp=base_time + (60 * 86400)
            ),

            # SSRF vulnerabilities
            Experience(
                env_features=["python", "ssrf", "security", "http"],
                goal="prevent SSRF in URL fetch feature",
                action="Added allowlist for allowed domains and blocked private IP ranges (RFC1918)",
                result="SSRF to internal services prevented. Only public URLs allowed.",
                success=True,
                timestamp=base_time + (70 * 86400)
            ),

            # Weak cryptography
            Experience(
                env_features=["python", "cryptography", "security", "passwords"],
                goal="upgrade from MD5 to secure password hashing",
                action="Migrated to Argon2id with 16MB memory cost and 3 iterations",
                result="Passwords now resistant to GPU brute-force attacks. Migration completed for 1M users.",
                success=True,
                timestamp=base_time + (80 * 86400)
            ),

            # API rate limiting
            Experience(
                env_features=["python", "api", "security", "rate-limiting"],
                goal="prevent API abuse and DoS attacks",
                action="Implemented token bucket rate limiting with Redis backend (100 req/min per IP)",
                result="DDoS attacks mitigated. Legitimate users unaffected.",
                success=True,
                timestamp=base_time + (90 * 86400)
            ),
        ]

        for exp in experiences:
            self.memory.store(exp)
            # Extract security facts
            text = f"{exp.action}. {exp.result}"
            self.fact_store.extract_facts(text, source_id=f"security_{int(exp.timestamp)}")

    def _load_security_knowledge(self):
        """Load OWASP and security standards into FactStore."""
        # OWASP Top 10 mapping
        owasp_knowledge = [
            "A01 = Broken Access Control. CWE-200, CWE-284, CWE-285.",
            "A02 = Cryptographic Failures. CWE-259, CWE-327, CWE-331.",
            "A03 = Injection. CWE-79 (XSS), CWE-89 (SQLi), CWE-94.",
            "A07 = Identification and Authentication Failures. CWE-287, CWE-384.",
            "OWASP recommends minimum 15-minute JWT expiration for sensitive operations.",
            "CSP = Content-Security-Policy header prevents XSS attacks.",
            "Argon2id is OWASP recommended for password hashing as of 2023.",
        ]

        for knowledge in owasp_knowledge:
            self.fact_store.extract_facts(knowledge, source_id="owasp_standards")

    def audit_application(
        self,
        application_name: str,
        code_files: Dict[str, str]
    ) -> AuditReport:
        """
        Perform comprehensive security audit.

        Args:
            application_name: Name of application being audited
            code_files: Dict of {file_path: code_content}

        Returns:
            AuditReport with findings and remediation guidance
        """
        print(f"\n{'=' * 70}")
        print(f"🔒 Security Audit: {application_name}")
        print(f"{'=' * 70}\n")

        findings = []

        # Audit each file
        for file_path, code_content in code_files.items():
            print(f"Auditing: {file_path}")
            self.rlm.add_document(file_path, code_content)

            # Run vulnerability detectors
            findings.extend(self._detect_sql_injection(file_path))
            findings.extend(self._detect_weak_crypto(file_path))
            findings.extend(self._detect_xss(file_path))
            findings.extend(self._detect_csrf_missing(file_path))
            findings.extend(self._detect_insecure_deserialization(file_path))
            findings.extend(self._detect_path_traversal(file_path))
            findings.extend(self._detect_ssrf(file_path))
            findings.extend(self._detect_auth_issues(file_path))

        # Generate summary
        summary = {}
        for severity in Severity:
            summary[severity.value] = sum(1 for f in findings if f.severity == severity)

        # OWASP coverage
        owasp_coverage = {}
        for category in OWASPCategory:
            count = sum(1 for f in findings if f.owasp_category == category)
            if count > 0:
                owasp_coverage[category.value] = count

        # Compliance notes
        compliance_notes = self._generate_compliance_notes(findings)

        report = AuditReport(
            application_name=application_name,
            audit_date=time.strftime("%Y-%m-%d"),
            findings=findings,
            summary=summary,
            owasp_coverage=owasp_coverage,
            compliance_notes=compliance_notes
        )

        print(f"\n✅ Audit complete: {len(findings)} findings")
        return report

    def _detect_sql_injection(self, file_path: str) -> List[Finding]:
        """Detect SQL injection vulnerabilities."""
        findings = []

        # Pattern: String formatting/concatenation in SQL
        sql_patterns = self.rlm.grep(
            r"execute\s*\(\s*['\"].*(%s|%d|\+|\.format|f['\"])",
            doc_id=file_path
        )

        if sql_patterns:
            # Recall past SQL injection fixes
            past_fixes = self.memory.recall(
                env_features=["python", "database", "security", "sql-injection"],
                goal="fix SQL injection",
                top_k=2
            )

            remediation = "Use parameterized queries with placeholders"
            if past_fixes:
                latest_fix = past_fixes[-1]
                remediation = latest_fix.action[:100]

            findings.append(Finding(
                title="SQL Injection Vulnerability",
                severity=Severity.CRITICAL,
                owasp_category=OWASPCategory.A03_INJECTION,
                file_path=file_path,
                line_number=None,
                description=f"Found {len(sql_patterns)} SQL queries using string formatting/concatenation",
                remediation=remediation,
                cwe_id="CWE-89",
                references=["https://owasp.org/www-community/attacks/SQL_Injection"]
            ))

        return findings

    def _detect_weak_crypto(self, file_path: str) -> List[Finding]:
        """Detect weak cryptographic practices."""
        findings = []

        # Weak hashing algorithms
        weak_hash = self.rlm.grep(r"hashlib\.(md5|sha1)\(", doc_id=file_path)

        if weak_hash:
            # Recall crypto upgrades
            past_fixes = self.memory.recall(
                env_features=["python", "cryptography", "security", "passwords"],
                goal="upgrade password hashing",
                top_k=1
            )

            remediation = "Use Argon2id, bcrypt, or scrypt for password hashing"
            if past_fixes:
                remediation = past_fixes[-1].action[:100]

            findings.append(Finding(
                title="Weak Cryptographic Algorithm",
                severity=Severity.HIGH,
                owasp_category=OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES,
                file_path=file_path,
                line_number=None,
                description=f"Found {len(weak_hash)} uses of MD5/SHA1 for password hashing",
                remediation=remediation,
                cwe_id="CWE-327",
                references=["https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html"]
            ))

        return findings

    def _detect_xss(self, file_path: str) -> List[Finding]:
        """Detect Cross-Site Scripting vulnerabilities."""
        findings = []

        # Unsafe template rendering
        unsafe_render = self.rlm.grep(r"render_template_string\(.*\+", doc_id=file_path)
        unsafe_safe = self.rlm.grep(r"\|\s*safe", doc_id=file_path)

        if unsafe_render or unsafe_safe:
            # Recall XSS fixes
            past_fixes = self.memory.recall(
                env_features=["python", "flask", "xss", "security"],
                goal="prevent XSS",
                top_k=1
            )

            remediation = "Enable auto-escaping and add Content-Security-Policy header"
            if past_fixes:
                remediation = past_fixes[-1].action[:100]

            findings.append(Finding(
                title="Cross-Site Scripting (XSS) Risk",
                severity=Severity.HIGH,
                owasp_category=OWASPCategory.A03_INJECTION,
                file_path=file_path,
                line_number=None,
                description="Found unsafe template rendering or |safe filter usage",
                remediation=remediation,
                cwe_id="CWE-79",
                references=["https://owasp.org/www-community/attacks/xss/"]
            ))

        return findings

    def _detect_csrf_missing(self, file_path: str) -> List[Finding]:
        """Detect missing CSRF protection."""
        findings = []

        # State-changing routes without CSRF protection
        post_routes = self.rlm.grep(r"@app\.route\(.*methods.*POST", doc_id=file_path)
        csrf_protection = self.rlm.grep(r"csrf", doc_id=file_path)

        if post_routes and not csrf_protection:
            past_fixes = self.memory.recall(
                env_features=["python", "flask", "csrf", "security"],
                goal="add CSRF protection",
                top_k=1
            )

            remediation = "Implement CSRF tokens for all state-changing requests"
            if past_fixes:
                remediation = past_fixes[-1].action[:100]

            findings.append(Finding(
                title="Missing CSRF Protection",
                severity=Severity.MEDIUM,
                owasp_category=OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
                file_path=file_path,
                line_number=None,
                description=f"Found {len(post_routes)} POST routes without CSRF protection",
                remediation=remediation,
                cwe_id="CWE-352",
                references=["https://owasp.org/www-community/attacks/csrf"]
            ))

        return findings

    def _detect_insecure_deserialization(self, file_path: str) -> List[Finding]:
        """Detect insecure deserialization."""
        findings = []

        pickle_loads = self.rlm.grep(r"pickle\.loads?\(", doc_id=file_path)

        if pickle_loads:
            past_fixes = self.memory.recall(
                env_features=["python", "pickle", "security", "deserialization"],
                goal="fix insecure deserialization",
                top_k=1
            )

            remediation = "Replace pickle with JSON or use signed serialization"
            if past_fixes:
                remediation = past_fixes[-1].action[:100]

            findings.append(Finding(
                title="Insecure Deserialization",
                severity=Severity.CRITICAL,
                owasp_category=OWASPCategory.A08_SOFTWARE_DATA_INTEGRITY,
                file_path=file_path,
                line_number=None,
                description=f"Found {len(pickle_loads)} uses of pickle deserialization",
                remediation=remediation,
                cwe_id="CWE-502",
                references=["https://owasp.org/www-community/vulnerabilities/Deserialization_of_untrusted_data"]
            ))

        return findings

    def _detect_path_traversal(self, file_path: str) -> List[Finding]:
        """Detect path traversal vulnerabilities."""
        findings = []

        # File operations with user input
        file_ops = self.rlm.grep(r"open\(.*request\.", doc_id=file_path)
        send_file = self.rlm.grep(r"send_file\(.*request\.", doc_id=file_path)

        if file_ops or send_file:
            past_fixes = self.memory.recall(
                env_features=["python", "flask", "path-traversal", "security"],
                goal="prevent directory traversal",
                top_k=1
            )

            remediation = "Validate file paths with os.path.abspath() and allowlist"
            if past_fixes:
                remediation = past_fixes[-1].action[:100]

            findings.append(Finding(
                title="Path Traversal Vulnerability",
                severity=Severity.HIGH,
                owasp_category=OWASPCategory.A01_BROKEN_ACCESS_CONTROL,
                file_path=file_path,
                line_number=None,
                description="Found file operations using unsanitized user input",
                remediation=remediation,
                cwe_id="CWE-22",
                references=["https://owasp.org/www-community/attacks/Path_Traversal"]
            ))

        return findings

    def _detect_ssrf(self, file_path: str) -> List[Finding]:
        """Detect Server-Side Request Forgery."""
        findings = []

        # HTTP requests with user-controlled URLs
        http_requests = self.rlm.grep(r"requests\.(get|post)\(.*request\.", doc_id=file_path)

        if http_requests:
            past_fixes = self.memory.recall(
                env_features=["python", "ssrf", "security", "http"],
                goal="prevent SSRF",
                top_k=1
            )

            remediation = "Validate URLs against allowlist and block private IP ranges"
            if past_fixes:
                remediation = past_fixes[-1].action[:100]

            findings.append(Finding(
                title="Server-Side Request Forgery (SSRF)",
                severity=Severity.HIGH,
                owasp_category=OWASPCategory.A10_SSRF,
                file_path=file_path,
                line_number=None,
                description=f"Found {len(http_requests)} HTTP requests with user-controlled URLs",
                remediation=remediation,
                cwe_id="CWE-918",
                references=["https://owasp.org/www-community/attacks/Server_Side_Request_Forgery"]
            ))

        return findings

    def _detect_auth_issues(self, file_path: str) -> List[Finding]:
        """Detect authentication/authorization issues."""
        findings = []

        # JWT without expiration
        jwt_decode = self.rlm.grep(r"jwt\.decode\(", doc_id=file_path)
        verify_exp = self.rlm.grep(r"verify_exp", doc_id=file_path)

        if jwt_decode and not verify_exp:
            past_fixes = self.memory.recall(
                env_features=["python", "jwt", "authentication", "security"],
                goal="fix JWT expiration",
                top_k=1
            )

            remediation = "Add exp claim validation with 15-minute expiration"
            if past_fixes:
                remediation = past_fixes[-1].action[:100]

            findings.append(Finding(
                title="JWT Token Without Expiration",
                severity=Severity.MEDIUM,
                owasp_category=OWASPCategory.A07_IDENTIFICATION_AUTH_FAILURES,
                file_path=file_path,
                line_number=None,
                description="JWT tokens decoded without expiration verification",
                remediation=remediation,
                cwe_id="CWE-613",
                references=["https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html"]
            ))

        return findings

    def _generate_compliance_notes(self, findings: List[Finding]) -> List[str]:
        """Generate compliance-related notes."""
        notes = []

        # Check for critical/high severity
        critical_count = sum(1 for f in findings if f.severity == Severity.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == Severity.HIGH)

        if critical_count > 0:
            notes.append(f"❌ {critical_count} CRITICAL findings must be fixed before production deployment")

        if high_count > 0:
            notes.append(f"⚠️  {high_count} HIGH severity findings should be addressed within 30 days")

        # GDPR compliance
        crypto_issues = sum(1 for f in findings if f.owasp_category == OWASPCategory.A02_CRYPTOGRAPHIC_FAILURES)
        if crypto_issues > 0:
            notes.append(f"🔒 {crypto_issues} cryptographic issues may affect GDPR Article 32 compliance")

        # OWASP Top 10 coverage
        owasp_categories = set(f.owasp_category for f in findings)
        notes.append(f"📊 Audit covered {len(owasp_categories)}/10 OWASP Top 10 categories")

        return notes


def main():
    """Run security audit demonstration."""
    print("\n" + "=" * 70)
    print("Security Audit Agent - Multi-Modal Memory Demo")
    print("=" * 70)

    agent = SecurityAuditAgent()

    print(f"\n📚 Memory initialized:")
    print(f"   - {agent.memory.size()} past security fixes")
    print(f"   - {agent.fact_store.count_facts()} security facts/standards")

    # Sample vulnerable application
    vulnerable_app = {
        "app.py": '''
import hashlib
import pickle
from flask import Flask, request, render_template_string
import requests
import jwt

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    # Weak crypto - MD5 for passwords
    password_hash = hashlib.md5(password.encode()).hexdigest()

    # SQL Injection vulnerability
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password_hash}'"
    cursor.execute(query)

    user = cursor.fetchone()
    if user:
        # JWT without expiration
        token = jwt.encode({'user_id': user[0]}, 'secret', algorithm='HS256')
        return {"token": token}

    return {"error": "Invalid credentials"}, 401

@app.route('/download', methods=['GET'])
def download_file():
    # Path traversal vulnerability
    filename = request.args.get('file')
    return send_file(f"/uploads/{filename}")

@app.route('/fetch', methods=['GET'])
def fetch_url():
    # SSRF vulnerability
    url = request.args.get('url')
    response = requests.get(url)
    return response.text

@app.route('/render', methods=['POST'])
def render_template():
    # XSS vulnerability
    template = request.form['template']
    return render_template_string(template + " Welcome!")

@app.route('/session', methods=['POST'])
def load_session():
    # Insecure deserialization
    session_data = request.form['data']
    session = pickle.loads(bytes.fromhex(session_data))
    return {"session": session}

@app.route('/update_profile', methods=['POST'])
def update_profile():
    # Missing CSRF protection
    user_id = request.form['user_id']
    email = request.form['email']
    # Update logic here
    return {"success": True}
''',
    }

    # Perform audit
    report = agent.audit_application(
        application_name="Sample Vulnerable Flask App",
        code_files=vulnerable_app
    )

    # Print report
    print(report)

    # Summary statistics
    print("\n" + "=" * 70)
    print("Audit Statistics")
    print("=" * 70)
    print(f"\n✅ Agent successfully:")
    print(f"   - Detected {len(report.findings)} vulnerabilities")
    print(f"   - Mapped findings to {len(report.owasp_coverage)} OWASP categories")
    print(f"   - Provided remediation from {agent.memory.size()} past fixes")
    print(f"   - Generated {len(report.compliance_notes)} compliance notes")

    print(f"\n💡 Multi-Modal Memory Benefits:")
    print(f"   - Procedural: Learned from {agent.memory.size()} security fixes")
    print(f"   - Semantic: {agent.fact_store.count_facts()} OWASP/CWE facts")
    print(f"   - RLM Context: Pattern matching across codebase")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
