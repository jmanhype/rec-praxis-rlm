# CLI Reference

Complete reference for rec-praxis-rlm command-line tools.

## rec-praxis-review

Intelligent code review tool with quality detection.

### Syntax

```bash
rec-praxis-review [FILES...] [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--severity` | Filter by severity (CRITICAL, HIGH, MEDIUM, LOW, INFO) | All |
| `--format` | Output format (human, json, html, sarif) | human |
| `--output` | Output file path (for html/sarif formats) | stdout |
| `--config` | Config file path (YAML/JSON) | None |

### Examples

```bash
# Review all Python files
rec-praxis-review src/**/*.py

# Only high-severity issues
rec-praxis-review src/**/*.py --severity=HIGH

# JSON output for IDE integration
rec-praxis-review src/**/*.py --format=json > findings.json

# Interactive HTML report
rec-praxis-review src/**/*.py --format=html --output=report.html

# SARIF for GitHub Security tab
rec-praxis-review src/**/*.py --format=sarif > results.sarif

# Multiple files
rec-praxis-review app.py utils.py models.py
```

### Exit Codes

- `0` - No issues found
- `1` - Issues found (any severity)
- `2` - Command error (invalid arguments, missing files)

### Output Formats

#### Human (Default)

```
🔍 Code Review Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 src/app.py

🔴 CRITICAL: Hardcoded Credentials (line 45)
   Hardcoded API key found in source code
   Remediation: Use environment variables: os.getenv('API_KEY')

🟠 HIGH: SQL Injection Risk (line 78)
   Potential SQL injection: String concatenation in execute()
   Remediation: Use parameterized queries
```

#### JSON

```json
{
  "files": [
    {
      "path": "src/app.py",
      "findings": [
        {
          "severity": "CRITICAL",
          "type": "Hardcoded Credentials",
          "line": 45,
          "message": "Hardcoded API key found",
          "remediation": "Use environment variables"
        }
      ]
    }
  ],
  "summary": {
    "total_files": 1,
    "total_findings": 2,
    "by_severity": {
      "CRITICAL": 1,
      "HIGH": 1
    }
  }
}
```

#### HTML

Interactive report with:
- Severity distribution pie chart
- File-by-file findings
- Copy-to-clipboard remediation
- Filter by severity
- Search functionality

#### SARIF

GitHub-compatible SARIF format for Security tab integration.

---

## rec-praxis-audit

OWASP-based security auditing tool.

### Syntax

```bash
rec-praxis-audit FILE [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--fail-on` | Exit with error on severity (CRITICAL, HIGH) | None |
| `--format` | Output format (human, json, sarif) | human |
| `--output` | Output file path | stdout |
| `--checks` | Specific checks to run (comma-separated) | All |

### Examples

```bash
# Audit a file
rec-praxis-audit app.py

# Fail on critical issues (for CI/CD)
rec-praxis-audit app.py --fail-on=CRITICAL

# JSON output
rec-praxis-audit app.py --format=json

# Run specific checks
rec-praxis-audit app.py --checks=sql_injection,xss,hardcoded_secrets
```

### Security Checks

| Check | Description | Severity |
|-------|-------------|----------|
| `sql_injection` | SQL injection vulnerabilities | CRITICAL |
| `xss` | Cross-site scripting risks | HIGH |
| `hardcoded_secrets` | Hardcoded passwords, API keys | CRITICAL |
| `command_injection` | OS command injection | CRITICAL |
| `path_traversal` | Path traversal vulnerabilities | HIGH |
| `insecure_deserialization` | Unsafe pickle/yaml usage | CRITICAL |
| `weak_crypto` | MD5, SHA1, weak encryption | MEDIUM |
| `unsafe_eval` | eval(), exec() usage | HIGH |

### Exit Codes

- `0` - No issues found (or below fail-on threshold)
- `1` - Issues found at or above fail-on severity
- `2` - Command error

### Output Example

```
🔒 Security Audit Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Findings: 3
Blocking (CRITICAL/HIGH): 1

✅ PASSED: No unsafe eval/exec detected
✅ PASSED: No pickle usage found
🔴 FAILED: Weak password hashing (MD5 detected)

📋 Detailed Findings:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 CRITICAL: Hardcoded API Key (line 23)
   API_KEY = "sk-1234567890abcdef"
   → Use environment variables: os.getenv('API_KEY')

🟠 HIGH: SQL Injection Risk (line 45)
   cursor.execute(f"SELECT * FROM users WHERE id={user_id}")
   → Use parameterized queries
```

---

## rec-praxis-deps

Dependency and secret scanning tool.

### Syntax

```bash
rec-praxis-deps [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--requirements` | Path to requirements.txt | None |
| `--files` | Python files to scan (glob pattern) | None |
| `--check-cves` | Check dependencies for known CVEs | False |
| `--check-secrets` | Scan for hardcoded secrets | False |
| `--format` | Output format (human, json) | human |
| `--output` | Output file path | stdout |

### Examples

```bash
# Scan requirements.txt
rec-praxis-deps --requirements=requirements.txt

# Scan files for imports
rec-praxis-deps --files src/**/*.py

# Check for CVEs
rec-praxis-deps --requirements=requirements.txt --check-cves

# Check for secrets
rec-praxis-deps --files src/**/*.py --check-secrets

# Full scan
rec-praxis-deps \
  --requirements=requirements.txt \
  --files src/**/*.py \
  --check-cves \
  --check-secrets \
  --format=json
```

### Output Example

```
📦 Dependency Scan Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Requirements.txt:
  ✅ requests==2.31.0 (no known CVEs)
  ⚠️  urllib3==1.26.5 (CVE-2023-43804: High severity)
  ✅ pydantic==2.4.2 (no known CVEs)

Secrets Found:
  🔴 AWS Access Key (src/config.py:15)
  🔴 GitHub Token (src/deploy.py:42)

Recommendations:
  1. Update urllib3 to >=2.0.7
  2. Move credentials to environment variables
  3. Add .env to .gitignore
```

---

## Global Options

All CLI tools support:

```bash
--help          Show help message
--version       Show version number
--verbose       Enable verbose logging
--quiet         Suppress all output except errors
--no-color      Disable colored output
```

---

## Configuration Files

All tools support YAML/JSON configuration files:

### Example config.yaml

```yaml
severity: HIGH
format: json
checks:
  - sql_injection
  - xss
  - hardcoded_secrets
exclude_patterns:
  - "test_*.py"
  - "*/migrations/*"
custom_patterns:
  - pattern: "password\s*=\s*['\"]"
    severity: CRITICAL
    message: "Hardcoded password detected"
```

### Usage

```bash
rec-praxis-review src/**/*.py --config=config.yaml
rec-praxis-audit app.py --config=config.yaml
```

---

## CI/CD Integration

### GitHub Actions

See [GitHub Actions Integration](integrations.md#github-actions) for using the official [rec-praxis-action](https://github.com/jmanhype/rec-praxis-action).

### Manual CLI Integration

```yaml
# .github/workflows/security.yml
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install rec-praxis-rlm
        run: pip install rec-praxis-rlm

      - name: Code Review
        run: rec-praxis-review src/**/*.py --severity=HIGH

      - name: Security Audit
        run: rec-praxis-audit src/app.py --fail-on=CRITICAL

      - name: Dependency Scan
        run: rec-praxis-deps --requirements=requirements.txt --check-cves
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: rec-praxis-review
        name: Code Review
        entry: rec-praxis-review
        language: system
        types: [python]
        args: [--severity=HIGH]

      - id: rec-praxis-audit
        name: Security Audit
        entry: rec-praxis-audit
        language: system
        types: [python]
        args: [--fail-on=CRITICAL]
```

---

## Exit Codes Summary

| Code | Meaning | When to Use |
|------|---------|-------------|
| 0 | Success | No issues or below threshold |
| 1 | Findings | Issues detected |
| 2 | Error | Invalid arguments, missing files |

Use exit codes in CI/CD:

```bash
# Fail build on high-severity issues
rec-praxis-review src/**/*.py --severity=HIGH || exit 1

# Fail on critical security issues
rec-praxis-audit app.py --fail-on=CRITICAL || exit 1
```
