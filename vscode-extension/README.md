# rec-praxis-rlm VS Code Extension

Intelligent code review, security auditing, and dependency scanning powered by procedural memory - right inside Visual Studio Code.

## Features

### 🔍 Code Review
- **Learn from past fixes**: The agent remembers successful code improvements
- **Context-aware suggestions**: Recommendations based on your team's coding patterns
- **Inline diagnostics**: See issues directly in your code with squiggly underlines

### 🔒 Security Audit
- **OWASP Top 10 detection**: Automatically finds SQL injection, XSS, weak crypto, etc.
- **CWE mapping**: Issues are tagged with industry-standard CWE identifiers
- **Experience-based remediation**: Fix suggestions based on past successful security patches

### 📦 Dependency Scanning
- **CVE detection**: Find vulnerabilities in your Python dependencies
- **Secret scanning**: Detect exposed API keys, tokens, and credentials
- **Entropy analysis**: Reduce false positives with Shannon entropy calculation

## Installation

1. Install the extension from VS Code Marketplace
2. Install rec-praxis-rlm Python package:
   ```bash
   pip install rec-praxis-rlm[all]
   ```

## Usage

### Code Review
- **Right-click** in a Python file → "REC Praxis: Review Current File"
- **Auto-review on save**: Enable in settings for real-time feedback
- **Command Palette**: `Ctrl+Shift+P` → "REC Praxis: Review Current File"

### Security Audit
- **Right-click** in a Python file → "REC Praxis: Security Audit Current File"
- **Command Palette**: `Ctrl+Shift+P` → "REC Praxis: Security Audit Current File"

### Dependency Scan
- **Right-click** on `requirements.txt` → "REC Praxis: Scan Dependencies"
- **Command Palette**: `Ctrl+Shift+P` → "REC Praxis: Scan Dependencies"

### Workspace Review
- **Command Palette**: `Ctrl+Shift+P` → "REC Praxis: Review Entire Workspace"
- Reviews all Python files in your workspace

## Configuration

Open VS Code settings and search for "rec-praxis-rlm":

```json
{
  "rec-praxis-rlm.pythonPath": "python",
  "rec-praxis-rlm.codeReview.severity": "HIGH",
  "rec-praxis-rlm.securityAudit.failOn": "CRITICAL",
  "rec-praxis-rlm.enableDiagnostics": true,
  "rec-praxis-rlm.memoryDir": ".rec-praxis-rlm"
}
```

### Settings

- **pythonPath**: Path to Python interpreter with rec-praxis-rlm installed
- **codeReview.severity**: Minimum severity to show (LOW, MEDIUM, HIGH, CRITICAL)
- **securityAudit.failOn**: Minimum severity to fail on (LOW, MEDIUM, HIGH, CRITICAL)
- **enableDiagnostics**: Show inline diagnostics in code
- **memoryDir**: Directory for procedural memory storage (shared across runs)

## How It Works

### Procedural Memory
Unlike traditional linters that use fixed rules, rec-praxis-rlm **learns from your team's fixes**:

1. **First review**: Detects SQL injection, suggests parameterized queries
2. **You fix it**: Replace f-strings with `cursor.execute()` and params
3. **Agent stores experience**: "SQL injection → parameterized queries = success"
4. **Next time**: Agent recalls this experience and provides the same fix faster

### Multi-Modal Memory
- **Procedural**: Past code reviews and security fixes
- **Semantic**: OWASP categories, CWE identifiers, CVE data
- **RLM Context**: Pattern matching in your actual code

## Example Workflow

1. **Write code** with potential security issue:
   ```python
   cursor.execute(f"SELECT * FROM users WHERE id={user_id}")
   ```

2. **Save file** → Extension auto-reviews

3. **See inline diagnostic**:
   ```
   HIGH: SQL Injection Risk
   Issue: String concatenation in SQL query
   Fix: Use parameterized queries with %s placeholders
   ```

4. **Apply fix**:
   ```python
   cursor.execute("SELECT * FROM users WHERE id=%s", (user_id,))
   ```

5. **Agent learns** from this fix for future reviews

## Requirements

- VS Code 1.85.0 or higher
- Python 3.10 or higher
- rec-praxis-rlm Python package installed

## Development

To contribute to this extension:

```bash
cd vscode-extension
npm install
npm run compile
# Press F5 in VS Code to launch Extension Development Host
```

## License

MIT

## Links

- [GitHub Repository](https://github.com/jmanhype/rec-praxis-rlm)
- [PyPI Package](https://pypi.org/project/rec-praxis-rlm/)
- [Documentation](https://github.com/jmanhype/rec-praxis-rlm#readme)
- [Report Issues](https://github.com/jmanhype/rec-praxis-rlm/issues)
