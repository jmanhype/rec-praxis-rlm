# Examples

This directory contains practical examples demonstrating rec-praxis-rlm features.

## Quick Start Examples

| File | Description | Features Demonstrated |
|------|-------------|----------------------|
| `quickstart.py` | Basic memory and context usage | ProceduralMemory, RLMContext, basic operations |
| `log_analyzer.py` | Log analysis with RLM context | Document inspection, grep, peek, tail |
| `web_agent.py` | Web scraping with procedural memory | Experience storage, recall, success tracking |

## Security & Code Quality

| File | Description | Features Demonstrated |
|------|-------------|----------------------|
| `code_review_agent.py` | Intelligent code review | Code quality detection, procedural memory learning |
| `security_audit_agent.py` | OWASP-based security auditing | Vulnerability detection, CWE mapping |
| `dependency_scan_agent.py` | CVE and secret scanning | Dependency analysis, credential detection |

## Advanced Features

| File | Description | Features Demonstrated |
|------|-------------|----------------------|
| `optimization.py` | DSPy MIPROv2 optimizer usage | Autonomous agent optimization, metrics tracking |

## Running Examples

```bash
# Install dependencies
pip install rec-praxis-rlm[all]

# Run any example
python examples/quickstart.py
python examples/code_review_agent.py src/myapp.py
python examples/security_audit_agent.py src/myapp.py
```

## Configuration

Most examples support environment variables for API keys:

```bash
# For DSPy-based examples (optimization.py)
export GROQ_API_KEY="gsk-..."  # Groq (recommended - free)
# OR
export OPENAI_API_KEY="sk-..."  # OpenAI
# OR
export OPENROUTER_API_KEY="sk-or-..."  # OpenRouter
```

## Example Output

### Code Review
```bash
$ python examples/code_review_agent.py src/app.py

🔍 Code Review Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 src/app.py

🔴 CRITICAL: Hardcoded Credentials (line 45)
   Hardcoded API key found in source code
   Remediation: Use environment variables: os.getenv('API_KEY')

🟠 HIGH: SQL Injection Risk (line 78)
   Potential SQL injection: String concatenation in execute()
   Remediation: Use parameterized queries: execute('SELECT * FROM users WHERE id=?', (user_id,))
```

### Security Audit
```bash
$ python examples/security_audit_agent.py src/auth.py

🔒 Security Audit Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Findings: 3
Blocking (CRITICAL/HIGH): 1

✅ PASSED: No unsafe eval/exec detected
✅ PASSED: No pickle usage found
🔴 FAILED: Weak password hashing (MD5 detected)
```

## Contributing

To add new examples:

1. Create a new `.py` file in this directory
2. Add docstring at the top explaining what it demonstrates
3. Include example output in comments
4. Update this README with the new example
5. Ensure it works with `python examples/your_example.py`

See [CONTRIBUTING.md](../CONTRIBUTING.md) for full guidelines.
