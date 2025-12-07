"""Code Review Agent - Week 3 Dogfooding Example.

This example demonstrates using rec-praxis-rlm for code review tasks:
1. Learn from past code review experiences
2. Detect common anti-patterns using RLM Context
3. Provide context-aware suggestions based on experience

Usage:
    python examples/code_review_agent.py
"""

import time
from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig, RLMContext, FactStore


class CodeReviewAgent:
    """Agent that learns from code review experiences."""

    def __init__(self):
        """Initialize agent with memory and context."""
        self.memory = ProceduralMemory(
            config=MemoryConfig(
                storage_path=":memory:",
                env_weight=0.6,
                goal_weight=0.4,
            )
        )
        self.rlm = RLMContext()
        self.fact_store = FactStore(storage_path=":memory:")

        # Populate with past review experiences
        self._populate_review_history()

    def _populate_review_history(self):
        """Add past code review experiences to memory."""
        base_time = time.time() - (90 * 24 * 3600)  # 90 days ago

        experiences = [
            # SQL Injection reviews
            Experience(
                env_features=["python", "flask", "database", "security"],
                goal="review authentication code for SQL injection",
                action="Found raw SQL with string formatting: cursor.execute(f'SELECT * FROM users WHERE id={user_id}')",
                result="Suggested parameterized queries. Developer fixed with cursor.execute('SELECT * FROM users WHERE id=?', (user_id,))",
                success=True,
                timestamp=base_time
            ),
            Experience(
                env_features=["python", "django", "database", "security"],
                goal="review user query endpoint",
                action="Found SQL injection via .raw() method with f-string",
                result="Recommended Django ORM QuerySet. Developer refactored to User.objects.filter(id=user_id)",
                success=True,
                timestamp=base_time + (10 * 86400)
            ),

            # Authentication reviews
            Experience(
                env_features=["python", "flask", "authentication"],
                goal="review password storage",
                action="Found plain MD5 hashing: hashlib.md5(password.encode()).hexdigest()",
                result="Suggested bcrypt. Developer implemented bcrypt.hashpw(password, bcrypt.gensalt())",
                success=True,
                timestamp=base_time + (20 * 86400)
            ),
            Experience(
                env_features=["python", "authentication", "security"],
                goal="review password validation",
                action="Found weak password check: len(password) >= 6",
                result="Suggested comprehensive validation (length, complexity, common passwords). Implemented with password_validator library",
                success=True,
                timestamp=base_time + (30 * 86400)
            ),

            # Error handling reviews
            Experience(
                env_features=["python", "error-handling"],
                goal="review exception handling",
                action="Found bare except: block swallowing all exceptions",
                result="Suggested specific exception types. Developer changed to 'except (ValueError, KeyError) as e:'",
                success=True,
                timestamp=base_time + (40 * 86400)
            ),
            Experience(
                env_features=["python", "logging", "error-handling"],
                goal="review error logging",
                action="Found print() for error logging in production code",
                result="Suggested logging module. Developer implemented logger.error() with proper formatting",
                success=True,
                timestamp=base_time + (50 * 86400)
            ),

            # Hardcoded credentials
            Experience(
                env_features=["python", "security", "credentials"],
                goal="review configuration management",
                action="Found hardcoded API key: api_key = 'sk-1234567890abcdef'",
                result="Suggested environment variables. Developer moved to os.getenv('API_KEY')",
                success=True,
                timestamp=base_time + (60 * 86400)
            ),

            # Race conditions
            Experience(
                env_features=["python", "threading", "concurrency"],
                goal="review concurrent file access",
                action="Found race condition in file writing without locks",
                result="Suggested threading.Lock(). Developer implemented lock-protected critical section",
                success=True,
                timestamp=base_time + (70 * 86400)
            ),
        ]

        for exp in experiences:
            self.memory.store(exp)

            # Extract coding standards as facts
            text = f"{exp.action}. {exp.result}"
            self.fact_store.extract_facts(text, source_id=f"review_{int(exp.timestamp)}")

    def review_code(self, file_path: str, code_content: str, language: str = "python") -> dict:
        """Review code file and provide suggestions.

        Args:
            file_path: Path to the code file being reviewed
            code_content: Content of the code file
            language: Programming language (default: python)

        Returns:
            dict with detected_issues, suggestions, and past_experiences
        """
        print(f"\n{'=' * 70}")
        print(f"Reviewing: {file_path}")
        print(f"{'=' * 70}\n")

        # Add code to RLM context
        self.rlm.add_document(file_path, code_content)

        # Detect common anti-patterns
        detected_issues = []

        # 1. SQL Injection patterns
        sql_injection = self.rlm.grep(
            r"execute\s*\(\s*['\"].*%s|execute\s*\(\s*f['\"]|\.format\(",
            doc_id=file_path
        )
        if sql_injection:
            detected_issues.append({
                "type": "SQL Injection Risk",
                "severity": "HIGH",
                "matches": len(sql_injection),
                "pattern": "String formatting in SQL queries"
            })

        # 2. Hardcoded credentials
        credentials = self.rlm.grep(
            r"(password|api_key|secret)\s*=\s*['\"][^'\"]+['\"]",
            doc_id=file_path
        )
        if credentials:
            detected_issues.append({
                "type": "Hardcoded Credentials",
                "severity": "HIGH",
                "matches": len(credentials),
                "pattern": "Credentials in source code"
            })

        # 3. Bare except blocks
        bare_except = self.rlm.grep(r"except\s*:", doc_id=file_path)
        if bare_except:
            detected_issues.append({
                "type": "Overly Broad Exception Handling",
                "severity": "MEDIUM",
                "matches": len(bare_except),
                "pattern": "Bare except: blocks"
            })

        # 4. MD5/SHA1 usage (weak hashing)
        weak_hash = self.rlm.grep(r"hashlib\.(md5|sha1)\(", doc_id=file_path)
        if weak_hash:
            detected_issues.append({
                "type": "Weak Cryptography",
                "severity": "HIGH",
                "matches": len(weak_hash),
                "pattern": "MD5/SHA1 for password hashing"
            })

        # 5. Print statements (should use logging)
        print_debug = self.rlm.grep(r"print\s*\(", doc_id=file_path)
        if print_debug:
            detected_issues.append({
                "type": "Debug Print Statements",
                "severity": "LOW",
                "matches": len(print_debug),
                "pattern": "print() instead of logging"
            })

        print(f"🔍 Detected {len(detected_issues)} potential issues:\n")
        for issue in detected_issues:
            print(f"   [{issue['severity']}] {issue['type']}")
            print(f"   - {issue['matches']} occurrence(s): {issue['pattern']}\n")

        # Recall relevant past reviews
        suggestions = []

        for issue in detected_issues:
            # Build search query based on issue type
            env_features = [language, "security"]
            goal = f"review code for {issue['type'].lower()}"

            if "SQL Injection" in issue['type']:
                env_features.extend(["database", "sql"])
            elif "Credentials" in issue['type']:
                env_features.extend(["credentials", "configuration"])
            elif "Exception" in issue['type']:
                env_features.extend(["error-handling"])
            elif "Cryptography" in issue['type']:
                env_features.extend(["authentication", "hashing"])
            elif "Print" in issue['type']:
                env_features.extend(["logging"])

            # Recall past experiences
            past_reviews = self.memory.recall(
                env_features=env_features,
                goal=goal,
                top_k=2
            )

            if past_reviews:
                latest_review = past_reviews[-1]  # Most recent

                suggestion = {
                    "issue_type": issue['type'],
                    "severity": issue['severity'],
                    "recommendation": latest_review.result,
                    "similar_fix": latest_review.action,
                    "context": f"Based on similar review from {int((time.time() - latest_review.timestamp) / 86400)} days ago"
                }
                suggestions.append(suggestion)

        print(f"\n💡 Suggestions based on past reviews:\n")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion['issue_type']} ({suggestion['severity']})")
            print(f"   Similar issue: {suggestion['similar_fix'][:80]}...")
            print(f"   Fix: {suggestion['recommendation'][:100]}...")
            print(f"   {suggestion['context']}\n")

        return {
            "file_path": file_path,
            "detected_issues": detected_issues,
            "suggestions": suggestions,
            "num_issues": len(detected_issues),
            "num_suggestions": len(suggestions),
        }


def main():
    """Run code review agent demonstration."""
    print("\n" + "=" * 70)
    print("Code Review Agent - Learning from Experience")
    print("=" * 70)

    agent = CodeReviewAgent()

    print(f"\n📚 Memory initialized with {agent.memory.size()} past code reviews")
    print(f"📝 Fact store contains {agent.fact_store.count_facts()} extracted coding standards\n")

    # Sample code with various issues
    sample_code = '''
import hashlib
from flask import Flask, request
import sqlite3

app = Flask(__name__)

# Configuration
API_KEY = "sk-1234567890abcdef"  # Hardcoded API key
DB_PASSWORD = "admin123"  # Hardcoded password

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    # Weak password hashing
    password_hash = hashlib.md5(password.encode()).hexdigest()

    # SQL Injection vulnerability
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password_hash}'"
    cursor.execute(query)

    user = cursor.fetchone()

    try:
        # Process user login
        if user:
            print(f"User {username} logged in")  # Should use logging
            return {"status": "success"}
    except:  # Bare except
        print("Login failed")
        return {"status": "error"}

@app.route('/api/data')
def get_data():
    # Another SQL injection
    user_id = request.args.get('id')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM data WHERE user_id={user_id}")
    return cursor.fetchall()
'''

    # Review the code
    result = agent.review_code("login_service.py", sample_code, language="python")

    print("\n" + "=" * 70)
    print("Review Summary")
    print("=" * 70)
    print(f"\n✅ Review complete!")
    print(f"   - Issues detected: {result['num_issues']}")
    print(f"   - Suggestions provided: {result['num_suggestions']}")
    print(f"   - Leveraged {agent.memory.size()} past review experiences")
    print(f"\n💡 The agent learned from past SQL injection, weak hashing, and")
    print(f"   error handling reviews to provide context-aware suggestions!")

    # Query fact store for learned standards
    print(f"\n📖 Learned Coding Standards (from FactStore):")
    bcrypt_facts = agent.fact_store.query("bcrypt")
    if bcrypt_facts:
        print(f"   - Password hashing: {bcrypt_facts[0].value[:60]}...")

    parameterized_facts = agent.fact_store.query("parameterized")
    if parameterized_facts:
        print(f"   - SQL safety: {parameterized_facts[0].value[:60]}...")

    logging_facts = agent.fact_store.query("logger")
    if logging_facts:
        print(f"   - Logging: {logging_facts[0].value[:60]}...")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
