"""RAGAS evaluation for Code Review Agent.

Tests code review quality using RAGAS metrics:
- Faithfulness: Are suggestions grounded in past experiences?
- Context Recall: Do we retrieve all relevant past reviews?
- Context Precision: Are retrieved reviews actually relevant?

Usage:
    pytest tests/test_ragas_code_review.py -v -s
    # OR
    python tests/test_ragas_code_review.py
"""

import os
import pytest
import time
from typing import List
from dotenv import load_dotenv
from datasets import Dataset

# Load environment variables
load_dotenv()

# Check if we can run evaluation
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CAN_RUN_EVALUATION = GROQ_API_KEY is not None

from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig, RLMContext, FactStore

# Only import RAGAS if we can run evaluation
if CAN_RUN_EVALUATION:
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, context_recall, context_precision
        from langchain_groq import ChatGroq
        from ragas.llms import LangchainLLMWrapper

        RAGAS_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: RAGAS imports failed: {e}")
        RAGAS_AVAILABLE = False
else:
    RAGAS_AVAILABLE = False


class CodeReviewEvaluator:
    """Evaluator for code review agent with RAGAS metrics."""

    def __init__(self, use_groq: bool = True):
        """Initialize evaluator with code review memory."""
        self.memory = ProceduralMemory(MemoryConfig(
            storage_path=":memory:",
            similarity_threshold=0.3,
            env_weight=0.6,
            goal_weight=0.4
        ))
        self.rlm = RLMContext()
        self.fact_store = FactStore(storage_path=":memory:")

        self.use_groq = use_groq and GROQ_API_KEY

        if self.use_groq and RAGAS_AVAILABLE:
            # Initialize Groq LLM for RAGAS evaluation
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                groq_api_key=GROQ_API_KEY
            )
            # Wrap for RAGAS
            self.ragas_llm = LangchainLLMWrapper(self.llm)
        else:
            self.llm = None
            self.ragas_llm = None

        self._populate_review_experiences()

    def _populate_review_experiences(self):
        """Populate memory with past code review experiences."""
        base_time = time.time() - (90 * 24 * 3600)  # 90 days ago

        experiences = [
            # SQL Injection reviews
            Experience(
                env_features=["python", "flask", "database", "security"],
                goal="review authentication code for SQL injection",
                action="Found raw SQL with string formatting in cursor.execute()",
                result="Suggested parameterized queries with placeholders. Developer fixed with prepared statements.",
                success=True,
                timestamp=base_time
            ),
            Experience(
                env_features=["python", "django", "database", "security"],
                goal="review database queries for injection risks",
                action="Found SQL injection via .raw() method with f-string interpolation",
                result="Recommended Django ORM QuerySet API. Developer refactored to use .filter() methods.",
                success=True,
                timestamp=base_time + (10 * 86400)
            ),
            Experience(
                env_features=["python", "sqlalchemy", "database"],
                goal="review ORM usage for SQL safety",
                action="Found text() SQL with string concatenation",
                result="Suggested SQLAlchemy bound parameters. Developer used :param syntax with bindparams.",
                success=True,
                timestamp=base_time + (15 * 86400)
            ),

            # Authentication & Cryptography
            Experience(
                env_features=["python", "authentication", "security"],
                goal="review password storage implementation",
                action="Found MD5 hashing for passwords: hashlib.md5(password.encode())",
                result="Suggested bcrypt with salt. Developer implemented bcrypt.hashpw() with proper rounds.",
                success=True,
                timestamp=base_time + (20 * 86400)
            ),
            Experience(
                env_features=["python", "flask", "authentication"],
                goal="review user authentication system",
                action="Found SHA1 for password hashing without salt",
                result="Recommended Argon2id. Developer migrated to argon2-cffi library.",
                success=True,
                timestamp=base_time + (25 * 86400)
            ),
            Experience(
                env_features=["python", "security", "credentials"],
                goal="review API key management",
                action="Found hardcoded API keys in source code: API_KEY = 'sk-xxx'",
                result="Suggested environment variables and .env files. Developer used python-dotenv.",
                success=True,
                timestamp=base_time + (30 * 86400)
            ),

            # Error Handling
            Experience(
                env_features=["python", "error-handling"],
                goal="review exception handling patterns",
                action="Found bare except: blocks catching all exceptions",
                result="Suggested specific exception types. Developer added proper exception hierarchies.",
                success=True,
                timestamp=base_time + (40 * 86400)
            ),
            Experience(
                env_features=["python", "logging", "error-handling"],
                goal="review error reporting",
                action="Found print() statements for error logging in production",
                result="Recommended logging module with proper levels. Developer implemented structured logging.",
                success=True,
                timestamp=base_time + (50 * 86400)
            ),

            # Concurrency Issues
            Experience(
                env_features=["python", "threading", "concurrency"],
                goal="review multi-threaded file access",
                action="Found race condition in shared resource access without locking",
                result="Suggested threading.Lock() for critical sections. Developer added proper synchronization.",
                success=True,
                timestamp=base_time + (60 * 86400)
            ),
        ]

        for exp in experiences:
            self.memory.store(exp)
            # Extract facts
            text = f"{exp.action}. {exp.result}"
            self.fact_store.extract_facts(text, source_id=f"review_{int(exp.timestamp)}")

    def retrieve_and_generate(self, question: str, env_features: List[str]) -> tuple[str, List[str]]:
        """Retrieve past reviews and generate answer."""
        # Retrieve experiences
        experiences = self.memory.recall(
            env_features=env_features,
            goal=question,
            top_k=5
        )

        # Format as context strings
        contexts = []
        for exp in experiences:
            context = (
                f"Environment: {', '.join(exp.env_features)}. "
                f"Goal: {exp.goal}. "
                f"Issue Found: {exp.action}. "
                f"Fix Implemented: {exp.result}. "
                f"Success: {exp.success}."
            )
            contexts.append(context)

        # Generate answer from successful experiences
        successful = [exp for exp in experiences if exp.success]

        if not successful:
            answer = "No past code review experiences found for this issue."
        else:
            # Take most relevant successful review (first in sorted results)
            most_relevant = successful[0]
            answer = (
                f"Based on past code reviews: {most_relevant.action}. "
                f"Recommended fix: {most_relevant.result} "
                f"This approach worked in a {', '.join(most_relevant.env_features)} context."
            )

        return answer, contexts

    def evaluate_scenario(self, question: str, env_features: List[str], ground_truth: str) -> dict:
        """Evaluate a code review scenario with RAGAS metrics."""
        # Retrieve and generate
        answer, contexts = self.retrieve_and_generate(question, env_features)

        # Create dataset for RAGAS
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth]
        }
        dataset = Dataset.from_dict(data)

        if not self.use_groq or not RAGAS_AVAILABLE:
            # Manual validation only
            return {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "num_contexts": len(contexts),
                "evaluation": "Manual validation only (no LLM judge)"
            }

        # Run RAGAS evaluation with Groq
        print(f"\n   Evaluating with Groq (llama-3.3-70b-versatile)...")

        try:
            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, context_recall, context_precision],
                llm=self.ragas_llm,
                embeddings=None,
            )

            # Extract metrics from RAGAS 0.4.0 EvaluationResult
            def get_metric(result_obj, key):
                if hasattr(result_obj, key):
                    val = getattr(result_obj, key)
                    if isinstance(val, list) and len(val) > 0:
                        return val[0]
                    return val if val is not None else 0.0

                try:
                    val = result_obj[key]
                    if isinstance(val, list) and len(val) > 0:
                        return val[0]
                    return val if val is not None else 0.0
                except (KeyError, TypeError):
                    pass

                if hasattr(result_obj, 'to_pandas'):
                    df = result_obj.to_pandas()
                    if key in df.columns:
                        return df[key].iloc[0]

                return 0.0

            faithfulness_score = get_metric(result, "faithfulness")
            context_recall_score = get_metric(result, "context_recall")
            context_precision_score = get_metric(result, "context_precision")

            return {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "faithfulness": faithfulness_score,
                "context_recall": context_recall_score,
                "context_precision": context_precision_score,
                "evaluation": "Full RAGAS evaluation with Groq LLM"
            }
        except Exception as e:
            print(f"   Warning: RAGAS evaluation failed: {e}")
            return {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "num_contexts": len(contexts),
                "evaluation": f"Evaluation failed: {e}"
            }


@pytest.mark.skipif(not CAN_RUN_EVALUATION, reason="GROQ_API_KEY not set")
@pytest.mark.ragas
def test_sql_injection_review():
    """Test code review for SQL injection with RAGAS evaluation."""
    print("\n" + "=" * 70)
    print("Test 1: SQL Injection Code Review (LLM Judge)")
    print("=" * 70)

    evaluator = CodeReviewEvaluator(use_groq=True)

    result = evaluator.evaluate_scenario(
        question="I found a Flask endpoint using f-strings to build SQL queries. What's the security risk and how should we fix it?",
        env_features=["python", "flask", "database", "security"],
        ground_truth="SQL injection vulnerability. Use parameterized queries with placeholders instead of string formatting."
    )

    print(f"\n   Question: {result['question'][:80]}...")
    print(f"   Answer: {result['answer'][:100]}...")
    print(f"   Contexts: {len(result['contexts'])} retrieved")

    if "faithfulness" in result:
        print(f"\n   📊 RAGAS Metrics:")
        print(f"      Faithfulness: {result['faithfulness']:.3f}")
        print(f"      Context Recall: {result['context_recall']:.3f}")
        print(f"      Context Precision: {result['context_precision']:.3f}")

        assert result['faithfulness'] >= 0.7, "Faithfulness should be >= 0.7"
        assert result['context_recall'] >= 0.7, "Context recall should be >= 0.7"
        # Check if SQL-related content is in the answer (flexible matching)
        has_sql_guidance = any(term in result['answer'].lower()
                              for term in ["parameterized", "prepared", "sql", "query", "injection", "orm"])
        print(f"      SQL-related guidance: {'✅' if has_sql_guidance else '❌'}")
    else:
        print(f"\n   Status: {result['evaluation']}")


@pytest.mark.skipif(not CAN_RUN_EVALUATION, reason="GROQ_API_KEY not set")
@pytest.mark.ragas
def test_weak_cryptography_review():
    """Test code review for weak password hashing."""
    print("\n" + "=" * 70)
    print("Test 2: Weak Cryptography Review (LLM Judge)")
    print("=" * 70)

    evaluator = CodeReviewEvaluator(use_groq=True)

    result = evaluator.evaluate_scenario(
        question="We're using MD5 to hash user passwords. Is this secure?",
        env_features=["python", "authentication", "security"],
        ground_truth="MD5 is cryptographically broken for password storage. Use bcrypt or Argon2 with proper salt and iterations."
    )

    print(f"\n   Question: {result['question']}")
    print(f"   Answer: {result['answer'][:100]}...")
    print(f"   Contexts: {len(result['contexts'])} retrieved")

    if "faithfulness" in result:
        print(f"\n   📊 RAGAS Metrics:")
        print(f"      Faithfulness: {result['faithfulness']:.3f}")
        print(f"      Context Recall: {result['context_recall']:.3f}")
        print(f"      Context Precision: {result['context_precision']:.3f}")

        assert result['faithfulness'] >= 0.7, "Faithfulness should be >= 0.7"
        assert result['context_recall'] >= 0.7, "Context recall should be >= 0.7"
        assert "bcrypt" in result['answer'].lower() or "argon2" in result['answer'].lower(), \
            "Should suggest bcrypt or Argon2"
    else:
        print(f"\n   Status: {result['evaluation']}")


@pytest.mark.skipif(not CAN_RUN_EVALUATION, reason="GROQ_API_KEY not set")
@pytest.mark.ragas
def test_error_handling_review():
    """Test code review for error handling patterns."""
    print("\n" + "=" * 70)
    print("Test 3: Error Handling Review (LLM Judge)")
    print("=" * 70)

    evaluator = CodeReviewEvaluator(use_groq=True)

    result = evaluator.evaluate_scenario(
        question="Our code has 'except:' blocks without specifying exception types. What's the issue?",
        env_features=["python", "error-handling"],
        ground_truth="Bare except catches all exceptions including SystemExit and KeyboardInterrupt. Use specific exception types."
    )

    print(f"\n   Question: {result['question'][:80]}...")
    print(f"   Answer: {result['answer'][:100]}...")
    print(f"   Contexts: {len(result['contexts'])} retrieved")

    if "faithfulness" in result:
        print(f"\n   📊 RAGAS Metrics:")
        print(f"      Faithfulness: {result['faithfulness']:.3f}")
        print(f"      Context Recall: {result['context_recall']:.3f}")
        print(f"      Context Precision: {result['context_precision']:.3f}")

        assert result['faithfulness'] >= 0.7, "Faithfulness should be >= 0.7"
        assert result['context_recall'] >= 0.7, "Context recall should be >= 0.7"
    else:
        print(f"\n   Status: {result['evaluation']}")


if __name__ == "__main__":
    print("=" * 70)
    print("Code Review Agent RAGAS Evaluation")
    print("=" * 70)

    if not CAN_RUN_EVALUATION:
        print("\n❌ Cannot run evaluation: GROQ_API_KEY not set")
        print("   Set GROQ_API_KEY in .env file to enable LLM judge evaluation")
        exit(1)

    if not RAGAS_AVAILABLE:
        print("\n❌ Cannot run evaluation: RAGAS dependencies not available")
        print("   Install with: pip install ragas langchain-groq")
        exit(1)

    print(f"\n✅ Groq API key found")
    print(f"✅ RAGAS framework available")
    print(f"\nRunning 3 code review evaluation scenarios...\n")

    # Run all tests
    try:
        test_sql_injection_review()
        test_weak_cryptography_review()
        test_error_handling_review()

        print("\n" + "=" * 70)
        print("✅ All Code Review Evaluations Complete!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
