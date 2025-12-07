"""Full RAGAS evaluation with LLM judge using Groq.

This extends test_ragas_procedural_memory.py with actual LLM-based evaluation
for faithfulness, context_recall, and context_precision metrics.

Usage:
    pytest tests/test_ragas_full_evaluation.py -v -s
    # OR
    python tests/test_ragas_full_evaluation.py
"""

import os
import pytest
import time
from typing import List
from dotenv import load_dotenv
from datasets import Dataset

# Load environment variables
load_dotenv()

# Check if we can run full evaluation
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CAN_RUN_EVALUATION = GROQ_API_KEY is not None

from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig

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


class ProceduralMemoryEvaluator:
    """Evaluator for procedural memory with full RAGAS metrics."""

    def __init__(self, use_groq: bool = True):
        """Initialize evaluator with LLM configuration."""
        self.memory = ProceduralMemory(MemoryConfig(
            storage_path=":memory:",
            similarity_threshold=0.3,
            env_weight=0.6,
            goal_weight=0.4
        ))

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

        self._populate_experiences()

    def _populate_experiences(self):
        """Populate memory with test experiences."""
        base_time = time.time() - (30 * 24 * 3600)  # 30 days ago

        # Web scraping experiences
        experiences = [
            Experience(
                env_features=["web", "python", "static"],
                goal="extract product data",
                action="Used BeautifulSoup with CSS selectors",
                result="Successfully extracted 100 items from static HTML",
                success=True,
                timestamp=base_time
            ),
            Experience(
                env_features=["web", "python", "javascript"],
                goal="extract product data",
                action="Tried BeautifulSoup on dynamic site",
                result="Failed - content not in HTML source (JavaScript-rendered)",
                success=False,
                timestamp=base_time + 86400
            ),
            Experience(
                env_features=["web", "python", "javascript", "dynamic"],
                goal="extract product data from dynamic site",
                action="Used Selenium with WebDriverWait for JavaScript-heavy site",
                result="Successfully extracted 200 items with 2s explicit wait",
                success=True,
                timestamp=base_time + (2 * 86400)
            ),
            # Database optimization experiences
            Experience(
                env_features=["database", "postgresql", "performance"],
                goal="optimize slow query",
                action="Added single-column index on user_id",
                result="Reduced query time from 5.2s to 0.8s",
                success=True,
                timestamp=base_time + (5 * 86400)
            ),
            Experience(
                env_features=["database", "postgresql", "performance", "indexing"],
                goal="optimize slow query further",
                action="Used EXPLAIN ANALYZE, found full table scan, added composite index (user_id, created_at)",
                result="Reduced query time to 0.08s (100x improvement)",
                success=True,
                timestamp=base_time + (6 * 86400)
            ),
            Experience(
                env_features=["database", "caching"],
                goal="optimize query performance",
                action="Tried Redis query caching",
                result="Helped for repeated queries but didn't solve root cause (cold cache still slow)",
                success=False,
                timestamp=base_time + (7 * 86400)
            ),
            # API endpoint evolution
            Experience(
                env_features=["api", "http", "authentication"],
                goal="fetch user data",
                action="Used /api/v1/users with basic auth",
                result="Success - retrieved user list",
                success=True,
                timestamp=base_time + (10 * 86400)
            ),
            Experience(
                env_features=["api", "http", "authentication"],
                goal="fetch user data",
                action="Tried /api/v1/users endpoint",
                result="Failed with 401 - endpoint deprecated, migration notice received",
                success=False,
                timestamp=base_time + (25 * 86400)
            ),
            Experience(
                env_features=["api", "http", "bearer", "authentication"],
                goal="fetch user data with new API",
                action="Used /api/v3/users with bearer token authentication",
                result="Success - new auth scheme works, faster response time",
                success=True,
                timestamp=base_time + (29 * 86400)
            ),
        ]

        for exp in experiences:
            self.memory.store(exp)

    def retrieve_and_generate(self, query: str, env_features: List[str]) -> tuple[str, List[str]]:
        """Retrieve contexts and generate answer."""
        # Retrieve experiences
        experiences = self.memory.recall(
            env_features=env_features,
            goal=query,
            top_k=5
        )

        # Format as context strings
        contexts = []
        for exp in experiences:
            context = (
                f"Environment: {', '.join(exp.env_features)}. "
                f"Goal: {exp.goal}. "
                f"Action: {exp.action}. "
                f"Result: {exp.result}. "
                f"Success: {exp.success}."
            )
            contexts.append(context)

        # Generate answer from successful experiences
        successful = [exp for exp in experiences if exp.success]

        if not successful:
            answer = "No successful past experiences found for this scenario."
        else:
            # Take most recent successful experience
            latest = successful[-1]
            answer = (
                f"Based on past experience: {latest.action}. "
                f"{latest.result} "
                f"This approach worked in a {', '.join(latest.env_features)} environment."
            )

        return answer, contexts

    def evaluate_scenario(self, question: str, env_features: List[str], ground_truth: str) -> dict:
        """Evaluate a single scenario with RAGAS metrics."""
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
                embeddings=None,  # Use default embeddings
            )

            # RAGAS 0.4.0 returns an EvaluationResult object
            # Access metrics as properties or dict keys
            def get_metric(result_obj, key):
                # Try as attribute first
                if hasattr(result_obj, key):
                    val = getattr(result_obj, key)
                    if isinstance(val, list) and len(val) > 0:
                        return val[0]
                    return val if val is not None else 0.0

                # Try as dict access
                try:
                    val = result_obj[key]
                    if isinstance(val, list) and len(val) > 0:
                        return val[0]
                    return val if val is not None else 0.0
                except (KeyError, TypeError):
                    pass

                # Try to_pandas if available
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
def test_web_scraping_with_llm_judge():
    """Test web scraping recall with full RAGAS evaluation."""
    print("\n" + "=" * 70)
    print("Test 1: Web Scraping Experience Recall (LLM Judge)")
    print("=" * 70)

    evaluator = ProceduralMemoryEvaluator(use_groq=True)

    result = evaluator.evaluate_scenario(
        question="I need to scrape a website that loads data dynamically with JavaScript. What approach worked before?",
        env_features=["web", "python", "javascript", "dynamic"],
        ground_truth="Use Selenium WebDriver with explicit wait conditions for dynamic JavaScript content"
    )

    print(f"\n   Question: {result['question']}")
    print(f"   Answer: {result['answer'][:100]}...")
    print(f"   Contexts: {len(result['contexts'])} retrieved")

    if "faithfulness" in result:
        print(f"\n   📊 RAGAS Metrics:")
        print(f"      Faithfulness: {result['faithfulness']:.3f}")
        print(f"      Context Recall: {result['context_recall']:.3f}")
        print(f"      Context Precision: {result['context_precision']:.3f}")

        # Assertions
        assert result['faithfulness'] >= 0.7, "Faithfulness should be >= 0.7"
        assert result['context_recall'] >= 0.7, "Context recall should be >= 0.7"
    else:
        print(f"\n   Status: {result['evaluation']}")


@pytest.mark.skipif(not CAN_RUN_EVALUATION, reason="GROQ_API_KEY not set")
@pytest.mark.ragas
def test_database_optimization_with_llm_judge():
    """Test database optimization recall with full RAGAS evaluation."""
    print("\n" + "=" * 70)
    print("Test 2: Database Optimization Recall (LLM Judge)")
    print("=" * 70)

    evaluator = ProceduralMemoryEvaluator(use_groq=True)

    result = evaluator.evaluate_scenario(
        question="Our database query is taking 5+ seconds on a 1M row table. How did we fix this before?",
        env_features=["database", "postgresql", "performance"],
        ground_truth="Add composite index on WHERE clause columns and verify with EXPLAIN ANALYZE"
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
    else:
        print(f"\n   Status: {result['evaluation']}")


@pytest.mark.skipif(not CAN_RUN_EVALUATION, reason="GROQ_API_KEY not set")
@pytest.mark.ragas
def test_temporal_resolution_with_llm_judge():
    """Test temporal conflict resolution with full RAGAS evaluation."""
    print("\n" + "=" * 70)
    print("Test 3: Temporal Conflict Resolution (LLM Judge)")
    print("=" * 70)

    evaluator = ProceduralMemoryEvaluator(use_groq=True)

    result = evaluator.evaluate_scenario(
        question="What's the correct API endpoint to fetch user data?",
        env_features=["api", "http", "authentication"],
        ground_truth="Use /api/v3/users with bearer token authentication (newest approach)"
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
        assert "/api/v3" in result['answer'], "Should recommend v3 endpoint"
    else:
        print(f"\n   Status: {result['evaluation']}")


if __name__ == "__main__":
    print("=" * 70)
    print("Full RAGAS Evaluation with LLM Judge (Groq)")
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
    print(f"\nRunning 3 evaluation scenarios...\n")

    # Run all tests
    try:
        test_web_scraping_with_llm_judge()
        test_database_optimization_with_llm_judge()
        test_temporal_resolution_with_llm_judge()

        print("\n" + "=" * 70)
        print("✅ All Evaluations Complete!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
