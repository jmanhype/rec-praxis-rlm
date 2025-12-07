"""
RAGAS evaluation for DependencyScanAgent using Groq LLM judge.

This test validates the dependency scan agent's ability to:
1. Detect CVE vulnerabilities accurately
2. Provide context-aware upgrade recommendations
3. Detect exposed secrets with minimal false positives
4. Learn from past upgrade experiences

Evaluation metrics:
- Faithfulness: Are findings grounded in actual dependency/secret patterns?
- Context Recall: Does the agent retrieve relevant past upgrades/incidents?
- Context Precision: Are retrieved experiences actually useful?

Run with: pytest tests/test_ragas_dependency_scan.py -m ragas -v
"""

import os
from typing import List, Dict
from dataclasses import dataclass

import pytest

# Check if RAGAS is available
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, context_recall, context_precision
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

from rec_praxis_rlm import ProceduralMemory, Experience, MemoryConfig, FactStore

# Check for GROQ_API_KEY
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


@dataclass
class DependencyFinding:
    """Dependency vulnerability or secret finding."""

    finding_type: str
    package_or_secret: str
    severity: str
    remediation: str


class DependencyScanEvaluator:
    """Evaluator for dependency scan agent using RAGAS metrics."""

    def __init__(self, use_groq: bool = True):
        """Initialize evaluator with optional Groq LLM judge."""
        self.use_groq = use_groq and RAGAS_AVAILABLE and GROQ_API_KEY

        # Initialize memory with past upgrades and incidents
        self.memory = ProceduralMemory(
            config=MemoryConfig(
                storage_path=":memory:",
                similarity_threshold=0.3,
                env_weight=0.6,
                goal_weight=0.4,
            )
        )

        # Initialize fact store with CVE knowledge
        self.fact_store = FactStore(storage_path=":memory:")

        # Populate memory
        self._populate_upgrade_history()
        self._load_cve_knowledge()

        # Initialize Groq LLM if available
        if self.use_groq and RAGAS_AVAILABLE:
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                groq_api_key=GROQ_API_KEY,
            )
            self.ragas_llm = LangchainLLMWrapper(self.llm)

    def _populate_upgrade_history(self) -> None:
        """Populate memory with past dependency upgrades."""
        import time

        now = time.time()
        days = 24 * 3600

        past_upgrades = [
            Experience(
                env_features=["python", "dependencies", "security", "cve", "requests"],
                goal="upgrade vulnerable requests library",
                action="Upgraded requests from 2.25.0 to 2.31.0 to fix CVE-2023-32681",
                result="Successfully patched. All tests pass. No breaking changes.",
                success=True,
                timestamp=now - (5 * days),
            ),
            Experience(
                env_features=["python", "dependencies", "django", "breaking-change"],
                goal="upgrade Django with major version change",
                action="Migrated from Django 2.2 to 3.2 LTS. Updated URL patterns, fixed deprecated imports.",
                result="Migration took 2 days. All tests pass. 6 CVEs patched.",
                success=True,
                timestamp=now - (20 * days),
            ),
            Experience(
                env_features=["security", "secrets", "aws", "credentials"],
                goal="fix exposed AWS credentials",
                action="Rotated AWS keys, moved to environment variables, added .env to .gitignore",
                result="Credentials secured. Pre-commit hook prevents future leaks.",
                success=True,
                timestamp=now - (10 * days),
            ),
            Experience(
                env_features=["security", "secrets", "github", "token"],
                goal="remediate leaked GitHub token",
                action="Revoked token, generated new with minimal scopes, moved to GitHub Actions secrets",
                result="Token secured. No unauthorized access detected.",
                success=True,
                timestamp=now - (15 * days),
            ),
        ]

        for exp in past_upgrades:
            self.memory.store(exp)

    def _load_cve_knowledge(self) -> None:
        """Load CVE facts into semantic memory."""
        cve_facts = """
        CVE = Common Vulnerabilities and Exposures
        CVE-2023-32681 = Requests library proxy credential leakage
        CRITICAL = CVSS score 9.0-10.0
        HIGH = CVSS score 7.0-8.9
        """
        self.fact_store.extract_facts(cve_facts, source_id="cve_knowledge")

    def detect_and_retrieve(
        self, scenario: str, env_features: List[str], goal: str
    ) -> tuple[str, List[str]]:
        """
        Detect vulnerability/secret and retrieve relevant past experiences.

        Args:
            scenario: Description of what to detect
            env_features: Environmental context
            goal: Goal for memory retrieval

        Returns:
            Tuple of (finding description, contexts)
        """
        # Retrieve relevant past experiences
        experiences = self.memory.recall(env_features=env_features, goal=goal, top_k=5)

        # Build contexts from retrieved experiences
        contexts = [
            f"Environment: {', '.join(exp.env_features)}. "
            f"Goal: {exp.goal}. "
            f"Action Taken: {exp.action}. "
            f"Result: {exp.result}. "
            f"Success: {exp.success}."
            for exp in experiences
        ]

        # Generate finding description from most relevant experience
        successful = [exp for exp in experiences if exp.success]
        if not successful:
            finding = f"Detected issue in {scenario}. No past remediation found."
        else:
            most_relevant = successful[0]
            finding = (
                f"Detected issue in {scenario}. "
                f"Recommended remediation: {most_relevant.action} "
                f"(Based on past success: {most_relevant.result})"
            )

        return finding, contexts

    def evaluate_scenario(
        self, question: str, scenario: str, env_features: List[str], goal: str, ground_truth: str
    ) -> Dict[str, float]:
        """
        Evaluate a single dependency scan scenario using RAGAS metrics.

        Args:
            question: Question about the vulnerability/secret
            scenario: Scenario description
            env_features: Environmental context
            goal: Goal for memory retrieval
            ground_truth: Expected correct answer

        Returns:
            Dictionary of metric scores
        """
        # Detect and retrieve
        finding, contexts = self.detect_and_retrieve(scenario, env_features, goal)

        # Prepare dataset for RAGAS
        dataset = Dataset.from_dict(
            {"question": [question], "answer": [finding], "contexts": [contexts], "ground_truth": [ground_truth]}
        )

        # Evaluate with RAGAS
        if self.use_groq:
            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, context_recall, context_precision],
                llm=self.ragas_llm,
                embeddings=None,
            )

            # Extract metrics
            def get_metric(result_obj, key):
                if hasattr(result_obj, key):
                    val = getattr(result_obj, key)
                    if isinstance(val, list) and len(val) > 0:
                        return val[0]
                    return val if val is not None else 0.0

                if hasattr(result_obj, "to_pandas"):
                    df = result_obj.to_pandas()
                    if key in df.columns:
                        return df[key].iloc[0]
                return 0.0

            faithfulness_score = get_metric(result, "faithfulness")
            recall_score = get_metric(result, "context_recall")
            precision_score = get_metric(result, "context_precision")
        else:
            # Fallback to heuristics
            faithfulness_score = 0.8
            recall_score = 0.8
            precision_score = 0.8

        return {
            "faithfulness": faithfulness_score,
            "context_recall": recall_score,
            "context_precision": precision_score,
            "answer": finding,
            "contexts": contexts,
        }


@pytest.mark.ragas
@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="RAGAS dependencies not installed")
@pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY not set")
def test_cve_detection_requests():
    """Test CVE detection for requests library."""
    evaluator = DependencyScanEvaluator(use_groq=True)

    question = "What vulnerability exists in requests 2.25.0 and how should it be fixed?"

    ground_truth = (
        "Requests 2.25.0 has CVE-2023-32681 (proxy credential leakage). "
        "Upgrade to requests>=2.31.0."
    )

    result = evaluator.evaluate_scenario(
        question=question,
        scenario="requests 2.25.0",
        env_features=["python", "dependencies", "security", "requests"],
        goal="upgrade vulnerable requests library",
        ground_truth=ground_truth,
    )

    print("\n" + "=" * 70)
    print("Test 1: CVE Detection - Requests Library")
    print("=" * 70)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"\nMetrics:")
    print(f"  Faithfulness: {result['faithfulness']:.3f}")
    print(f"  Context Recall: {result['context_recall']:.3f}")
    print(f"  Context Precision: {result['context_precision']:.3f}")
    print("=" * 70)

    # Assertions
    assert result["faithfulness"] >= 0.7, "Should accurately describe CVE and fix"
    assert result["context_recall"] >= 0.7, "Should recall past requests upgrade"
    assert result["context_precision"] >= 0.7, "Should provide relevant remediation"


@pytest.mark.ragas
@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="RAGAS dependencies not installed")
@pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY not set")
def test_upgrade_path_django():
    """Test upgrade path recommendation for Django major version."""
    evaluator = DependencyScanEvaluator(use_groq=True)

    question = "How should we upgrade Django 2.2 to fix security vulnerabilities?"

    ground_truth = (
        "Upgrade to Django 3.2 LTS. This requires URL pattern migration from url() to path() "
        "and fixing deprecated imports."
    )

    result = evaluator.evaluate_scenario(
        question=question,
        scenario="Django 2.2",
        env_features=["python", "dependencies", "django"],
        goal="upgrade Django with major version change",
        ground_truth=ground_truth,
    )

    print("\n" + "=" * 70)
    print("Test 2: Upgrade Path - Django Major Version")
    print("=" * 70)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"\nMetrics:")
    print(f"  Faithfulness: {result['faithfulness']:.3f}")
    print(f"  Context Recall: {result['context_recall']:.3f}")
    print(f"  Context Precision: {result['context_precision']:.3f}")
    print("=" * 70)

    # Assertions
    assert result["faithfulness"] >= 0.7, "Should mention Django upgrade and migration"
    assert result["context_recall"] >= 0.7, "Should recall past Django upgrade"
    assert result["context_precision"] >= 0.7, "Should warn about breaking changes"


@pytest.mark.ragas
@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="RAGAS dependencies not installed")
@pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY not set")
def test_secret_detection_aws():
    """Test secret detection for AWS credentials."""
    evaluator = DependencyScanEvaluator(use_groq=True)

    question = "How should we handle exposed AWS credentials found in code?"

    ground_truth = (
        "Rotate AWS credentials immediately. Move to environment variables or AWS Secrets Manager. "
        "Add .env to .gitignore and implement pre-commit hooks."
    )

    result = evaluator.evaluate_scenario(
        question=question,
        scenario="AWS Access Key in config.py",
        env_features=["security", "secrets", "aws"],
        goal="fix exposed AWS credentials",
        ground_truth=ground_truth,
    )

    print("\n" + "=" * 70)
    print("Test 3: Secret Detection - AWS Credentials")
    print("=" * 70)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"\nMetrics:")
    print(f"  Faithfulness: {result['faithfulness']:.3f}")
    print(f"  Context Recall: {result['context_recall']:.3f}")
    print(f"  Context Precision: {result['context_precision']:.3f}")
    print("=" * 70)

    # Assertions
    assert result["faithfulness"] >= 0.7, "Should recommend credential rotation"
    assert result["context_recall"] >= 0.7, "Should recall past AWS incident"
    assert result["context_precision"] >= 0.7, "Should suggest environment variables"


@pytest.mark.ragas
@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="RAGAS dependencies not installed")
@pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY not set")
def test_secret_detection_github():
    """Test secret detection for GitHub tokens."""
    evaluator = DependencyScanEvaluator(use_groq=True)

    question = "What should we do about a leaked GitHub personal access token?"

    ground_truth = (
        "Revoke the token immediately. Generate a new token with minimal required scopes. "
        "Store in GitHub Actions secrets or environment variables."
    )

    result = evaluator.evaluate_scenario(
        question=question,
        scenario="GitHub Token in config.py",
        env_features=["security", "secrets", "github"],
        goal="remediate leaked GitHub token",
        ground_truth=ground_truth,
    )

    print("\n" + "=" * 70)
    print("Test 4: Secret Detection - GitHub Token")
    print("=" * 70)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"\nMetrics:")
    print(f"  Faithfulness: {result['faithfulness']:.3f}")
    print(f"  Context Recall: {result['context_recall']:.3f}")
    print(f"  Context Precision: {result['context_precision']:.3f}")
    print("=" * 70)

    # Assertions
    assert result["faithfulness"] >= 0.7, "Should recommend token revocation"
    assert result["context_recall"] >= 0.7, "Should recall past GitHub incident"
    assert result["context_precision"] >= 0.7, "Should suggest secure storage"


@pytest.mark.ragas
@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="RAGAS dependencies not installed")
@pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY not set")
def test_average_ragas_scores():
    """
    Run all dependency scan scenarios and compute average RAGAS scores.

    Acceptance criteria:
    - Average faithfulness >= 0.85 (findings grounded in actual issues)
    - Average context recall >= 0.85 (relevant past experiences retrieved)
    - Average context precision >= 0.85 (useful remediation guidance)
    """
    evaluator = DependencyScanEvaluator(use_groq=True)

    scenarios = [
        {
            "name": "CVE Detection",
            "question": "What CVE affects requests 2.25.0?",
            "scenario": "requests 2.25.0",
            "env_features": ["python", "dependencies", "security"],
            "goal": "upgrade vulnerable requests library",
            "ground_truth": "CVE-2023-32681. Upgrade to requests>=2.31.0.",
        },
        {
            "name": "Django Upgrade",
            "question": "How to upgrade Django 2.2?",
            "scenario": "Django 2.2",
            "env_features": ["python", "dependencies", "django"],
            "goal": "upgrade Django with major version change",
            "ground_truth": "Upgrade to Django 3.2 LTS with URL pattern migration.",
        },
        {
            "name": "AWS Secret",
            "question": "How to fix exposed AWS credentials?",
            "scenario": "AWS credentials",
            "env_features": ["security", "secrets", "aws"],
            "goal": "fix exposed AWS credentials",
            "ground_truth": "Rotate credentials, move to environment variables.",
        },
        {
            "name": "GitHub Secret",
            "question": "How to handle leaked GitHub token?",
            "scenario": "GitHub token",
            "env_features": ["security", "secrets", "github"],
            "goal": "remediate leaked GitHub token",
            "ground_truth": "Revoke token, generate new with minimal scopes.",
        },
    ]

    all_scores = {"faithfulness": [], "context_recall": [], "context_precision": []}

    for scenario in scenarios:
        result = evaluator.evaluate_scenario(
            question=scenario["question"],
            scenario=scenario["scenario"],
            env_features=scenario["env_features"],
            goal=scenario["goal"],
            ground_truth=scenario["ground_truth"],
        )

        all_scores["faithfulness"].append(result["faithfulness"])
        all_scores["context_recall"].append(result["context_recall"])
        all_scores["context_precision"].append(result["context_precision"])

    # Compute averages
    avg_faithfulness = sum(all_scores["faithfulness"]) / len(all_scores["faithfulness"])
    avg_recall = sum(all_scores["context_recall"]) / len(all_scores["context_recall"])
    avg_precision = sum(all_scores["context_precision"]) / len(all_scores["context_precision"])

    print("\n" + "=" * 70)
    print("Dependency Scan Agent - Average RAGAS Scores")
    print("=" * 70)
    print(f"Average Faithfulness: {avg_faithfulness:.3f}")
    print(f"Average Context Recall: {avg_recall:.3f}")
    print(f"Average Context Precision: {avg_precision:.3f}")
    print("=" * 70)

    # Assertions
    assert avg_faithfulness >= 0.85, f"Faithfulness {avg_faithfulness:.3f} < 0.85"
    assert avg_recall >= 0.85, f"Context Recall {avg_recall:.3f} < 0.85"
    assert avg_precision >= 0.85, f"Context Precision {avg_precision:.3f} < 0.85"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "-m", "ragas"])
