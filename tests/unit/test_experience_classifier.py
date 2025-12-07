"""Unit tests for experience classification module."""

import time
from rec_praxis_rlm.experience_classifier import ExperienceClassifier
from rec_praxis_rlm.memory import Experience


def test_classify_learn_experience():
    """Test classification of learning experience."""
    classifier = ExperienceClassifier()

    exp_type = classifier.classify(
        goal="Learn how to use pandas DataFrame groupby",
        action="Read documentation and tried examples",
        result="Successfully grouped data by category",
        success=True,
    )

    assert exp_type == "learn"


def test_classify_recover_experience():
    """Test classification of error recovery experience."""
    classifier = ExperienceClassifier()

    exp_type = classifier.classify(
        goal="Fix timeout error in API call",
        action="Added retry logic with exponential backoff",
        result="API calls now succeed consistently",
        success=True,
    )

    assert exp_type == "recover"


def test_classify_optimize_experience():
    """Test classification of optimization experience."""
    classifier = ExperienceClassifier()

    exp_type = classifier.classify(
        goal="Improve database query performance",
        action="Added index on user_id column",
        result="Query latency reduced from 2s to 50ms",
        success=True,
    )

    assert exp_type == "optimize"


def test_classify_explore_experience():
    """Test classification of exploratory experience."""
    classifier = ExperienceClassifier()

    exp_type = classifier.classify(
        goal="Explore alternative caching strategies",
        action="Tested Redis vs Memcached for session storage",
        result="Redis provided better persistence guarantees",
        success=True,
    )

    assert exp_type == "explore"


def test_classify_failed_experience_as_recover():
    """Test that failed experiences are classified as recovery."""
    classifier = ExperienceClassifier()

    exp_type = classifier.classify(
        goal="Process user uploads",
        action="Implemented file validation",
        result="File upload still failing",
        success=False,
    )

    # Failed experiences should be weighted toward "recover"
    assert exp_type == "recover"


def test_classify_default_to_learn():
    """Test default classification when no keywords match."""
    classifier = ExperienceClassifier()

    exp_type = classifier.classify(
        goal="Simple task",
        action="Did something",
        result="Got result",
        success=True,
    )

    # Should default to "learn" when no keywords match
    assert exp_type == "learn"


def test_classify_experience_object():
    """Test classifying an Experience object."""
    classifier = ExperienceClassifier()

    experience = Experience(
        env_features=["api", "database"],
        goal="Debug slow API endpoint",
        action="Added logging to identify bottleneck",
        result="Found N+1 query issue",
        success=True,
        timestamp=time.time(),
    )

    tagged_exp = classifier.classify_experience(experience)

    assert tagged_exp.experience_type == "recover"


def test_classify_multiple_keywords():
    """Test classification with multiple matching keyword types."""
    classifier = ExperienceClassifier()

    # Text with both "optimize" and "fix" keywords
    exp_type = classifier.classify(
        goal="Fix performance bug in rendering",
        action="Optimized component re-renders",
        result="Rendering now 10x faster",
        success=True,
    )

    # Should pick the dominant type (in this case "optimize" due to success)
    assert exp_type in ["recover", "optimize"]


def test_classify_with_error_keywords():
    """Test classification with error-related keywords."""
    classifier = ExperienceClassifier()

    exp_type = classifier.classify(
        goal="Handle exception in payment processing",
        action="Added try-catch with proper error handling",
        result="Exceptions now logged and user notified",
        success=True,
    )

    assert exp_type == "recover"


def test_classify_with_research_keywords():
    """Test classification with research/investigation keywords."""
    classifier = ExperienceClassifier()

    exp_type = classifier.classify(
        goal="Investigate new database migration tool",
        action="Researched Alembic vs Flyway comparison",
        result="Alembic better suited for Python projects",
        success=True,
    )

    assert exp_type in ["learn", "explore"]


def test_classify_benchmark_experience():
    """Test classification of benchmarking experience."""
    classifier = ExperienceClassifier()

    exp_type = classifier.classify(
        goal="Benchmark cache performance",
        action="Measured throughput with different cache sizes",
        result="1MB cache provides best latency/memory tradeoff",
        success=True,
    )

    assert exp_type == "optimize"
