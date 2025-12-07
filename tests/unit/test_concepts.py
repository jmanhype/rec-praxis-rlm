"""Unit tests for concept tagging module."""

from rec_praxis_rlm.concepts import ConceptTagger
from rec_praxis_rlm.memory import Experience
import time


def test_extract_database_tags():
    """Test extraction of database-related tags."""
    tagger = ConceptTagger()
    text = "SELECT * FROM users WHERE id = 123"

    tags = tagger.extract_tags(text)

    assert "database" in tags
    assert "sql" in tags


def test_extract_api_tags():
    """Test extraction of API-related tags."""
    tagger = ConceptTagger()
    text = "HTTP GET request to /api/v1/users endpoint returned 200 OK"

    tags = tagger.extract_tags(text)

    assert "api" in tags


def test_extract_file_references():
    """Test extraction of file path references."""
    tagger = ConceptTagger()
    text = "Reading config from ./config.json and script from src/main.py"

    tags = tagger.extract_tags(text)

    assert "file" in tags
    assert "python" in tags
    assert "config" in tags


def test_extract_programming_languages():
    """Test extraction of programming language mentions."""
    tagger = ConceptTagger()
    text = "Running pytest tests on Python codebase"

    tags = tagger.extract_tags(text)

    assert "python" in tags
    assert "test" in tags


def test_extract_auth_tags():
    """Test extraction of authentication-related tags."""
    tagger = ConceptTagger()
    text = "User login with JWT token authentication successful"

    tags = tagger.extract_tags(text)

    assert "auth" in tags


def test_tag_experience():
    """Test tagging an Experience object."""
    tagger = ConceptTagger()

    experience = Experience(
        env_features=["api", "database"],
        goal="Fetch user data from database",
        action="SELECT * FROM users WHERE email = user@example.com",
        result="Query returned 1 row successfully",
        success=True,
        timestamp=time.time(),
    )

    tagged_exp = tagger.tag_experience(experience)

    # Should have extracted database and sql tags
    assert "database" in tagged_exp.tags
    assert "sql" in tagged_exp.tags


def test_tag_experience_merges_existing():
    """Test that tagging merges with existing tags."""
    tagger = ConceptTagger()

    experience = Experience(
        env_features=["api"],
        goal="API request",
        action="GET /api/users",
        result="200 OK",
        success=True,
        timestamp=time.time(),
        tags=["custom_tag"],  # Existing tag
    )

    tagged_exp = tagger.tag_experience(experience)

    # Should keep existing tag and add new ones
    assert "custom_tag" in tagged_exp.tags
    assert "api" in tagged_exp.tags


def test_empty_text():
    """Test extraction from empty text."""
    tagger = ConceptTagger()
    tags = tagger.extract_tags("")

    assert tags == []


def test_max_tags_limit():
    """Test max_tags parameter."""
    tagger = ConceptTagger()

    # Text with many different keywords
    text = "database sql query select api http rest auth login password file read write test assert"

    tags = tagger.extract_tags(text, max_tags=3)

    # Should limit to 3 tags
    assert len(tags) <= 3
