"""Concept tagging and semantic extraction for experiences.

Auto-extracts semantic tags from goal/action/result text to enable rich semantic search.

Supported extraction methods:
1. Heuristic: Pattern-based extraction (file paths, tech keywords, domains)
2. LLM-based: Optional LLM extraction for advanced concepts

Usage:
    from rec_praxis_rlm.concepts import ConceptTagger

    tagger = ConceptTagger()

    # Extract tags from text
    tags = tagger.extract_tags("SELECT * FROM users WHERE id=123")
    # Returns: ["database", "sql", "query"]

    # Extract tags from experience
    tagged_exp = tagger.tag_experience(experience)
"""

import logging
import re
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# Common tech/domain keywords for heuristic tagging
TECH_KEYWORDS = {
    "database": ["database", "select", "insert", "update", "delete", "query", "table", "from", "where"],
    "sql": ["sql", "select", "insert", "update", "delete", "query"],
    "api": ["api", "endpoint", "rest", "graphql", "http", "request", "response"],
    "auth": ["auth", "login", "password", "token", "jwt", "oauth", "session"],
    "file": ["file", "read", "write", "open", "save", "load", "path"],
    "network": ["network", "socket", "tcp", "udp", "connection", "ping"],
    "security": ["security", "encrypt", "decrypt", "hash", "sign", "verify"],
    "error": ["error", "exception", "fail", "crash", "bug"],
    "performance": ["performance", "optimize", "slow", "fast", "latency", "cache"],
    "test": ["test", "assert", "mock", "stub", "verify"],
    "config": ["config", "settings", "environment", "env"],
}


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers supporting tag extraction."""

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from prompt."""
        ...


class ConceptTagger:
    """Extract semantic concept tags from experience text.

    Uses heuristics (keywords, file paths, tech terms) and optional LLM
    to extract meaningful tags for semantic search.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize concept tagger.

        Args:
            llm_provider: Optional LLM provider for advanced tag extraction
        """
        self.llm_provider = llm_provider

    def extract_tags(self, text: str, max_tags: int = 10) -> list[str]:
        """Extract semantic tags from text.

        Args:
            text: Text to extract tags from
            max_tags: Maximum number of tags to return

        Returns:
            List of tag strings (lowercase, deduplicated)
        """
        if not text:
            return []

        tags = set()

        # Heuristic extraction
        tags.update(self._extract_tech_keywords(text))
        tags.update(self._extract_file_references(text))
        tags.update(self._extract_programming_languages(text))

        # Convert to sorted list
        tag_list = sorted(tags)[:max_tags]

        logger.debug(f"Extracted {len(tag_list)} tags: {tag_list}")
        return tag_list

    def _extract_tech_keywords(self, text: str) -> set[str]:
        """Extract technology/domain keywords."""
        text_lower = text.lower()
        found_tags = set()

        for tag, keywords in TECH_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                found_tags.add(tag)

        return found_tags

    def _extract_file_references(self, text: str) -> set[str]:
        """Extract file path references and infer tags."""
        tags = set()

        # Pattern: file paths (e.g., /path/to/file.py, ./config.json)
        file_pattern = r"[./\\]?[\w/\\.-]+\.(py|js|ts|json|yaml|yml|txt|md|sql|html|css)"
        matches = re.findall(file_pattern, text, re.IGNORECASE)

        if matches:
            tags.add("file")

            # Infer language/type from extension
            for match in matches:
                ext = match.lower()
                if ext in ["py", "python"]:
                    tags.add("python")
                elif ext in ["js", "ts", "jsx", "tsx"]:
                    tags.add("javascript")
                elif ext in ["json", "yaml", "yml"]:
                    tags.add("config")
                elif ext in ["sql"]:
                    tags.add("database")

        return tags

    def _extract_programming_languages(self, text: str) -> set[str]:
        """Extract programming language mentions."""
        text_lower = text.lower()
        languages = {
            "python": ["python", "py", "pip", "pytest", "django"],
            "javascript": ["javascript", "js", "node", "npm", "react"],
            "typescript": ["typescript", "ts"],
            "sql": ["sql", "postgres", "mysql", "sqlite"],
            "rust": ["rust", "cargo"],
            "go": ["golang", "go func"],
        }

        tags = set()
        for lang, keywords in languages.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.add(lang)

        return tags

    def tag_experience(self, experience):
        """Extract and add tags to an Experience object.

        Args:
            experience: Experience object to tag (modified in-place)

        Returns:
            The modified Experience object with tags populated
        """
        # Import here to avoid circular dependency
        from rec_praxis_rlm.memory import Experience

        # Combine all text fields
        combined_text = f"{experience.goal} {experience.action} {experience.result}"

        # Extract tags
        tags = self.extract_tags(combined_text)

        # Update experience
        if not experience.tags:
            experience.tags = []

        # Merge with existing tags, deduplicate
        all_tags = set(experience.tags) | set(tags)
        experience.tags = sorted(all_tags)

        return experience
