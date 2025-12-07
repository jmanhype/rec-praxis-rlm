"""Semantic memory: Fact extraction and storage.

Inspired by HMLR's FactScrubber, adapted for rec-praxis-rlm's architecture.

The FactStore extracts "hard facts" (definitions, acronyms, entities, metrics)
from agent experiences and stores them for fast exact-match retrieval.

Key Features:
- Categorical organization (Definition, Acronym, Entity, Metric)
- Fast exact-match retrieval via indexed SQLite storage
- Links facts to source experiences for provenance
- Supports both heuristic and LLM-based extraction

Usage:
    store = FactStore(storage_path="facts.db")

    # Extract facts from text
    facts = store.extract_facts(
        text="FAISS = Facebook AI Similarity Search. It provides 10x speedup.",
        source_id="exp_001"
    )

    # Query facts
    faiss_facts = store.query("FAISS")
    speedup_facts = store.query("speedup", category="Metric")
"""

import sqlite3
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class Fact:
    """A hard fact extracted from agent experiences.

    Attributes:
        key: The identifier (e.g., "FAISS", "response_time", "user_preference")
        value: The fact content (e.g., "Facebook AI Similarity Search", "0.08s")
        category: Classification (Definition, Acronym, Entity, Metric)
        evidence: 10-30 words of context around the fact
        source_id: Experience ID or document ID containing the fact
        created_at: ISO-8601 timestamp
    """
    key: str
    value: str
    category: str  # Definition | Acronym | Entity | Metric
    evidence: str
    source_id: Optional[str] = None
    created_at: str = ""

    def __post_init__(self):
        """Set created_at if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat() + "Z"


class FactStore:
    """Semantic memory for storing and retrieving structured facts.

    The FactStore extracts key-value facts from text using either:
    1. Heuristic patterns (acronyms, key=value, metrics)
    2. LLM-based extraction (optional, requires API key)

    Facts are stored in SQLite with indexes for fast retrieval.
    """

    def __init__(self, storage_path: str = ":memory:"):
        """Initialize fact store with SQLite backend.

        Args:
            storage_path: Path to SQLite database file, or ":memory:" for in-memory
        """
        self.storage_path = storage_path

        # Create parent directory if needed
        if storage_path != ":memory:":
            Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(storage_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        """Create fact_store table and indexes."""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_store (
                fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT NOT NULL,
                evidence TEXT,
                source_id TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Indexes for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_key ON fact_store(key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_category ON fact_store(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_source ON fact_store(source_id)")

        self.conn.commit()

    def extract_facts(
        self,
        text: str,
        source_id: Optional[str] = None,
        use_heuristics: bool = True
    ) -> List[Fact]:
        """Extract facts from text and store them.

        Args:
            text: Text to extract facts from
            source_id: ID of the source (experience ID, document ID, etc.)
            use_heuristics: If True, use pattern-based extraction

        Returns:
            List of extracted Fact objects

        Raises:
            TypeError: If text is not a string
        """
        # Type validation
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        # Early return for empty text
        if not text:
            return []

        facts = []

        if use_heuristics:
            facts.extend(self._heuristic_extract(text, source_id))

        # Store all extracted facts
        for fact in facts:
            self._save_fact(fact)

        return facts

    def _heuristic_extract(self, text: str, source_id: Optional[str]) -> List[Fact]:
        """Extract facts using heuristic patterns.

        Patterns:
        - Acronym: "X = Y" or "X stands for Y" (uppercase)
        - Metric: Numbers with units (10x, 0.08s, 100ms, 5MB)
        - Key-value: "key: value" or "key = value"
        """
        facts = []

        # Pattern 1: Acronym expansion (e.g., "FAISS = Facebook AI...")
        acronym_pattern = r'([A-Z][A-Z0-9]+)\s*=\s*([^.]+?)(?:\.|$)'
        for match in re.finditer(acronym_pattern, text):
            acronym = match.group(1).strip()
            expansion = match.group(2).strip()

            # Get evidence (30 chars before and after)
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            evidence = text[start:end].strip()

            facts.append(Fact(
                key=acronym,
                value=expansion,
                category="Acronym",
                evidence=evidence,
                source_id=source_id
            ))

        # Pattern 2: "stands for" (e.g., "FAISS stands for...")
        stands_for_pattern = r'([A-Z][A-Z0-9]+)\s+stands for\s+([^.]+?)(?:\.|$)'
        for match in re.finditer(stands_for_pattern, text, re.IGNORECASE):
            acronym = match.group(1).strip()
            expansion = match.group(2).strip()

            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            evidence = text[start:end].strip()

            facts.append(Fact(
                key=acronym,
                value=expansion,
                category="Acronym",
                evidence=evidence,
                source_id=source_id
            ))

        # Pattern 3: Metrics (e.g., "10x speedup", "0.08s", "100ms")
        metric_pattern = r'(\d+\.?\d*)\s*(x|ms|s|MB|GB|%)\s+(faster|speedup|improvement|latency|size|time)'
        for match in re.finditer(metric_pattern, text, re.IGNORECASE):
            number = match.group(1)
            unit = match.group(2)
            metric_type = match.group(3)

            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            evidence = text[start:end].strip()

            facts.append(Fact(
                key=f"{metric_type}_{unit}",
                value=f"{number}{unit}",
                category="Metric",
                evidence=evidence,
                source_id=source_id
            ))

        # Pattern 4: Standalone metrics (e.g., "reduced to 0.08s")
        standalone_metric_pattern = r'(\d+\.?\d*)(x|ms|s|MB|GB|%)\b'
        for match in re.finditer(standalone_metric_pattern, text):
            # Skip if already captured by metric_pattern
            if any(f.evidence and match.group(0) in f.evidence for f in facts):
                continue

            number = match.group(1)
            unit = match.group(2)

            # Get surrounding context for key
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 30)
            evidence = text[start:end].strip()

            # Try to extract what this metric measures
            key = "performance_metric"
            if "query" in evidence.lower() or "time" in evidence.lower():
                key = "query_time"
            elif "speedup" in evidence.lower():
                key = "speedup"
            elif "latency" in evidence.lower():
                key = "latency"

            facts.append(Fact(
                key=key,
                value=f"{number}{unit}",
                category="Metric",
                evidence=evidence,
                source_id=source_id
            ))

        # Pattern 5: Key-value pairs (e.g., "max_pool = 50", "endpoint: /api/v3")
        # Match key = value where value is: number, path, or short string (no spaces)
        kv_pattern = r'([a-z_]+)\s*[:=]\s*([^\s,;]+)'
        for match in re.finditer(kv_pattern, text, re.IGNORECASE):
            key = match.group(1).strip()
            value = match.group(2).strip()

            # Skip if value is too long (likely not a fact)
            if len(value) > 100:
                continue

            # Skip if key is a common word (not a config key)
            common_words = {'and', 'or', 'the', 'for', 'with', 'from', 'set'}
            if key.lower() in common_words:
                continue

            # Skip if value contains "and" (indicates incomplete match)
            if 'and' in value.lower():
                continue

            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            evidence = text[start:end].strip()

            facts.append(Fact(
                key=key,
                value=value,
                category="Definition",
                evidence=evidence,
                source_id=source_id
            ))

        return facts

    def _save_fact(self, fact: Fact):
        """Persist fact to database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO fact_store (key, value, category, evidence, source_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            fact.key,
            fact.value,
            fact.category,
            fact.evidence,
            fact.source_id,
            fact.created_at
        ))
        self.conn.commit()

    def query(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Fact]:
        """Query fact store for matching facts.

        Args:
            query: Search query (matches key or value)
            category: Optional category filter (Definition, Acronym, Entity, Metric)
            limit: Maximum number of results

        Returns:
            List of matching Fact objects, sorted by recency
        """
        cursor = self.conn.cursor()

        if category:
            cursor.execute("""
                SELECT key, value, category, evidence, source_id, created_at
                FROM fact_store
                WHERE (key LIKE ? OR value LIKE ?) AND category = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", category, limit))
        else:
            cursor.execute("""
                SELECT key, value, category, evidence, source_id, created_at
                FROM fact_store
                WHERE key LIKE ? OR value LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit))

        facts = []
        for row in cursor.fetchall():
            facts.append(Fact(
                key=row[0],
                value=row[1],
                category=row[2],
                evidence=row[3],
                source_id=row[4],
                created_at=row[5]
            ))

        return facts

    def get_by_key(self, key: str) -> Optional[Fact]:
        """Get the most recent fact for an exact key match.

        Args:
            key: Exact key to match

        Returns:
            Most recent Fact object or None
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT key, value, category, evidence, source_id, created_at
            FROM fact_store
            WHERE key = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (key,))

        row = cursor.fetchone()
        if not row:
            return None

        return Fact(
            key=row[0],
            value=row[1],
            category=row[2],
            evidence=row[3],
            source_id=row[4],
            created_at=row[5]
        )

    def get_all_by_category(self, category: str, limit: int = 50) -> List[Fact]:
        """Get all facts in a category.

        Args:
            category: Fact category (Definition, Acronym, Entity, Metric)
            limit: Maximum number of results

        Returns:
            List of Fact objects
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT key, value, category, evidence, source_id, created_at
            FROM fact_store
            WHERE category = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (category, limit))

        facts = []
        for row in cursor.fetchall():
            facts.append(Fact(
                key=row[0],
                value=row[1],
                category=row[2],
                evidence=row[3],
                source_id=row[4],
                created_at=row[5]
            ))

        return facts

    def count_facts(self) -> int:
        """Count total number of facts stored."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM fact_store")
        return cursor.fetchone()[0]

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
