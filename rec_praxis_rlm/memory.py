"""Procedural memory storage and retrieval with hybrid similarity scoring."""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from rec_praxis_rlm.config import MemoryConfig
from rec_praxis_rlm.embeddings import SentenceTransformerEmbedding, EmbeddingProvider
from rec_praxis_rlm.exceptions import StorageError, EmbeddingError
from rec_praxis_rlm.telemetry import emit_event

# Optional FactStore import (for semantic memory integration)
try:
    from rec_praxis_rlm.fact_store import FactStore
    FACTSTORE_AVAILABLE = True
except ImportError:
    FACTSTORE_AVAILABLE = False
    FactStore = None  # type: ignore

logger = logging.getLogger(__name__)

# Storage format version for backward compatibility
STORAGE_VERSION = "2.0"

# Try to import FAISS for fast similarity search
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    FAISS_AVAILABLE = False


class Experience(BaseModel):
    """A single procedural memory experience.

    Attributes:
        env_features: List of environmental feature strings
        goal: Goal description string
        action: Action taken (code, selector, etc.)
        result: Result of the action
        success: Whether the action was successful
        timestamp: Unix timestamp when experience was created
        embedding: Optional pre-computed goal embedding vector
        cost: Optional cost in dollars (for LLM calls)
        metadata: Optional metadata dictionary
    """

    model_config = {"strict": True}

    env_features: list[str] = Field(
        ...,
        description="List of environmental feature strings",
    )
    goal: str = Field(
        ...,
        description="Goal description string",
    )
    action: str = Field(
        ...,
        description="Action taken",
    )
    result: str = Field(
        ...,
        description="Result of the action",
    )
    success: bool = Field(
        ...,
        description="Whether the action was successful",
    )
    timestamp: float = Field(
        ...,
        gt=0.0,
        description="Unix timestamp when experience was created",
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Optional pre-computed goal embedding vector",
    )
    cost: Optional[float] = Field(
        default=None,
        description="Optional cost in dollars",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Optional metadata dictionary",
    )


class ProceduralMemory:
    """Procedural memory with hybrid Jaccard + cosine similarity retrieval.

    This class manages a collection of experiences stored in JSONL format,
    supporting efficient similarity-based retrieval using environmental
    features (Jaccard) and goal embeddings (cosine).
    """

    def __init__(
        self,
        config: MemoryConfig = MemoryConfig(),
        use_faiss: bool = True,
        embedding_provider: Optional[EmbeddingProvider] = None,
        fact_store: Optional["FactStore"] = None,
    ) -> None:
        """Initialize procedural memory.

        Args:
            config: Memory configuration
            use_faiss: If True and FAISS available, use FAISS index for fast retrieval
            embedding_provider: Optional pre-configured embedding provider for dependency injection.
                               If None, will create default provider from config.embedding_model
            fact_store: Optional FactStore for semantic memory integration.
                       If provided, facts will be auto-extracted from experiences.
        """
        self.config = config
        self.experiences: list[Experience] = []
        self.use_faiss = use_faiss and FAISS_AVAILABLE

        # Initialize embedding provider (dependency injection or default creation)
        if embedding_provider is not None:
            # Use injected provider
            self.embedding_provider: Optional[EmbeddingProvider] = embedding_provider
        else:
            # Create default provider from config
            self.embedding_provider = self._create_embedding_provider(config)

        # Initialize FAISS index (will be built after loading experiences)
        self._faiss_index: Optional["faiss.Index"] = None
        self._embedding_dimension: Optional[int] = None

        # Initialize ThreadPoolExecutor for async operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="memory-async")

        # Store fact_store reference for semantic memory integration
        self.fact_store = fact_store

        # Corruption statistics (tracked during load)
        self.corruption_stats = {
            "checksum_failures": 0,
            "parse_errors": 0,
            "total_lines_scanned": 0,
        }

        # Load existing experiences if storage file exists
        if config.storage_path != ":memory:" and os.path.exists(config.storage_path):
            self._load_experiences()

        # Build FAISS index if we have experiences with embeddings
        if self.use_faiss and self.experiences:
            self._rebuild_faiss_index()

    def _create_embedding_provider(self, config: MemoryConfig) -> Optional[EmbeddingProvider]:
        """Create default embedding provider from configuration.

        Args:
            config: Memory configuration with embedding_model setting

        Returns:
            EmbeddingProvider instance or None if model not configured or fails to load
        """
        if config.embedding_model:
            try:
                return SentenceTransformerEmbedding(config.embedding_model)
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                return None
        else:
            return None

    def _load_experiences(self) -> None:
        """Load experiences from JSONL storage file with version migration."""
        try:
            with open(self.config.storage_path, "r") as f:
                lines = f.readlines()

            if not lines:
                return

            # Check for version marker in first line
            first_line = lines[0].strip()
            if first_line and first_line.startswith('{"__version__"'):
                try:
                    version_obj = json.loads(first_line)
                    file_version = version_obj.get("__version__", "0.0")
                    logger.info(f"Loading storage version {file_version}")

                    # Migrate if needed
                    if file_version != STORAGE_VERSION:
                        lines = self._migrate_storage(file_version, lines[1:])
                    else:
                        lines = lines[1:]  # Skip version marker
                except json.JSONDecodeError:
                    # Not a version marker, treat as regular experience
                    logger.warning("Invalid version marker, treating as legacy format")
            else:
                # No version marker - legacy format (pre-1.0)
                logger.info("Loading legacy storage format (no version marker)")

            # Load experiences from remaining lines
            for line_num, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue

                self.corruption_stats["total_lines_scanned"] += 1

                try:
                    # For v2.0+, lines have format: checksum|json_data
                    if file_version >= "2.0" and "|" in line:
                        checksum, json_data = line.split("|", 1)

                        # Validate checksum
                        computed_checksum = hashlib.sha256(json_data.encode()).hexdigest()
                        if computed_checksum != checksum:
                            self.corruption_stats["checksum_failures"] += 1
                            logger.error(
                                f"Checksum mismatch on line {line_num}: "
                                f"expected {checksum}, got {computed_checksum}. "
                                f"Data may be corrupted. Skipping this experience."
                            )
                            continue

                        obj = json.loads(json_data)
                    else:
                        # Legacy format (no checksum)
                        obj = json.loads(line)

                    exp = Experience(**obj)
                    self.experiences.append(exp)
                except Exception as e:
                    self.corruption_stats["parse_errors"] += 1
                    logger.warning(
                        f"Skipping corrupted line {line_num} in {self.config.storage_path}: {e}"
                    )

            # Emit corruption statistics
            if self.corruption_stats["checksum_failures"] > 0 or self.corruption_stats["parse_errors"] > 0:
                emit_event(
                    "memory.corruption_detected",
                    {
                        "checksum_failures": self.corruption_stats["checksum_failures"],
                        "parse_errors": self.corruption_stats["parse_errors"],
                        "total_lines_scanned": self.corruption_stats["total_lines_scanned"],
                        "corruption_rate": (
                            (self.corruption_stats["checksum_failures"] + self.corruption_stats["parse_errors"])
                            / max(self.corruption_stats["total_lines_scanned"], 1)
                        ),
                    },
                )
        except FileNotFoundError:
            # File doesn't exist yet - that's okay
            pass

    def _migrate_storage(self, from_version: str, lines: list[str]) -> list[str]:
        """Migrate storage from one version to another.

        Args:
            from_version: Source version string
            lines: Storage lines to migrate (excluding version marker)

        Returns:
            Migrated lines

        Raises:
            StorageError: If migration fails or version is unsupported
        """
        logger.info(f"Migrating storage from version {from_version} to {STORAGE_VERSION}")

        if from_version == "0.0" or from_version == "1.0":
            # Migrate to 2.0 by adding checksums
            logger.info(f"Migrating from {from_version} to 2.0 (adding checksums)")
            migrated_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Compute checksum for existing JSON data
                checksum = hashlib.sha256(line.encode()).hexdigest()

                # New format: checksum|json_data
                migrated_line = f"{checksum}|{line}"
                migrated_lines.append(migrated_line)

            return migrated_lines
        elif from_version == "2.0":
            # Already at 2.0, no migration needed
            return lines
        else:
            # Unsupported version
            raise StorageError(
                f"Unsupported storage version {from_version}. "
                f"Cannot migrate to {STORAGE_VERSION}. "
                f"Please use an older version of rec_praxis_rlm to read this file."
            )

    def _append_experience(self, experience: Experience) -> None:
        """Append an experience to JSONL file atomically with version marker.

        Args:
            experience: Experience to append
        """
        if self.config.storage_path == ":memory:":
            # In-memory mode - no file I/O
            return

        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(self.config.storage_path), exist_ok=True)

            # Atomic write: write to temp file, then rename
            fd, temp_path = tempfile.mkstemp(
                dir=os.path.dirname(self.config.storage_path),
                prefix=".memory_",
                suffix=".tmp",
            )

            try:
                # Copy existing content + new experience
                with os.fdopen(fd, "w") as temp_file:
                    if os.path.exists(self.config.storage_path):
                        # File exists - copy existing content
                        with open(self.config.storage_path, "r") as existing:
                            temp_file.write(existing.read())
                    else:
                        # New file - write version marker as first line
                        version_marker = json.dumps({"__version__": STORAGE_VERSION})
                        temp_file.write(version_marker + "\n")

                    # Append new experience with checksum (v2.0 format)
                    json_data = experience.model_dump_json()
                    checksum = hashlib.sha256(json_data.encode()).hexdigest()
                    temp_file.write(f"{checksum}|{json_data}\n")

                # Atomic rename
                os.replace(temp_path, self.config.storage_path)
            except Exception:
                # Clean up temp file on error
                if os.path.exists(temp_path):  # pragma: no branch
                    os.unlink(temp_path)
                raise

        except Exception as e:
            raise StorageError(f"Failed to append experience: {e}")

    def _jaccard_similarity(self, set_a: set[str], set_b: set[str]) -> float:
        """Compute Jaccard similarity between two sets.

        Args:
            set_a: First set
            set_b: Second set

        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if len(set_a) == 0 and len(set_b) == 0:
            return 0.0
        if len(set_a) == 0 or len(set_b) == 0:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / union if union > 0 else 0.0

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec_a: First vector
            vec_b: Second vector

        Returns:
            Cosine similarity score (-1.0 to 1.0)
        """
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must have same dimension")

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _rebuild_faiss_index(self) -> None:
        """Rebuild FAISS index from current experiences.

        This method extracts all embeddings from experiences and builds
        a FAISS IndexFlatIP (Inner Product) index for fast similarity search.

        Embeddings with inconsistent dimensions are filtered out and logged.
        If all embeddings are invalid, the FAISS index is disabled.
        """
        if not self.use_faiss:
            return

        try:
            # Collect all embeddings
            embeddings_list = []
            for exp in self.experiences:
                if exp.embedding is not None:
                    # Validate embedding is a list/array
                    if not isinstance(exp.embedding, (list, np.ndarray)):
                        continue
                    embeddings_list.append(exp.embedding)

            if not embeddings_list:
                # No embeddings available
                self._faiss_index = None
                self._embedding_dimension = None
                logger.debug("No embeddings available for FAISS index")
                return

            # Determine embedding dimension from first embedding
            expected_dim = len(embeddings_list[0])

            # Validate all embeddings have same dimension
            filtered_embeddings = []
            for i, emb in enumerate(embeddings_list):
                if len(emb) != expected_dim:
                    logger.warning(
                        f"Skipping embedding {i} with dimension {len(emb)} "
                        f"(expected {expected_dim})"
                    )
                    continue
                filtered_embeddings.append(emb)

            if not filtered_embeddings:
                # All embeddings were filtered out due to dimension mismatch
                self._faiss_index = None
                self._embedding_dimension = None
                logger.warning("No valid embeddings after dimension validation")
                return

            self._embedding_dimension = expected_dim

            # Convert to numpy array and normalize for cosine similarity
            embeddings_np = np.array(filtered_embeddings, dtype=np.float32)

            # Normalize vectors for cosine similarity (then use inner product)
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_normalized = embeddings_np / norms

            # Create FAISS index (IndexFlatIP for normalized vectors = cosine similarity)
            self._faiss_index = faiss.IndexFlatIP(self._embedding_dimension)
            self._faiss_index.add(embeddings_normalized)

            logger.info(
                f"Built FAISS index with {len(embeddings_list)} embeddings "
                f"(dimension={self._embedding_dimension})"
            )
        except Exception as e:
            # If FAISS index building fails, log warning and disable FAISS
            logger.warning(f"Failed to build FAISS index: {e}. Falling back to linear scan.")
            self._faiss_index = None
            self._embedding_dimension = None

    def _compute_similarity_score(
        self,
        experience: Experience,
        query_env_features: list[str],
        query_goal_embedding: Optional[list[float]],
    ) -> float:
        """Compute hybrid similarity score for an experience.

        Args:
            experience: Experience to score
            query_env_features: Query environmental features
            query_goal_embedding: Query goal embedding (optional)

        Returns:
            Hybrid similarity score (0.0 to 1.0)
        """
        # Environmental similarity (Jaccard)
        env_sim = self._jaccard_similarity(set(experience.env_features), set(query_env_features))

        # Goal similarity (cosine)
        if query_goal_embedding and experience.embedding:
            goal_sim = self._cosine_similarity(experience.embedding, query_goal_embedding)
        else:
            # No embeddings available - use env similarity only
            goal_sim = 0.0

        # Weighted combination
        score = self.config.env_weight * env_sim + self.config.goal_weight * goal_sim

        return score

    def store(self, experience: Experience) -> None:
        """Store an experience in memory.

        Args:
            experience: Experience to store
        """
        # Compute embedding if missing and provider available
        if experience.embedding is None and self.embedding_provider:
            try:
                experience.embedding = self.embedding_provider.embed(experience.goal)
            except Exception as e:
                logger.warning(f"Failed to compute embedding: {e}")

        # Add to in-memory list
        self.experiences.append(experience)

        # Extract facts if FactStore is configured
        if self.fact_store is not None:
            try:
                # Generate source_id from timestamp
                source_id = f"exp_{int(experience.timestamp)}"

                # Extract facts from action + result text
                text = f"{experience.action}. {experience.result}"
                facts = self.fact_store.extract_facts(text, source_id=source_id)

                logger.debug(f"Extracted {len(facts)} facts from experience")
            except Exception as e:
                logger.warning(f"Failed to extract facts: {e}")

        # Update FAISS index if available
        if self.use_faiss and experience.embedding is not None:
            if self._faiss_index is None:
                # First experience with embedding - rebuild index
                self._rebuild_faiss_index()
            else:
                # Incremental add to existing index
                embedding_np = np.array([experience.embedding], dtype=np.float32)
                # Normalize for cosine similarity
                norm = np.linalg.norm(embedding_np)
                if norm > 0:
                    embedding_normalized = embedding_np / norm
                    self._faiss_index.add(embedding_normalized)

        # Persist to storage
        self._append_experience(experience)

        # Emit telemetry
        emit_event(
            "memory.store",
            {
                "env_features": experience.env_features,
                "success": experience.success,
                "has_embedding": experience.embedding is not None,
            },
        )

    def recall(
        self,
        env_features: list[str],
        goal: str,
        top_k: Optional[int] = None,
    ) -> list[Experience]:
        """Recall top-k most similar experiences using FAISS-accelerated search.

        Args:
            env_features: Environmental features to match
            goal: Goal to match
            top_k: Number of experiences to return (defaults to config.top_k)

        Returns:
            List of experiences sorted by similarity (descending)
        """
        if top_k is None:
            top_k = self.config.top_k

        # Compute goal embedding
        goal_embedding: Optional[list[float]] = None
        if self.embedding_provider:
            try:
                goal_embedding = self.embedding_provider.embed(goal)
            except Exception as e:
                logger.warning(f"Failed to compute query embedding: {e}")

        # Fast path: Use FAISS if available
        if self.use_faiss and self._faiss_index is not None and goal_embedding is not None:
            return self._recall_with_faiss(env_features, goal_embedding, top_k)

        # Slow path: Linear scan (fallback or no embeddings)
        return self._recall_linear(env_features, goal_embedding, top_k)

    def _recall_with_faiss(
        self,
        env_features: list[str],
        goal_embedding: list[float],
        top_k: int,
    ) -> list[Experience]:
        """Fast recall using FAISS index.

        Strategy:
        1. Use FAISS to get top candidates by goal similarity (fast)
        2. Re-rank with full hybrid score (env + goal)
        3. Return top-k after re-ranking

        Args:
            env_features: Environmental features to match
            goal_embedding: Query goal embedding
            top_k: Number of experiences to return

        Returns:
            List of experiences sorted by similarity (descending)
        """
        # Get candidate pool from FAISS (over-fetch for better recall after re-ranking)
        # Fetch more candidates than needed because environmental filtering might reduce count
        candidate_multiplier = 5  # Fetch 5x more candidates for re-ranking
        k_candidates = min(top_k * candidate_multiplier, len(self.experiences))

        # Normalize query embedding for cosine similarity
        query_np = np.array([goal_embedding], dtype=np.float32)
        norm = np.linalg.norm(query_np)
        if norm > 0:
            query_normalized = query_np / norm
        else:
            # Zero vector - fall back to linear scan
            return self._recall_linear(env_features, goal_embedding, top_k)

        # FAISS search (returns distances and indices)
        distances, indices = self._faiss_index.search(query_normalized, k_candidates)

        # Map FAISS indices to experiences (only those with embeddings)
        experiences_with_embeddings = [exp for exp in self.experiences if exp.embedding is not None]

        # Re-rank candidates with full hybrid score
        scored_experiences: list[tuple[Experience, float]] = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(experiences_with_embeddings):
                continue  # Invalid index

            exp = experiences_with_embeddings[idx]

            # Filter by success if required
            if self.config.require_success and not exp.success:
                continue

            # Compute full hybrid score (env + goal)
            score = self._compute_similarity_score(exp, env_features, goal_embedding)

            # Filter by threshold
            if score >= self.config.similarity_threshold:
                scored_experiences.append((exp, score))

        # Sort by hybrid score descending
        scored_experiences.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        results = [exp for exp, score in scored_experiences[:top_k]]

        # Emit telemetry
        emit_event(
            "memory.recall",
            {
                "query_env_features": env_features,
                "query_goal": None,  # Don't log full goal text
                "results_count": len(results),
                "total_candidates": len(self.experiences),
                "used_faiss": True,
                "faiss_candidates": k_candidates,
            },
        )

        return results

    def _recall_linear(
        self,
        env_features: list[str],
        goal_embedding: Optional[list[float]],
        top_k: int,
    ) -> list[Experience]:
        """Linear scan recall (fallback when FAISS not available).

        Args:
            env_features: Environmental features to match
            goal_embedding: Query goal embedding (optional)
            top_k: Number of experiences to return

        Returns:
            List of experiences sorted by similarity (descending)
        """
        # Score all experiences
        scored_experiences: list[tuple[Experience, float]] = []
        for exp in self.experiences:
            # Filter by success if required
            if self.config.require_success and not exp.success:
                continue

            score = self._compute_similarity_score(exp, env_features, goal_embedding)

            # Filter by threshold
            if score >= self.config.similarity_threshold:
                scored_experiences.append((exp, score))

        # Sort by score descending
        scored_experiences.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        results = [exp for exp, score in scored_experiences[:top_k]]

        # Emit telemetry
        emit_event(
            "memory.recall",
            {
                "query_env_features": env_features,
                "query_goal": None,  # Don't log full goal text
                "results_count": len(results),
                "total_candidates": len(self.experiences),
                "used_faiss": False,
            },
        )

        return results

    def size(self) -> int:
        """Return number of experiences in memory.

        Returns:
            Number of experiences
        """
        return len(self.experiences)

    def get_corruption_stats(self) -> dict[str, int]:
        """Get corruption statistics from last load operation.

        Returns:
            Dictionary with checksum_failures, parse_errors, and total_lines_scanned
        """
        return self.corruption_stats.copy()

    def compact(self, keep_recent_n: Optional[int] = None) -> int:
        """Compact memory by removing old experiences.

        Args:
            keep_recent_n: Keep only the N most recent experiences

        Returns:
            Number of experiences removed
        """
        if keep_recent_n is None:
            return 0

        initial_size = len(self.experiences)

        # Sort by timestamp descending, keep top N
        self.experiences.sort(key=lambda x: x.timestamp, reverse=True)
        self.experiences = self.experiences[:keep_recent_n]

        # Rebuild FAISS index after compaction
        if self.use_faiss:
            self._rebuild_faiss_index()

        # Rewrite storage file with version marker and checksums
        if self.config.storage_path != ":memory:":
            try:
                with open(self.config.storage_path, "w") as f:
                    # Write version marker first
                    version_marker = json.dumps({"__version__": STORAGE_VERSION})
                    f.write(version_marker + "\n")

                    # Write experiences with checksums (v2.0 format)
                    for exp in self.experiences:
                        json_data = exp.model_dump_json()
                        checksum = hashlib.sha256(json_data.encode()).hexdigest()
                        f.write(f"{checksum}|{json_data}\n")
            except Exception as e:
                raise StorageError(f"Failed to compact storage: {e}")

        removed_count = initial_size - len(self.experiences)
        return removed_count

    def recompute_embeddings(self, new_model: str) -> None:
        """Recompute all embeddings with a new model.

        Args:
            new_model: Name of new embedding model
        """
        # Load new model
        try:
            new_provider = SentenceTransformerEmbedding(new_model)
        except Exception as e:
            raise EmbeddingError(f"Failed to load new model {new_model}: {e}")

        # Recompute all embeddings
        for exp in self.experiences:
            try:
                exp.embedding = new_provider.embed(exp.goal)
            except Exception as e:
                logger.warning(f"Failed to recompute embedding for experience: {e}")

        # Rebuild FAISS index with new embeddings
        if self.use_faiss:
            self._rebuild_faiss_index()

        # Rewrite storage with version marker and checksums
        if self.config.storage_path != ":memory:":
            try:
                with open(self.config.storage_path, "w") as f:
                    # Write version marker first
                    version_marker = json.dumps({"__version__": STORAGE_VERSION})
                    f.write(version_marker + "\n")

                    # Write experiences with checksums (v2.0 format)
                    for exp in self.experiences:
                        json_data = exp.model_dump_json()
                        checksum = hashlib.sha256(json_data.encode()).hexdigest()
                        f.write(f"{checksum}|{json_data}\n")
            except Exception as e:
                raise StorageError(f"Failed to rewrite storage: {e}")

        # Update provider
        self.embedding_provider = new_provider

    async def arecall(
        self,
        env_features: list[str],
        goal: str,
        top_k: Optional[int] = None,
    ) -> list[Experience]:
        """Async version of recall() for high-concurrency deployments.

        Uses ThreadPoolExecutor to run the synchronous recall() in a thread pool,
        preventing blocking of the event loop.

        Args:
            env_features: Environmental features to match
            goal: Goal to match
            top_k: Number of experiences to return

        Returns:
            List of experiences sorted by similarity (descending)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.recall, env_features, goal, top_k)
