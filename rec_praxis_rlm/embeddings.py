"""Embedding abstraction with local and API fallback support."""

import hashlib
import re
import sys
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional

# Default cache size for embedding LRU caches
# Balances memory usage (~40MB for 384-dim embeddings) with hit rate
DEFAULT_EMBEDDING_CACHE_SIZE = 10000

# Suppress bitsandbytes stdout errors during sentence_transformers import
_stdout_backup = sys.stdout
sys.stdout = sys.stderr

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore[assignment]
finally:
    sys.stdout = _stdout_backup

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENAI_AVAILABLE = False


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed

        Returns:
            List of floating point numbers representing the embedding
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        pass


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Local embedding using sentence-transformers library with LRU cache.

    This provider runs models locally without requiring API keys.
    Embeddings are cached with LRU eviction for 10-100x performance improvement.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_size: int = DEFAULT_EMBEDDING_CACHE_SIZE,
    ) -> None:
        """Initialize sentence-transformers model with embedding cache.

        Args:
            model_name: Name of the sentence-transformers model to use
            cache_size: Maximum number of embeddings to cache (default: 10,000)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is None:  # type: ignore[name-defined]
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        try:
            self.model = SentenceTransformer(model_name)  # type: ignore[name-defined]
        except Exception as e:
            raise ImportError(
                "Failed to initialize sentence-transformers model. "
                "Ensure sentence-transformers and its dependencies are installed correctly."
            ) from e
        self.cache_size = cache_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text using SHA256 hash.

        Args:
            text: Text to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> list[float]:
        """Embed a single text string with LRU caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (cached if available)
        """
        # Check cache
        cache_key = self._get_cache_key(text)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                # Cache hit - move to end (most recent)
                self._cache.move_to_end(cache_key)
                return cached

        # Cache miss - compute embedding
        embeddings = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        # Handle both numpy arrays and lists (for mocking)
        if isinstance(embeddings, list):
            embedding = embeddings[0]
        else:
            embedding = embeddings[0].tolist()

        # Add to cache (LRU eviction)
        with self._cache_lock:
            self._cache[cache_key] = embedding
            if len(self._cache) > self.cache_size:
                # Evict oldest entry (FIFO when cache is full)
                self._cache.popitem(last=False)

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts with cache awareness.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (some from cache, some computed)
        """
        results: list[list[float]] = []
        to_compute: list[tuple[int, str]] = []  # (index, text) pairs

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            with self._cache_lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    # Cache hit
                    self._cache.move_to_end(cache_key)
                    results.append(cached)
                    continue
            # Cache miss - mark for computation
            results.append([])  # Placeholder
            to_compute.append((i, text))

        # Compute uncached embeddings in batch
        if to_compute:
            texts_to_compute = [text for _, text in to_compute]
            embeddings = self.model.encode(
                texts_to_compute, convert_to_numpy=True, show_progress_bar=False
            )
            # Handle both numpy arrays and lists (for mocking)
            if isinstance(embeddings, list):
                computed = embeddings
            else:
                computed = embeddings.tolist()

            # Update results and cache
            for (index, text), embedding in zip(to_compute, computed):
                results[index] = embedding
                cache_key = self._get_cache_key(text)
                with self._cache_lock:
                    self._cache[cache_key] = embedding
                    if len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)

        return results


class APIEmbedding(EmbeddingProvider):
    """API-based embedding fallback (OpenAI, Cohere, Voyage) with LRU cache.

    This provider requires API keys and makes network requests.
    Embeddings are cached with LRU eviction to reduce API costs and latency.
    """

    def __init__(
        self,
        api_provider: str = "openai",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        cache_size: int = DEFAULT_EMBEDDING_CACHE_SIZE,
    ) -> None:
        """Initialize API embedding provider with cache.

        Args:
            api_provider: API provider name (openai, cohere, voyage)
            api_key: API key for the provider
            model_name: Optional model name override
            cache_size: Maximum number of embeddings to cache (default: 10,000)
        """
        self.api_provider = api_provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.cache_size = cache_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_lock = threading.Lock()

        if self.api_provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not installed. Install with: pip install openai")
            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = model_name or "text-embedding-3-small"
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text using SHA256 hash.

        Args:
            text: Text to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> list[float]:
        """Embed a single text string using API with LRU caching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (cached if available, saving API cost)
        """
        # Check cache
        cache_key = self._get_cache_key(text)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                # Cache hit - move to end (most recent)
                self._cache.move_to_end(cache_key)
                return cached

        # Cache miss - call API
        response = self.client.embeddings.create(input=[text], model=self.model_name)
        embedding = response.data[0].embedding

        # Add to cache (LRU eviction)
        with self._cache_lock:
            self._cache[cache_key] = embedding
            if len(self._cache) > self.cache_size:
                # Evict oldest entry (FIFO when cache is full)
                self._cache.popitem(last=False)

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts using API with cache awareness.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (some from cache, some from API)
        """
        results: list[list[float]] = []
        to_compute: list[tuple[int, str]] = []  # (index, text) pairs

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            with self._cache_lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    # Cache hit
                    self._cache.move_to_end(cache_key)
                    results.append(cached)
                    continue
            # Cache miss - mark for API call
            results.append([])  # Placeholder
            to_compute.append((i, text))

        # Compute uncached embeddings via API batch call
        if to_compute:
            texts_to_compute = [text for _, text in to_compute]
            response = self.client.embeddings.create(input=texts_to_compute, model=self.model_name)
            computed = [item.embedding for item in response.data]

            # Update results and cache
            for (index, text), embedding in zip(to_compute, computed):
                results[index] = embedding
                cache_key = self._get_cache_key(text)
                with self._cache_lock:
                    self._cache[cache_key] = embedding
                    if len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)

        return results


class TextSimilarityFallback:
    """Text-based similarity fallback using Jaccard similarity.

    This does not produce embeddings but can compute similarity
    directly between texts when embeddings are unavailable.
    """

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        # Tokenize: lowercase and split on whitespace/punctuation
        tokens1 = set(self._tokenize(text1.lower()))
        tokens2 = set(self._tokenize(text2.lower()))

        # Handle empty sets
        if len(tokens1) == 0 and len(tokens2) == 0:
            return 0.0
        if len(tokens1) == 0 or len(tokens2) == 0:
            return 0.0

        # Jaccard: intersection / union
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text by splitting on whitespace and punctuation."""
        # Split on whitespace and remove punctuation
        tokens = re.findall(r"\w+", text)
        return tokens
