"""Unit tests for embedding abstraction."""
import pytest
from unittest.mock import Mock, patch, MagicMock

from rec_praxis_rlm.embeddings import (
    EmbeddingProvider,
    SentenceTransformerEmbedding,
    APIEmbedding,
    TextSimilarityFallback,
)


class TestSentenceTransformerEmbedding:
    """Tests for local sentence-transformers embedding."""

    @patch("rec_praxis_rlm.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE", False)
    def test_import_error_when_not_installed(self) -> None:
        """Test ImportError when sentence-transformers not installed."""
        with pytest.raises(ImportError, match="sentence-transformers not installed"):
            SentenceTransformerEmbedding()

    @patch("rec_praxis_rlm.embeddings.SentenceTransformer")
    def test_model_loading(self, mock_st: Mock) -> None:
        """Test that sentence-transformers model loads correctly."""
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        provider = SentenceTransformerEmbedding(model_name="test-model")
        mock_st.assert_called_once_with("test-model")

    @patch("rec_praxis_rlm.embeddings.SentenceTransformer")
    def test_embed_single_text(self, mock_st: Mock) -> None:
        """Test embedding a single text string."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_st.return_value = mock_model

        provider = SentenceTransformerEmbedding()
        embedding = provider.embed("test text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once_with(
            ["test text"], convert_to_numpy=True, show_progress_bar=False
        )

    @patch("rec_praxis_rlm.embeddings.SentenceTransformer")
    def test_embed_batch(self, mock_st: Mock) -> None:
        """Test embedding multiple texts."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_st.return_value = mock_model

        provider = SentenceTransformerEmbedding()
        embeddings = provider.embed_batch(["text1", "text2"])

        assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
        mock_model.encode.assert_called_once()

    @patch("rec_praxis_rlm.embeddings.SentenceTransformer")
    def test_dimension_consistency(self, mock_st: Mock) -> None:
        """Test that embedding dimensions are consistent."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_st.return_value = mock_model

        provider = SentenceTransformerEmbedding()
        emb1 = provider.embed("text1")
        emb2 = provider.embed("text2")

        assert len(emb1) == len(emb2)

    @patch("rec_praxis_rlm.embeddings.SentenceTransformer")
    def test_embed_with_numpy_array(self, mock_st: Mock) -> None:
        """Test embed when model returns numpy array."""
        import numpy as np
        mock_model = MagicMock()
        # Return numpy array instead of list
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st.return_value = mock_model

        provider = SentenceTransformerEmbedding()
        embedding = provider.embed("test")

        # Should convert numpy array to list
        assert isinstance(embedding, list)
        assert len(embedding) == 3

    @patch("rec_praxis_rlm.embeddings.SentenceTransformer")
    def test_embed_batch_with_numpy_array(self, mock_st: Mock) -> None:
        """Test embed_batch when model returns numpy array."""
        import numpy as np
        mock_model = MagicMock()
        # Return numpy array
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_st.return_value = mock_model

        provider = SentenceTransformerEmbedding()
        embeddings = provider.embed_batch(["text1", "text2"])

        # Should convert to list of lists
        assert isinstance(embeddings, list)
        assert all(isinstance(e, list) for e in embeddings)


class TestAPIEmbedding:
    """Tests for API-based embedding fallback."""

    @patch("rec_praxis_rlm.embeddings.OPENAI_AVAILABLE", False)
    def test_import_error_when_openai_not_installed(self) -> None:
        """Test ImportError when openai not installed."""
        with pytest.raises(ImportError, match="openai not installed"):
            APIEmbedding(api_provider="openai", api_key="test-key")

    def test_unsupported_provider_raises_error(self) -> None:
        """Test ValueError for unsupported API provider."""
        with patch("rec_praxis_rlm.embeddings.openai"):
            with pytest.raises(ValueError, match="Unsupported API provider"):
                APIEmbedding(api_provider="unsupported", api_key="test-key")

    def test_openai_provider(self) -> None:
        """Test OpenAI API embedding provider."""
        with patch("rec_praxis_rlm.embeddings.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3])
            ]
            mock_openai.OpenAI.return_value = mock_client

            provider = APIEmbedding(api_provider="openai", api_key="test-key")
            embedding = provider.embed("test text")

            assert embedding == [0.1, 0.2, 0.3]

    def test_openai_embed_batch(self) -> None:
        """Test OpenAI batch embedding."""
        with patch("rec_praxis_rlm.embeddings.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value.data = [
                MagicMock(embedding=[0.1, 0.2]),
                MagicMock(embedding=[0.3, 0.4]),
            ]
            mock_openai.OpenAI.return_value = mock_client

            provider = APIEmbedding(api_provider="openai", api_key="test-key")
            embeddings = provider.embed_batch(["text1", "text2"])

            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2]
            assert embeddings[1] == [0.3, 0.4]

    def test_fallback_on_api_failure(self) -> None:
        """Test that API failures raise appropriate errors."""
        with patch("rec_praxis_rlm.embeddings.openai") as mock_openai:
            mock_openai.OpenAI.return_value.embeddings.create.side_effect = Exception(
                "API error"
            )

            provider = APIEmbedding(api_provider="openai", api_key="test-key")
            with pytest.raises(Exception, match="API error"):
                provider.embed("test text")


class TestTextSimilarityFallback:
    """Tests for text-based similarity fallback."""

    def test_jaccard_similarity(self) -> None:
        """Test Jaccard similarity computation."""
        provider = TextSimilarityFallback()

        # Identical texts should have similarity 1.0
        sim1 = provider.compute_similarity("hello world", "hello world")
        assert sim1 == 1.0

        # No overlap should have similarity 0.0
        sim2 = provider.compute_similarity("hello world", "foo bar")
        assert sim2 == 0.0

        # Partial overlap
        sim3 = provider.compute_similarity("hello world foo", "hello world bar")
        assert 0.0 < sim3 < 1.0

    def test_case_sensitivity(self) -> None:
        """Test that similarity is case-insensitive."""
        provider = TextSimilarityFallback()

        sim = provider.compute_similarity("Hello World", "hello world")
        assert sim == 1.0

    def test_tokenization(self) -> None:
        """Test that text is properly tokenized."""
        provider = TextSimilarityFallback()

        # Punctuation should be handled
        sim = provider.compute_similarity("hello, world!", "hello world")
        assert sim > 0.5  # Should have high similarity despite punctuation

    def test_empty_strings(self) -> None:
        """Test handling of empty strings."""
        provider = TextSimilarityFallback()

        # Empty strings should have 0.0 similarity
        sim1 = provider.compute_similarity("", "")
        assert sim1 == 0.0

        sim2 = provider.compute_similarity("hello", "")
        assert sim2 == 0.0


class TestEmbeddingProvider:
    """Tests for base EmbeddingProvider interface."""

    def test_abstract_methods(self) -> None:
        """Test that EmbeddingProvider is abstract."""
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore


# ============================================================================
# Cache-specific tests for LRU embedding cache
# ============================================================================


class TestSentenceTransformerCache:
    """Tests for SentenceTransformer LRU cache functionality."""

    @patch("rec_praxis_rlm.embeddings.SentenceTransformer")
    def test_cache_hit(self, mock_st: Mock) -> None:
        """Test that cache hit returns cached embedding without recomputing."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_st.return_value = mock_model

        embedding = SentenceTransformerEmbedding()

        # First call - cache miss, should compute
        result1 = embedding.embed("hello world")
        assert result1 == [0.1, 0.2, 0.3]

        # Second call - cache hit, should return same embedding
        result2 = embedding.embed("hello world")
        assert result2 == [0.1, 0.2, 0.3]

        # Verify mock was only called once (cache hit didn't trigger computation)
        assert mock_model.encode.call_count == 1

    @patch("rec_praxis_rlm.embeddings.SentenceTransformer")
    def test_cache_miss(self, mock_st: Mock) -> None:
        """Test that cache miss computes and caches new embedding."""
        mock_model = MagicMock()
        # Return different embeddings for different texts
        mock_model.encode.side_effect = [
            [[0.1, 0.2, 0.3]],
            [[0.4, 0.5, 0.6]]
        ]
        mock_st.return_value = mock_model

        embedding = SentenceTransformerEmbedding()

        # Different texts should be cache misses
        result1 = embedding.embed("first text")
        result2 = embedding.embed("second text")

        # Results should be different
        assert result1 != result2

        # Both should now be in cache
        assert len(embedding._cache) == 2

    @patch("rec_praxis_rlm.embeddings.SentenceTransformer")
    def test_cache_lru_eviction(self, mock_st: Mock) -> None:
        """Test that LRU eviction removes oldest entries when cache is full."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_st.return_value = mock_model

        # Create embedding with small cache size
        embedding = SentenceTransformerEmbedding(cache_size=3)

        # Fill cache to capacity
        embedding.embed("text1")
        embedding.embed("text2")
        embedding.embed("text3")
        assert len(embedding._cache) == 3

        # Access text2 to make it more recent
        embedding.embed("text2")

        # Add new text - should evict text1 (oldest)
        embedding.embed("text4")
        assert len(embedding._cache) == 3

        # text1 should be evicted, others should remain
        cache_keys = list(embedding._cache.keys())
        text1_key = embedding._get_cache_key("text1")
        text2_key = embedding._get_cache_key("text2")
        text3_key = embedding._get_cache_key("text3")
        text4_key = embedding._get_cache_key("text4")

        assert text1_key not in cache_keys  # Evicted
        assert text2_key in cache_keys      # Most recent
        assert text3_key in cache_keys      # Second most recent
        assert text4_key in cache_keys      # Just added

    @patch("rec_praxis_rlm.embeddings.SentenceTransformer")
    def test_cache_key_generation(self, mock_st: Mock) -> None:
        """Test that cache keys are generated correctly using SHA256."""
        mock_model = MagicMock()
        mock_st.return_value = mock_model

        embedding = SentenceTransformerEmbedding()

        # Same text should produce same key
        key1 = embedding._get_cache_key("hello world")
        key2 = embedding._get_cache_key("hello world")
        assert key1 == key2

        # Different text should produce different key
        key3 = embedding._get_cache_key("different text")
        assert key1 != key3

        # Key should be SHA256 hex (64 characters)
        assert len(key1) == 64
        assert all(c in "0123456789abcdef" for c in key1)

    @patch("rec_praxis_rlm.embeddings.SentenceTransformer")
    def test_batch_cache_aware(self, mock_st: Mock) -> None:
        """Test that batch processing uses cache for some texts and computes others."""
        mock_model = MagicMock()
        # First calls for pre-populating cache
        mock_model.encode.side_effect = [
            [[0.1, 0.2, 0.3]],  # cached1
            [[0.4, 0.5, 0.6]],  # cached2
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]  # new1, new2 batch
        ]
        mock_st.return_value = mock_model

        embedding = SentenceTransformerEmbedding()

        # Pre-populate cache with some texts
        embedding.embed("cached1")
        embedding.embed("cached2")

        # Batch with mix of cached and uncached texts
        results = embedding.embed_batch([
            "cached1",    # Cache hit
            "new1",       # Cache miss
            "cached2",    # Cache hit
            "new2"        # Cache miss
        ])

        # Should have 4 results
        assert len(results) == 4

        # Mock should be called 3 times total (2 singles + 1 batch of 2)
        assert mock_model.encode.call_count == 3

        # Last call should be for the 2 uncached texts only
        last_call_args = mock_model.encode.call_args[0][0]
        assert len(last_call_args) == 2  # Only 2 uncached texts
        assert "new1" in last_call_args
        assert "new2" in last_call_args

    @patch("rec_praxis_rlm.embeddings.SentenceTransformer")
    def test_custom_cache_size(self, mock_st: Mock) -> None:
        """Test that custom cache size is respected."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_st.return_value = mock_model

        # Create embedding with custom cache size
        embedding = SentenceTransformerEmbedding(cache_size=5)

        # Fill cache
        for i in range(5):
            embedding.embed(f"text{i}")

        assert len(embedding._cache) == 5

        # Add one more - should evict oldest
        embedding.embed("text5")
        assert len(embedding._cache) == 5


class TestAPIEmbeddingCache:
    """Tests for APIEmbedding LRU cache functionality."""

    def test_cache_hit_avoids_api_call(self) -> None:
        """Test that cache hit avoids API call."""
        with patch("rec_praxis_rlm.embeddings.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3])
            ]
            mock_openai.OpenAI.return_value = mock_client

            embedding = APIEmbedding(api_key="test-key")

            # First call - cache miss, should call API
            result1 = embedding.embed("hello world")
            assert result1 == [0.1, 0.2, 0.3]
            assert mock_client.embeddings.create.call_count == 1

            # Second call - cache hit, should NOT call API
            result2 = embedding.embed("hello world")
            assert result2 == [0.1, 0.2, 0.3]
            assert mock_client.embeddings.create.call_count == 1  # Still 1

    def test_cache_reduces_api_costs(self) -> None:
        """Test that caching reduces API calls and costs."""
        with patch("rec_praxis_rlm.embeddings.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3])
            ]
            mock_openai.OpenAI.return_value = mock_client

            embedding = APIEmbedding(api_key="test-key")

            # Embed same text 100 times
            for _ in range(100):
                embedding.embed("repeated text")

            # Should only call API once (first time)
            assert mock_client.embeddings.create.call_count == 1

    def test_batch_cache_aware(self) -> None:
        """Test that API batch processing is cache-aware."""
        with patch("rec_praxis_rlm.embeddings.openai") as mock_openai:
            mock_client = MagicMock()
            # Different responses for pre-population and batch
            mock_client.embeddings.create.side_effect = [
                MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])]),  # cached1
                MagicMock(data=[MagicMock(embedding=[0.4, 0.5, 0.6])]),  # cached2
                MagicMock(data=[  # new1, new2 batch
                    MagicMock(embedding=[0.7, 0.8, 0.9]),
                    MagicMock(embedding=[1.0, 1.1, 1.2])
                ])
            ]
            mock_openai.OpenAI.return_value = mock_client

            embedding = APIEmbedding(api_key="test-key")

            # Pre-populate cache
            embedding.embed("cached1")
            embedding.embed("cached2")

            # Batch with mix of cached and uncached
            results = embedding.embed_batch([
                "cached1",    # Cache hit
                "new1",       # Cache miss
                "cached2",    # Cache hit
                "new2"        # Cache miss
            ])

            # Should have 4 results
            assert len(results) == 4

            # Should call API 3 times total (2 singles + 1 batch)
            assert mock_client.embeddings.create.call_count == 3

            # Last call should be for uncached texts only
            last_call = mock_client.embeddings.create.call_args[1]["input"]
            assert len(last_call) == 2  # Only 2 uncached texts
            assert "new1" in last_call
            assert "new2" in last_call

    def test_cache_lru_eviction(self) -> None:
        """Test that API embedding cache uses LRU eviction."""
        with patch("rec_praxis_rlm.embeddings.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3])
            ]
            mock_openai.OpenAI.return_value = mock_client

            # Create embedding with small cache
            embedding = APIEmbedding(api_key="test-key", cache_size=3)

            # Fill cache
            embedding.embed("text1")
            embedding.embed("text2")
            embedding.embed("text3")
            assert len(embedding._cache) == 3

            # Access text2 to make it recent
            embedding.embed("text2")

            # Add new text - should evict text1
            embedding.embed("text4")
            assert len(embedding._cache) == 3

            # Verify text1 was evicted
            text1_key = embedding._get_cache_key("text1")
            assert text1_key not in embedding._cache

    def test_cache_key_generation(self) -> None:
        """Test that API embedding cache keys are generated correctly."""
        with patch("rec_praxis_rlm.embeddings.openai") as mock_openai:
            mock_client = MagicMock()
            mock_openai.OpenAI.return_value = mock_client

            embedding = APIEmbedding(api_key="test-key")

            # Same text should produce same key
            key1 = embedding._get_cache_key("hello world")
            key2 = embedding._get_cache_key("hello world")
            assert key1 == key2

            # Different text should produce different key
            key3 = embedding._get_cache_key("different text")
            assert key1 != key3

            # Key should be SHA256 hex (64 characters)
            assert len(key1) == 64
            assert all(c in "0123456789abcdef" for c in key1)
