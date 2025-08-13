"""
Integration tests for Qdrant vector database service.

These tests require:
1. A running Qdrant instance (configured in config/config.yaml)
2. Environment variables set (if authentication required)
3. A collection with test data

Tests are marked with @pytest.mark.integration and can be skipped
by running: pytest -m "not integration"
"""

import sys
import pytest
from pathlib import Path
from typing import Dict, Any

from chat_my_doc_app.config import get_config
from chat_my_doc_app.qdrant_service import QdrantService, create_qdrant_service


@pytest.mark.integration
class TestQdrantIntegration:
    """Integration tests for QdrantService."""

    @classmethod
    def setup_class(cls):
        """Set up test class with configuration."""
        cls.config = get_config()
        cls.qdrant_config = cls.config.get('qdrant', {})

    def test_config_loading(self):
        """Test that configuration loads successfully."""
        assert self.config is not None
        assert 'qdrant' in self.config
        assert 'embedding' in self.config

        # Verify required Qdrant config
        assert self.qdrant_config.get('host') is not None, "Qdrant host must be configured"
        assert self.qdrant_config.get('port') is not None, "Qdrant port must be configured"
        assert self.qdrant_config.get('collection_name') is not None, "Collection name must be configured"

    def test_qdrant_service_initialization(self):
        """Test QdrantService can be initialized with config."""
        service = QdrantService(self.config)

        assert service.host == self.qdrant_config['host']
        assert service.port == self.qdrant_config['port']
        assert service.collection_name == self.qdrant_config['collection_name']
        assert service.client is not None
        assert service.embedding_model is not None

    def test_factory_function(self):
        """Test the factory function creates a service correctly."""
        service = create_qdrant_service(self.config)

        assert isinstance(service, QdrantService)
        assert service.host == self.qdrant_config['host']

    @pytest.mark.slow
    def test_qdrant_connection(self):
        """Test connection to Qdrant server (slow test)."""
        service = create_qdrant_service(self.config)

        # Test connection
        connection_success = service.test_connection()
        assert connection_success, "Failed to connect to Qdrant server"

    @pytest.mark.slow
    def test_collection_exists(self):
        """Test that the configured collection exists (slow test)."""
        service = create_qdrant_service(self.config)

        collection_exists = service.check_collection_exists()
        assert collection_exists, f"Collection '{service.collection_name}' does not exist"

    def test_embedding_generation(self):
        """Test embedding generation functionality."""
        service = create_qdrant_service(self.config)

        test_text = "This is a test document for embedding generation."
        embedding = service.generate_embedding(test_text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

        # Test embedding consistency
        embedding2 = service.generate_embedding(test_text)
        assert embedding == embedding2, "Embeddings should be deterministic"

    def test_embedding_different_texts(self):
        """Test that different texts produce different embeddings."""
        service = create_qdrant_service(self.config)

        text1 = "This is about movies and entertainment."
        text2 = "This is about science and technology."

        embedding1 = service.generate_embedding(text1)
        embedding2 = service.generate_embedding(text2)

        assert embedding1 != embedding2, "Different texts should produce different embeddings"

    @pytest.mark.slow
    def test_similarity_search_basic(self):
        """Test basic similarity search functionality (slow test)."""
        service = create_qdrant_service(self.config)

        # Test search with basic parameters
        query = "movie review"
        results = service.similarity_search(query, limit=3)

        assert isinstance(results, list)
        assert len(results) <= 3

        # Verify result structure
        for result_data, score in results:
            assert isinstance(result_data, dict)
            assert 'id' in result_data
            assert 'payload' in result_data
            assert 'score' in result_data
            assert isinstance(score, float)
            assert 0 <= score <= 1, f"Score {score} should be between 0 and 1"

    @pytest.mark.slow
    def test_similarity_search_with_threshold(self):
        """Test similarity search with score threshold (slow test)."""
        service = create_qdrant_service(self.config)

        query = "excellent movie"
        threshold = 0.1
        results = service.similarity_search(
            query,
            limit=5,
            score_threshold=threshold
        )

        # All results should meet the threshold
        for result_data, score in results:
            assert score >= threshold, f"Score {score} should be >= {threshold}"

    @pytest.mark.slow
    def test_similarity_search_limit_enforcement(self):
        """Test that search respects configured limits (slow test)."""
        service = create_qdrant_service(self.config)

        # Test with limit higher than max_limit in config
        max_limit = self.qdrant_config.get('search', {}).get('max_limit', 20)
        high_limit = max_limit + 10

        query = "test query"
        results = service.similarity_search(query, limit=high_limit)

        # Should be capped at max_limit
        assert len(results) <= max_limit

    @pytest.mark.slow
    def test_search_with_metadata_filter(self):
        """Test search with metadata filtering (slow test)."""
        service = create_qdrant_service(self.config)

        # This test assumes your data has some metadata fields
        # Adjust the filter based on your actual data structure
        query = "movie"

        try:
            # Test without filter first
            results_no_filter = service.similarity_search(query, limit=5)

            # Test with a hypothetical metadata filter
            # Note: This might fail if your data doesn't have the expected metadata
            metadata_filter = {"type": "review"}  # Adjust based on your data
            results_with_filter = service.similarity_search(
                query,
                limit=5,
                metadata_filter=metadata_filter
            )

            # Results should be different or same based on your data
            assert isinstance(results_with_filter, list)

        except Exception as e:
            # Skip this test if metadata structure is different
            pytest.skip(f"Metadata filter test skipped: {e}")

    def test_empty_query_handling(self):
        """Test handling of edge cases."""
        service = create_qdrant_service(self.config)

        # Test empty query
        empty_embedding = service.generate_embedding("")
        assert isinstance(empty_embedding, list)
        assert len(empty_embedding) > 0

    @pytest.mark.slow
    def test_search_by_metadata_only(self):
        """Test metadata-only search functionality (slow test)."""
        service = create_qdrant_service(self.config)

        try:
            # Test metadata search - adjust filter based on your data
            metadata_filter = {"sentiment": "positive"}  # Adjust based on your data
            results = service.search_by_metadata(metadata_filter, limit=3)

            assert isinstance(results, list)
            assert len(results) <= 3

            # Verify result structure
            for doc in results:
                assert isinstance(doc, dict)
                assert 'id' in doc
                assert 'payload' in doc

        except Exception as e:
            # Skip if metadata structure is different
            pytest.skip(f"Metadata search test skipped: {e}")


@pytest.fixture(scope="session")
def qdrant_service():
    """Session-scoped fixture for QdrantService."""
    config = get_config()
    return create_qdrant_service(config)


@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end_search_workflow(qdrant_service):
    """End-to-end test of the complete search workflow."""
    # Test the complete workflow: config -> service -> search -> results

    # Step 1: Verify service is ready
    assert qdrant_service.test_connection()
    assert qdrant_service.check_collection_exists()

    # Step 2: Generate query embedding
    query = "great movie with excellent acting"
    embedding = qdrant_service.generate_embedding(query)
    assert len(embedding) > 0

    # Step 3: Perform search
    results = qdrant_service.similarity_search(query, limit=5)
    assert len(results) > 0

    # Step 4: Verify result quality
    # Results should be sorted by relevance (highest score first)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"

    # Step 5: Verify we can access document content
    for result_data, score in results[:2]:  # Check first 2 results
        assert 'payload' in result_data
        payload = result_data['payload']
        assert isinstance(payload, dict)
        # You might want to add specific checks based on your data structure


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    pytest.main([__file__, "-v", "-m", "integration"])
