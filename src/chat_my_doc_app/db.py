"""
Qdrant Vector Database Service

This module provides connection and search functionality for the Qdrant vector database
deployed on Google Cloud Engine. It handles document retrieval, similarity search,
and metadata filtering for RAG applications.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from chat_my_doc_app.config import get_config


class QdrantService:
    """Service class for interacting with Qdrant vector database."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize Qdrant service.

        Args:
            config_dict: Configuration dictionary. If None, loads from default config.
        """
        if config_dict is None:
            config_dict = get_config()

        self.config = config_dict
        qdrant_config = config_dict.get('qdrant', {})
        embedding_config = config_dict.get('embedding', {})

        self.host = qdrant_config.get('host')
        self.port = qdrant_config.get('port', 6333)
        self.collection_name = qdrant_config.get('collection_name', 'documents')

        if not self.host:
            raise ValueError("Qdrant host must be specified in config.yaml")

        # Initialize Qdrant client
        self.client = QdrantClient(
            host=self.host,
            port=self.port,
            timeout=qdrant_config.get('timeout', 30)
        )

        # Initialize embedding model
        model_name = embedding_config.get('model_name')
        logger.info(f"Loading embedding model: {model_name}")

        self.embedding_model = SentenceTransformer(model_name)
        logger.info("QdrantService initialized successfully")

    def test_connection(self) -> bool:
        """
        Test connection to Qdrant server.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            collections = self.client.get_collections()
            logger.info(f"Successfully connected to Qdrant. Available collections: {[c.name for c in collections.collections]}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            return False

    def check_collection_exists(self) -> bool:
        """
        Check if the specified collection exists.

        Returns:
            bool: True if collection exists, False otherwise
        """
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            exists = self.collection_name in collection_names

            if exists:
                logger.info(f"Collection '{self.collection_name}' exists")
            else:
                logger.warning(f"Collection '{self.collection_name}' does not exist. Available: {collection_names}")

            return exists
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False

    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the collection.

        Returns:
            Dict with collection information or None if error
        """
        try:
            info = self.client.get_collection(self.collection_name)
            # Get the default vector config (usually the first one or named vector)
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict) and vectors_config:
                # Get first vector config
                _, vector_config = next(iter(vectors_config.items()))
                vector_size = vector_config.size
                distance_metric = vector_config.distance.name
            else:
                vector_size = 0
                distance_metric = "unknown"

            collection_info = {
                "name": self.collection_name,
                "vector_size": vector_size,
                "distance_metric": distance_metric,
                "points_count": info.points_count,
                "status": info.status.name
            }
            logger.info(f"Collection info: {collection_info}")
            return collection_info
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return None

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for given text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def similarity_search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform similarity search in the vector database.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            metadata_filter: Optional metadata filter conditions

        Returns:
            List of tuples containing (document_data, similarity_score)
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query)
            logger.debug(f"Generated embedding for query: '{query[:50]}...'")

            # Prepare filter conditions
            search_filter = None
            if metadata_filter:
                conditions = []
                for key, value in metadata_filter.items():
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
                if conditions:
                    search_filter = models.Filter(must=cast(List[Union[
                            models.FieldCondition,
                            models.IsEmptyCondition,
                            models.IsNullCondition,
                            models.HasIdCondition,
                            models.HasVectorCondition,
                            models.NestedCondition,
                            models.Filter]
                        ], conditions)
                    )

            # Perform search
            # Use configured default values if not specified
            qdrant_config = self.config.get('qdrant', {})
            search_config = qdrant_config.get('search', {})

            if limit > search_config.get('max_limit', 20):
                limit = search_config.get('max_limit', 20)
                logger.warning(f"Limit reduced to maximum allowed: {limit}")

            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True
            )

            # Process results
            results = []
            for result in search_results.points:
                document_data = {
                    "id": result.id,
                    "payload": result.payload,
                    "score": result.score
                }
                results.append((document_data, result.score))

            logger.info(f"Found {len(results)} documents for query: '{query[:50]}...'")
            return results

        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise

    def search_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search documents by metadata only (no vector similarity).

        Args:
            metadata_filter: Metadata filter conditions
            limit: Maximum number of results to return

        Returns:
            List of document data dictionaries
        """
        try:
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )

            search_filter = models.Filter(must=cast(List[Union[
                    models.FieldCondition,
                    models.IsEmptyCondition,
                    models.IsNullCondition,
                    models.HasIdCondition,
                    models.HasVectorCondition,
                    models.NestedCondition,
                    models.Filter
                ]
            ], conditions))

            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True
            )

            documents = []
            for point in results[0]:  # results is a tuple (points, next_page_offset)
                document_data = {
                    "id": point.id,
                    "payload": point.payload
                }
                documents.append(document_data)

            logger.info(f"Found {len(documents)} documents matching metadata filter")
            return documents

        except Exception as e:
            logger.error(f"Error searching by metadata: {str(e)}")
            raise
