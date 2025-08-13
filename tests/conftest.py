"""
Pytest configuration and shared fixtures.

This file contains:
- Test configuration setup
- Shared fixtures for integration tests
- Environment validation for integration tests
"""

import os
import sys
import pytest
from pathlib import Path

from chat_my_doc_app.config import get_config


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring external services"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (may take several seconds)"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (fast)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and handle skipping."""
    skip_integration = pytest.mark.skip(reason="Integration tests skipped. Use --integration to run.")
    skip_slow = pytest.mark.skip(reason="Slow tests skipped. Use --slow to run.")

    # Check command line options
    run_integration = config.getoption("--integration", default=False)
    run_slow = config.getoption("--slow", default=False)

    for item in items:
        # Skip integration tests unless explicitly requested
        if "integration" in item.keywords and not run_integration:
            item.add_marker(skip_integration)

        # Skip slow tests unless explicitly requested
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


@pytest.fixture(scope="session")
def app_config():
    """Session-scoped fixture for application configuration."""
    try:
        config = get_config()
        return config
    except Exception as e:
        pytest.skip(f"Could not load configuration: {e}")


@pytest.fixture(scope="session")
def qdrant_available(app_config):
    """Check if Qdrant is available for testing."""
    qdrant_config = app_config.get('qdrant', {})
    host = qdrant_config.get('host')

    if not host:
        pytest.skip("Qdrant host not configured")

    # Test basic connectivity
    try:
        from chat_my_doc_app.db import QdrantService
        service = QdrantService(app_config)
        if not service.test_connection():
            pytest.skip("Qdrant server not available")
        return True
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "test_1",
            "text": "This is a great movie with excellent acting and cinematography.",
            "metadata": {"type": "review", "sentiment": "positive", "rating": 5}
        },
        {
            "id": "test_2",
            "text": "The plot was confusing and the characters were poorly developed.",
            "metadata": {"type": "review", "sentiment": "negative", "rating": 2}
        },
        {
            "id": "test_3",
            "text": "A fascinating documentary about marine life and ocean conservation.",
            "metadata": {"type": "documentary", "category": "nature", "rating": 4}
        }
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for testing search functionality."""
    return [
        "excellent movie review",
        "documentary about nature",
        "poor character development",
        "ocean conservation marine life",
        "great acting cinematography"
    ]


class IntegrationTestHelper:
    """Helper class for integration tests."""

    @staticmethod
    def assert_valid_search_result(result_data, score):
        """Assert that a search result has the expected structure."""
        assert isinstance(result_data, dict)
        assert 'id' in result_data
        assert 'payload' in result_data
        assert 'score' in result_data
        assert isinstance(score, float)
        assert 0 <= score <= 1, f"Score {score} should be between 0 and 1"

    @staticmethod
    def assert_valid_embedding(embedding):
        """Assert that an embedding has the expected properties."""
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    @staticmethod
    def assert_search_results_sorted(results):
        """Assert that search results are sorted by score in descending order."""
        if len(results) <= 1:
            return  # Cannot check sorting with 0 or 1 results

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), \
            "Search results should be sorted by score in descending order"


@pytest.fixture
def integration_helper():
    """Fixture providing integration test helper methods."""
    return IntegrationTestHelper()
