"""
Pytest configuration and shared fixtures.

This file contains:
- Test configuration setup
- Shared fixtures for integration tests
- Environment validation for integration tests
"""

from pathlib import Path

import pytest

from chat_my_doc_app.config import get_config


def pytest_configure(config):
    """Configure pytest with custom settings."""

    config.addinivalue_line(
        "markers", "slow: mark test as slow (may take several seconds)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and handle skipping."""
    skip_slow = pytest.mark.skip(reason="Slow tests skipped. Use --slow to run.")

    # Check command line options
    run_slow = config.getoption("--slow", default=False)

    for item in items:
        # Skip slow tests unless explicitly requested
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
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
