# Integration Tests

This directory contains integration tests for the Chat My Doc App that require external services (Qdrant vector database).

## Overview

Integration tests verify that different components work together correctly and that the application can communicate with external services. These tests are more comprehensive but slower than unit tests.

## Test Structure

```
tests/integration/
├── README.md                    # This file
└── test_qdrant_integration.py   # Qdrant database integration tests
```

## Test Categories

Tests are marked with pytest markers:

- `@pytest.mark.integration` - Requires external services
- `@pytest.mark.slow` - May take several seconds to complete
- `@pytest.mark.unit` - Fast unit tests (in tests/unit/)

## Prerequisites

Before running integration tests, ensure:

1. **Qdrant Server**: A running Qdrant instance configured in `src/chat_my_doc_app/config/config.yaml`
2. **Environment Variables**: Set any required API keys in `.env` file
3. **Test Data**: Your Qdrant collection should contain some test documents

## Running Tests

### Quick Methods

```bash
# Run all integration tests
uv run python scripts/run_integration_tests.py

# Run only fast integration tests (skip slow ones)
uv run pytest tests/integration/ --integration -m "not slow"

# Run a specific test
uv run pytest tests/integration/test_qdrant_integration.py::TestQdrantIntegration::test_config_loading --integration -v
```

### Using pytest directly

```bash
# Run all integration tests
uv run pytest tests/integration/ --integration

# Run integration tests with verbose output
uv run pytest tests/integration/ --integration -v

# Run slow tests too
uv run pytest tests/integration/ --integration --slow

# Skip integration tests (run only if they don't require external services)
uv run pytest tests/integration/ -m "not integration"
```

### Using the test runner script

```bash
# Check environment setup
uv run python scripts/run_integration_tests.py --quick-check

# Run integration tests
uv run python scripts/run_integration_tests.py --type integration

# Run with verbose output
uv run python scripts/run_integration_tests.py --type integration --verbose

# Run both integration and slow tests
uv run python scripts/run_integration_tests.py --type integration_and_slow
```

## Test Configuration

### Environment Setup

Integration tests use the same configuration as the main application:

- **Configuration File**: `src/chat_my_doc_app/config/config.yaml`
- **Environment Variables**: `.env` file for sensitive data
- **Test Fixtures**: Defined in `tests/conftest.py`

### Pytest Configuration

Test markers and settings are defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "integration: marks tests as integration tests (may be slow)",
    "slow: marks tests as slow (may take several seconds)",
    "unit: marks tests as unit tests (fast)",
]
```

## Test Coverage

Integration tests cover:

### Qdrant Integration (`test_qdrant_integration.py`)

1. **Configuration Loading**
   - YAML config file parsing
   - Environment variable integration
   - Configuration validation

2. **Service Initialization**
   - QdrantService creation
   - Client connection setup
   - Embedding model loading

3. **Database Operations**
   - Connection testing
   - Collection existence verification
   - Collection metadata retrieval

4. **Embedding Operations**
   - Text embedding generation
   - Embedding consistency
   - Different text handling

5. **Search Operations**
   - Basic similarity search
   - Score threshold filtering
   - Result limit enforcement
   - Metadata filtering
   - Result structure validation

6. **End-to-End Workflows**
   - Complete search pipeline
   - Result quality verification
   - Score ordering validation

## Writing New Integration Tests

### Best Practices

1. **Use Appropriate Markers**
   ```python
   @pytest.mark.integration
   @pytest.mark.slow  # If test takes >2 seconds
   def test_my_integration():
       pass
   ```

2. **Handle External Dependencies**
   ```python
   def test_external_service():
       try:
           # Test external service
           result = external_service.call()
           assert result.success
       except ServiceUnavailableError:
           pytest.skip("External service not available")
   ```

3. **Use Fixtures for Setup**
   ```python
   def test_with_service(qdrant_service):
       # Use the qdrant_service fixture
       result = qdrant_service.search("test query")
       assert len(result) > 0
   ```

4. **Validate External State**
   ```python
   def test_database_state():
       # Verify external system is in expected state
       assert service.collection_exists()
       assert service.get_document_count() > 0
   ```

5. **Clean Up Resources**
   ```python
   def test_with_cleanup():
       try:
           # Test logic
           pass
       finally:
           # Clean up any test data
           service.cleanup_test_data()
   ```

### Test Structure Template

```python
@pytest.mark.integration
class TestMyIntegration:

    @classmethod
    def setup_class(cls):
        """One-time setup for the test class."""
        cls.config = get_config()

    def test_basic_functionality(self):
        """Test basic integration functionality."""
        # Arrange
        service = create_service(self.config)

        # Act
        result = service.do_something()

        # Assert
        assert result.success

    @pytest.mark.slow
    def test_slow_operation(self):
        """Test that may take several seconds."""
        # Test implementation
        pass
```

## Troubleshooting

### Common Issues

1. **Tests Skipped**: Check that you're using the `--integration` flag
2. **Connection Errors**: Verify Qdrant server is running and accessible
3. **Import Errors**: Ensure the `src` directory is in Python path
4. **Configuration Errors**: Check `config/config.yaml` and `.env` files

### Debug Mode

Run tests with maximum verbosity:

```bash
uv run pytest tests/integration/ --integration -vvv -s --tb=long
```

### Test Coverage

Generate coverage reports:

```bash
uv run pytest tests/integration/ --integration --cov=src/chat_my_doc_app --cov-report=html
```

## CI/CD Integration

For continuous integration, create separate jobs:

```yaml
# Example GitHub Actions
- name: Run Unit Tests
  run: uv run pytest tests/unit/ -v

- name: Run Integration Tests
  run: uv run pytest tests/integration/ --integration
  env:
    QDRANT_HOST: ${{ secrets.QDRANT_HOST }}
```

Integration tests can be configured to run:
- On pull requests (with test database)
- On main branch commits (with staging environment)
- Nightly (with production-like setup)
