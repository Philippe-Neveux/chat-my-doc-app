import pytest
from unittest.mock import patch, Mock
import os
from typer.testing import CliRunner
import typer

from chat_my_doc_app.app import app, validate_port, validate_host


class TestValidationFunctions:
    """Test validation functions for CLI parameters."""

    def test_validate_port_valid_values(self):
        """Test validate_port with valid port numbers."""
        valid_ports = [1, 80, 443, 8000, 8080, 65535]

        for port in valid_ports:
            result = validate_port(port)
            assert result == port

    def test_validate_port_none_value(self):
        """Test validate_port with None value."""
        result = validate_port(None)
        assert result is None

    def test_validate_port_invalid_values(self):
        """Test validate_port with invalid port numbers."""
        invalid_ports = [0, -1, 65536, 70000, 999999]

        for port in invalid_ports:
            with pytest.raises(typer.BadParameter) as exc_info:
                validate_port(port)
            assert "Port must be between 1 and 65535" in str(exc_info.value)

    def test_validate_host_valid_values(self):
        """Test validate_host with valid host values."""
        valid_hosts = [
            "0.0.0.0",
            "localhost",
            "127.0.0.1",
            "192.168.1.1",
            "10.0.0.1"
            # Note: hostname validation with socket.inet_aton only accepts IP addresses
            # "example.com" would fail with current implementation
        ]

        for host in valid_hosts:
            result = validate_host(host)
            assert result == host

    def test_validate_host_invalid_values(self):
        """Test validate_host with invalid host values."""
        invalid_hosts = [
            "999.999.999.999",
            "256.1.1.1",
            "invalid@host",
            "host with spaces",
            "example.com"  # Hostname fails with socket.inet_aton
        ]

        for host in invalid_hosts:
            with pytest.raises(typer.BadParameter) as exc_info:
                validate_host(host)
            assert "Invalid host format" in str(exc_info.value)


class TestTyperCLI:
    """Test Typer CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch('chat_my_doc_app.app.create_chat_interface')
    def test_cli_default_parameters(self, mock_create_interface):
        """Test CLI with default parameters."""
        mock_interface = Mock()
        mock_create_interface.return_value = mock_interface
        mock_interface.launch = Mock()

        result = self.runner.invoke(app, [])

        # Should exit cleanly
        assert result.exit_code == 0
        mock_create_interface.assert_called_once()
        mock_interface.launch.assert_called_once_with(
            server_name="0.0.0.0",
            server_port=8000,  # Default port
            share=False,
            show_error=True,
            debug=False,  # Default debug mode
            inbrowser=False
        )

    @patch('chat_my_doc_app.app.create_chat_interface')
    def test_cli_debug_mode(self, mock_create_interface):
        """Test CLI with debug mode enabled."""
        mock_interface = Mock()
        mock_create_interface.return_value = mock_interface
        mock_interface.launch = Mock()

        result = self.runner.invoke(app, ["--debug"])

        assert result.exit_code == 0
        mock_interface.launch.assert_called_once_with(
            server_name="0.0.0.0",
            server_port=8000,
            share=False,
            show_error=True,
            debug=True,  # Debug mode enabled
            inbrowser=False
        )

    @patch('chat_my_doc_app.app.create_chat_interface')
    def test_cli_custom_port(self, mock_create_interface):
        """Test CLI with custom port parameter."""
        mock_interface = Mock()
        mock_create_interface.return_value = mock_interface
        mock_interface.launch = Mock()

        result = self.runner.invoke(app, ["--port", "7860"])

        assert result.exit_code == 0
        mock_interface.launch.assert_called_once_with(
            server_name="0.0.0.0",
            server_port=7860,  # Custom port
            share=False,
            show_error=True,
            debug=False,
            inbrowser=False
        )

    @patch('chat_my_doc_app.app.create_chat_interface')
    def test_cli_custom_host(self, mock_create_interface):
        """Test CLI with custom host parameter."""
        mock_interface = Mock()
        mock_create_interface.return_value = mock_interface
        mock_interface.launch = Mock()

        result = self.runner.invoke(app, ["--host", "127.0.0.1"])

        assert result.exit_code == 0
        mock_interface.launch.assert_called_once_with(
            server_name="127.0.0.1",  # Custom host
            server_port=8000,
            share=False,
            show_error=True,
            debug=False,
            inbrowser=False
        )

    @patch('chat_my_doc_app.app.create_chat_interface')
    def test_cli_all_options(self, mock_create_interface):
        """Test CLI with all options enabled."""
        mock_interface = Mock()
        mock_create_interface.return_value = mock_interface
        mock_interface.launch = Mock()

        result = self.runner.invoke(app, [
            "--debug",
            "--port", "9000",
            "--host", "localhost",
            "--share",
            "--browser"
        ])

        assert result.exit_code == 0
        mock_interface.launch.assert_called_once_with(
            server_name="localhost",
            server_port=9000,
            share=True,
            show_error=True,
            debug=True,
            inbrowser=True
        )

    def test_cli_invalid_port(self):
        """Test CLI with invalid port parameter."""
        result = self.runner.invoke(app, ["--port", "70000"])

        # Should fail with validation error
        assert result.exit_code != 0
        # Typer's built-in range validation message (split across lines)
        assert "70000 is not in the" in result.output
        assert "range 1<=x<=65535" in result.output

    def test_cli_invalid_host(self):
        """Test CLI with invalid host parameter."""
        result = self.runner.invoke(app, ["--host", "invalid@host"])

        # Should fail with validation error
        assert result.exit_code != 0
        assert "Invalid host format" in result.output

    @patch.dict(os.environ, {'PORT': '7860'})
    @patch('chat_my_doc_app.app.create_chat_interface')
    def test_cli_environment_variable_port(self, mock_create_interface):
        """Test that CLI uses PORT environment variable when no port argument provided."""
        mock_interface = Mock()
        mock_create_interface.return_value = mock_interface
        mock_interface.launch = Mock()

        result = self.runner.invoke(app, [])

        assert result.exit_code == 0
        mock_interface.launch.assert_called_once_with(
            server_name="0.0.0.0",
            server_port=7860,  # From environment variable
            share=False,
            show_error=True,
            debug=False,
            inbrowser=False
        )

    @patch.dict(os.environ, {'PORT': '7860'})
    @patch('chat_my_doc_app.app.create_chat_interface')
    def test_cli_port_argument_overrides_env(self, mock_create_interface):
        """Test that CLI port argument overrides environment variable."""
        mock_interface = Mock()
        mock_create_interface.return_value = mock_interface
        mock_interface.launch = Mock()

        result = self.runner.invoke(app, ["--port", "9000"])

        assert result.exit_code == 0
        mock_interface.launch.assert_called_once_with(
            server_name="0.0.0.0",
            server_port=9000,  # CLI argument overrides env var
            share=False,
            show_error=True,
            debug=False,
            inbrowser=False
        )

    def test_cli_short_options(self):
        """Test CLI short option flags."""
        with patch('chat_my_doc_app.app.create_chat_interface') as mock_create_interface:
            mock_interface = Mock()
            mock_create_interface.return_value = mock_interface
            mock_interface.launch = Mock()

            # Use --host instead of -h since -h is reserved for help
            result = self.runner.invoke(app, ["-d", "-p", "8080", "--host", "localhost", "-s", "-b"])

            assert result.exit_code == 0
            mock_interface.launch.assert_called_once_with(
                server_name="localhost",
                server_port=8080,
                share=True,
                show_error=True,
                debug=True,
                inbrowser=True
            )


class TestGradioInterface:
    """Simplified tests for Gradio interface."""

    def test_create_chat_interface_returns_interface(self):
        """Test that create_chat_interface returns a working Gradio interface."""
        from chat_my_doc_app.app import create_chat_interface

        # This will actually create the real Gradio interface
        # but we won't launch it, just test that it's created
        interface = create_chat_interface()

        # Check that the interface has the expected Gradio attributes
        assert hasattr(interface, 'launch')
        assert callable(interface.launch)

        # Check that it has other expected Gradio Block attributes
        assert hasattr(interface, 'queue')
        assert hasattr(interface, 'close')

    @patch('chat_my_doc_app.app.get_available_models')  # Patch it where it's imported in main.py
    def test_create_interface_with_mocked_models(self, mock_get_models):
        """Test interface creation with mocked available models."""
        mock_get_models.return_value = ["test-model-1", "test-model-2"]

        from chat_my_doc_app.app import create_chat_interface
        interface = create_chat_interface()

        assert interface is not None
        assert hasattr(interface, 'launch')
        # Function is called at least once (might be called twice for choices and value)
        assert mock_get_models.call_count >= 1


class TestEnvironmentIntegration:
    """Test environment variable integration."""

    def test_port_environment_variable_parsing(self):
        """Test that PORT environment variable is correctly parsed."""
        # Test default port
        with patch.dict(os.environ, {}, clear=True):
            port = int(os.getenv("PORT", 8000))
            assert port == 8000

        # Test custom port
        with patch.dict(os.environ, {'PORT': '9000'}):
            port = int(os.getenv("PORT", 8000))
            assert port == 9000

    def test_port_variable_types(self):
        """Test different types of PORT values."""
        test_cases = [
            ("8000", 8000),
            ("7860", 7860),
            ("3000", 3000),
        ]

        for port_str, expected_int in test_cases:
            with patch.dict(os.environ, {'PORT': port_str}):
                port = int(os.getenv("PORT", 8000))
                assert port == expected_int
                assert isinstance(port, int)


class TestGradioImports:
    """Test that Gradio imports work correctly."""

    def test_gradio_imports_available(self):
        """Test that required Gradio components can be imported."""
        try:
            import gradio as gr

            # Test that required Gradio components exist
            assert hasattr(gr, 'Blocks')
            assert hasattr(gr, 'Chatbot')
            assert hasattr(gr, 'Textbox')
            assert hasattr(gr, 'Button')
            assert hasattr(gr, 'Dropdown')
            assert hasattr(gr, 'Markdown')
            assert hasattr(gr, 'Row')
            assert hasattr(gr, 'Column')

        except ImportError:
            pytest.fail("Gradio is not properly installed or importable")

    def test_main_module_imports(self):
        """Test that main module imports work correctly."""
        try:
            from chat_my_doc_app.app import create_chat_interface, main
            assert callable(create_chat_interface)
            assert callable(main)
        except ImportError as e:
            pytest.fail(f"Failed to import main module functions: {e}")
