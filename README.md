# Chat My Doc App

A modern Gradio-based chat application that leverages Google's Gemini AI models through a custom LangChain implementation. Chat with powerful AI models using a clean, intuitive web interface with conversation memory and streaming responses.

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Gradio](https://img.shields.io/badge/gradio-5.42+-green.svg)
![LangChain](https://img.shields.io/badge/langchain-latest-orange.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

## Features

- **Modern Chat Interface**: Clean Gradio-based web UI with real-time streaming responses
- **Multiple AI Models**: Support for Gemini 2.0 Flash, Gemini 1.5 Pro, and more
- **Conversation Memory**: Maintains context across conversations with session management
- **Custom LangChain Integration**: Direct connection to your deployed Gemini API
- **Advanced CLI**: Full-featured command-line interface with Typer
- **Input Validation**: Robust parameter validation and error handling
- **Comprehensive Testing**: 37+ passing tests with full coverage
- **Cloud Ready**: Docker support with Google Cloud Run deployment
- **Development Tools**: Hot-reload, debug mode, and development utilities

## Quick Start

### Prerequisites

- Python 3.12+
- UV package manager (recommended) or pip
- Google API access for Gemini models

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd chat-my-doc-app

# Install dependencies with UV (recommended)
uv sync

# Or with pip
pip install -e .
```

### Environment Setup

Create a `.env` file in the project root:

```env
# Required: Your deployed API URL
CLOUD_RUN_API_URL=https://your-api-endpoint.run.app

# Optional: Custom port (defaults to 8000)
PORT=8000
```

### Launch the Application

```bash
# Basic launch
uv run python src/app/main.py

# Development mode with debug and auto-open browser
uv run python src/app/main.py --debug --browser --port 7860

# Production mode
uv run python src/app/main.py --host 0.0.0.0 --port 8000
```

## Usage

### Command Line Interface

The application includes a full-featured CLI built with Typer:

```bash
# Show all available options
uv run python src/app/main.py --help

# Basic options
uv run python src/app/main.py [OPTIONS]

Options:
  --debug, -d              Enable debug mode with auto-reload
  --port, -p INTEGER       Port to run server on (1-65535)
  --host TEXT             Host to bind server to [default: 0.0.0.0]
  --share, -s             Create public shareable link
  --browser, -b           Auto-open in browser
  --help                  Show this message and exit
```

### Development Examples

```bash
# Development with hot-reload and browser auto-open
uv run python src/app/main.py -d -b -p 7860

# Share publicly (creates Gradio public link)
uv run python src/app/main.py --share

# Custom host and port
uv run python src/app/main.py --host localhost --port 3000

# Production deployment
uv run python src/app/main.py --host 0.0.0.0 --port 8080
```

### Web Interface

1. **Start the application** using one of the methods above
2. **Open your browser** to `http://localhost:8000` (or your custom port)
3. **Select an AI model** from the dropdown (Gemini 2.0 Flash Lite, etc.)
4. **Start chatting** - your conversation history is maintained automatically
5. **Use the Clear button** to reset the conversation

## Architecture

### Project Structure

```
chat-my-doc-app/
├── src/
│   ├── app/
│   │   └── main.py              # Gradio interface & Typer CLI
│   └── chat_my_doc_app/
│       ├── chats.py             # Chat functionality & session management
│       └── llms.py              # Custom LangChain model implementation
├── tests/                       # Comprehensive test suite (37+ tests)
│   ├── app/test_main.py         # CLI & interface tests
│   └── chat_my_doc_app/         # Model & chat functionality tests
├── .github/workflows/           # CI/CD pipelines
├── Dockerfile                   # Container configuration
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

### Key Components

#### 1. **Gradio Interface** (`src/app/main.py`)
- Modern web UI with streaming chat
- Multi-model selection dropdown
- Session-based conversation memory
- Real-time response streaming

#### 2. **Custom LangChain Model** (`src/chat_my_doc_app/llms.py`)
- `GeminiChat` class extending `BaseChatModel`
- Direct API integration with your deployed endpoint
- Streaming and async support
- Full LangChain compatibility

#### 3. **Chat Management** (`src/chat_my_doc_app/chats.py`)
- Conversation history management
- Multi-session support
- Streaming response handling
- Error handling and logging

#### 4. **CLI Application** (`src/app/main.py`)
- Built with Typer for professional CLI experience
- Input validation and helpful error messages
- Development and production modes
- Environment variable integration

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `CLOUD_RUN_API_URL` | Your deployed API endpoint | None | ✅ Yes |
| `PORT` | Server port | 8000 | ❌ No |

### Model Configuration

Available models (configured in `chats.py`):
- `gemini-2.0-flash-lite` (default)
- `gemini-2.0-flash`
- `gemini-1.5-pro`

## Testing

The project includes comprehensive testing with 37+ passing tests:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test categories
uv run pytest tests/app/                    # CLI & interface tests
uv run pytest tests/chat_my_doc_app/        # Model & chat tests

# Run with verbose output
uv run pytest -v

# Run specific test
uv run pytest tests/app/test_main.py::TestTyperCLI::test_cli_debug_mode -v
```

### Test Coverage

- ✅ **CLI Functionality**: All Typer CLI options and validation
- ✅ **Gradio Interface**: Component creation and functionality
- ✅ **LangChain Integration**: Custom model implementation
- ✅ **Chat Management**: Session handling and conversation memory
- ✅ **Input Validation**: Port ranges, host formats, parameter validation
- ✅ **Error Handling**: Invalid inputs and edge cases

## Deployment

### Docker Deployment

```bash
# Build the container
docker build -t chat-my-doc-app .

# Run locally
docker run -p 8000:8000 \
  -e CLOUD_RUN_API_URL=your-api-url \
  chat-my-doc-app
```

### Google Cloud Run

The project includes automated deployment via GitHub Actions:

1. **Set up secrets** in your GitHub repository:
   - `GCP_PROJECT_ID`
   - `GCP_SA_KEY`
   - `CLOUD_RUN_API_URL`

2. **Push to main branch** - deployment happens automatically

3. **Manual deployment**:
```bash
# Deploy to Cloud Run
gcloud run deploy chat-my-doc-app \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars CLOUD_RUN_API_URL=your-api-url
```

## Development

### Development Setup

```bash
# Install development dependencies
uv sync --group dev

# Install pre-commit hooks (recommended)
pre-commit install

# Run linting
uv run ruff check src/ tests/

# Run type checking
uv run mypy src/

# Run tests with coverage
uv run pytest --cov=src/chat_my_doc_app
```

### Development Commands

```bash
# Start in development mode
uv run python src/app/main.py --debug --browser

# Run with hot-reload on custom port
uv run python src/app/main.py -d -b -p 7860

# Create a public share link for testing
uv run python src/app/main.py --debug --share
```

### Adding New Features

1. **Models**: Add new model names to `get_available_models()` in `chats.py`
2. **CLI Options**: Extend the Typer command in `main.py`
3. **UI Components**: Modify the Gradio interface in `create_chat_interface()`
4. **Tests**: Add corresponding tests in the `tests/` directory

## API Reference

### Chat Functions

```python
from chat_my_doc_app.chats import chat_with_gemini_stream, get_available_models

# Stream chat responses
for chunk in chat_with_gemini_stream(
    message="Hello, how are you?",
    model_name="gemini-2.0-flash-lite",
    session_id="user_123"
):
    print(chunk, end="")

# Get available models
models = get_available_models()
```

### Custom LangChain Model

```python
from chat_my_doc_app.llms import GeminiChat

# Initialize custom model
llm = GeminiChat(
    api_url="https://your-api-endpoint.run.app",
    model_name="gemini-2.0-flash-lite"
)

# Use with LangChain
response = llm.invoke("What is the capital of France?")
```

## Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run tests**: `uv run pytest`
5. **Run linting**: `uv run ruff check src/ tests/`
6. **Commit your changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Gradio** - For the excellent web UI framework
- **LangChain** - For the flexible LLM integration framework
- **Typer** - For the modern CLI framework
- **Google** - For the powerful Gemini AI models
- **UV** - For fast and reliable Python package management

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/chat-my-doc-app/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/chat-my-doc-app/discussions)
- **Documentation**: This README and inline code documentation

---

**Built with ❤️ using Python, Gradio, and LangChain**
