# Rag-Chat-Bot Setup Guide

Complete setup instructions for getting Rag-Chat-Bot up and running on your system.

## Project Structure

```
rag-chatbot-main/
├── chatbot/                    # Core chatbot implementation
│   ├── bot/                   # Main bot logic
│   │   ├── client/           # LLM client implementations
│   │   │   ├── lama_cpp_client.py
│   │   │   ├── ollama_client.py
│   │   │   ├── openai_client.py
│   │   │   └── prompt.py
│   │   ├── conversation/     # Chat history & context management
│   │   ├── memory/           # RAG memory & embeddings
│   │   └── model/            # Model registry & settings
│   ├── document_loader/      # PDF & Markdown document loading
│   ├── entities/             # Data models
│   ├── cli/                  # Command-line interfaces
│   ├── chatbot_app.py        # Streamlit chatbot UI
│   ├── rag_chatbot_app.py    # Streamlit RAG chatbot UI
│   └── memory_builder.py     # Build vector embeddings
├── tests/                    # Test suite
├── docs/                     # Documentation folder
├── images/                   # Architecture diagrams & screenshots
├── vector_store/             # Chroma vector database storage
├── Makefile                  # Development tasks
├── pyproject.toml            # Poetry dependencies
├── setup.sh                  # Linux/Mac setup script
├── setup.bat                 # Windows setup script
└── README.md                 # Project documentation
```

## Prerequisites

- **Python**: 3.10 or higher
- **Poetry**: 1.7.0 (dependency manager)
- **GPU** (Optional): NVIDIA GPU with CUDA 12.1+ for acceleration
- **OS**: Windows, Linux, or macOS

## Quick Start (Automated)

### Linux / macOS

```bash
bash setup.sh
```

Configurable environment variables:
```bash
MODEL=llama-3.1 K=2 SYNTHESIS_STRATEGY=async-tree-summarization bash setup.sh
```

### Windows

```cmd
setup.bat
```

Or with CUDA support:
```cmd
set INSTALL_CUDA=1
setup.bat
```

## Manual Setup

### Step 1: Install Python 3.10+

Verify Python installation:
```bash
python3 --version  # Should be 3.10 or higher
```

### Step 2: Install Poetry

Follow the [official Poetry installation guide](https://python-poetry.org/docs/#installing-with-the-official-installer):

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"  # Add to your shell profile
```

Verify installation:
```bash
poetry --version  # Should be 1.7.0
```

### Step 3: Configure Poetry Environment

```bash
cd rag-chatbot-main
poetry env use python3  # Use your Python 3.10+ installation
```

### Step 4: Install Dependencies

**Without CUDA** (CPU only):
```bash
poetry install
```

**With CUDA 12.1** (NVIDIA GPU):
```bash
poetry install -E cuda-acceleration
```

**With Metal GPU** (macOS):
```bash
poetry install
```

### Step 5: Build Vector Embeddings

```bash
poetry run python chatbot/memory_builder.py --chunk-size 1000 --chunk-overlap 50
```

This will:
- Load Markdown files from `docs/` folder
- Process PDF files if available
- Create embeddings using `all-MiniLM-L6-v2`
- Store vectors in Chroma database (`vector_store/`)

## Running the Applications

### Conversation-Aware Chatbot

Interactive chatbot with memory of previous conversations:

```bash
poetry run streamlit run chatbot/chatbot_app.py -- --model llama-3.1 --max-new-tokens 1024
```

### RAG Chatbot

Document-based question answering with retrieval:

```bash
poetry run streamlit run chatbot/rag_chatbot_app.py -- --model llama-3.1 --k 2 --synthesis-strategy async-tree-summarization
```

## Configuration Options

### Models

Supported models for local inference:

| Model | Size | Recommended | Download Link |
|-------|------|-------------|---------------|
| llama-3.1 | 8B | Yes | [HuggingFace](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) |
| llama-3.2 | 3B-8B | Yes | [HuggingFace](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |
| qwen-2.5 | 3B | Yes | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) |
| deep-seek-r1 | 7B | Experimental | [HuggingFace](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF) |
| phi-3.5 | 3.8B | Yes | [HuggingFace](https://huggingface.co/MaziyarPanahi/Phi-3.5-mini-instruct-GGUF) |

### Response Synthesis Strategies

- `create-and-refine`: Process documents sequentially
- `tree-summarization`: Hierarchical answer synthesis
- `async-tree-summarization`: Fast hierarchical processing (recommended)

### LLM Backends

Switch between different inference engines by modifying client selection in the app.

**Supported:**
- llama-cpp (local, optimized)
- Ollama (local, Docker-friendly)
- OpenAI (cloud-based)

## Troubleshooting

### Python Version Issues

```bash
python3 --version
poetry env info
```

Must be Python 3.10+.

### Poetry Cache Issues

```bash
poetry cache clear . --all
poetry install
```

### GPU/CUDA Issues

Check NVIDIA setup:
```bash
nvidia-smi
```

If using M1/M2 Mac with x86 Python:
- Install arm64 Python from [python.org](https://www.python.org)
- Reinstall torch and dependencies

### Memory Builder Errors

Ensure `docs/` folder has Markdown files:
```bash
ls docs/
```

Verify NLTK data is loaded:
```bash
python -c "import nltk; nltk.download('punkt')"
```

### Streamlit Won't Start

Clear cache:
```bash
rm -rf ~/.streamlit
poetry run streamlit run chatbot/rag_chatbot_app.py --logger.level=debug
```

## Development

### Using Makefile

```bash
make check      # Verify Python & pip paths
make setup_cuda # Setup with CUDA
make update     # Update dependencies
make tidy       # Format code with Ruff
make test       # Run test suite
make clean      # Remove environment
```

### Running Tests

```bash
poetry run pytest tests/
poetry run pytest tests/ -v --cov=chatbot  # With coverage
```

### Code Style

Format with Ruff:
```bash
poetry run ruff check chatbot/ --fix
poetry run ruff format chatbot/
```

## Using Ollama for Local Inference

Alternative to llama-cpp for easier model management:

### Install Ollama

- [Ollama Download](https://ollama.ai)
- Windows: Run installer
- Linux/Mac: `curl https://ollama.ai/install.sh | sh`

### Pull a Model

```bash
ollama pull llama2
```

### Set Environment Variables

```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama2
```

### Run the App

```bash
poetry run streamlit run chatbot/rag_chatbot_app.py
```

## System Requirements

| Component | CPU Only | With GPU |
|-----------|----------|----------|
| Python | 3.10+ | 3.10+ |
| RAM | 8GB+ | 8GB+ |
| VRAM | N/A | 4GB+ |
| Storage | 5GB+ | 10GB+ |
| CUDA | N/A | 12.1+ |

## Common Commands Reference

```bash
# Enter Poetry shell
poetry shell

# Run tests
poetry run pytest

# Update dependencies
poetry update

# Build the project
poetry build

# Install in development mode
poetry install --with dev

# Check package info
poetry show

# Lock dependencies
poetry lock
```

## Environment Variables

Set before running scripts:

```bash
# Model selection
MODEL=llama-3.1

# RAG parameters
K=2  # Number of documents to retrieve
SYNTHESIS_STRATEGY=async-tree-summarization

# Text splitting
CHUNK_SIZE=1000
CHUNK_OVERLAP=50

# Installation
INSTALL_CUDA=1  # Set to 1 for CUDA support

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
```

## Next Steps

1. Place your documentation in `docs/` folder
2. Run memory builder to create embeddings
3. Launch Streamlit app
4. Start asking questions about your documents

## Support & Resources

- [RAG Chatbot Architecture](docs/demo.md)
- [LLaMA CPP Python](https://github.com/abetlen/llama-cpp-python)
- [Chroma Vector DB](https://www.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Ollama Documentation](https://ollama.ai)

## License

See LICENSE file for details.
