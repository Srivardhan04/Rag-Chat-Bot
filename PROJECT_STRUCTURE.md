# Project Structure Documentation

Detailed breakdown of the Rag-Chat-Bot project organization and module responsibilities.

## Directory Layout

### Root Level Files

```
Makefile              - Build automation for development tasks
pyproject.toml        - Poetry project configuration and dependencies
poetry.lock           - Locked dependency versions
setup.sh              - Linux/macOS automated setup script
setup.bat             - Windows automated setup script
README.md             - Main project documentation
LICENSE               - MIT License
```

## chatbot/ - Main Application

Core implementation of the chatbot and RAG system.

### chatbot/bot/ - Bot Engine

Core business logic and integrations.

**client/** - Language Model Clients
- `lama_cpp_client.py` - Local inference via llama-cpp
- `ollama_client.py` - Ollama backend integration
- `openai_client.py` - OpenAI API client
- `langchain_openai_client.py` - LangChain OpenAI wrapper
- `prompt.py` - Prompt templates and management
- `__init__.py` - Module initialization

**conversation/** - Chat History & Context
- `chat_history.py` - In-memory chat history management
- `conversation_handler.py` - Conversation state management
- `ctx_strategy.py` - Context window overflow strategies
- `__init__.py` - Module initialization

**memory/** - Vector Embeddings & RAG
- `embedder.py` - Embedding generation using sentence-transformers
- `vector_database/`
  - `chroma.py` - Chroma vector store integration
  - `distance_metric.py` - Similarity metrics (cosine, euclidean)
  - `__pycache__/` - Compiled Python cache

**model/** - Model Registry & Configuration
- `base_model.py` - Abstract base model class
- `model_registry.py` - Model loader and registry
- `settings/` - Model-specific configurations
  - `llama.py` - Llama model parameters
  - `qwen.py` - Qwen model parameters
  - `phi.py` - Phi model parameters
  - `deep_seek.py` - DeepSeek model parameters
  - `openchat.py` - OpenChat model parameters
  - `starling.py` - Starling model parameters
  - `stablelm_zephyr.py` - StableLM Zephyr parameters

### chatbot/cli/ - Command-Line Interfaces

- `chatbot.py` - CLI for conversation-aware chatbot
- `rag_chatbot.py` - CLI for RAG chatbot
- `__init__.py` - Module initialization

### chatbot/document_loader/ - Document Processing

- `loader.py` - PDF and Markdown document loading
- `text_splitter.py` - Text chunking (RecursiveCharacterTextSplitter)
- `format.py` - Document format detection and parsing
- `__init__.py` - Module initialization

### chatbot/entities/ - Data Models

- `document.py` - Document entity with metadata
- `__init__.py` - Module initialization

### chatbot/helpers/ - Utility Functions

- `log.py` - Logging configuration
- `prettier.py` - Output formatting
- `reader.py` - File reading utilities
- `__init__.py` - Module initialization

### chatbot/experiments/ - Research & Experimentation

- `explore_memory.py` - Vector memory exploration scripts
- `exp_lama_cpp/`
  - `chat_template.py` - LLaMA template exploration
  - `function_calling.py` - Function calling experiments

### Top-Level Scripts

- `chatbot_app.py` - Streamlit app for conversation-aware chatbot
- `rag_chatbot_app.py` - Streamlit app for RAG chatbot
- `memory_builder.py` - Script to build vector database
- `__pycache__/` - Python compiled bytecode cache

## tests/ - Test Suite

Pytest-based testing organized by module:

```
tests/
├── conftest.py                                  # Pytest fixtures
├── bot/
│   ├── client/
│   │   └── test_lamacpp_client.py              # LLaMA CPP client tests
│   ├── conversation/
│   │   └── test_conversation_handler.py        # Conversation handler tests
│   └── memory/
│       └── vector_database/
│           └── test_chroma.py                  # Chroma vector DB tests
└── document_loader/
    ├── test_loader_pdf.py                      # PDF loader tests
    └── test_text_splitter.py                   # Text splitting tests
```

## docs/ - Documentation

- `demo.md` - Example usage and demonstration
- `.gitkeep` - Git directory marker

## images/ - Visual Assets

```
images/
├── rag-chatbot-architecture-1.png              # System architecture
├── create-and-refine-the-context.png           # Synthesis strategy
├── hierarchical-summarization.png              # Synthesis strategy
├── conversation-aware-chatbot.gif              # App demo
├── rag_chatbot_example.gif                     # RAG demo
├── bot.png                                      # Bot icon
└── bot-small.png                               # Bot icon (small)
```

## models/ - Model Storage

Directory for storing GGUF quantized models:

```
models/
└── .gitkeep                                    # Placeholder for models
```

## vector_store/ - Vector Database

Chroma database storage for document embeddings:

```
vector_store/
├── chroma.sqlite3                              # Vector database
└── docs_index/                                 # Index metadata
    ├── chroma.sqlite3
    └── ba52c292-a51b-4503-b326-6f4e747d52b4/ # Collection directory
        ├── data_level0.bin
        ├── header.bin
        ├── length.bin
        └── link_lists.bin
```

## nltk_data/ - NLP Resources

Pre-downloaded NLTK data for text processing:

```
nltk_data/
├── tokenizers/
│   └── punkt/                                  # Sentence tokenizer
├── taggers/
│   └── averaged_perceptron_tagger/            # POS tagger
└── punkt.zip, averaged_perceptron_tagger.zip # Archived data
```

## version/ - Version Files

```
version/
├── poetry                                      # Required Poetry version (1.7.0)
└── llama_cpp                                   # Required llama-cpp version
```

## .github/ - GitHub Configuration

```
.github/
└── workflows/
    ├── ci.yaml                                 # CI/CD pipeline
    └── pre-commit.yaml                        # Pre-commit checks
```

## Configuration Files

- `.gitignore` - Git ignore patterns
- `.pre-commit-config.yaml` - Pre-commit hooks configuration
- `pyproject.toml` - Poetry dependencies and project metadata

## Data Flow Architecture

```
User Input
    |
    v
[Streamlit UI]
    |
    +-- [ChatBot App] ------+
    |                       |
    +-- [RAG ChatBot] ------+
                            |
                            v
                    [Conversation Handler]
                            |
            +-------+-------+-------+-------+
            |       |       |       |       |
            v       v       v       v       v
        [Chat   [Memory  [LLM    [Context [Response
         History] Builder] Client] Manager] Synthesizer]
            |       |       |       |       |
            +---+---+---+---+---+---+---+---+
                    |
                    v
            [Vector Database (Chroma)]
                    |
                    v
            [Document Embeddings]
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ~2.1.2 | Deep learning framework |
| sentence-transformers | ~5.0.0 | Embedding generation |
| streamlit | ~1.37.0 | Web UI framework |
| chromadb | ~0.4.18 | Vector database |
| llama-cpp-python | - | Local LLM inference |
| transformers | ~4.50.0 | Model loading |
| openai | 0.28.1 | OpenAI API (pinned) |
| PyPDF2 | ^3.0.0 | PDF processing |
| unstructured[md] | ~0.14.3 | Markdown parsing |
| clean-text | ~0.6.0 | Text cleaning |
| rich | ~13.4.2 | Terminal formatting |

## Module Responsibilities

### chatbot.bot.client
- **Responsibility**: Interface with different LLM backends
- **Abstracts**: OpenAI, Ollama, Local LLaMA-cpp
- **Returns**: Completions with streaming support

### chatbot.bot.conversation
- **Responsibility**: Manage conversation state and history
- **Features**: Chat memory, context management, overflow handling
- **Strategies**: Create-and-refine, tree-summarization, async options

### chatbot.bot.memory
- **Responsibility**: Store and retrieve document embeddings
- **Uses**: Sentence-transformers for embeddings
- **Backend**: Chroma vector database
- **Supports**: Similarity and distance metrics

### chatbot.document_loader
- **Responsibility**: Load and process documents
- **Formats**: Markdown (.md), PDF (.pdf)
- **Output**: Chunked documents with metadata

### chatbot.helpers
- **Responsibility**: Cross-cutting utilities
- **Features**: Logging, formatting, file I/O
- **Use**: Throughout the application

## Configuration & Runtime

### Environment Variables

Set before running:
```bash
MODEL=llama-3.1
K=2
SYNTHESIS_STRATEGY=async-tree-summarization
CHUNK_SIZE=1000
CHUNK_OVERLAP=50
INSTALL_CUDA=1
```

### Model Loading

Models are loaded via `model_registry.py` and configured in `model/settings/`.

### Vector Database

Initialized in `memory_builder.py` and accessed via `bot.memory.vector_database.chroma.py`.

## Adding New Features

### New LLM Backend
1. Create client in `bot/client/new_client.py`
2. Implement base model interface
3. Register in `bot/model/model_registry.py`

### New Document Format
1. Add parser to `document_loader/format.py`
2. Update `document_loader/loader.py`
3. Add tests in `tests/document_loader/`

### New Model
1. Add settings to `bot/model/settings/new_model.py`
2. Register in `model_registry.py`
3. Test via CLI and apps

## Testing

- **Framework**: Pytest
- **Location**: `tests/`
- **Run**: `poetry run pytest`
- **Coverage**: `poetry run pytest --cov=chatbot`
