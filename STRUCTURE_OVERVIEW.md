# Complete Project Folder Structure

Comprehensive directory tree of the Rag-Chat-Bot project.

```
rag-chatbot-main/
│
├── .git/                                    # Git repository data
│
├── .github/                                 # GitHub configuration
│   └── workflows/
│       ├── ci.yaml                          # Continuous Integration pipeline
│       └── pre-commit.yaml                  # Pre-commit checks workflow
│
├── .gitignore                               # Git ignore patterns
├── .pre-commit-config.yaml                  # Pre-commit hooks config
├── .pytest_cache/                           # Pytest cache directory
│
├── chatbot/                                 # Main application package
│   │
│   ├── __init__.py
│   ├── chatbot_app.py                       # Streamlit app (conversation-aware chatbot)
│   ├── rag_chatbot_app.py                   # Streamlit app (RAG chatbot with documents)
│   ├── memory_builder.py                    # Build vector embeddings from documents
│   │
│   ├── bot/                                 # Core bot engine
│   │   ├── __init__.py
│   │   │
│   │   ├── client/                          # LLM client implementations
│   │   │   ├── __init__.py
│   │   │   ├── lama_cpp_client.py           # Local inference via llama-cpp
│   │   │   ├── ollama_client.py             # Ollama backend integration
│   │   │   ├── openai_client.py             # OpenAI API client
│   │   │   ├── langchain_openai_client.py   # LangChain OpenAI wrapper
│   │   │   ├── prompt.py                    # Prompt templates & management
│   │   │   └── __pycache__/
│   │   │
│   │   ├── conversation/                    # Chat history & context management
│   │   │   ├── __init__.py
│   │   │   ├── chat_history.py              # In-memory chat history
│   │   │   ├── conversation_handler.py      # Conversation state management
│   │   │   ├── ctx_strategy.py              # Context overflow strategies
│   │   │   └── __pycache__/
│   │   │
│   │   ├── memory/                          # Vector embeddings & RAG
│   │   │   ├── __init__.py
│   │   │   ├── embedder.py                  # Embedding generation (sentence-transformers)
│   │   │   ├── __pycache__/
│   │   │   │
│   │   │   └── vector_database/             # Vector database interface
│   │   │       ├── __init__.py
│   │   │       ├── chroma.py                # Chroma vector store integration
│   │   │       ├── distance_metric.py       # Similarity metrics (cosine, euclidean)
│   │   │       └── __pycache__/
│   │   │
│   │   ├── model/                           # Model registry & configuration
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py                # Abstract base model class
│   │   │   ├── model_registry.py            # Model loader & registry
│   │   │   ├── __pycache__/
│   │   │   │
│   │   │   └── settings/                    # Model-specific configurations
│   │   │       ├── __init__.py
│   │   │       ├── llama.py                 # Llama model parameters
│   │   │       ├── qwen.py                  # Qwen model parameters
│   │   │       ├── phi.py                   # Phi model parameters
│   │   │       ├── deep_seek.py             # DeepSeek model parameters
│   │   │       ├── openchat.py              # OpenChat model parameters
│   │   │       ├── starling.py              # Starling model parameters
│   │   │       ├── stablelm_zephyr.py       # StableLM Zephyr parameters
│   │   │       └── __pycache__/
│   │   │
│   │   └── __pycache__/
│   │
│   ├── cli/                                 # Command-line interfaces
│   │   ├── __init__.py
│   │   ├── chatbot.py                       # CLI for conversation chatbot
│   │   ├── rag_chatbot.py                   # CLI for RAG chatbot
│   │   └── __pycache__/
│   │
│   ├── document_loader/                     # Document processing
│   │   ├── __init__.py
│   │   ├── loader.py                        # PDF & Markdown document loader
│   │   ├── text_splitter.py                 # Text chunking (RecursiveCharacterTextSplitter)
│   │   ├── format.py                        # Document format detection & parsing
│   │   └── __pycache__/
│   │
│   ├── entities/                            # Data models
│   │   ├── __init__.py
│   │   ├── document.py                      # Document entity with metadata
│   │   └── __pycache__/
│   │
│   ├── helpers/                             # Utility functions
│   │   ├── __init__.py
│   │   ├── log.py                           # Logging configuration
│   │   ├── prettier.py                      # Output formatting
│   │   ├── reader.py                        # File reading utilities
│   │   └── __pycache__/
│   │
│   ├── experiments/                         # Research & experimentation
│   │   ├── explore_memory.py                # Vector memory exploration
│   │   │
│   │   └── exp_lama_cpp/                    # LLaMA-cpp experiments
│   │       ├── chat_template.py             # LLaMA template exploration
│   │       └── function_calling.py          # Function calling experiments
│   │
│   └── __pycache__/
│
├── tests/                                   # Test suite (pytest)
│   ├── conftest.py                          # Pytest fixtures & configuration
│   ├── __pycache__/
│   │
│   ├── bot/                                 # Bot tests
│   │   ├── __pycache__/
│   │   │
│   │   ├── client/                          # LLM client tests
│   │   │   ├── test_lamacpp_client.py       # LLaMA-cpp client tests
│   │   │   └── __pycache__/
│   │   │
│   │   ├── conversation/                    # Conversation handler tests
│   │   │   ├── test_conversation_handler.py
│   │   │   └── __pycache__/
│   │   │
│   │   └── memory/                          # Memory/vector DB tests
│   │       ├── __pycache__/
│   │       │
│   │       └── vector_database/
│   │           ├── test_chroma.py           # Chroma vector DB tests
│   │           └── __pycache__/
│   │
│   ├── document_loader/                     # Document loader tests
│   │   ├── test_loader_pdf.py               # PDF loader tests
│   │   ├── test_text_splitter.py            # Text splitting tests
│   │   └── __pycache__/
│   │
│   └── __pycache__/
│
├── docs/                                    # Documentation
│   ├── .gitkeep                             # Git directory marker
│   ├── demo.md                              # Example usage & demonstration
│
├── images/                                  # Visual assets
│   ├── rag-chatbot-architecture-1.png       # System architecture diagram
│   ├── create-and-refine-the-context.png    # Response synthesis strategy
│   ├── hierarchical-summarization.png       # Response synthesis strategy
│   ├── conversation-aware-chatbot.gif       # Chatbot demo GIF
│   ├── rag_chatbot_example.gif              # RAG chatbot demo GIF
│   ├── bot.png                              # Bot icon
│   └── bot-small.png                        # Bot icon (small)
│
├── models/                                  # Model storage directory
│   └── .gitkeep                             # Placeholder (models go here)
│
├── nltk_data/                               # NLP resources (pre-downloaded)
│   ├── punkt.zip                            # Archived sentence tokenizer
│   ├── averaged_perceptron_tagger.zip       # Archived POS tagger
│   │
│   ├── tokenizers/                          # Tokenization data
│   │   └── punkt/                           # Sentence tokenizer
│   │       ├── README
│   │       ├── czech.pickle
│   │       ├── danish.pickle
│   │       ├── dutch.pickle
│   │       ├── english.pickle
│   │       ├── estonian.pickle
│   │       ├── finnish.pickle
│   │       ├── french.pickle
│   │       ├── german.pickle
│   │       ├── greek.pickle
│   │       ├── italian.pickle
│   │       ├── malayalam.pickle
│   │       ├── norwegian.pickle
│   │       ├── polish.pickle
│   │       ├── portuguese.pickle
│   │       ├── russian.pickle
│   │       ├── slovene.pickle
│   │       ├── spanish.pickle
│   │       ├── swedish.pickle
│   │       ├── turkish.pickle
│   │       ├── .DS_Store
│   │       │
│   │       └── PY3/                         # Python 3 specific tokenizers
│   │           ├── README
│   │           ├── czech.pickle
│   │           ├── danish.pickle
│   │           ├── dutch.pickle
│   │           ├── english.pickle
│   │           ├── estonian.pickle
│   │           ├── finnish.pickle
│   │           ├── french.pickle
│   │           ├── german.pickle
│   │           ├── greek.pickle
│   │           ├── italian.pickle
│   │           ├── malayalam.pickle
│   │           ├── norwegian.pickle
│   │           ├── polish.pickle
│   │           ├── portuguese.pickle
│   │           ├── russian.pickle
│   │           ├── slovene.pickle
│   │           ├── spanish.pickle
│   │           ├── swedish.pickle
│   │           └── turkish.pickle
│   │
│   └── taggers/                             # Tagging data
│       └── averaged_perceptron_tagger/      # POS tagger
│           └── averaged_perceptron_tagger.pickle
│
├── vector_store/                            # Vector database (Chroma) storage
│   ├── chroma.sqlite3                       # Main vector database file
│   │
│   └── docs_index/                          # Index metadata
│       ├── chroma.sqlite3
│       │
│       └── ba52c292-a51b-4503-b326-6f4e747d52b4/  # Collection directory
│           ├── data_level0.bin              # HNSW graph data
│           ├── header.bin                   # Header information
│           ├── length.bin                   # Length metadata
│           └── link_lists.bin               # Link lists for HNSW
│
├── version/                                 # Version specifications
│   ├── poetry                               # Required Poetry version (1.7.0)
│   └── llama_cpp                            # Required llama-cpp version
│
├── .pytest_cache/                           # Pytest cache
│
├── Makefile                                 # Build automation & development tasks
├── pyproject.toml                           # Poetry project configuration
├── poetry.lock                              # Locked dependency versions
│
├── setup.sh                                 # Linux/macOS automated setup script
├── setup.bat                                # Windows automated setup script
│
├── README.md                                # Main project documentation
├── SETUP_GUIDE.md                           # Complete setup instructions
├── PROJECT_STRUCTURE.md                     # Detailed architecture docs
├── STRUCTURE_OVERVIEW.md                    # This file
│
├── LICENSE                                  # MIT License
├── demo.md                                  # Demo documentation
├── llama-server-docker.md                   # Docker setup for LLaMA server
├── notes.md                                 # Project notes
├── todo.md                                  # TODO items
├── log.txt                                  # Log file
│
└── demo.md                                  # Demo markdown
```

## Directory Descriptions

### Root Level
- **Configuration Files**: `pyproject.toml`, `poetry.lock`, `.gitignore`, etc.
- **Setup Scripts**: `setup.sh` (Unix/Mac), `setup.bat` (Windows)
- **Documentation**: `README.md`, `SETUP_GUIDE.md`, `PROJECT_STRUCTURE.md`
- **Build Automation**: `Makefile` for development tasks

### chatbot/ - Main Application
Core implementation organized by functionality:

| Directory | Purpose |
|-----------|---------|
| `bot/client/` | LLM backend integrations (llama-cpp, Ollama, OpenAI) |
| `bot/conversation/` | Chat history & context management |
| `bot/memory/` | Vector embeddings & retrieval (Chroma) |
| `bot/model/` | Model registry & configuration |
| `cli/` | Command-line interfaces |
| `document_loader/` | PDF & Markdown processing |
| `entities/` | Data models (Document, etc.) |
| `helpers/` | Utilities (logging, formatting, file I/O) |
| `experiments/` | Research & experimentation |

### tests/
Parallel structure to chatbot/ with test files:
- `tests/bot/client/` - LLM client tests
- `tests/bot/conversation/` - Conversation handler tests
- `tests/bot/memory/` - Vector DB tests
- `tests/document_loader/` - Document processing tests

### Data & Storage Directories
- **nltk_data/**: Pre-downloaded NLTK tokenizers & taggers
- **vector_store/**: Chroma vector database (embeddings)
- **models/**: Storage for GGUF quantized models
- **docs/**: User documentation folder

### Support Files
- **.github/workflows/**: CI/CD pipelines
- **images/**: Architecture diagrams & screenshots
- **version/**: Pinned dependency versions

## Key Metrics

| Metric | Count |
|--------|-------|
| Python Modules | 40+ |
| Test Files | 6 |
| Configuration Files | 5 |
| Documentation Files | 5 |
| LLM Clients | 4 |
| Model Settings | 7 |
| Supported Languages (Tokenizers) | 19 |

## File Organization by Type

### Python Code
- Core logic: `chatbot/bot/`
- Applications: `chatbot/chatbot_app.py`, `chatbot/rag_chatbot_app.py`
- Builders: `chatbot/memory_builder.py`
- Tests: `tests/`

### Configuration
- Poetry: `pyproject.toml`, `poetry.lock`
- Git: `.gitignore`, `.pre-commit-config.yaml`
- GitHub: `.github/workflows/`

### Data
- Vector DB: `vector_store/`
- Models: `models/`
- NLTK Data: `nltk_data/`

### Documentation
- User Guides: `README.md`, `SETUP_GUIDE.md`, `PROJECT_STRUCTURE.md`
- Technical: `demo.md`, `notes.md`
- Docker: `llama-server-docker.md`

### Build & Setup
- Scripts: `setup.sh`, `setup.bat`
- Automation: `Makefile`

## Total Project Size
- **Source Code**: ~2-3 MB
- **NLTK Data**: ~500 KB
- **Vector Store**: ~100 KB
- **Total**: ~4-5 MB (without models)

## Models Directory
Note: Place downloaded GGUF models in `models/` directory:
```
models/
├── llama-3.1-8b.gguf
├── phi-3.5-mini.gguf
├── qwen-2.5-3b.gguf
└── ...
```

## Git Structure
```
.git/
├── objects/         # Git objects
├── refs/            # Branch & tag references
├── hooks/           # Git hooks
├── config           # Repository config
└── HEAD             # Current branch pointer
```

## Cache & Generated Directories (in .gitignore)
- `__pycache__/` - Python bytecode cache
- `.pytest_cache/` - Pytest cache
- `*.egg-info/` - Package info
- `.venv/` or `.venv_*` - Virtual environments
