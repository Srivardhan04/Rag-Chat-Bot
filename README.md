# Rag-Chat-Bot

Offline RAG Chatbot with Semantic Search powered by Ollama and Vector Database. A production-ready system for local LLM inference with document-based question answering.

**No cloud API required. Fully privacy-focused. Run completely offline.**

Maintained by Srivardhan04.


## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [How to Run Locally](#how-to-run-locally)
- [Example Queries](#example-queries)
- [Extensibility](#extensibility)
- [Conclusion](#conclusion)
- [License](#license)

## Overview

Rag-Chat-Bot is a Retrieval-Augmented Generation (RAG) system that enables semantic search over documents using local Large Language Models. The system works completely offline with no dependence on cloud services, making it ideal for privacy-sensitive applications, research, and enterprise use cases.

### Core Technology Stack

- **Vector Database**: Chroma (in-memory vector storage with similarity search)
- **Local LLM Inference**: Ollama (Docker-based local model serving)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Web Interface**: Streamlit (interactive UI for querying)
- **Document Processing**: PDF & Markdown support
- **Context Management**: Hierarchical summarization strategies

## Key Features

### Offline Semantic Search

- Index documents and search using semantic meaning, not keywords
- Vector embeddings stored locally in Chroma vector database
- No data sent to external servers
- Complete privacy for sensitive documents

### Local LLM Inference

- Accessed through local HTTP API at `http://localhost:11434`
- Supports multiple model architectures (Llama, Qwen, Phi, DeepSeek)
- Model switching without code changes
- Streaming responses for better UX

### OpenAI Integration (Optional Fallback)

- Can be enabled if an API key is provided
- Automatically disabled when Ollama is available
- Not required for offline execution
- Optional cost control

### Document Support

- Markdown files (.md) - Full support
- PDF files (.pdf) - Full support with metadata
- Automatic document chunking with configurable overlap
- Source attribution in responses

### Multiple Context Synthesis Strategies

- `create-and-refine`: Sequential context processing
- `tree-summarization`: Hierarchical answer synthesis
- `async-tree-summarization`: Fast parallel processing

## Architecture

```
User Query
    |
    v
[Streamlit Web Interface]
    |
    +---+---+---+
        |
        v
[Vector Search (Chroma)]
    |
    v
[Retrieved Documents]
    |
    v
[Context Manager]
    |
    v
[Ollama (Local LLM)]
    |
    v
[Final Response]
```

## Prerequisites

- **Operating System**: Windows, macOS, or Linux
- **Memory**: 8GB+ RAM
- **Storage**: 10GB+ (for models and vector database)
- **Internet**: Required for initial setup only (to download Ollama and models)
- **Python**: 3.10 or higher (optional, comes with Ollama)

## Installation

### Step 1: Install Ollama

Download and install Ollama from: [https://ollama.com/download](https://ollama.com/download)

**Windows**: Run the installer executable  
**macOS**: Install via Homebrew or DMG  
**Linux**: Run the installation script

### Step 2: Pull an LLM Model

Open terminal/command prompt and run:

```bash
ollama pull llama2
```

Or choose another model:

```bash
ollama pull llama3
ollama pull mistral
ollama pull neural-chat
```

### Step 3: Install Project Dependencies

```bash
# Clone or download the repository
cd rag-chatbot-main

# Install Python dependencies using Poetry
poetry install

# Or use pip
pip install -r requirements.txt
```

### Step 4: Build Vector Embeddings (Optional)

If you have documents in the `docs/` folder:

```bash
poetry run python chatbot/memory_builder.py --chunk-size 1000 --chunk-overlap 50
```

This creates embeddings for semantic search.

## Configuration

### Environment Variables

Set these before running the application:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2

# RAG Parameters
K=2                                      # Number of documents to retrieve
SYNTHESIS_STRATEGY=async-tree-summarization

# OpenAI (Optional)
OPENAI_API_KEY=your-key-here           # If using OpenAI fallback

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=50
```

### Using a .env File

Create a `.env` file in the project root:

```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
K=2
SYNTHESIS_STRATEGY=async-tree-summarization
```

Load it automatically by running:

```bash
set -a
source .env
set +a
```

On Windows:

```cmd
for /f "tokens=*" %a in ('type .env') do set "%a"
```

## How to Run Locally

### Windows

#### Step 1: Start Ollama

```cmd
ollama serve
```

Or if Ollama is installed as a service, it should run automatically.

#### Step 2: Verify Ollama is Running

```cmd
curl http://localhost:11434/api/tags
```

Should return a JSON list of models.

#### Step 3: Install Dependencies

```cmd
pip install poetry
poetry install
```

#### Step 4: Run the Application

```cmd
poetry run streamlit run chatbot/rag_chatbot_app.py
```

#### Step 5: Open in Browser

The application automatically opens at: `http://localhost:8501`

Or manually open: `http://127.0.0.1:8501`

### macOS / Linux

```bash
# Start Ollama in background
ollama serve &

# Install dependencies
poetry install

# Run application
poetry run streamlit run chatbot/rag_chatbot_app.py
```

### Configuration for Your System

**For CPU-only systems:**
```bash
OLLAMA_MODEL=mistral
poetry run streamlit run chatbot/rag_chatbot_app.py
```

**For GPU-enabled systems:**
```bash
OLLAMA_MODEL=llama2
poetry run streamlit run chatbot/rag_chatbot_app.py
```

**For Mac with Apple Silicon:**
```bash
OLLAMA_MODEL=neural-chat
poetry run streamlit run chatbot/rag_chatbot_app.py
```

## Example Queries

Once the application is running, try these queries:

- "Summarize the uploaded documents"
- "What topics are covered in the files?"
- "Explain the main idea of the project"
- "List all key concepts mentioned"
- "What are the technical requirements?"
- "Provide a detailed analysis of section X"
- "Compare and contrast different approaches mentioned"

### Query Flow

1. User enters query in Streamlit interface
2. Query is converted to embedding using Sentence-Transformers
3. Chroma searches vector database for similar documents
4. Top K documents are retrieved (default K=2)
5. Documents are passed to Ollama as context
6. Ollama generates response based on context and query
7. Response is streamed to user interface

## Extensibility

### Replace Vector Database

Currently using **Chroma**. Can be replaced with:

- **Qdrant** - Production-grade vector database
- **Weaviate** - Open-source vector search
- **Milvus** - Scalable vector database
- **Pinecone** - Cloud-hosted (loses offline capability)

### Add New Document Loaders

Custom loaders can be added in `chatbot/document_loader/`:

- Support for Excel files (.xlsx, .csv)
- Web scraping (HTML, API responses)
- Database queries
- Email archive parsing

### Support Additional LLM Backends

Current support:

- Ollama (primary)
- OpenAI (optional)

Can add:

- Local llama-cpp
- Hugging Face Transformers
- Cohere
- Anthropic Claude

### Customize Context Synthesis

Implement custom strategies in `chatbot/bot/conversation/ctx_strategy.py`:

- Custom summarization algorithms
- Multi-stage processing
- Domain-specific context management
- Custom prompt engineering

### Model Configuration

Add new model configs in `chatbot/bot/model/settings/`:

```python
# custom_model.py
CUSTOM_MODEL = {
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "context_window": 4096
}
```

## Performance Considerations

| Component | Typical Performance |
|-----------|-------------------|
| Document Indexing | 1000 docs/minute |
| Vector Search | <100ms |
| Embedding Generation | 5-10s per document |
| LLM Inference | 10-50 tokens/second |
| Total Query Time | 2-5 seconds (end-to-end) |

Performance depends on:

- Model size and complexity
- System hardware
- Number of retrieved documents
- Query complexity

## Troubleshooting

**Ollama not responding?**
```bash
# Check if running
curl http://localhost:11434/api/tags

# Restart service
ollama serve
```

**Out of memory errors?**
- Use smaller model (Phi-3, Neural-Chat instead of Llama)
- Reduce K (number of retrieved documents)
- Increase CHUNK_OVERLAP

**Slow performance?**
- Use GPU if available
- Reduce document chunk size
- Use smaller embedding model
- Reduce K value

**Models not appearing?**
```bash
ollama list
ollama pull llama2
```

## Conclusion

Rag-Chat-Bot demonstrates a production-ready offline RAG system that combines:

- Semantic search via vector databases
- Local LLM inference without cloud dependency
- Document processing with multiple formats
- Modular, extensible architecture
- Privacy-focused design (no data leaves your system)
- Cost-free execution (no API charges)

This makes it suitable for:

- Academic research
- Corporate knowledge assistance
- Privacy-sensitive applications
- Offline document analysis
- Educational demonstrations

The modular design allows customization for specific use cases while maintaining ease of use for general document Q&A.

## License

Free for educational, research, and commercial use.

---

**For detailed setup instructions**, see [SETUP_GUIDE.md](SETUP_GUIDE.md)

**For architecture details**, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

**For complete folder structure**, see [STRUCTURE_OVERVIEW.md](STRUCTURE_OVERVIEW.md)
