Offline RAG Chatbot with Semantic Search (Ollama + Vector DB)
Project Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers user questions based on their own documents using semantic search and a local Large Language Model (LLM).

Unlike traditional chatbots that rely on paid cloud APIs, this system runs fully offline using Ollama. This makes the application cost-free, privacy-friendly, and suitable for local development, academic projects, and enterprise environments.

The chatbot retrieves the most relevant document chunks from a vector database and then generates answers grounded strictly in that retrieved context.

Objectives

Build a document-aware chatbot using RAG

Enable semantic search over custom documents

Avoid cloud-based LLM dependencies

Provide streaming answers in a simple web UI

Maintain a modular and extensible architecture

Key Features

Semantic search using vector embeddings

Retrieval-Augmented Generation (RAG)

Fully offline LLM inference using Ollama

No API keys or usage limits required

Streaming chatbot responses in real time

Modular backend with interchangeable LLMs

High-Level Architecture

User Query → Query Embedding → Vector Database (Semantic Search) → Relevant Document Chunks → Prompt Construction (RAG) → Local LLM via Ollama → Streamed Answer to UI

Project Structure
rag-chatbot-main/
│
├── chatbot/
│   ├── memory/                         # Vector embeddings & RAG logic
│   │   └── vector_database/            # Chroma vector store implementation
│   │
│   ├── model/                          # Model registry and configuration
│   │   └── settings/                   # Model configs (llama, qwen, phi, etc.)
│   │
│   ├── cli/                            # Command-line interfaces
│   ├── document_loader/                # PDF & Markdown document processing
│   ├── entities/                       # Data models and schemas
│   ├── helpers/                        # Utility functions (logging, formatting)
│   ├── experiments/                    # Research experiments and prototypes
│   │
│   ├── chatbot_app.py                  # Streamlit conversational chatbot
│   ├── rag_chatbot_app.py              # Streamlit RAG chatbot (main app)
│   └── memory_builder.py               # Builds vector embeddings from documents
│
├── tests/                              # Test suite
│   ├── bot/                            # Bot-related tests
│   │   ├── client/
│   │   ├── conversation/
│   │   └── memory/
│   │
│   └── document_loader/                # Document loader tests
│
├── docs/                               # Documentation and sample documents
│
├── images/                             # Architecture diagrams and screenshots
│
├── models/                             # Local GGUF model storage (user-provided)
│
├── nltk_data/                          # NLTK tokenizers and taggers
│   ├── tokenizers/
│   │   └── punkt/
│   └── taggers/
│       └── averaged_perceptron_tagger/
│
├── vector_store/                       # Persistent Chroma vector database
│   ├── chroma.sqlite3
│   └── docs_index/
│
├── version/                            # Version tracking
│   ├── poetry                          # Poetry version information
│   └── llama_cpp                       # llama-cpp version reference
│
├── configuration-and-setup/
│   ├── Makefile                        # Build and automation tasks
│   ├── pyproject.toml                  # Poetry dependencies
│   ├── poetry.lock                     # Locked dependency versions
│   ├── setup.sh                        # Linux/macOS setup script
│   └── setup.bat                       # Windows setup script
│
└── documentation/
    ├── README.md                       # Main project documentation
    ├── SETUP_GUIDE.md                  # Detailed setup instructions
    ├── PROJECT_STRUCTURE.md            # Architecture explanation
    ├── STRUCTURE_OVERVIEW.md            # Full structure overview
    ├── LICENSE                         # MIT License
    └── demo.md                         #

Document Handling

All documents are stored in the docs/ directory

Supported formats include Markdown (PDF support can be added)

Documents are split into manageable chunks

Each chunk is converted into vector embeddings

Embeddings are stored in a vector database for fast similarity search

Semantic Search Workflow

The user enters a natural language query

The query is converted into an embedding

Vector similarity search retrieves the most relevant chunks

Low-relevance chunks are filtered using a threshold

The selected chunks are passed to the LLM as context

Retrieval-Augmented Generation (RAG)

The system does not allow the LLM to answer from general knowledge alone.

Instead:

Retrieved document chunks are injected into the prompt

The model is instructed to answer only using the provided context

This approach significantly reduces hallucinations

Responses remain grounded in the source documents

LLM Backend
Ollama (Primary Backend)

Runs local language models such as:

llama3

stablelm-zephyr

mistral

No internet connection or API keys required

Accessed through a local HTTP API at http://localhost:11434

OpenAI (Optional Fallback)

Can be enabled if an API key is provided

Automatically disabled when Ollama is available

Not required for offline execution

Environment Variables
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3:latest

Environment variables can be set in the terminal or via a .env file.

How to Run Locally (Windows)
Step 1: Install Ollama

Download and install Ollama from: https://ollama.com/download

Step 2: Pull a Model
ollama pull llama3
Step 3: Install Project Dependencies
poetry install
Step 4: Run the Application
poetry run streamlit run chatbot/rag_chatbot_app.py

Open the application in a browser at: http://localhost:8501

Example Queries

Summarize the documents

What topics are covered in the uploaded files

Explain the main idea of the project

Extensibility

The vector database can be replaced with Endee

Additional document loaders can be added

Multiple LLM backends are supported

Context synthesis strategies are modular and configurable

Conclusion

This project demonstrates a production-ready offline RAG system that combines semantic search with local LLM inference. The design is modular, cost-free, and privacy-focused, making it suitable for academic, research, and real-world knowledge assistant applications.

License

This project is intended for educational and research use.
