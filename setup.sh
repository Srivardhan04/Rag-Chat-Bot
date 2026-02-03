#!/usr/bin/env bash
set -euo pipefail

# Default configurable env vars (can be exported before running)
: "${MODEL:=llama-3.1}"
: "${K:=2}"
: "${SYNTHESIS_STRATEGY:=async-tree-summarization}"
: "${CHUNK_SIZE:=1000}"
: "${CHUNK_OVERLAP:=50}"
: "${INSTALL_CUDA:=0}"
: "${SAMPLE_DOC_URL:=https://raw.githubusercontent.com/adam-p/markdown-here/master/README.md}"
: "${SKIP_RUN:=0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "Checking Python 3.10+..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Please install Python 3.10+ and re-run this script." >&2
  exit 1
fi
python3 - <<'PY' || { echo "ERROR: Python 3.10+ is required."; exit 1; }
import sys
if sys.version_info < (3,10):
    print('too-old')
    raise SystemExit(1)
print('ok')
PY

# Install Poetry (1.7.0) if missing
if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry not found. Installing Poetry 1.7.0..."
  curl -sSL https://install.python-poetry.org | python3 - --version 1.7.0
  # ensure poetry is on PATH for this session (default installer puts it in $HOME/.local/bin)
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "Poetry found: $(poetry --version)"
fi

# Use the current python for poetry venv (creates env if missing)
PYTHON_PATH="$(command -v python3)"
if [ -z "${PYTHON_PATH}" ]; then
  echo "Could not resolve python3 path. Exiting." >&2
  exit 1
fi

echo "Ensuring Poetry environment uses: $PYTHON_PATH"
poetry env use "$PYTHON_PATH" || echo "Note: poetry env use reported non-zero status (ok to continue)."

# Install dependencies (optionally with CUDA extras)
if [ "$INSTALL_CUDA" = "1" ]; then
  echo "Installing dependencies with CUDA acceleration extras..."
  poetry install -E cuda-acceleration
else
  echo "Installing dependencies..."
  poetry install
fi

# Prepare docs folder and sample docs
mkdir -p docs
if [ ! -f docs/demo.md ]; then
  if [ -f demo.md ]; then
    cp demo.md docs/
    echo "Copied local demo.md into docs/"
  else
    echo "No local demo.md found. Downloading sample markdown into docs/sample.md..."
    curl -fsSL "$SAMPLE_DOC_URL" -o docs/sample.md || echo "Failed to download sample doc from $SAMPLE_DOC_URL"
  fi
else
  echo "docs/demo.md already exists - skipping docs copy/download"
fi

# Build vector index
echo "Building vector embedding index (chunk-size=$CHUNK_SIZE chunk-overlap=$CHUNK_OVERLAP)..."
poetry run python chatbot/memory_builder.py --chunk-size "$CHUNK_SIZE" --chunk-overlap "$CHUNK_OVERLAP"

# Run Streamlit app unless SKIP_RUN set
if [ "$SKIP_RUN" != "1" ]; then
  echo "Starting Streamlit RAG Chatbot (model=$MODEL)..."
  echo "If this is the first time running, the selected model will be downloaded automatically into ./models (may be large)."
  poetry run streamlit run chatbot/rag_chatbot_app.py -- --model "$MODEL" --k "$K" --synthesis-strategy "$SYNTHESIS_STRATEGY"
else
  echo "SKIP_RUN=1 set - setup is complete. You can start the app with:"
  echo "  poetry run streamlit run chatbot/rag_chatbot_app.py -- --model $MODEL --k $K --synthesis-strategy $SYNTHESIS_STRATEGY"
fi
