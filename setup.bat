@echo off
setlocal enabledelayedexpansion

REM Defaults (can be overridden by setting env vars before running)
if "%MODEL%"=="" set MODEL=llama-3.1
if "%K%"=="" set K=2
if "%SYNTHESIS_STRATEGY%"=="" set SYNTHESIS_STRATEGY=async-tree-summarization
if "%CHUNK_SIZE%"=="" set CHUNK_SIZE=1000
if "%CHUNK_OVERLAP%"=="" set CHUNK_OVERLAP=50
if "%INSTALL_CUDA%"=="" set INSTALL_CUDA=0
if "%SAMPLE_DOC_URL%"=="" set SAMPLE_DOC_URL=https://raw.githubusercontent.com/adam-p/markdown-here/master/README.md
if "%SKIP_RUN%"=="" set SKIP_RUN=0

echo Checking for Python 3.10+
py -3 --version >nul 2>&1 || (echo ERROR: Python 3.x not found. Install Python 3.10+ and re-run. & exit /b 1)
for /f "tokens=1,2" %%a in ('py -3 -c "import sys; print(sys.version_info.major, sys.version_info.minor)"') do (
  set PY_MAJOR=%%a
  set PY_MINOR=%%b
)
if %PY_MAJOR% LSS 3 (
  echo ERROR: Python 3.10+ is required. & exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 (
  echo ERROR: Python 3.10+ is required. & exit /b 1
)

echo Checking for Poetry...
where poetry >nul 2>&1 || (
  echo Poetry not found. Installing Poetry 1.7.0 via the official installer using PowerShell...
  powershell -Command "(Invoke-WebRequest -Uri 'https://install.python-poetry.org' -UseBasicParsing).Content | py -3 - --version 1.7.0"
)

REM Ensure poetry command is available in this session
where poetry >nul 2>&1 || echo WARNING: poetry executable not found in PATH. You may need to reopen your shell.

REM Use Python executable for poetry venv
for /f "delims=" %%i in ('py -3 -c "import sys; print(sys.executable)"') do set PYEXE=%%i
if defined PYEXE (
  echo Setting Poetry to use: %PYEXE%
  poetry env use "%PYEXE%" || echo Note: poetry env use returned non-zero (that can be okay)
) else (
  echo Could not detect python path to feed to Poetry. Continuing...
)

REM Install dependencies
if "%INSTALL_CUDA%"=="1" (
  echo Installing dependencies with CUDA extras...
  poetry install -E cuda-acceleration
) else (
  echo Installing dependencies...
  poetry install
)

REM Prepare docs
if not exist docs mkdir docs
if exist docs\demo.md (
  echo docs\demo.md already exists - skipping copy
) else (
  if exist demo.md (
    copy /Y demo.md docs\ >nul
    echo Copied demo.md into docs\
  ) else (
    echo Downloading example markdown into docs\sample.md
    powershell -Command "Invoke-WebRequest -Uri '%SAMPLE_DOC_URL%' -OutFile 'docs\sample.md'"
  )
)

REM Build index
echo Building vector embedding index (chunk-size=%CHUNK_SIZE% chunk-overlap=%CHUNK_OVERLAP%)...
poetry run python chatbot/memory_builder.py --chunk-size %CHUNK_SIZE% --chunk-overlap %CHUNK_OVERLAP%

if "%SKIP_RUN%"=="1" (
  echo Setup complete. To start the app, run:
  echo   poetry run streamlit run chatbot/rag_chatbot_app.py -- --model %MODEL% --k %K% --synthesis-strategy %SYNTHESIS_STRATEGY%
  exit /b 0
)

REM Run the Streamlit app
echo Starting Streamlit RAG Chatbot (model=%MODEL%)
echo If this is the first run the model will be downloaded into ./models (may take a long time and a lot of disk space)
poetry run streamlit run chatbot/rag_chatbot_app.py -- --model %MODEL% --k %K% --synthesis-strategy %SYNTHESIS_STRATEGY%
