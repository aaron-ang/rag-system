# RAG System

A Retrieval-Augmented Generation system for medical research papers.

## Setup

### Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installation
```bash
# Install dependencies
uv sync

# Run the download script
uv run download.py

# Run the main application
uv run main.py
```

### Quick Start
```bash
# One-liner setup and run
uv sync && uv run download.py && uv run main.py
```

## Project Structure
- `download.py` - Downloads PubMed abstracts and Semantic Scholar papers
- `main.py` - Main RAG application
- `data/` - Downloaded datasets
