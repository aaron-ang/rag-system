# RAG System

A Retrieval-Augmented Generation system for medical research papers using [SciNCL methodology](https://github.com/malteos/scincl) for scientific document representations and similarity search.

## Features

- **SciNCL Integration**: Uses the official SciNCL model for document embeddings
- **Milvus Vector DB**: Milvus server with IVF index by default; Milvus Lite (flat) fallback available
- **Multiple Data Sources**: PubMed abstracts and Semantic Scholar papers
- **Smart Caching**: Automatic artifact caching
- **Document Deduplication**: Automatic removal of duplicate documents by title
- **CLI & Interactive**: Both command-line and interactive interfaces available
- **LLM Assist (optional)**: LLM can rewrite user queries and generate concise answers using retrieved context via a single `retrieve` call

## Setup

### Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Python 3.13+
- CUDA (optional, for GPU acceleration)

### Installation
```bash
# Install dependencies
uv sync

# Download data
uv run download.py

# Run the main application
uv run main.py
```

### Quick Start
```bash
# One-liner setup and run
uv sync && uv run download.py && uv run main.py
```

## Usage

### Main Application
```bash
# Interactive RAG system with SciNCL methodology
uv run main.py

# Interactive with LLM-assisted rewrite + answers
uv run main.py --llm
```

### Milvus Server (default) vs Lite
- Server (default, "v2" profile): connects to `http://localhost:19530` using an IVF index. Start Milvus via Docker (see below).
- Lite ("v1" profile): add `--v1` to CLI commands for embedded Milvus Lite (local `milvus.db`, FLAT index, full-document embeddings without sliding window).

### CLI Interface

**Examples:**
```bash
# Ingest all data using SciNCL (default: Milvus server IVF at http://localhost:19530)
uv run -m scincl.cli ingest

# Use Milvus Lite (v1 profile) instead
uv run -m scincl.cli ingest --v1

# Query the system
uv run -m scincl.cli query "machine learning in medical diagnosis" --k 5

# Query against Milvus Lite
uv run -m scincl.cli query "machine learning in medical diagnosis" --k 5 --v1

# Query with more results and LLM-assisted answers
uv run -m scincl.cli query "tuberculosis treatment" --k 10 --llm
```

### Start/Stop Milvus Server (Docker, standalone)
```bash
# OR using the bundled helper script
bash standalone_embed.sh start

# Check health
curl -f http://localhost:9091/healthz

# Stop Milvus
bash standalone_embed.sh stop
```

### Programmatic Usage

```python
from scincl import load_or_create_artifacts

# Load existing artifacts or create new ones (enable_llm optional)
retrieval = load_or_create_artifacts(enable_llm=True)

# Query the system; returns RetrievalResult
result = retrieval.retrieve("machine learning in medical diagnosis", k=5)

# Process results
for chunk in result.retrieval_chunks:
    doc = chunk.document
    print(f"{doc.title} (Score: {chunk.sim_score:.3f}) | Source: {doc.source}")

if result.llm_answer:
    print("\nLLM Answer:")
    print(result.llm_answer)
```

### LLM Assist (optional)
- Add `--llm` to CLI or main to enable LLM-assisted query rewrite and concise answers from retrieved context. If LLM is unavailable, the system falls back to the original query and skips the answer.
- The `retrieve` method handles rewrite + answer generation automatically when LLM is enabled.

## Data Sources

### PubMed Abstracts (`data/pubmed_abstracts.csv`)
- Medical research abstracts from PubMed
- Title extracted from first sentence of abstract
- Includes metadata: PMID, language, publication info
- ~100 abstracts included in sample dataset

### Semantic Scholar Papers (`data/semanticscholar.json`)
- Academic papers from Semantic Scholar API
- Includes full abstracts, titles, and metadata
- Open access papers with PDF links
- ~100 papers included in sample dataset

## Project Structure
- `download.py` - Downloads PubMed abstracts and Semantic Scholar papers
- `main.py` - Main RAG application with interactive interface
- `scincl/`
  - `__init__.py` - Package initialization and exports
  - `core.py` - SciNCL ingestion and retrieval system
  - `cli.py` - Command-line interface
- `data/` - Downloaded datasets and generated artifacts
- `data/scincl_artifacts/` - database, embeddings metadata, and documents

## Advanced Features

### Artifact Caching
- Ingestion automatically skips if artifacts already exist
- Saves time on repeated runs

### Document Deduplication
- Documents are automatically deduplicated by title during ingestion
- Duplicate information is skipped
- Check console output for deduplication statistics

### CLI Options
- **Ingest command**: `uv run -m scincl.cli ingest [--model MODEL] [--output-dir DIR] [--v1]`
- **Query command**: `uv run -m scincl.cli query QUERY [--k N] [--artifacts-dir DIR] [--v1]`
- **Milvus**: server default is `http://localhost:19530` with IVF index; add `--v1` for local Milvus Lite (FLAT index).
- **Help**: `uv run -m scincl.cli --help` or `uv run -m scincl.cli ingest --help`

## SciNCL Methodology

This implementation uses the [SciNCL model](https://huggingface.co/malteos/scincl) which provides:

1. **Scientific Document Embeddings**: Specialized embeddings for scientific papers
2. **Title-Abstract Concatenation**: Combines title and abstract with [SEP] token
3. **Milvus Integration**: Efficient cosine search via Milvus Lite (FLAT) or Milvus server (IVF)
4. **Cosine Similarity**: Uses normalized embeddings for accurate similarity matching

## Evaluation

```bash
# Default (SciNCL + Milvus server at localhost:19530)
uv run -m eval.benchmark

# Run against Milvus Lite artifacts (v1 profile; full-document embeddings)
uv run -m eval.benchmark --v1

# Specific backend: scincl | qdrant_st | qdrant_tfidf | all
uv run -m eval.benchmark --backend all -k 10

# Add LLM judge (SciNCL only): Deepeval metrics
uv run -m eval.benchmark --llm
```

**Metrics:** Recall@k, Precision@k, nDCG@k, MAP, MRR@k
**Test Set:** 10 queries in `eval/queries_latest.csv`
**LLM Judge (optional, SciNCL backend):** Contextual Relevancy, Answer Relevancy, Faithfulness via deepeval. The `--llm` flag enables LLM-generated answers needed for answer-level scoring.

## Alternative Backend: Qdrant

Production-ready vector database with persistent storage. Two embedding approaches:
- **SentenceTransformer** (recommended): Neural embeddings
- **TF-IDF** (baseline): Fast, no GPU required

```bash
# Setup & run
python -m qdrant.setup
python -m qdrant.sentence_transformer  # or: qdrant.tfidf
python -m qdrant.cli

# Options: --backend tfidf | --query "your question"
# Also: qdrant.examples, qdrant.test --compare
```

**When to use:**
- **SciNCL + Milvus Lite (v1)**: Local/dev, fast setup, flat index
- **SciNCL + Milvus Server (v2 default)**: Requires Milvus server at `http://localhost:19530`
- **Qdrant+ST**: Production, persistent storage, scalable
- **Qdrant+TF-IDF**: Baselines, prototyping, CPU-only

## Troubleshooting

**Memory Issues:** Reduce batch size in `scincl/core.py` or use CPU
**Artifacts Not Found:** Run `uv run -m scincl.cli ingest` first
**Qdrant Issues:** Check Docker (`docker ps`), restart (`docker-compose restart`), or reprocess data (`uv run -m qdrant.sentence_transformer`)
