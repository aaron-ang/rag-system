# RAG System

A Retrieval-Augmented Generation system for medical research papers using [SciNCL methodology](https://github.com/malteos/scincl) for scientific document representations and similarity search.

## Features

- **SciNCL Integration**: Uses the official SciNCL model for document embeddings
- **FAISS Indexing**: Fast similarity search using Facebook's FAISS library
- **Multiple Data Sources**: PubMed abstracts and Semantic Scholar papers
- **Smart Caching**: Automatic artifact caching with force re-ingestion option
- **Document Deduplication**: Automatic removal of duplicate documents by title
- **CLI & Interactive**: Both command-line and interactive interfaces available

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
```

### CLI Interface

**Examples:**
```bash
# Ingest all data using SciNCL
uv run -m scincl.cli ingest

# Force re-ingestion (delete and rebuild artifacts)
uv run -m scincl.cli ingest --force

# Query the system
uv run -m scincl.cli query "machine learning in medical diagnosis" --k 5

# Query with more results
uv run -m scincl.cli query "tuberculosis treatment" --k 10
```

### Programmatic Usage

```python
from scincl import load_or_create_artifacts

# Load existing artifacts or create new ones
retrieval, documents = load_or_create_artifacts()

# Query the system
results = retrieval.retrieve_similar_documents("machine learning in medical diagnosis", k=5)

# Process results
for result in results:
    doc = result.document
    print(f"{doc.title} (Score: {result.score:.3f})")
    print(f"Source: {doc.source}")
    print(f"Abstract: {doc.abstract[:200]}...")
    print()
```

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
- `data/scincl_artifacts/` - FAISS indices, embeddings, and documents

## Advanced Features

### Artifact Caching
- Ingestion automatically skips if artifacts already exist
- Saves time on repeated runs
- Use `--force` flag to re-ingest:
  ```bash
  uv run -m scincl.cli ingest --force
  ```

### Document Deduplication
- Documents are automatically deduplicated by title during ingestion
- Duplicate information is skipped
- Check console output for deduplication statistics

### CLI Options
- **Ingest command**: `python -m scincl.cli ingest [--force] [--model MODEL] [--index-type TYPE] [--output-dir DIR]`
- **Query command**: `python -m scincl.cli query QUERY [--k N] [--artifacts-dir DIR]`
- **Help**: `python -m scincl.cli --help` or `python -m scincl.cli ingest --help`

## SciNCL Methodology

This implementation uses the [SciNCL model](https://huggingface.co/malteos/scincl) which provides:

1. **Scientific Document Embeddings**: Specialized embeddings for scientific papers
2. **Title-Abstract Concatenation**: Combines title and abstract with [SEP] token
3. **FAISS Integration**: Efficient similarity search with multiple index types
4. **Cosine Similarity**: Uses normalized embeddings for accurate similarity matching

## Evaluation

```bash
# Run benchmark evaluation
uv run -m eval.benchmark
```

### Metrics
- **Recall@k**: Percentage of queries with ≥1 relevant document in top-k
- **Precision@k**: Average precision across all queries  
- **nDCG@k**: Ranking quality with position weighting
- **MAP**: Mean average precision across all queries

### Test Queries
10 diverse queries covering pediatric medicine, infectious diseases, AI in healthcare, precision medicine, oncology, and more. Stored in `eval/retrieval_queries.csv`.

### Custom Evaluation
1. Edit `eval/retrieval_queries.csv` with new queries and target papers
2. Run `uv run -m eval.benchmark`
3. Interpret results: higher metrics = better performance

## Troubleshooting

### Common Issues

**Memory Issues During Embedding Generation:**
- Reduce batch size in `scincl/core.py` (currently set to 16)
- Use CPU instead of GPU if running out of memory

**Artifacts Not Found:**
- Run `python -m scincl.cli ingest` first to create artifacts
- Check that `data/` directory contains the required data files

**Import Errors:**
- Ensure you're using `uv run -m scincl.cli` (not direct Python execution)
- Verify all dependencies are installed with `uv sync`
