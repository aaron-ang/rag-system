"""
Qdrant-based Medical RAG System Package.

This package provides a Retrieval-Augmented Generation (RAG) system for medical
research papers using Qdrant vector database. It supports two embedding approaches:

1. SentenceTransformer: Neural embeddings for semantic understanding (recommended)
2. TF-IDF: Traditional text vectorization for baseline comparison

Modules:
    sentence_transformer: SentenceTransformer-based RAG implementation
    tfidf: TF-IDF-based RAG implementation (baseline)
    cli: Interactive command-line interface
    setup: Docker and environment setup utilities
    utils: Database inspection and debugging tools
    examples: Example usage and query patterns
    test: Test suite for both backends

Quick Start:
    # Setup Qdrant
    python -m qdrant.setup

    # Process data with SentenceTransformer
    python -m qdrant.sentence_transformer

    # Run interactive CLI
    python -m qdrant.cli

    # Run tests
    python -m qdrant.test

Example Usage:
    from qdrant.sentence_transformer import SentenceTransformerRAG

    # Initialize RAG system
    rag = SentenceTransformerRAG()

    # Query the system
    result = rag.query("What are the symptoms of hypoxaemia?", limit=5)

    # Display results
    for doc in result['documents']:
        print(f"{doc['title']} (Score: {doc['score']:.3f})")
"""

from qdrant.sentence_transformer import SentenceTransformerRAG
from qdrant.tfidf import TfidfRAG

__version__ = "1.0.0"
__all__ = [
    "SentenceTransformerRAG",
    "TfidfRAG",
]
