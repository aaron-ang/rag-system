"""
Shared utilities for SciNCL system.
Common functions for data availability checks and artifact management.
"""

import os
import json

from scincl.core import SciNCLIngestion, SciNCLRetrieval


def load_artifacts(artifacts_dir="data/scincl_artifacts"):
    """
    Load existing artifacts from disk.

    Args:
        artifacts_dir: Directory containing artifacts

    Returns:
        Tuple of (retrieval, documents)

    Raises:
        FileNotFoundError: If artifacts directory doesn't exist
        ValueError: If required artifact files are missing
    """
    if not os.path.exists(artifacts_dir):
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    required_files = [
        "faiss_index.bin",
        "documents.json",
        "embeddings.pkl",
    ]
    missing_files = [
        f for f in required_files if not os.path.exists(os.path.join(artifacts_dir, f))
    ]

    if missing_files:
        raise ValueError(f"Missing artifact files: {missing_files}")

    print("Loading existing artifacts...")

    # Load ingestion system
    ingestion = SciNCLIngestion()
    ingestion.load_artifacts(artifacts_dir)

    # Load documents
    with open(os.path.join(artifacts_dir, "documents.json"), "r") as f:
        documents = json.load(f)

    # Initialize retrieval system
    retrieval = SciNCLRetrieval(ingestion)
    retrieval.load_documents(documents)

    print(f"Loaded {len(documents)} documents from artifacts")
    return retrieval, documents


def create_artifacts(
    model_name="malteos/scincl",
    index_type="flat",
    artifacts_dir="data/scincl_artifacts",
):
    """
    Create new artifacts from raw data.

    Args:
        model_name: SciNCL model name to use
        index_type: FAISS index type
        artifacts_dir: Directory to save artifacts

    Returns:
        Tuple of (retrieval, documents)
    """
    print("Creating new artifacts...")

    # Check data availability
    if not _check_data_availability():
        raise ValueError("Required data files are missing")

    # Initialize ingestion system
    ingestion = SciNCLIngestion(model_name=model_name)

    # Process PubMed data
    pubmed_docs = []
    if os.path.exists("data/pubmed_abstracts.csv"):
        print("Processing PubMed abstracts...")
        pubmed_docs = ingestion.process_pubmed_data("data/pubmed_abstracts.csv")
    else:
        print("PubMed data not found, skipping...")

    # Process Semantic Scholar data
    ss_docs = []
    if os.path.exists("data/semanticscholar.json"):
        print("Processing Semantic Scholar papers...")
        ss_docs = ingestion.process_semantic_scholar_data("data/semanticscholar.json")
    else:
        print("Semantic Scholar data not found, skipping...")

    # Combine and deduplicate documents
    all_documents = pubmed_docs + ss_docs
    print(f"Total documents (before deduplication): {len(all_documents)}")

    # Deduplicate by title
    seen_titles = set()
    unique_documents = []
    for doc in all_documents:
        title = doc.get("title")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        unique_documents.append(doc)

    all_documents = unique_documents
    print(f"Total documents (after deduplication): {len(all_documents)}")

    if len(all_documents) == 0:
        raise ValueError("No documents found. Please run download.py first.")


    # Generate embeddings
    embeddings = ingestion.generate_document_embeddings(all_documents)

    # Create FAISS index
    ingestion.create_faiss_index(embeddings, index_type=index_type)

    # Save artifacts
    os.makedirs(artifacts_dir, exist_ok=True)
    ingestion.save_artifacts(artifacts_dir)

    # Save documents
    with open(os.path.join(artifacts_dir, "documents.json"), "w") as f:
        json.dump(all_documents, f, indent=2)

    # Initialize retrieval system
    retrieval = SciNCLRetrieval(ingestion)
    retrieval.load_documents(all_documents)

    print(f"Created artifacts for {len(all_documents)} documents")
    return retrieval, all_documents


def _check_data_availability():
    """Internal function to check if all required data files are present."""
    required_files = ["data/pubmed_abstracts.csv", "data/semanticscholar.json"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("Missing data files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run 'uv run download.py' first to download the data.")
        return False

    return True
