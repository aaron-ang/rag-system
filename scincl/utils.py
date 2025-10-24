"""
Shared utilities for SciNCL system.
Common functions for data availability checks and artifact management.
"""

import os
import json
from typing import Callable

from scincl.core import SciNCLIngestion, SciNCLRetrieval, Document


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
        "doc_ids.pkl",
    ]
    missing_files = [
        f for f in required_files if not os.path.exists(os.path.join(artifacts_dir, f))
    ]

    if missing_files:
        raise ValueError(f"Missing artifact files: {missing_files}")

    print("Loading existing artifacts...")

    ingestion = SciNCLIngestion()
    ingestion.load_artifacts(artifacts_dir)

    with open(os.path.join(artifacts_dir, "documents.json"), "r") as f:
        documents_list = json.load(f)

    documents = {}
    for doc in documents_list:
        doc_id = doc.pop("id")
        documents[doc_id] = Document.from_dict(doc)

    retrieval = SciNCLRetrieval(ingestion, documents)

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
        index_type: FAISS index type ("flat", "ivf", "hnsw", etc.)
        artifacts_dir: Directory to save artifacts

    Returns:
        Tuple of (retrieval, documents)
    """
    print("Creating new artifacts...")

    if not _check_data_availability():
        raise ValueError("Required data files are missing")

    ingestion = SciNCLIngestion(model_name=model_name)

    def _try_process_dataset(path: str, name: str, process_func: Callable):
        if os.path.exists(path):
            print(f"Processing {name}...")
            return process_func(path)
        print(f"{name} not found, skipping...")
        return {}

    pubmed_docs = _try_process_dataset(
        "data/pubmed_abstracts.csv", "PubMed abstracts", ingestion.process_pubmed_data
    )
    ss_docs = _try_process_dataset(
        "data/semanticscholar.json",
        "Semantic Scholar papers",
        ingestion.process_semantic_scholar_data,
    )

    # Combine and deduplicate by title
    all_documents = {**pubmed_docs, **ss_docs}
    print(f"Total documents (before deduplication): {len(all_documents)}")

    seen_titles = set()
    deduped_documents: dict[str, Document] = {}
    for doc_id, doc in all_documents.items():
        if doc.title not in seen_titles:
            deduped_documents[doc_id] = doc
            seen_titles.add(doc.title)

    all_documents = deduped_documents
    print(f"Total documents (after deduplication): {len(all_documents)}")

    if not all_documents:
        raise ValueError("No documents found. Please run download.py first.")

    embeddings = ingestion.generate_document_embeddings(all_documents)
    ingestion.create_faiss_index(embeddings, index_type=index_type)

    os.makedirs(artifacts_dir, exist_ok=True)
    ingestion.save_artifacts(artifacts_dir)

    # Save documents as a JSON list
    documents_list = [
        {"id": doc_id, **doc.to_dict()} for doc_id, doc in all_documents.items()
    ]
    with open(os.path.join(artifacts_dir, "documents.json"), "w") as f:
        json.dump(documents_list, f, indent=2)

    retrieval = SciNCLRetrieval(ingestion, all_documents)
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
