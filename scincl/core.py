"""
SciNCL-based data ingestion and FAISS retrieval system.
"""

import ast
import json
import logging
import os
import pickle
from dataclasses import dataclass, asdict

import faiss
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a research literature document with metadata."""

    title: str
    abstract: str
    source: str
    metadata: dict[str]

    def __post_init__(self):
        """Validate document data after initialization."""
        if self.title is None:
            raise ValueError("Document title cannot be None")
        if self.abstract is None:
            self.abstract = ""

        if not self.title.strip():
            raise ValueError("Document title cannot be empty")
        if not self.abstract.strip():
            logger.warning(f"Document '{self.title}' has empty abstract")

    def to_dict(self):
        """Convert document to dictionary format."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str]):
        """Create document from dictionary."""
        return cls(**data)


class SciNCLIngestion:
    def __init__(self, model_name: str = "malteos/scincl"):
        """
        Initialize the SciNCL ingestion system.

        Args:
            model_name: Name of the SciNCL model
        """
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.device = device
        self.faiss_index: faiss.Index | None = None
        self._doc_ids: list[str] | None = None

        logger.info(f"Loading SciNCL model on {self.device}")
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model: AutoModel = AutoModel.from_pretrained(model_name).to(
            self.device
        )
        logger.info("SciNCL model loaded successfully")

    def process_pubmed_data(self, csv_path: str):
        """
        Process PubMed abstracts data.

        Args:
            csv_path: Path to the PubMed CSV file

        Returns:
            Dictionary of processed documents (doc_id -> Document)
        """
        df = pd.read_csv(csv_path)
        documents: dict[str, Document] = {}

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Processing PubMed data from {csv_path}",
            disable=True,
        ):
            try:
                meta = row["meta"]
                if isinstance(meta, str):
                    meta = ast.literal_eval(meta)

                text = str(row["text"]).strip()
                title, _, abstract = text.partition(".")

                doc_id = f"pubmed_{meta.get('pmid', 'unknown')}"
                documents[doc_id] = Document(
                    title=title.strip(),
                    abstract=abstract.strip(),
                    metadata=meta,
                    source="pubmed",
                )

            except Exception as e:
                logger.warning(f"Error processing row: {e}")

        logger.info(f"Processed {len(documents)} PubMed documents")
        return documents

    def process_semantic_scholar_data(self, json_path: str):
        """
        Process Semantic Scholar data.

        Args:
            json_path: Path to the Semantic Scholar JSON file

        Returns:
            Dictionary of processed documents (doc_id -> Document)
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        documents: dict[str, Document] = {}
        papers = data.get("data", [])

        for paper in tqdm(
            papers, desc=f"Processing Semantic Scholar data from {json_path}"
        ):
            try:
                doc_id = paper.get("paperId") or f"ss_{len(documents)}"
                title = paper.get("title", "")
                abstract = paper.get("abstract", "")
                if not abstract:
                    tldr = paper.get("tldr")
                    if isinstance(tldr, dict):
                        abstract = tldr.get("text", "")

                metadata = {
                    "year": paper.get("year"),
                    "authors": paper.get("authors", []),
                    "citationCount": paper.get("citationCount", 0),
                    "referenceCount": paper.get("referenceCount", 0),
                    "influentialCitationCount": paper.get(
                        "influentialCitationCount", 0
                    ),
                    "url": paper.get("url", ""),
                }

                documents[doc_id] = Document(
                    title=title,
                    abstract=abstract,
                    metadata=metadata,
                    source="semantic_scholar",
                )
            except Exception as e:
                logger.warning(f"Error processing paper: {e}")

        logger.info(f"Processed {len(documents)} Semantic Scholar documents")
        return documents

    def generate_document_embeddings(self, documents: dict[str, Document]):
        """
        Generate document embeddings using the SciNCL model.

        Args:
            documents: dict of {doc_id: Document}

        Returns:
            np.ndarray: Embeddings array for all documents (order follows input dict)
        """
        logger.info("Generating document embeddings using SciNCL model")

        batch_size = 16
        doc_items = list(documents.items())
        embeddings = []

        def _prepare_text(doc: Document):
            sep = self.tokenizer.sep_token or "[SEP]"
            return f"{doc.title}{sep}{doc.abstract}"

        for i in tqdm(
            range(0, len(doc_items), batch_size), desc="Generating embeddings"
        ):
            batch = doc_items[i : i + batch_size]
            batch_texts = [_prepare_text(doc) for _, doc in batch]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.base_model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_emb)

        embeddings_array = np.vstack(embeddings)
        self._doc_ids = [doc_id for doc_id, _ in doc_items]

        logger.info(f"Generated embeddings for {len(embeddings_array)} documents")
        return embeddings_array

    # TODO: verify if this is the best way to create the index
    def create_faiss_index(self, embeddings: np.ndarray, index_type: str = "flat"):
        """
        Create FAISS index for efficient similarity search.

        Args:
            embeddings: Document embeddings array
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")

        Returns:
            FAISS index
        """
        logger.info(f"Creating FAISS '{index_type}' index")

        dim = embeddings.shape[1]

        if index_type == "flat":
            index = faiss.IndexFlatIP(dim)
        elif index_type == "ivf":
            nlist = min(100, max(1, len(embeddings) // 10))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Normalize embeddings (L2) for cosine similarity
        faiss.normalize_L2(embeddings)

        if index_type == "ivf":
            index.train(embeddings)

        index.add(embeddings)
        self.faiss_index = index

        logger.info(f"FAISS index created with {index.ntotal} vectors")
        return index

    def save_artifacts(self, output_dir: str):
        """Save all artifacts for later use."""
        os.makedirs(output_dir, exist_ok=True)

        if self.faiss_index is not None:
            faiss.write_index(
                self.faiss_index, os.path.join(output_dir, "faiss_index.bin")
            )

        if self._doc_ids is not None:
            with open(os.path.join(output_dir, "doc_ids.pkl"), "wb") as f:
                pickle.dump(self._doc_ids, f)

        logger.info(f"Artifacts saved to {output_dir}")

    def load_artifacts(self, output_dir: str):
        """Load previously saved artifacts."""
        index_path = os.path.join(output_dir, "faiss_index.bin")
        if os.path.exists(index_path):
            self.faiss_index = faiss.read_index(index_path)

        doc_ids_path = os.path.join(output_dir, "doc_ids.pkl")
        if os.path.exists(doc_ids_path):
            with open(doc_ids_path, "rb") as f:
                self._doc_ids = pickle.load(f)

        logger.info(f"Artifacts loaded from {output_dir}")


@dataclass
class RetrievalResult:
    """Represents a document retrieval result with similarity score."""

    document: Document
    score: float
    similarity: float

    def __post_init__(self):
        """Validate retrieval result."""
        if not 0.0 <= self.score <= 1.0:
            logger.warning(f"Score {self.score} outside expected range [0.0, 1.0]")

    def to_dict(self):
        """Convert result to dictionary format."""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "similarity": self.similarity,
        }


class SciNCLRetrieval:
    def __init__(
        self, ingestion_system: SciNCLIngestion, documents: dict[str, Document]
    ):
        """
        Initialize retrieval system.

        Args:
            ingestion_system: Initialized SciNCLIngestion system
            documents: Dictionary of documents (doc_id -> Document)
        """
        self.ingestion = ingestion_system
        self.documents = documents

    def retrieve_similar_documents(self, query: str, k: int):
        """
        Retrieve similar documents using FAISS.

        Args:
            query: Query text
            k: Number of documents to retrieve

        Returns:
            List of RetrievalResult objects
        """
        if self.ingestion.faiss_index is None:
            raise ValueError("FAISS index not initialized")

        # Encode the query as [CLS] embedding (title-only)
        inputs = self.ingestion.tokenizer(
            [query], padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.ingestion.device)

        with torch.no_grad():
            outputs = self.ingestion.base_model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Normalize the embedding for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search FAISS index for top-k results
        scores, indices = self.ingestion.faiss_index.search(query_embedding, k)

        # Convert FAISS indices to document results
        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            doc_id = self.ingestion._doc_ids[idx]
            doc = self.documents.get(doc_id)
            results.append(
                RetrievalResult(
                    document=doc,
                    score=float(score),
                    similarity=float(score),
                )
            )

        return results
