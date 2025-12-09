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

    id: str
    title: str
    abstract: str
    source: str
    metadata: dict[str]

    def __post_init__(self):
        """Validate document data after initialization."""
        if self.id is None:
            raise ValueError("Document ID cannot be None")
        if self.title is None:
            raise ValueError("Document title cannot be None")
        if self.abstract is None:
            self.abstract = ""

        if not self.id.strip():
            raise ValueError("Document ID cannot be empty")
        if not self.title.strip():
            raise ValueError("Document title cannot be empty")
        if not self.abstract.strip():
            logger.warning(f"Document '{self.id}' has empty abstract")

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

        self.max_length = 512
        self.stride = 384  # 512 - 128 overlap

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

                doc_id = str(meta["pmid"])
                documents[doc_id] = Document(
                    id=doc_id,
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
                doc_id = paper["paperId"]
                title = paper.get("title", "")
                abstract = paper.get("abstract", "")
                if not abstract:
                    tldr = paper["tldr"]
                    if isinstance(tldr, dict):
                        abstract = tldr.get("text", "")

                metadata = {
                    "year": paper["year"],
                    "authors": paper.get("authors", []),
                    "citationCount": paper.get("citationCount", 0),
                    "referenceCount": paper.get("referenceCount", 0),
                    "influentialCitationCount": paper.get(
                        "influentialCitationCount", 0
                    ),
                    "url": paper.get("url", ""),
                }

                documents[doc_id] = Document(
                    id=doc_id,
                    title=title,
                    abstract=abstract,
                    metadata=metadata,
                    source="semantic_scholar",
                )
            except Exception as e:
                logger.warning(f"Error processing paper: {e}")

        logger.info(f"Processed {len(documents)} Semantic Scholar documents")
        return documents

    def generate_document_embeddings(
        self, documents: dict[str, Document], batch_size: int = 16
    ):
        """
        Generate document embeddings using Sliding Window chunking.

        Args:
            documents: dict of {doc_id: Document}
            batch_size: Number of chunks to embed at once

        Returns:
            np.ndarray: Embeddings array for all document chunks (order follows input dict)
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        logger.info("Generating embeddings with Sliding Window strategy...")

        self._doc_ids = []
        all_input_ids: list[torch.Tensor] = []
        all_attention_masks: list[torch.Tensor] = []

        sep = self.tokenizer.sep_token or "[SEP]"
        doc_items = list(documents.items())
        overlap = max(self.max_length - self.stride, 0)

        # Collect all chunks for every document
        for doc_id, doc in tqdm(doc_items, desc="Chunking documents"):
            full_text = f"{doc.title}{sep}{doc.abstract}"

            encodings = self.tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                stride=overlap,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )

            chunk_count = encodings.input_ids.size(0)
            all_input_ids.extend([chunk for chunk in encodings.input_ids])
            all_attention_masks.extend([mask for mask in encodings.attention_mask])
            self._doc_ids.extend([doc_id] * chunk_count)

        if not all_input_ids:
            logger.warning("No chunks found to embed; returning empty embeddings array")
            return np.empty((0, self.base_model.config.hidden_size))

        # Embed chunks in batches
        embeddings = []
        for i in tqdm(
            range(0, len(all_input_ids), batch_size), desc="Embedding chunks"
        ):
            batch_input_ids = torch.stack(all_input_ids[i : i + batch_size]).to(
                self.device
            )
            batch_attention = torch.stack(all_attention_masks[i : i + batch_size]).to(
                self.device
            )
            batch_embeddings = self._generate_batch_embeddings(
                batch_input_ids, batch_attention
            )
            embeddings.append(batch_embeddings)

        embeddings_array = np.vstack(embeddings)
        logger.info(
            f"Generated {len(embeddings_array)} embeddings for {len(documents)} documents"
        )
        return embeddings_array

    def _generate_batch_embeddings(
        self, input_ids_batch: torch.Tensor, attention_mask_batch: torch.Tensor
    ):
        """Helper to run the model on padded batches of chunk tensors."""
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch
            )
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

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
        n_vectors = len(embeddings)

        if index_type == "flat":
            index = faiss.IndexFlatIP(dim)
        elif index_type == "ivf":
            # Calculate nlist based on FAISS guidance: K = 4*sqrt(N) to 16*sqrt(N)
            sqrt_n = int(n_vectors**0.5)
            nlist = 4 * sqrt_n
            nlist = min(nlist, n_vectors)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.nprobe = max(1, nlist // 4)
            logger.info(f"IVF parameters: nlist={nlist}, nprobe={index.nprobe}")
        elif index_type == "hnsw":
            # HNSW parameter: 4 <= M <= 64
            hnsw_m = 16
            index = faiss.IndexHNSWFlat(dim, hnsw_m)
            # Set efSearch for speed-accuracy tradeoff (default: 16)
            ef_search = 16
            index.hnsw.efSearch = ef_search
            logger.info(f"HNSW parameters: M={hnsw_m}, efSearch={ef_search}")
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Normalize embeddings (L2) for cosine similarity
        faiss.normalize_L2(embeddings)

        if index_type == "ivf":
            logger.info("Training IVF index...")
            index.train(embeddings)

        logger.info("Adding embeddings to index...")
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
    sim_score: float

    def __post_init__(self):
        """Validate retrieval result."""
        if not 0.0 <= self.sim_score <= 1.0:
            logger.warning(f"Score {self.sim_score} outside expected range [0.0, 1.0]")

    def to_dict(self):
        """Convert result to dictionary format."""
        return {
            "document": self.document.to_dict(),
            "score": self.sim_score,
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
        With chunking, multiple chunks may map to the same document.
        We group by document and take the best score.

        Args:
            query: Query text
            k: Number of documents to retrieve

        Returns:
            List of RetrievalResult objects
        """
        if self.ingestion.faiss_index is None:
            raise ValueError("FAISS index not initialized")

        # Encode the query as [CLS] embedding
        inputs = self.ingestion.tokenizer(
            [query], padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.ingestion.device)

        with torch.no_grad():
            outputs = self.ingestion.base_model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Normalize the embedding for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search for more chunks to account for multiple chunks per document
        # Search for k*3 chunks to ensure we get k unique documents
        search_k = min(k * 3, self.ingestion.faiss_index.ntotal)
        scores, indices = self.ingestion.faiss_index.search(query_embedding, search_k)

        # Group chunks by document ID and take the best score per document
        doc_scores: dict[str, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            doc_id = self.ingestion._doc_ids[idx]
            # Keep the maximum score for each document
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = float(score)

        # Sort by score and take top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Convert to RetrievalResult objects
        results: list[RetrievalResult] = []
        for doc_id, score in sorted_docs:
            doc = self.documents[doc_id]
            results.append(RetrievalResult(document=doc, sim_score=score))

        return results
