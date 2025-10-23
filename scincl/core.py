"""
SciNCL-based data ingestion and FAISS retrieval system.
Implements neighborhood contrastive learning for scientific document representations.
"""

import json
import os
import pickle
import logging

import numpy as np
import pandas as pd
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SciNCLIngestion:
    """
    Data ingestion system based on SciNCL methodology.
    Handles document processing and embedding generation.
    """

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
        self.model_name = model_name
        self.tokenizer = None
        self.base_model = None
        self.document_embeddings = {}
        self.faiss_index = None

    def load_model(self):
        """Load the SciNCL model and tokenizer."""
        logger.info(f"Loading SciNCL model on {self.device}")

        # Load SciNCL model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base_model = AutoModel.from_pretrained(self.model_name).to(self.device)

        logger.info("SciNCL model loaded successfully")

    def process_pubmed_data(self, csv_path: str) -> list[dict[str]]:
        """
        Process PubMed abstracts data.

        Args:
            csv_path: Path to the PubMed CSV file

        Returns:
            List of processed documents
        """
        logger.info(f"Processing PubMed data from {csv_path}")

        df = pd.read_csv(csv_path)
        documents = []

        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Processing PubMed abstracts"
        ):
            try:
                # Parse metadata field safely
                meta = row["meta"]
                if isinstance(meta, str):
                    meta = eval(meta)

                abstract_text = str(row["text"]).strip()
                title = abstract_text.partition(".")[0].strip() if abstract_text else ""

                documents.append(
                    {
                        "id": f"pubmed_{meta.get('pmid', 'unknown')}",
                        "title": title,
                        "abstract": abstract_text,
                        "metadata": meta,
                        "source": "pubmed",
                    }
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
            List of processed documents
        """
        logger.info(f"Processing Semantic Scholar data from {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        documents = []
        papers = data.get("data", [])

        for paper in tqdm(papers, desc="Processing Semantic Scholar papers"):
            try:
                document = {
                    "id": paper.get("paperId") or f"ss_{len(documents)}",
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract") or paper.get("tldr", ""),
                    "metadata": {
                        "year": paper.get("year"),
                        "authors": paper.get("authors", []),
                        "citationCount": paper.get("citationCount", 0),
                        "referenceCount": paper.get("referenceCount", 0),
                        "influentialCitationCount": paper.get(
                            "influentialCitationCount", 0
                        ),
                        "url": paper.get("url", ""),
                    },
                    "source": "semantic_scholar",
                }
                documents.append(document)
            except Exception as e:
                logger.warning(f"Error processing paper: {e}")

        logger.info(f"Processed {len(documents)} Semantic Scholar documents")
        return documents

    def generate_document_embeddings(self, documents: list[dict[str]]):
        """
        Generate document embeddings using SciNCL model.

        Args:
            documents: List of processed documents

        Returns:
            Document embeddings array
        """
        if self.base_model is None:
            self.load_model()

        logger.info("Generating document embeddings using SciNCL model")

        batch_size = 16
        embeddings = []

        def concatenate_title_abstract(doc):
            title = doc.get("title", "")
            abstract = doc.get("abstract", "")
            return f"{title}{self.tokenizer.sep_token}{abstract}"

        for i in tqdm(
            range(0, len(documents), batch_size), desc="Generating embeddings"
        ):
            batch_docs = documents[i : i + batch_size]
            inputs = self.tokenizer(
                [concatenate_title_abstract(doc) for doc in batch_docs],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.base_model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(cls_embeddings)

        embeddings_array = np.vstack(embeddings)
        self.document_embeddings = {
            doc["id"]: emb for doc, emb in zip(documents, embeddings_array)
        }

        logger.info(f"Generated embeddings for {len(embeddings_array)} documents")
        return embeddings_array

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

        # Save FAISS index
        if self.faiss_index is not None:
            faiss.write_index(
                self.faiss_index, os.path.join(output_dir, "faiss_index.bin")
            )

        # Save document embeddings
        if self.document_embeddings:
            with open(os.path.join(output_dir, "document_embeddings.pkl"), "wb") as f:
                pickle.dump(self.document_embeddings, f)

        logger.info(f"Artifacts saved to {output_dir}")

    def load_artifacts(self, output_dir: str):
        """Load previously saved artifacts."""
        # Load FAISS index
        index_path = os.path.join(output_dir, "faiss_index.bin")
        if os.path.exists(index_path):
            self.faiss_index = faiss.read_index(index_path)

        # Load document embeddings
        embeddings_path = os.path.join(output_dir, "document_embeddings.pkl")
        if os.path.exists(embeddings_path):
            with open(embeddings_path, "rb") as f:
                self.document_embeddings = pickle.load(f)

        logger.info(f"Artifacts loaded from {output_dir}")


class SciNCLRetrieval:
    """
    Retrieval system using FAISS and SciNCL methodology.
    Implements neighborhood contrastive learning for better retrieval.
    """

    def __init__(self, ingestion_system: SciNCLIngestion):
        """
        Initialize retrieval system.

        Args:
            ingestion_system: Initialized SciNCLIngestion system
        """
        self.ingestion = ingestion_system
        self.documents = []
        self.document_ids = []

    def load_documents(self, documents: list[dict[str]]):
        """Load documents for retrieval."""
        self.documents = documents
        self.document_ids = [doc["id"] for doc in documents]

    def retrieve_similar_documents(self, query: str, k: int = 10):
        """
        Retrieve similar documents using FAISS.

        Args:
            query: Query text
            k: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores
        """
        if self.ingestion.faiss_index is None:
            raise ValueError("FAISS index not initialized")

        if self.ingestion.base_model is None:
            self.ingestion.load_model()

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

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.document_ids):
                doc = next(
                    (d for d in self.documents if d["id"] == self.document_ids[idx]),
                    None,
                )
                if doc:
                    results.append(
                        {
                            "document": doc,
                            "score": float(score),
                            "similarity": float(score),
                        }
                    )

        return results
