"""
SciNCL-based data ingestion and retrieval system.
"""

import ast
import json
import logging
import os
import time
from dataclasses import dataclass, asdict

import torch
import numpy as np
import pandas as pd
from pymilvus import MilvusClient
from pymilvus.model.reranker import BGERerankFunction
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
        logger.info(f"Loading SciNCL model on {self.device}")
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model: AutoModel = AutoModel.from_pretrained(model_name).to(
            self.device
        )
        self.base_model.eval()

        self.client: MilvusClient | None = None
        self.collection_name = "scincl_chunks"
        self.vector_field_name = "vector"
        self.index_name = "scincl_index"
        self.search_params: dict = {}

        self.max_length = 512
        self.overlap = 128
        self._doc_ids: list[str] | None = None
        self._chunk_texts: list[str] | None = None

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
        self._chunk_texts = []
        all_input_ids: list[torch.Tensor] = []
        all_attention_masks: list[torch.Tensor] = []

        sep = self.tokenizer.sep_token or "[SEP]"
        doc_items = list(documents.items())

        # Collect all chunks for every document
        for doc_id, doc in tqdm(doc_items, desc="Chunking documents"):
            full_text = f"{doc.title}{sep}{doc.abstract}"

            encodings = self.tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                stride=self.overlap,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )

            chunk_count = encodings.input_ids.size(0)
            all_input_ids.extend([chunk for chunk in encodings.input_ids])
            all_attention_masks.extend([mask for mask in encodings.attention_mask])
            self._doc_ids.extend([doc_id] * chunk_count)

            chunk_texts = self.tokenizer.batch_decode(
                encodings.input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            self._chunk_texts.extend(chunk_texts)

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

    def create_index(
        self,
        embeddings: np.ndarray,
        index_type: str = "flat",
        db_path: str | None = None,
    ):
        """
        Args:
            embeddings: Document embeddings array
            index_type: Index type to create (only 'flat' is supported)
        """
        if embeddings.size == 0:
            raise ValueError("No embeddings provided to index")

        db_path = db_path or "milvus.db"
        is_lite = self._is_lite_uri(db_path)
        if is_lite and index_type.lower() != "flat":
            raise ValueError("Milvus Lite backend currently supports only 'flat' index")

        if self._doc_ids is None or self._chunk_texts is None:
            raise ValueError("Document IDs and chunk texts must be generated first")

        if len(self._doc_ids) != len(embeddings) or len(self._chunk_texts) != len(
            embeddings
        ):
            raise ValueError("Mismatch between embeddings and stored chunk metadata")

        normalized_index_type = index_type.lower()
        db_dir = os.path.dirname(db_path)
        if is_lite:
            os.makedirs(db_dir, exist_ok=True)

        logger.info(f"Creating collection '{self.collection_name}' at {db_path}")
        self.client = MilvusClient(uri=db_path)

        if self.client.has_collection(self.collection_name):
            logger.info("Existing collection found; dropping before recreation")
            self.client.drop_collection(self.collection_name)

        index_params = self.client.prepare_index_params()
        index_type, index_kwargs, search_kwargs = self._resolve_index_params(
            index_type=normalized_index_type,
            total_vectors=len(embeddings),
            is_lite=is_lite,
        )
        index_params.add_index(
            field_name=self.vector_field_name,
            index_type=index_type,
            index_name=self.index_name,
            metric_type="COSINE",
            params=index_kwargs,
        )
        self.search_params = search_kwargs

        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=embeddings.shape[1],
            auto_id=True,
            index_params=index_params,
        )

        logger.info("Adding embeddings to collection...")
        records = [
            {
                self.vector_field_name: embedding.tolist(),
                "doc_id": doc_id,
                "chunk_text": chunk_text,
            }
            for doc_id, chunk_text, embedding in zip(
                self._doc_ids, self._chunk_texts, embeddings
            )
        ]
        self.client.insert(self.collection_name, records)
        self.client.flush(self.collection_name)

        logger.info(f"Collection ready with {len(embeddings)} vectors")

    def load_collection(self, db_path: str | None = None):
        """
        Attach to an existing Milvus database (Lite or server) and load the collection.
        """
        path = db_path or "milvus.db"
        if path is None:
            raise ValueError("Milvus database path is not set")

        is_lite = self._is_lite_uri(path)
        if is_lite and not os.path.exists(path):
            raise FileNotFoundError(f"Milvus database not found at {path}")

        self.client = MilvusClient(uri=path)
        if not self.client.has_collection(self.collection_name):
            raise ValueError(f"Collection '{self.collection_name}' missing in {path}")
        self.client.load_collection(self.collection_name)

        index_type = self._get_index_from_collection()
        _, _, search_kwargs = self._resolve_index_params(
            index_type,
            total_vectors=self.get_vector_count(),
            is_lite=is_lite,
        )
        self.search_params = search_kwargs

    def get_vector_count(self):
        """Return number of vectors stored in the collection."""
        if self._doc_ids is not None:
            return len(self._doc_ids)
        if self.client is None or self.collection_name is None:
            return 0
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return int(stats.get("row_count", 0))
        except Exception as exc:
            logger.warning(f"Could not fetch collection stats: {exc}")
            return 0

    @staticmethod
    def _is_lite_uri(uri: str) -> bool:
        """Heuristic to decide if the target is Milvus Lite (file-backed)."""
        return uri.endswith(".db") or uri.startswith("file:")

    def _resolve_index_params(self, index_type: str, total_vectors: int, is_lite: bool):
        """
        Map friendly index types to Milvus index params and default search params.
        """
        index_type = index_type.lower()
        if is_lite and index_type != "flat":
            raise ValueError("Milvus Lite only supports 'flat' index type")

        if index_type == "flat":
            return "FLAT", {}, {}

        if index_type.startswith("ivf"):
            sqrt_n = max(1, int(total_vectors**0.5))
            nlist = min(max(4 * sqrt_n, 1), max(total_vectors, 1))
            search_params = {
                "params": {"nprobe": min(32, nlist)},
            }
            return "IVF_FLAT", {"nlist": nlist}, search_params

        if index_type == "hnsw":
            search_params = {"params": {"ef": 64}}
            return "HNSW", {"M": 16, "efConstruction": 200}, search_params

        raise ValueError(f"Unsupported index type: {index_type}")

    def _get_index_from_collection(self):
        if self.client is None:
            raise ValueError("Client not initialized")

        index_info = self.client.describe_index(
            collection_name=self.collection_name, index_name=self.index_name
        )

        if not index_info:
            return "flat"

        return index_info["index_type"]


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
        self.device = self.ingestion.device
        self.reranker = BGERerankFunction(device=self.ingestion.device, use_fp16=True)

    def retrieve_similar_documents(self, query: str, k: int):
        """
        Retrieve similar documents and rerank chunk-level hits.
        With chunking, multiple chunks may map to the same document.

        Args:
            query: Query text
            k: Number of documents to retrieve
        """
        if self.ingestion.client is None:
            raise ValueError("Collection not loaded")

        if k <= 0:
            raise ValueError("k must be positive")

        n_vectors = self.ingestion.get_vector_count()
        if n_vectors == 0:
            return []

        # Encode the query as [CLS] embedding
        inputs = self.ingestion.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=self.ingestion.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.ingestion.base_model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Search for extra chunks to increase odds of k unique documents
        oversample_factor = n_vectors / max(len(self.documents), 1)
        target = int(np.ceil(k * max(oversample_factor, 1.0)))
        search_k = min(max(target, k), n_vectors)

        search_start = time.perf_counter()
        search_results = self.ingestion.client.search(
            collection_name=self.ingestion.collection_name,
            data=query_embedding.tolist(),
            anns_field=self.ingestion.vector_field_name,
            limit=search_k,
            search_params=self.ingestion.search_params,
            output_fields=["doc_id", "chunk_text"],
        )
        search_end = time.perf_counter()
        logger.info(f"Search time: {search_end - search_start:.2f} seconds")

        if not search_results:
            return []

        doc_chunk_hits = {}
        chunk_to_doc = {}
        for hit in search_results[0]:
            entity = hit.get("entity", {})
            doc_id, chunk_text = entity.get("doc_id"), entity.get("chunk_text")
            if not doc_id or not chunk_text:
                logger.warning(
                    "Missing doc_id or chunk_text in search result; skipping"
                )
                continue
            doc_chunk_hits.setdefault(doc_id, chunk_text)
            chunk_to_doc.setdefault(chunk_text, doc_id)

        if not doc_chunk_hits:
            return []

        rerank_start = time.perf_counter()
        rerank_results = self.reranker(query, list(doc_chunk_hits.values()))
        rerank_end = time.perf_counter()
        logger.info(f"Rerank time: {rerank_end - rerank_start:.2f} seconds")

        results = []
        seen = set()
        for result in rerank_results:
            if len(results) >= k:
                break
            doc_id = chunk_to_doc.get(result.text)
            if not doc_id or doc_id in seen:
                continue
            doc = self.documents.get(doc_id)
            if doc is None:
                logger.warning(f"Document ID {doc_id} not found in store; skipping")
                continue
            results.append(RetrievalResult(document=doc, sim_score=result.score))
            seen.add(doc_id)

        return results
