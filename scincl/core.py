"""
SciNCL-based data ingestion and retrieval system.
"""

import ast
import json
import logging
import os
import re
import time
import hashlib
from dataclasses import dataclass, asdict

import coloredlogs
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient
from pymilvus.model.reranker import BGERerankFunction
from pymilvus.milvus_client.index import IndexParams
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from verboselogs import VerboseLogger

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
LOG_LEVEL = logging.INFO
logger = VerboseLogger(__name__)
coloredlogs.install(level=LOG_LEVEL, logger=logger)


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


class LLMAssistant:
    """LLM utility for query rewrite and answer generation."""

    def __init__(
        self,
        model_id: str = "gpt-5-nano",
        cache_path: str = "data/llm_cache.json",
    ):
        self.model_id = model_id
        self.client = OpenAI()
        self.cache_path = cache_path
        self._cache = self._load_cache()

    def rewrite_query(self, query: str):
        """Rewrite the user query into a concise search query."""
        cache_key = f"rewrite::{query.strip()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        system = (
            "Rewrite the following user query into a concise search phrase for scientific literature. "
            "Example rewritten query: risk factors for hypoxaemia in pediatric pneumonia and its prevalence. "
            "Return only the rewritten query text."
        )
        prompt = f"User query: {query}\n\nRewritten query: "

        try:
            result = self._chat_completion(system, prompt)
            self._cache[cache_key] = result
            self._save_cache()
            return result
        except Exception as exc:
            logger.warning(
                f"LLM query rewrite failed; falling back to original query: {exc}"
            )
            return query

    def generate_answer(self, query: str, contexts: list[str]):
        """Generate an answer using retrieved context."""
        key_payload = {
            "q": query.strip(),
            "contexts": contexts,
        }
        digest = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True).encode()
        ).hexdigest()
        cache_key = f"answer::{digest}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        context_block = (
            "\n\n".join([f"- {c}" for c in contexts if c and c.strip()])
            or "No additional context provided."
        )

        system = (
            "You are a concise assistant for scientific literature. "
            "Given a user query and retrieved context, "
            "produce a concise answer in a single paragraph that summarizes the relevant findings. "
            "If the context is insufficient, say so briefly."
        )
        prompt = f"Query: {query}\nContext: {context_block}\n\nAnswer: "

        try:
            result = self._chat_completion(system, prompt)
            self._cache[cache_key] = result
            self._save_cache()
            return result
        except Exception as exc:
            logger.warning(f"LLM answer generation failed: {exc}")
            return ""

    def _chat_completion(self, system: str, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_id,
            instructions=system,
            input=prompt,
        )
        return response.output_text.strip()

    def _load_cache(self) -> dict:
        try:
            if self.cache_path and os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    return json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to load LLM cache: {exc}")
        return {}

    def _save_cache(self):
        if not self.cache_path:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as exc:
            logger.warning(f"Failed to save LLM cache: {exc}")


class SciNCLIngestion:
    def __init__(self, model_name: str = "malteos/scincl", use_v1: bool = False):
        """
        Initialize the SciNCL ingestion system.

        Args:
            model_name: Name of the SciNCL model
            use_v1: Whether to use v1 profile (Lite/FLAT, no rerank)
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

        self.db_path: str | None = None
        self.client: MilvusClient | None = None
        self.collection_name = "scincl_chunks"
        self.vector_field_name = "vector"
        self.index_name = "scincl_index"

        self.max_window_len = 256
        self.overlap = 64
        self._doc_ids: list[str] | None = None
        self._chunk_texts: list[str] | None = None
        self.use_v1 = use_v1

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
        self,
        documents: dict[str, Document],
        batch_size: int = 16,
    ):
        """
        Generate document embeddings.

        Args:
            documents: dict of {doc_id: Document}
            batch_size: Number of chunks to embed at once

        Returns:
            np.ndarray: Embeddings array for all document chunks (order follows input dict)
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        use_sliding_window = not self.use_v1
        logger.info(
            "Generating embeddings with Sliding Window strategy..."
            if use_sliding_window
            else "Generating embeddings without chunking (full documents)..."
        )

        self._doc_ids = []
        self._chunk_texts = []
        all_input_ids: list[torch.Tensor] = []
        all_attention_masks: list[torch.Tensor] = []

        sep = self.tokenizer.sep_token or "[SEP]"
        doc_items = list(documents.items())

        for doc_id, doc in tqdm(doc_items, desc="Preparing document inputs"):
            if use_sliding_window:
                abstract_chunks = self._sentence_chunks(doc.abstract)
                prefix = f"{doc.title}{sep}"
                for abstract_chunk in abstract_chunks:
                    full_text = f"{prefix}{abstract_chunk}"
                    encodings = self.tokenizer(
                        full_text,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_window_len,
                        return_tensors="pt",
                    )
                    all_input_ids.append(encodings.input_ids.squeeze(0))
                    all_attention_masks.append(encodings.attention_mask.squeeze(0))
                    self._doc_ids.append(doc_id)
                    self._chunk_texts.append(full_text)
            else:
                full_text = f"{doc.title}{sep}{doc.abstract}"
                encodings = self.tokenizer(
                    full_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_window_len * 2,
                    return_tensors="pt",
                )
                all_input_ids.append(encodings.input_ids.squeeze(0))
                all_attention_masks.append(encodings.attention_mask.squeeze(0))
                self._doc_ids.append(doc_id)
                self._chunk_texts.append(full_text)

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
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return self._normalize_embeddings(embeddings)

    def _normalize_embeddings(self, embeddings: np.ndarray):
        """L2-normalize embedding matrix; safeguards zero vectors."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _sentence_chunks(self, text: str):
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return [text]

        chunks = []
        current = []

        def token_len(parts: list[str]):
            return len(self.tokenizer.tokenize(" ".join(parts)))

        for sentence in sentences:
            candidate = current + [sentence]
            if token_len(candidate) <= self.max_window_len:
                current = candidate
                continue

            if current:
                chunks.append(" ".join(current))

            # keep a short tail for context overlap
            tail = []
            for s in reversed(current):
                if token_len([s] + tail) <= self.overlap:
                    tail.insert(0, s)
                else:
                    break
            current = tail + [sentence]

        if current:
            chunks.append(" ".join(current))

        return chunks

    def create_index(
        self,
        embeddings: np.ndarray,
        db_path: str | None = None,
    ):
        """
        Args:
            embeddings: Document embeddings array
            index_type: Index type to create ('flat' for Lite; 'flat', 'ivf', or 'hnsw' for server)
        """
        if embeddings.size == 0:
            raise ValueError("No embeddings provided to index")

        index_type = "flat" if self.use_v1 else "ivf"

        is_lite = self.is_lite_uri(db_path)
        if is_lite and index_type.lower() != "flat":
            raise ValueError("Milvus Lite backend currently supports only 'flat' index")

        if self._doc_ids is None or self._chunk_texts is None:
            raise ValueError("Document IDs and chunk texts must be generated first")

        if len(self._doc_ids) != len(embeddings) or len(self._chunk_texts) != len(
            embeddings
        ):
            raise ValueError("Mismatch between embeddings and stored chunk metadata")

        if is_lite:
            db_dir = os.path.dirname(db_path)
            os.makedirs(db_dir, exist_ok=True)

        logger.info(f"Creating collection '{self.collection_name}'")
        self.client = self._make_client(db_path)

        if self.client.has_collection(self.collection_name):
            logger.info("Existing collection found; dropping before recreation")
            self.client.drop_collection(self.collection_name)

        index_params = self.client.prepare_index_params()
        self._resolve_index_params(
            index_params,
            index_type=index_type,
            total_vectors=len(embeddings),
            is_lite=is_lite,
        )

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
        is_lite = self.is_lite_uri(db_path)
        self.use_v1 = self.use_v1 or is_lite
        if is_lite and not os.path.exists(db_path):
            raise FileNotFoundError(f"Milvus database not found at {db_path}")

        self.db_path = db_path
        self.client = self._make_client(self.db_path)
        if not self.client.has_collection(self.collection_name):
            raise ValueError(f"Collection '{self.collection_name}' missing")
        self.client.load_collection(self.collection_name)

    def get_vector_count(self):
        """Return number of vectors stored in the collection."""
        if self._doc_ids is not None:
            return len(self._doc_ids)
        if self.client is None or self.collection_name is None:
            return 0
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return int(stats["row_count"])
        except Exception as exc:
            logger.warning(f"Could not fetch collection stats: {exc}")
            return 0

    def search_params(self):
        index_type = self._get_index_from_collection().lower()
        is_lite = self.is_lite_uri(self.db_path)

        if is_lite and index_type != "flat":
            raise ValueError("Milvus Lite only supports 'flat' index type")

        if index_type == "flat" or index_type == "hnsw":
            # use default search params
            return None

        if index_type.startswith("ivf"):
            total_vectors = self.get_vector_count()
            sqrt_n = max(1, int(total_vectors**0.5))
            nlist = min(max(4 * sqrt_n, 1), max(total_vectors, 1))
            return {"params": {"nprobe": nlist}}

        raise ValueError(f"Unsupported index type: {index_type}")

    @staticmethod
    def is_lite_uri(uri: str | None):
        return uri is not None and uri.endswith(".db")

    def _resolve_index_params(
        self,
        index_params: IndexParams,
        index_type: str,
        total_vectors: int,
        is_lite: bool,
    ):
        index_type = index_type.lower()

        if is_lite and index_type != "flat":
            raise ValueError("Milvus Lite only supports 'flat' index type")

        params = {
            "field_name": self.vector_field_name,
            "index_name": self.index_name,
            "metric_type": "COSINE",
        }

        if index_type == "flat":
            params["index_type"] = "FLAT"
        elif index_type.startswith("ivf"):
            sqrt_n = max(1, int(total_vectors**0.5))
            nlist = min(max(4 * sqrt_n, 1), max(total_vectors, 1))
            params.update(
                {
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": nlist},
                }
            )
        elif index_type == "hnsw":
            params["index_type"] = "HNSW"
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        logger.info(f"Creating {params['index_type']} index")

        index_params.add_index(**params)

    def _get_index_from_collection(self):
        if self.client is None:
            raise ValueError("Client not initialized")
        index_info = self.client.describe_index(
            collection_name=self.collection_name, index_name=self.index_name
        )
        if not index_info:
            return "flat"
        return str(index_info["index_type"])

    def _make_client(self, db_path: str | None):
        """Create a MilvusClient, defaulting to localhost when no path/URI is provided."""
        return MilvusClient(uri=db_path) if db_path else MilvusClient()


@dataclass
class RetrievalChunk:
    document: Document
    score: float
    chunk_text: str | None = None

    def __post_init__(self):
        """Validate retrieval chunk."""
        if not 0.0 <= self.score <= 1.0:
            logger.warning(f"Score {self.score} outside expected range [0.0, 1.0]")


@dataclass
class RetrievalResult:
    retrieval_chunks: list[RetrievalChunk]
    llm_answer: str | None = None


class SciNCLRetrieval:
    def __init__(
        self,
        ingestion_system: SciNCLIngestion,
        documents: dict[str, Document],
        enable_llm: bool = False,
    ):
        """
        Initialize retrieval system.

        Args:
            ingestion_system: Initialized SciNCLIngestion system
            documents: Dictionary of documents (doc_id -> Document)
            enable_llm: Explicitly enable/disable LLM query rewrite/answering (default: False)
        """
        self.ingestion = ingestion_system
        self.documents = documents
        self.device = self.ingestion.device
        self.reranker = (
            BGERerankFunction(device=self.ingestion.device, use_fp16=True)
            if not self.ingestion.use_v1
            else None
        )
        self.llm_assistant = LLMAssistant() if enable_llm else None

    def retrieve(self, query: str, k: int):
        """
        Retrieve documents, optionally rewriting the query and generating an LLM answer.
        With chunking, multiple chunks may map to the same document.

        Args:
            query: Query text
            k: Number of documents to retrieve

        Returns:
            RetrievalResult containing retrieved chunks and optional LLM answer
        """
        if k <= 0:
            raise ValueError("k must be positive")

        original_query = query
        if self.llm_assistant:
            rewrite_start = time.perf_counter()
            query = self.llm_assistant.rewrite_query(query)
            rewrite_end = time.perf_counter()
            logger.notice(f"Rewritten query: {query}")
            logger.notice(f"Rewrite time: {rewrite_end - rewrite_start:.2f} seconds")

        retrieval_chunks = self._search_and_rerank(query, k)

        llm_answer = None
        if self.llm_assistant and retrieval_chunks:
            contexts = [
                chunk.chunk_text for chunk in retrieval_chunks if chunk.chunk_text
            ]
            generate_start = time.perf_counter()
            llm_answer = self.llm_assistant.generate_answer(original_query, contexts)
            generate_end = time.perf_counter()
            logger.notice(f"LLM answer: {llm_answer}")
            logger.notice(f"Generate time: {generate_end - generate_start:.2f} seconds")

        return RetrievalResult(retrieval_chunks=retrieval_chunks, llm_answer=llm_answer)

    def _search_and_rerank(self, query: str, k: int):
        """
        Internal search and rerank pipeline without LLM usage.

        Args:
            query: Query text (already rewritten if LLM is enabled)
            k: Number of documents to retrieve
        """
        if self.ingestion.client is None:
            raise ValueError("Collection not loaded")

        n_vectors = self.ingestion.get_vector_count()
        if n_vectors == 0:
            return []

        # Encode the query as [CLS] embedding
        inputs = self.ingestion.tokenizer(
            [query],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.ingestion.base_model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        avg_chunks_per_doc = n_vectors / max(len(self.documents), 1)
        search_k = int(np.ceil(k * max(avg_chunks_per_doc, 1.0)))
        search_k = min(max(search_k, k), n_vectors)

        search_start = time.perf_counter()
        search_results = self.ingestion.client.search(
            collection_name=self.ingestion.collection_name,
            data=query_embedding.tolist(),
            anns_field=self.ingestion.vector_field_name,
            limit=search_k,
            search_params=self.ingestion.search_params(),
            output_fields=["doc_id", "chunk_text"],
        )
        search_end = time.perf_counter()
        logger.notice(f"Search time: {search_end - search_start:.2f} seconds")

        if not search_results or not search_results[0]:
            return []

        aggregated_hits = self._aggregate_hits(search_results[0])
        if not aggregated_hits:
            return []

        if self.reranker is None:
            return self._top_k_without_rerank(aggregated_hits, k)

        return self._rerank_results(query, aggregated_hits, k)

    def _aggregate_hits(self, hits: list[dict]):
        """
        Merge multiple chunk hits per document into a single entry.
        For each doc_id, keep the first score and concatenate the body texts.
        """
        per_doc = {}
        for hit in hits:
            entity = hit["entity"]
            doc_id = entity["doc_id"]
            if not doc_id or doc_id not in self.documents:
                continue

            chunk_text = entity["chunk_text"]
            score = 1.0 - (hit["distance"] / 2.0)

            doc_entry = per_doc.setdefault(doc_id, {"score": float(score), "texts": []})
            doc_entry["texts"].append(chunk_text)

        sep_token = self.ingestion.tokenizer.sep_token or "[SEP]"
        aggregated = []
        for doc_id, data in per_doc.items():
            texts = [t for t in data["texts"] if t]
            bodies = [t.split(sep_token, 1)[1].strip() for t in texts]
            merged_abstract = " ".join(bodies).strip()
            doc_title = self.documents[doc_id].title
            merged_text = f"{doc_title}: {merged_abstract}"
            aggregated.append((doc_id, data["score"], merged_text))

        return aggregated

    def _top_k_without_rerank(self, aggregated_hits, k: int):
        return [
            RetrievalChunk(
                document=self.documents[doc_id], score=score, chunk_text=chunk_text
            )
            for doc_id, score, chunk_text in aggregated_hits[:k]
            if doc_id in self.documents
        ]

    def _rerank_results(self, query: str, aggregated_hits: list[tuple], k: int):
        chunk_texts = [chunk_text for _, _, chunk_text in aggregated_hits if chunk_text]

        if not chunk_texts:
            return self._top_k_without_rerank(aggregated_hits, k)

        chunk_to_doc = {chunk_text: doc_id for doc_id, _, chunk_text in aggregated_hits}

        rerank_start = time.perf_counter()
        rerank_results = self.reranker(query, chunk_texts, top_k=k)
        rerank_end = time.perf_counter()
        logger.notice(f"Rerank time: {rerank_end - rerank_start:.2f} seconds")

        results = []
        for result in rerank_results:
            doc_id = chunk_to_doc[result.text]
            doc = self.documents[doc_id]
            results.append(
                RetrievalChunk(
                    document=doc,
                    score=result.score,
                    chunk_text=result.text,
                )
            )

        return results
