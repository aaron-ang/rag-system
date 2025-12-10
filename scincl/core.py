"""
SciNCL-based data ingestion and retrieval system.
"""

import ast
import json
import logging
import os
import time
from dataclasses import dataclass, asdict

import boto3
import coloredlogs
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
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


class BedrockLLMAssistant:
    """LLM utility for query rewrite and answer generation via Amazon Bedrock."""

    def __init__(
        self,
        model_id: str = "amazon.titan-text-express-v1",
        region: str = "us-west-2",
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def rewrite_query(self, query: str):
        """Rewrite the user query into a concise search query."""
        prompt = (
            "Rewrite the following user query into a concise search query for scientific literature. "
            "Preserve core intent and key entities. Return only the rewritten query text."
            f"\n\nUser query:\n{query}"
        )

        request = json.dumps(
            {
                "inputText": prompt,
                "textGenerationConfig": {
                    "temperature": self.temperature,
                    "maxTokenCount": self.max_tokens,
                },
            }
        )

        try:
            response = self.client.invoke_model(modelId=self.model_id, body=request)
            model_response = json.loads(response["body"].read())
            return str(model_response["results"][0]["outputText"])
        except Exception as exc:
            logger.warning(
                f"Bedrock query rewrite failed; falling back to original query: {exc}"
            )
            return query

    def generate_answer(self, query: str, contexts: list[str]):
        """Generate an answer using retrieved context."""
        context_block = (
            "\n\n".join([f"- {c}" for c in contexts if c and c.strip()])
            or "No additional context provided."
        )

        prompt = (
            "You are a concise assistant for scientific literature.\n"
            "Given a user query and retrieved passages, "
            "produce a short, directly answer or summarize relevant findings.\n"
            "If the context is insufficient, say so briefly.\n\n"
            f"Query: {query}\n"
            f"Context:\n{context_block}\n\n"
            "Answer:"
        )

        request = json.dumps(
            {
                "inputText": prompt,
                "textGenerationConfig": {
                    "temperature": self.temperature,
                    "maxTokenCount": self.max_tokens,
                },
            }
        )

        try:
            response = self.client.invoke_model(modelId=self.model_id, body=request)
            model_response = json.loads(response["body"].read())
            return str(model_response["results"][0]["outputText"])
        except Exception as exc:
            logger.warning(f"Bedrock answer generation failed: {exc}")
            return ""


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
            full_text = f"{doc.title}{sep}{doc.abstract}"

            if use_sliding_window:
                encodings = self.tokenizer(
                    full_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_window_len,
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
            else:
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
            index_type: Index type to create ('flat' for Lite; 'flat', 'ivf', or 'hnsw' for server)
        """
        if embeddings.size == 0:
            raise ValueError("No embeddings provided to index")

        is_lite = self.is_lite_uri(db_path)
        self.use_v1 = self.use_v1 or is_lite
        if is_lite and index_type.lower() != "flat":
            raise ValueError("Milvus Lite backend currently supports only 'flat' index")

        if self._doc_ids is None or self._chunk_texts is None:
            raise ValueError("Document IDs and chunk texts must be generated first")

        if len(self._doc_ids) != len(embeddings) or len(self._chunk_texts) != len(
            embeddings
        ):
            raise ValueError("Mismatch between embeddings and stored chunk metadata")

        normalized_index_type = index_type.lower()

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
            index_type=normalized_index_type,
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
            return int(stats.get("row_count", 0))
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
    def is_lite_uri(uri: str | None) -> bool:
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

    def _make_client(self, db_path: str | None) -> MilvusClient:
        """Create a MilvusClient, defaulting to localhost when no path/URI is provided."""
        return MilvusClient(uri=db_path) if db_path else MilvusClient()


@dataclass
class RetrievalChunk:
    document: Document
    sim_score: float
    chunk_text: str | None = None

    def __post_init__(self):
        """Validate retrieval chunk."""
        if not 0.0 <= self.sim_score <= 1.0:
            logger.warning(f"Score {self.sim_score} outside expected range [0.0, 1.0]")


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
        self.llm_assistant = BedrockLLMAssistant() if enable_llm else None

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

        if self.llm_assistant:
            rewrite_start = time.perf_counter()
            query = self.llm_assistant.rewrite_query(query)
            rewrite_end = time.perf_counter()
            logger.notice(f"Rewrite time: {rewrite_end - rewrite_start:.2f} seconds")

        retrieval_chunks = self._search_and_rerank(query, k)

        llm_answer = None
        if self.llm_assistant and retrieval_chunks:
            contexts = [
                chunk.chunk_text for chunk in retrieval_chunks if chunk.chunk_text
            ]
            generate_start = time.perf_counter()
            llm_answer = self.llm_assistant.generate_answer(query, contexts)
            generate_end = time.perf_counter()
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
            search_params=self.ingestion.search_params(),
            output_fields=["doc_id", "chunk_text"],
        )
        search_end = time.perf_counter()
        logger.notice(f"Search time: {search_end - search_start:.2f} seconds")

        if not search_results or not search_results[0]:
            return []

        search_result = search_results[0]
        if len(search_result) == 1:
            hit = search_result[0]
            entity = hit.get("entity", {})
            doc_id = entity.get("doc_id")
            document = self.documents.get(doc_id)
            if document is None:
                logger.warning(f"Document ID {doc_id} not found in store; skipping")
                return []
            return [RetrievalChunk(document=document, sim_score=hit.get("distance"))]

        if self.reranker is None:
            results = []
            seen = set()
            for hit in search_result:
                entity = hit.get("entity", {})
                doc_id = entity.get("doc_id")
                if not doc_id or doc_id in seen:
                    continue
                doc = self.documents.get(doc_id)
                if doc is None:
                    logger.warning(f"Document ID {doc_id} not found in store; skipping")
                    continue
                results.append(
                    RetrievalChunk(document=doc, sim_score=hit.get("distance"))
                )
                seen.add(doc_id)
                if len(results) >= k:
                    break
            return results

        return self._rerank_results(query, search_result, k)

    def _rerank_results(self, query: str, search_result: list[dict], k: int):
        doc_chunk_hits = {}
        chunk_to_doc = {}
        for hit in search_result:
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
            results.append(
                RetrievalChunk(
                    document=doc,
                    sim_score=result.score,
                    chunk_text=result.text,
                )
            )
            seen.add(doc_id)

        return results
