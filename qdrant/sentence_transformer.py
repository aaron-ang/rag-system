"""
RAG System Implementation using Qdrant with SentenceTransformer embeddings.
Handles PubMed abstracts and Semantic Scholar papers for medical research queries.

This implementation uses persistent vectorizer storage for consistent querying.
"""

import json
import os
import uuid
from typing import List, Dict, Any, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SentenceTransformerRAG:
    """RAG system for medical research papers using Qdrant vector database with SentenceTransformer embeddings."""

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "medical_papers",
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize the RAG system.

        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Name of the Qdrant collection
            embedding_model: Sentence transformer model for embeddings
            openai_api_key: OpenAI API key for generation
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        except Exception:
            print(f"Error loading {embedding_model}, trying alternative...")
            # Fallback to a simpler model
            self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        print(f"✅ Embedding model loaded (dimension: {self.embedding_dim})")

        # Initialize OpenAI client
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            else:
                self.openai_client = None
                print("⚠️  OpenAI API key not provided. Generation will be disabled.")

    def setup_collection(self):
        """Create or recreate the Qdrant collection."""
        try:
            # Delete existing collection if it exists
            self.qdrant_client.delete_collection(self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except Exception:
            pass  # Collection doesn't exist, which is fine

        # Create new collection
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim, distance=Distance.COSINE
            ),
        )
        print(f"✅ Created collection: {self.collection_name}")

    def process_pubmed_data(self, csv_path: str) -> List[Dict[str, Any]]:
        """Process PubMed abstracts CSV data."""
        print("Processing PubMed abstracts...")
        df = pd.read_csv(csv_path)

        documents = []
        for idx, row in df.iterrows():
            # Parse metadata
            meta = eval(row["meta"])  # Convert string to dict

            document = {
                "id": str(uuid.uuid4()),
                "source": "pubmed",
                "pmid": meta.get("pmid"),
                "language": meta.get("language"),
                "text": row["text"],
                "title": row["text"].split(".")[0][:100]
                + "...",  # Extract title from first sentence
                "year": None,  # PubMed doesn't have year in this dataset
                "citations": None,
                "authors": None,
                "abstract": row["text"],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{meta.get('pmid')}/"
                if meta.get("pmid")
                else None,
                "metadata": {"stats": eval(row["stats"]), "simhash": row["simhash"]},
            }
            documents.append(document)

        print(f"✅ Processed {len(documents)} PubMed abstracts")
        return documents

    def process_semantic_scholar_data(self, json_path: str) -> List[Dict[str, Any]]:
        """Process Semantic Scholar JSON data."""
        print("Processing Semantic Scholar papers...")

        with open(json_path, "r") as f:
            data = json.load(f)

        documents = []
        for paper in data.get("data", []):
            # Extract abstract or use title if no abstract
            abstract = paper.get("abstract", "")
            if not abstract:
                abstract = paper.get("title", "")

            # Extract authors
            authors = []
            if "authors" in paper and paper["authors"]:
                authors = [author.get("name", "") for author in paper["authors"]]

            document = {
                "id": str(uuid.uuid4()),
                "source": "semantic_scholar",
                "paper_id": paper.get("paperId"),
                "url": paper.get("url"),
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "citations": paper.get("citationCount", 0),
                "authors": authors,
                "abstract": abstract,
                "text": abstract,  # Use abstract as main text
                "metadata": {
                    "reference_count": paper.get("referenceCount", 0),
                    "influential_citations": paper.get("influentialCitationCount", 0),
                    "open_access": paper.get("openAccessPdf", {}).get("url")
                    if paper.get("openAccessPdf")
                    else None,
                    "tldr": paper.get("tldr", {}).get("text")
                    if paper.get("tldr")
                    else None,
                },
            }
            documents.append(document)

        print(f"✅ Processed {len(documents)} Semantic Scholar papers")
        return documents

    def generate_embeddings(self, documents: List[Dict[str, Any]]) -> List[PointStruct]:
        """Generate embeddings for documents and create Qdrant points."""
        print("Generating embeddings...")

        texts: list[str] = [doc["text"] for doc in documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        points = []
        for doc, embedding in zip(documents, embeddings):
            # Create Qdrant point
            point = PointStruct(
                id=doc["id"],
                vector=embedding.tolist(),
                payload={
                    "source": doc["source"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "abstract": doc["abstract"],
                    "year": doc["year"],
                    "citations": doc["citations"],
                    "authors": doc["authors"],
                    "metadata": doc["metadata"],
                },
            )
            points.append(point)

        print(f"✅ Generated embeddings for {len(points)} documents")
        return points

    def upload_to_qdrant(self, points: List[PointStruct]):
        """Upload points to Qdrant collection."""
        print("Uploading to Qdrant...")

        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=batch
            )
            print(
                f"Uploaded batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1}"
            )

        print(f"✅ Uploaded {len(points)} documents to Qdrant")

    def search_similar_documents(
        self,
        query: str,
        limit: int = 5,
        source_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Build filter
        filter_conditions = []
        if source_filter:
            filter_conditions.append(
                FieldCondition(key="source", match=MatchValue(value=source_filter))
            )
        if year_filter:
            filter_conditions.append(
                FieldCondition(key="year", match=MatchValue(value=year_filter))
            )

        search_filter = Filter(must=filter_conditions) if filter_conditions else None

        # Search
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=search_filter,
        )

        # Format results
        results = []
        for result in search_results:
            results.append(
                {
                    "id": result.id,
                    "score": result.score,
                    "source": result.payload["source"],
                    "title": result.payload["title"],
                    "text": result.payload["text"],
                    "abstract": result.payload["abstract"],
                    "year": result.payload["year"],
                    "citations": result.payload["citations"],
                    "authors": result.payload["authors"],
                    "pmid": result.payload.get("pmid"),  # PubMed ID
                    "paper_id": result.payload.get("paper_id"),  # Semantic Scholar ID
                    "url": result.payload.get("url"),  # Paper URL
                    "metadata": result.payload["metadata"],
                }
            )

        return results

    def generate_answer(
        self, query: str, context_documents: List[Dict[str, Any]]
    ) -> str:
        """Generate answer using OpenAI GPT."""
        if not self.openai_client:
            return "OpenAI API key not configured. Cannot generate answer."

        # Prepare context
        context = ""
        for i, doc in enumerate(context_documents, 1):
            context += f"\n--- Document {i} ---\n"
            context += f"Title: {doc['title']}\n"
            context += f"Source: {doc['source']}\n"
            if doc["year"]:
                context += f"Year: {doc['year']}\n"
            if doc["authors"]:
                context += (
                    f"Authors: {', '.join(doc['authors'][:3])}\n"  # First 3 authors
                )
            context += f"Content: {doc['abstract'][:500]}...\n"  # First 500 chars

        # Create prompt
        prompt = f"""You are a medical research assistant. Answer the following question based on the provided research papers.

Question: {query}

Context from research papers:
{context}

Please provide a comprehensive answer based on the research papers above. If the papers don't contain enough information to answer the question, say so clearly."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful medical research assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def query(
        self,
        question: str,
        limit: int = 5,
        source_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        generate_answer: bool = True,
    ) -> Dict[str, Any]:
        """Main query method that retrieves documents and optionally generates an answer."""
        print(f"🔍 Searching for: {question}")

        # Search for relevant documents
        documents = self.search_similar_documents(
            query=question,
            limit=limit,
            source_filter=source_filter,
            year_filter=year_filter,
        )

        result = {"question": question, "documents": documents, "answer": None}

        # Generate answer if requested and OpenAI is available
        if generate_answer and documents and self.openai_client:
            print("🤖 Generating answer...")
            result["answer"] = self.generate_answer(question, documents)

        return result

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status,
                "config": collection_info.config,
            }
        except Exception as e:
            return {"error": str(e)}


def main():
    """Main function to set up and test the RAG system."""
    print("🚀 Initializing Medical RAG System with SentenceTransformers...")

    # Initialize RAG system
    rag = SentenceTransformerRAG(
        openai_api_key=os.getenv("OPENAI_API_KEY")  # Set this in .env file
    )

    # Setup collection
    rag.setup_collection()

    # Process and upload data
    print("\n📚 Processing data sources...")

    # Process PubMed data
    pubmed_docs = rag.process_pubmed_data("data/pubmed_abstracts.csv")

    # Process Semantic Scholar data
    semantic_docs = rag.process_semantic_scholar_data("data/semanticscholar.json")

    # Combine all documents
    all_documents = pubmed_docs + semantic_docs
    print(f"📊 Total documents: {len(all_documents)}")

    # Generate embeddings and upload
    print("\n🔢 Generating embeddings...")
    points = rag.generate_embeddings(all_documents)

    print("\n📤 Uploading to Qdrant...")
    rag.upload_to_qdrant(points)

    # Test the system
    print("\n🧪 Testing the system...")

    # Test queries
    test_queries = [
        "What are the symptoms of hypoxaemia in children?",
        "How is oxygen therapy administered?",
        "What are the risk factors for pneumonia in children?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("=" * 60)

        result = rag.query(query, limit=3)

        print(f"\n📄 Found {len(result['documents'])} relevant documents:")
        for i, doc in enumerate(result["documents"], 1):
            print(f"\n{i}. {doc['title']}")
            print(f"   Source: {doc['source']}")
            print(f"   Score: {doc['score']:.3f}")
            if doc["year"]:
                print(f"   Year: {doc['year']}")
            if doc["citations"]:
                print(f"   Citations: {doc['citations']}")

        if result["answer"]:
            print("\n🤖 Generated Answer:")
            print(result["answer"])

    # Collection info
    print("\n📊 Collection Info:")
    info = rag.get_collection_info()
    print(f"Documents in collection: {info.get('vectors_count', 'Unknown')}")

    print("\n✅ RAG system setup complete!")


if __name__ == "__main__":
    main()
