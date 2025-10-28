"""
Simple RAG System Implementation using Qdrant with TF-IDF embeddings.
This is a baseline implementation for comparison with transformer-based models.
"""

import json
import os
import uuid
import pickle
from typing import List, Dict, Any, Optional
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings("ignore")

# Try to import OpenAI, but make it optional
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. AI generation will be disabled.")


class TfidfRAG:
    """Simple RAG system using TF-IDF for embeddings - baseline for comparison."""

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "medical_papers_tfidf",
        vectorizer_path: str = "data/tfidf_vectorizer.pkl",
    ):
        """Initialize the TF-IDF RAG system."""
        self.collection_name = collection_name
        self.vectorizer_path = vectorizer_path

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Initialize TF-IDF vectorizer
        print("Initializing TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )
        self.embedding_dim = None  # Will be set after fitting the vectorizer

        # Try to load existing vectorizer
        self._load_vectorizer()

        # Initialize OpenAI if available
        self.openai_client = None
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                print("✅ OpenAI client initialized")
            else:
                print("⚠️  OpenAI API key not found. AI generation disabled.")
        else:
            print("⚠️  OpenAI not available. AI generation disabled.")

    def _load_vectorizer(self):
        """Load the fitted vectorizer if it exists."""
        if os.path.exists(self.vectorizer_path):
            try:
                with open(self.vectorizer_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
                self.embedding_dim = len(self.vectorizer.vocabulary_)
                print("✅ Loaded existing vectorizer")
                return True
            except Exception as e:
                print(f"⚠️  Could not load vectorizer: {e}")
        return False

    def _save_vectorizer(self):
        """Save the fitted vectorizer."""
        try:
            os.makedirs(os.path.dirname(self.vectorizer_path), exist_ok=True)
            with open(self.vectorizer_path, "wb") as f:
                pickle.dump(self.vectorizer, f)
            print("✅ Saved vectorizer")
        except Exception as e:
            print(f"⚠️  Could not save vectorizer: {e}")

    def setup_collection(self, documents: List[Dict[str, Any]]) -> List[PointStruct]:
        """Generate embeddings and create Qdrant collection with correct dimensions."""
        # Generate embeddings (this fits the vectorizer and sets embedding_dim)
        print("Generating TF-IDF embeddings...")
        texts = [doc["text"] for doc in documents]

        print("Fitting TF-IDF vectorizer...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.embedding_dim = tfidf_matrix.shape[1]
        print(f"✅ Vectorizer fitted (dimension: {self.embedding_dim})")

        # Save the fitted vectorizer
        self._save_vectorizer()

        # Convert to dense arrays
        embeddings = tfidf_matrix.toarray()

        # Now create the collection with the correct dimension
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
        print(
            f"✅ Created collection: {self.collection_name} (dimension: {self.embedding_dim})"
        )

        # Create Qdrant points
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            if i % 20 == 0:
                print(f"Creating points: {i + 1}/{len(documents)}")

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
                    "pmid": doc.get("pmid"),
                    "paper_id": doc.get("paper_id"),
                    "url": doc.get("url"),
                    "metadata": doc["metadata"],
                },
            )
            points.append(point)

        print(f"✅ Generated {len(points)} points")
        return points

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
                "year": None,
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
                "text": abstract,
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
        """Search for similar documents using TF-IDF."""
        # Check if vectorizer is fitted
        if not hasattr(self.vectorizer, "vocabulary_"):
            raise ValueError(
                "The TF-IDF vectorizer is not fitted. Please run the setup process first."
            )

        # Generate query embedding
        query_embedding = self.vectorizer.transform([query]).toarray()[0]

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
            query_vector=query_embedding.tolist(),
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
                context += f"Authors: {', '.join(doc['authors'][:3])}\n"
            context += f"Content: {doc['abstract'][:500]}...\n"

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
    """Main function to set up and test the TF-IDF RAG system."""
    print("🚀 Initializing TF-IDF Medical RAG System (Baseline)...")

    # Initialize RAG system
    rag = TfidfRAG()

    print("\n📚 Processing data sources...")
    pubmed_docs = rag.process_pubmed_data("data/pubmed_abstracts.csv")
    semantic_docs = rag.process_semantic_scholar_data("data/semanticscholar.json")

    # Combine all documents
    all_documents = pubmed_docs + semantic_docs
    print(f"📊 Total documents: {len(all_documents)}")

    print("\n🔢 Setting up collection with TF-IDF embeddings...")
    points = rag.setup_collection(all_documents)

    print("\n📤 Uploading to Qdrant...")
    rag.upload_to_qdrant(points)

    print("\n🧪 Testing the system...")
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

    print("\n📊 Collection Info:")
    info = rag.get_collection_info()
    print(f"Documents in collection: {info.get('vectors_count', 'Unknown')}")

    print("\n✅ TF-IDF RAG system setup complete!")


if __name__ == "__main__":
    main()
