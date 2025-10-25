"""
SciNCL-based RAG System for Medical Research
Uses SciNCL embeddings for better semantic understanding of scientific literature
"""

import json
import os
import uuid
import pickle
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# Try to import OpenAI, but make it optional
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. AI generation will be disabled.")


class SciNCLMedicalRAG:
    """SciNCL-based RAG system for medical research."""
    
    def __init__(self, 
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6333,
                 collection_name: str = "medical_papers_scincl",
                 model_name: str = "malteos/scincl"):
        """Initialize the SciNCL-based RAG system."""
        self.collection_name = collection_name
        self.model_name = model_name
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Initialize SciNCL model
        print("🔄 Loading SciNCL model...")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = 768  # SciNCL embedding dimension
        
        # Initialize OpenAI client if available
        self.openai_client = None
        if OPENAI_AVAILABLE:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key and api_key != 'your_openai_api_key_here':
                    self.openai_client = OpenAI(api_key=api_key)
                    print("✅ OpenAI client initialized")
                else:
                    print("⚠️  OpenAI API key not found. AI generation disabled.")
            except Exception as e:
                print(f"⚠️  OpenAI initialization failed: {e}")
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists with correct configuration."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                print(f"📦 Creating collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Collection '{self.collection_name}' created")
            else:
                print(f"✅ Collection '{self.collection_name}' already exists")
        except Exception as e:
            print(f"❌ Error managing collection: {e}")
            raise
    
    def load_pubmed_data(self) -> List[Dict[str, Any]]:
        """Load PubMed abstracts data."""
        print("📚 Loading PubMed abstracts...")
        df = pd.read_csv("data/pubmed_abstracts.csv")
        
        documents = []
        for _, row in df.iterrows():
            # Parse metadata
            meta = eval(row['meta']) if isinstance(row['meta'], str) else row['meta']
            pmid = meta.get('pmid', '')
            
            # Create document
            doc = {
                'id': str(uuid.uuid4()),
                'source': 'pubmed',
                'title': '',  # PubMed abstracts don't have titles in this format
                'text': row['text'],
                'abstract': row['text'],
                'year': None,  # Not available in this dataset
                'citations': None,
                'authors': [],
                'pmid': pmid,
                'paper_id': None,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                'metadata': meta
            }
            documents.append(doc)
        
        print(f"✅ Loaded {len(documents)} PubMed abstracts")
        return documents
    
    def load_semantic_scholar_data(self) -> List[Dict[str, Any]]:
        """Load Semantic Scholar data."""
        print("📚 Loading Semantic Scholar data...")
        
        with open("data/semanticscholar.json", "r") as f:
            data = json.load(f)
        
        documents = []
        for paper in data['data']:
            # Create document
            doc = {
                'id': str(uuid.uuid4()),
                'source': 'semantic_scholar',
                'title': paper.get('title', ''),
                'text': paper.get('abstract', '') or paper.get('title', ''),
                'abstract': paper.get('abstract', ''),
                'year': paper.get('year'),
                'citations': paper.get('citationCount', 0),
                'authors': [author.get('name', '') for author in paper.get('authors', [])],
                'pmid': None,
                'paper_id': paper.get('paperId'),
                'url': paper.get('url'),
                'metadata': {
                    'venue': paper.get('venue'),
                    'fieldsOfStudy': paper.get('fieldsOfStudy', []),
                    'isOpenAccess': paper.get('isOpenAccess', False)
                }
            }
            documents.append(doc)
        
        print(f"✅ Loaded {len(documents)} Semantic Scholar papers")
        return documents
    
    def generate_embeddings(self, documents: List[Dict[str, Any]]) -> List[PointStruct]:
        """Generate SciNCL embeddings for documents."""
        print("🔢 Generating SciNCL embeddings...")
        
        # Prepare texts for embedding
        texts = []
        for doc in documents:
            # Combine title and abstract for better representation
            text_parts = []
            if doc['title']:
                text_parts.append(doc['title'])
            if doc['abstract']:
                text_parts.append(doc['abstract'])
            
            combined_text = " ".join(text_parts)
            texts.append(combined_text)
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            all_embeddings.extend(batch_embeddings)
        
        # Create Qdrant points
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, all_embeddings)):
            if i % 100 == 0:
                print(f"Creating points: {i+1}/{len(documents)}")
            
            point = PointStruct(
                id=doc['id'],
                vector=embedding.tolist(),
                payload={
                    'source': doc['source'],
                    'title': doc['title'],
                    'text': doc['text'],
                    'abstract': doc['abstract'],
                    'year': doc['year'],
                    'citations': doc['citations'],
                    'authors': doc['authors'],
                    'pmid': doc.get('pmid'),
                    'paper_id': doc.get('paper_id'),
                    'url': doc.get('url'),
                    'metadata': doc['metadata']
                }
            )
            points.append(point)
        
        print(f"✅ Generated embeddings for {len(points)} documents")
        return points
    
    def upload_to_qdrant(self, points: List[PointStruct]):
        """Upload points to Qdrant collection."""
        print("📤 Uploading to Qdrant...")
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
        
        print(f"✅ Uploaded {len(points)} documents to Qdrant")
    
    def search_similar_documents(self, 
                               query: str, 
                               limit: int = 5,
                               source_filter: Optional[str] = None,
                               year_filter: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for similar documents using SciNCL embeddings."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        # Build filter
        filter_conditions = []
        if source_filter:
            filter_conditions.append(FieldCondition(key="source", match=MatchValue(value=source_filter)))
        if year_filter:
            filter_conditions.append(FieldCondition(key="year", match=MatchValue(value=year_filter)))
        
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Search
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            query_filter=search_filter
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                'id': result.id,
                'score': result.score,
                'source': result.payload['source'],
                'title': result.payload['title'],
                'text': result.payload['text'],
                'abstract': result.payload['abstract'],
                'year': result.payload['year'],
                'citations': result.payload['citations'],
                'authors': result.payload['authors'],
                'pmid': result.payload.get('pmid'),
                'paper_id': result.payload.get('paper_id'),
                'url': result.payload.get('url'),
                'metadata': result.payload['metadata']
            })
        
        return results
    
    def generate_answer(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """Generate answer using OpenAI GPT."""
        if not self.openai_client:
            return "OpenAI API key not configured. Cannot generate answer."
        
        # Prepare context
        context = ""
        for i, doc in enumerate(context_documents, 1):
            context += f"\n--- Document {i} ---\n"
            context += f"Title: {doc['title']}\n"
            context += f"Source: {doc['source']}\n"
            if doc['year']:
                context += f"Year: {doc['year']}\n"
            if doc['authors']:
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
                    {"role": "system", "content": "You are a helpful medical research assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(self, 
              question: str, 
              limit: int = 5,
              source_filter: Optional[str] = None,
              year_filter: Optional[int] = None,
              generate_answer: bool = True) -> Dict[str, Any]:
        """Main query method that retrieves documents and optionally generates an answer."""
        print(f"🔍 Searching for: {question}")
        
        # Search for relevant documents
        documents = self.search_similar_documents(
            query=question,
            limit=limit,
            source_filter=source_filter,
            year_filter=year_filter
        )
        
        # Generate answer if requested and OpenAI is available
        answer = None
        if generate_answer and self.openai_client:
            answer = self.generate_answer(question, documents)
        
        return {
            'documents': documents,
            'answer': answer,
            'query': question
        }
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                'vectors_count': collection_info.vectors_count,
                'indexed_vectors_count': collection_info.indexed_vectors_count,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count
            }
        except Exception as e:
            return {'error': str(e)}


def main():
    """Main function to set up the SciNCL RAG system."""
    print("🏥 SciNCL Medical RAG System Setup")
    print("=" * 50)
    
    # Initialize RAG system
    rag = SciNCLMedicalRAG()
    
    # Load data
    print("\n📚 Loading data...")
    pubmed_docs = rag.load_pubmed_data()
    semantic_docs = rag.load_semantic_scholar_data()
    
    # Combine all documents
    all_documents = pubmed_docs + semantic_docs
    print(f"📊 Total documents: {len(all_documents)}")
    
    # Generate embeddings and upload
    print("\n🔢 Generating SciNCL embeddings...")
    points = rag.generate_embeddings(all_documents)
    
    print("\n📤 Uploading to Qdrant...")
    rag.upload_to_qdrant(points)
    
    # Test the system
    print("\n🧪 Testing the system...")
    
    # Test queries
    test_queries = [
        "What are the applications of artificial intelligence in clinical medicine?",
        "How does precision medicine work?",
        "What are the latest developments in stem cell therapy?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = rag.query(query, limit=3)
        
        print(f"\n📄 Found {len(result['documents'])} relevant documents:")
        for i, doc in enumerate(result['documents'], 1):
            print(f"\n{i}. {doc['title']}")
            print(f"   Source: {doc['source']}")
            print(f"   Score: {doc['score']:.3f}")
            if doc['year']:
                print(f"   Year: {doc['year']}")
            if doc['citations']:
                print(f"   Citations: {doc['citations']}")
        
        if result['answer']:
            print(f"\n🤖 Generated Answer:")
            print(result['answer'])
    
    # Collection info
    print(f"\n📊 Collection Info:")
    info = rag.get_collection_info()
    print(f"Documents in collection: {info.get('vectors_count', 'Unknown')}")
    
    print("\n✅ SciNCL RAG system setup complete!")


if __name__ == "__main__":
    main()
