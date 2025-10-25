"""
Fresh TF-IDF RAG System Setup
Creates new embeddings from scratch with a clean collection
"""

import json
import os
import uuid
import pickle
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")


class FreshTFIDFRAG:
    """Fresh TF-IDF RAG system with clean setup."""
    
    def __init__(self, 
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6333,
                 collection_name: str = "medical_papers_tfidf_fresh"):
        """Initialize the fresh TF-IDF RAG system."""
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Initialize TF-IDF vectorizer
        print("🔄 Initializing fresh TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.embedding_dim = 1000  # Will be set after fitting
        
        # Create fresh collection
        self._create_fresh_collection()
    
    def _create_fresh_collection(self):
        """Create a fresh collection, deleting any existing one."""
        try:
            # Delete existing collection if it exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                print(f"🗑️  Deleting existing collection: {self.collection_name}")
                self.qdrant_client.delete_collection(self.collection_name)
            
            # Create new collection
            print(f"📦 Creating fresh collection: {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Fresh collection '{self.collection_name}' created")
            
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
        """Generate TF-IDF embeddings for documents."""
        print("🔢 Generating fresh TF-IDF embeddings...")
        
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
        
        # Fit TF-IDF vectorizer on all texts
        print("🔧 Fitting TF-IDF vectorizer...")
        embeddings = self.vectorizer.fit_transform(texts)
        self.embedding_dim = embeddings.shape[1]
        
        print(f"✅ TF-IDF vectorizer fitted. Embedding dimension: {self.embedding_dim}")
        
        # Create Qdrant points
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings.toarray())):
            if i % 50 == 0:
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
    
    def save_vectorizer(self, path: str = "vectorizer_fresh.pkl"):
        """Save the fitted vectorizer."""
        print(f"💾 Saving vectorizer to {path}...")
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"✅ Vectorizer saved to {path}")
    
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
    """Main function to set up the fresh TF-IDF RAG system."""
    print("🏥 Fresh TF-IDF RAG System Setup")
    print("=" * 50)
    
    # Initialize RAG system
    rag = FreshTFIDFRAG()
    
    # Load data
    print("\n📚 Loading data...")
    pubmed_docs = rag.load_pubmed_data()
    semantic_docs = rag.load_semantic_scholar_data()
    
    # Combine all documents
    all_documents = pubmed_docs + semantic_docs
    print(f"📊 Total documents: {len(all_documents)}")
    
    # Generate embeddings and upload
    print("\n🔢 Generating fresh TF-IDF embeddings...")
    points = rag.generate_embeddings(all_documents)
    
    print("\n📤 Uploading to Qdrant...")
    rag.upload_to_qdrant(points)
    
    # Save vectorizer
    rag.save_vectorizer()
    
    # Collection info
    print(f"\n📊 Collection Info:")
    info = rag.get_collection_info()
    print(f"Documents in collection: {info.get('vectors_count', 'Unknown')}")
    
    print("\n✅ Fresh TF-IDF RAG system setup complete!")


if __name__ == "__main__":
    main()
