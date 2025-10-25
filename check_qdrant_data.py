"""
Script to check the structure of data in Qdrant database
"""

from rag_with_persistence import PersistentMedicalRAG
import json


def main():
    """Check the structure of data in Qdrant."""
    print("🔍 Checking Qdrant Database Structure")
    print("=" * 50)
    
    # Initialize RAG system
    try:
        rag = PersistentMedicalRAG()
        print("✅ RAG system initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return
    
    # Check collection info
    info = rag.get_collection_info()
    if 'error' in info:
        print(f"❌ Collection error: {info['error']}")
        return
    
    print(f"📊 Collection: {info.get('name', 'Unknown')}")
    print(f"📊 Vectors count: {info.get('vectors_count', 'Unknown')}")
    
    # Search for a sample document
    print("\n🔍 Searching for a sample document...")
    try:
        # Get a sample document
        sample_results = rag.search_similar_documents("medical research", limit=1)
        
        if sample_results:
            doc = sample_results[0]
            print("\n📄 Sample Document Structure:")
            print("-" * 40)
            print(f"🆔 Point ID: {doc['id']}")
            print(f"📊 Score: {doc['score']}")
            print(f"📚 Source: {doc['source']}")
            print(f"📝 Title: {doc['title'][:100]}...")
            print(f"📅 Year: {doc['year']}")
            print(f"📈 Citations: {doc['citations']}")
            print(f"👥 Authors: {doc['authors']}")
            
            print(f"\n📋 Full Payload Structure:")
            print("-" * 40)
            print(json.dumps({
                'id': doc['id'],
                'source': doc['source'],
                'title': doc['title'][:50] + "...",
                'text': doc['text'][:100] + "...",
                'abstract': doc['abstract'][:100] + "...",
                'year': doc['year'],
                'citations': doc['citations'],
                'authors': doc['authors'],
                'metadata': doc['metadata']
            }, indent=2))
            
            # Check if original IDs are preserved
            print(f"\n🔍 Original Paper IDs:")
            print("-" * 40)
            if doc['source'] == 'pubmed':
                # Look for PMID in metadata
                if 'stats' in doc['metadata']:
                    print("📄 PubMed ID found in metadata.stats")
                else:
                    print("❌ No PubMed ID found in metadata")
            elif doc['source'] == 'semantic_scholar':
                # Look for paper_id in metadata
                if 'paper_id' in doc['metadata']:
                    print(f"📄 Semantic Scholar ID: {doc['metadata']['paper_id']}")
                else:
                    print("❌ No Semantic Scholar ID found in metadata")
            
            print(f"\n📊 Metadata Keys:")
            print("-" * 40)
            for key, value in doc['metadata'].items():
                print(f"  {key}: {type(value).__name__}")
                if key == 'stats' and isinstance(value, dict):
                    print(f"    Stats keys: {list(value.keys())}")
        else:
            print("❌ No documents found in collection")
            
    except Exception as e:
        print(f"❌ Error fetching sample document: {e}")
    
    print(f"\n✅ Database structure check completed!")


if __name__ == "__main__":
    main()
