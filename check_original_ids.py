"""
Script to check if original paper IDs are stored in the document structure
"""

from rag_with_persistence import PersistentMedicalRAG
import json


def main():
    """Check if original paper IDs are preserved."""
    print("🔍 Checking Original Paper IDs in Document Structure")
    print("=" * 60)
    
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
    
    # Search for both PubMed and Semantic Scholar samples
    print("\n🔍 Searching for PubMed sample...")
    try:
        pubmed_results = rag.search_similar_documents("hypoxaemia children", limit=1, source_filter="pubmed")
        
        if pubmed_results:
            doc = pubmed_results[0]
            print(f"\n📄 PubMed Sample:")
            print("-" * 40)
            print(f"🆔 Point ID: {doc['id']}")
            print(f"📚 Source: {doc['source']}")
            print(f"📝 Title: {doc['title'][:80]}...")
            
            # Check if PMID is in the payload
            print(f"\n🔍 Checking for PMID in payload...")
            if 'pmid' in doc:
                print(f"✅ PMID found: {doc['pmid']}")
            else:
                print("❌ No PMID found in payload")
            
            print(f"\n📋 Available payload keys:")
            for key in doc.keys():
                print(f"  - {key}")
        else:
            print("❌ No PubMed documents found")
            
    except Exception as e:
        print(f"❌ Error fetching PubMed sample: {e}")
    
    print("\n" + "="*60)
    print("🔍 Searching for Semantic Scholar sample...")
    try:
        semantic_results = rag.search_similar_documents("nanotechnology healthcare", limit=1, source_filter="semantic_scholar")
        
        if semantic_results:
            doc = semantic_results[0]
            print(f"\n📄 Semantic Scholar Sample:")
            print("-" * 40)
            print(f"🆔 Point ID: {doc['id']}")
            print(f"📚 Source: {doc['source']}")
            print(f"📝 Title: {doc['title'][:80]}...")
            
            # Check if paper_id is in the payload
            print(f"\n🔍 Checking for paper_id in payload...")
            if 'paper_id' in doc:
                print(f"✅ Paper ID found: {doc['paper_id']}")
            else:
                print("❌ No paper_id found in payload")
            
            print(f"\n📋 Available payload keys:")
            for key in doc.keys():
                print(f"  - {key}")
        else:
            print("❌ No Semantic Scholar documents found")
            
    except Exception as e:
        print(f"❌ Error fetching Semantic Scholar sample: {e}")
    
    print(f"\n✅ Original ID check completed!")
    print(f"\n💡 Note: Original paper IDs should be stored in the payload for easy access")


if __name__ == "__main__":
    main()
