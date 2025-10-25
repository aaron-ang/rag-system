"""
Interactive query script for your Medical RAG System
Run your own custom queries!
"""

from rag_with_persistence import PersistentMedicalRAG


def run_custom_query(rag, question, source_filter=None, year_filter=None, limit=5):
    """Run a custom query and display results."""
    print(f"\n{'='*80}")
    print(f"🔍 Your Query: {question}")
    if source_filter:
        print(f"📚 Source Filter: {source_filter}")
    if year_filter:
        print(f"📅 Year Filter: {year_filter}")
    print('='*80)
    
    try:
        result = rag.query(
            question=question,
            limit=limit,
            source_filter=source_filter,
            year_filter=year_filter,
            generate_answer=False
        )
        
        print(f"\n📄 Found {len(result['documents'])} relevant documents:")
        print("-" * 50)
        
        for i, doc in enumerate(result['documents'], 1):
            print(f"\n{i}. {doc['title']}")
            print(f"   📚 Source: {doc['source']}")
            print(f"   📊 Score: {doc['score']:.3f}")
            if doc['year']:
                print(f"   📅 Year: {doc['year']}")
            if doc['citations']:
                print(f"   📈 Citations: {doc['citations']}")
            if doc['authors']:
                print(f"   👥 Authors: {', '.join(doc['authors'][:3])}")
            print(f"   📝 Abstract: {doc['abstract'][:300]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return None


def main():
    """Main function for custom queries."""
    print("🏥 Medical Research RAG System - Custom Queries")
    print("=" * 60)
    
    # Initialize RAG system
    try:
        rag = PersistentMedicalRAG()
        print("✅ RAG system initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return
    
    # Check collection status
    info = rag.get_collection_info()
    if 'error' in info:
        print(f"❌ Collection error: {info['error']}")
        return
    
    print(f"📊 Collection has {info.get('vectors_count', 0)} documents")
    
    # Example queries you can try
    print("\n💡 Example queries you can try:")
    print("- What are the symptoms of hypoxaemia in children?")
    print("- How is oxygen therapy administered?")
    print("- What are the risk factors for pneumonia?")
    print("- pubmed What are the clinical signs of respiratory distress?")
    print("- semantic What are the latest treatments for COVID-19?")
    print("- year:2024 What are recent advances in medical research?")
    print("\n🔧 Filter options:")
    print("- Add 'pubmed' to search only PubMed papers")
    print("- Add 'semantic' to search only Semantic Scholar papers")
    print("- Add 'year:2024' to filter by year")
    
    # Run some example custom queries
    print("\n" + "="*60)
    print("🧪 Running some example custom queries...")
    print("="*60)
    
    # Example 1: General medical query
    run_custom_query(rag, "What are the treatment options for respiratory infections?")
    
    # Example 2: PubMed only
    run_custom_query(rag, "pubmed What are the clinical signs of hypoxaemia?", source_filter="pubmed")
    
    # Example 3: Semantic Scholar only
    run_custom_query(rag, "semantic What are the latest AI applications in healthcare?", source_filter="semantic_scholar")
    
    # Example 4: Recent papers
    run_custom_query(rag, "year:2024 What are the latest developments in precision medicine?", year_filter=2024)
    
    # Example 5: COVID-19 research
    run_custom_query(rag, "What are the latest treatments for COVID-19?")
    
    print(f"\n🎉 Custom query examples completed!")
    print(f"\n💡 To run your own queries, modify the queries in this script or create new ones!")


if __name__ == "__main__":
    main()
