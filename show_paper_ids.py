"""
Script to show original paper IDs in search results
"""

from rag_with_persistence import PersistentMedicalRAG


def main():
    """Show original paper IDs in search results."""
    print("🔍 Showing Original Paper IDs in Search Results")
    print("=" * 60)
    
    # Initialize RAG system
    try:
        rag = PersistentMedicalRAG()
        print("✅ RAG system initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return
    
    # Test queries to show paper IDs
    test_queries = [
        ("pubmed", "What are the clinical signs of hypoxaemia?"),
        ("semantic_scholar", "What are the latest AI applications in healthcare?")
    ]
    
    for source_filter, query in test_queries:
        print(f"\n{'='*80}")
        print(f"🔍 Query: {query}")
        print(f"📚 Source Filter: {source_filter}")
        print('='*80)
        
        try:
            results = rag.search_similar_documents(query, limit=3, source_filter=source_filter)
            
            if results:
                print(f"\n📄 Found {len(results)} documents:")
                print("-" * 50)
                
                for i, doc in enumerate(results, 1):
                    print(f"\n{i}. {doc['title'][:70]}...")
                    print(f"   📚 Source: {doc['source']}")
                    print(f"   📊 Score: {doc['score']:.3f}")
                    
                    # Show original paper IDs
                    if doc['source'] == 'pubmed' and doc.get('pmid'):
                        print(f"   🆔 PMID: {doc['pmid']}")
                        print(f"   🔗 URL: https://pubmed.ncbi.nlm.nih.gov/{doc['pmid']}/")
                    elif doc['source'] == 'semantic_scholar' and doc.get('paper_id'):
                        print(f"   🆔 Paper ID: {doc['paper_id']}")
                        print(f"   🔗 URL: https://www.semanticscholar.org/paper/{doc['paper_id']}")
                    
                    if doc.get('url'):
                        print(f"   🌐 Direct URL: {doc['url']}")
                    
                    if doc['year']:
                        print(f"   📅 Year: {doc['year']}")
                    if doc['citations']:
                        print(f"   📈 Citations: {doc['citations']}")
                    if doc['authors']:
                        print(f"   👥 Authors: {', '.join(doc['authors'][:2])}")
            else:
                print("❌ No documents found")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    print(f"\n🎉 Paper ID demonstration completed!")
    print(f"\n💡 Now you can access original paper IDs for referencing back to source databases!")


if __name__ == "__main__":
    main()
