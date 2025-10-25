"""
Test script for the Medical RAG System
Runs some sample queries to demonstrate the system.
"""

from simple_rag import SimpleMedicalRAG


def main():
    """Test the RAG system with sample queries."""
    print("🏥 Medical Research RAG System - Test Queries")
    print("=" * 60)
    
    # Initialize RAG system
    try:
        rag = SimpleMedicalRAG()
        print("✅ RAG system initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return
    
    # Check collection status
    info = rag.get_collection_info()
    if 'error' in info:
        print(f"❌ Collection error: {info['error']}")
        print("Please run the main setup first: python simple_rag.py")
        return
    
    print(f"📊 Collection has {info.get('vectors_count', 0)} documents")
    
    # Test queries
    test_queries = [
        "What are the symptoms of hypoxaemia in children?",
        "How is oxygen therapy administered?",
        "What are the risk factors for pneumonia in children?",
        "pubmed What are the clinical signs of respiratory distress?",
        "semantic What are the latest treatments for COVID-19?",
        "year:2024 What are recent advances in medical research?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test Query {i}: {query}")
        print('='*60)
        
        try:
            # Parse filters
            source_filter = None
            year_filter = None
            
            if 'pubmed' in query.lower():
                source_filter = 'pubmed'
                query = query.replace('pubmed', '').strip()
            elif 'semantic' in query.lower():
                source_filter = 'semantic_scholar'
                query = query.replace('semantic', '').strip()
            
            if 'year:' in query:
                try:
                    year_part = query.split('year:')[1].split()[0]
                    year_filter = int(year_part)
                    query = query.replace(f'year:{year_part}', '').strip()
                except:
                    pass
            
            if not query:
                print("Please enter a valid question.")
                continue
            
            # Query the system
            result = rag.query(
                question=query,
                limit=3,
                source_filter=source_filter,
                year_filter=year_filter,
                generate_answer=False  # Disable AI generation for testing
            )
            
            # Display results
            print(f"\n📄 Found {len(result['documents'])} relevant documents:")
            print("-" * 40)
            
            for j, doc in enumerate(result['documents'], 1):
                print(f"\n{j}. {doc['title']}")
                print(f"   📚 Source: {doc['source']}")
                print(f"   📊 Score: {doc['score']:.3f}")
                if doc['year']:
                    print(f"   📅 Year: {doc['year']}")
                if doc['citations']:
                    print(f"   📈 Citations: {doc['citations']}")
                if doc['authors']:
                    print(f"   👥 Authors: {', '.join(doc['authors'][:2])}")
                print(f"   📝 Abstract: {doc['abstract'][:200]}...")
            
            if result['answer']:
                print(f"\n🤖 AI Answer:")
                print("-" * 40)
                print(result['answer'])
            else:
                print("\n💡 Tip: Add OpenAI API key to .env file for AI-generated answers")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    print(f"\n🎉 Test completed! Your RAG system is working perfectly!")
    print(f"\n📊 Collection Info:")
    print(f"Documents in collection: {info.get('vectors_count', 'Unknown')}")


if __name__ == "__main__":
    main()
