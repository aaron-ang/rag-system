"""
Interactive query interface for the Medical RAG System
"""

import os
from dotenv import load_dotenv
from simple_rag import SimpleMedicalRAG

load_dotenv()


def main():
    """Interactive query interface."""
    print("🏥 Medical Research RAG System")
    print("=" * 50)
    
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
        print("Please run the main setup first: python rag_system.py")
        return
    
    print(f"📊 Collection has {info.get('vectors_count', 0)} documents")
    
    print("\n💡 Example queries:")
    print("- What are the symptoms of hypoxaemia in children?")
    print("- How is oxygen therapy administered?")
    print("- What are the risk factors for pneumonia?")
    print("- Search only PubMed papers: 'pubmed'")
    print("- Search only Semantic Scholar: 'semantic'")
    print("- Filter by year: 'year:2024'")
    print("- Type 'quit' to exit")
    
    while True:
        print("\n" + "="*50)
        query = input("\n🔍 Enter your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            continue
        
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
        
        try:
            # Query the system
            result = rag.query(
                question=query,
                limit=5,
                source_filter=source_filter,
                year_filter=year_filter,
                generate_answer=True
            )
            
            # Display results
            print(f"\n📄 Found {len(result['documents'])} relevant documents:")
            print("-" * 30)
            
            for i, doc in enumerate(result['documents'], 1):
                print(f"\n{i}. {doc['title']}")
                print(f"   📚 Source: {doc['source']}")
                print(f"   📊 Score: {doc['score']:.3f}")
                if doc['year']:
                    print(f"   📅 Year: {doc['year']}")
                if doc['citations']:
                    print(f"   📈 Citations: {doc['citations']}")
                if doc['authors']:
                    print(f"   👥 Authors: {', '.join(doc['authors'][:2])}")
                print(f"   📝 Abstract: {doc['abstract'][:200]}...")
            
            # Display generated answer
            if result['answer']:
                print(f"\n🤖 AI Answer:")
                print("-" * 30)
                print(result['answer'])
            else:
                print("\n⚠️  No AI answer generated (OpenAI API key not configured)")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")


if __name__ == "__main__":
    main()
