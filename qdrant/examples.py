"""
Example usage of the Qdrant-based Medical RAG System.
Demonstrates various query patterns and filtering options.
"""

import argparse


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

            # Show original paper IDs
            if doc.get('pmid'):
                print(f"   🆔 PMID: {doc['pmid']}")
            if doc.get('paper_id'):
                print(f"   🆔 Paper ID: {doc['paper_id']}")
            if doc.get('url'):
                print(f"   🔗 URL: {doc['url']}")

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


def run_examples(backend='sentence_transformer'):
    """Run example queries demonstrating the RAG system."""
    print("🏥 Medical Research RAG System - Example Queries")
    print("=" * 60)

    # Initialize RAG system
    if backend == 'sentence_transformer':
        from qdrant.sentence_transformer import SentenceTransformerRAG
        rag = SentenceTransformerRAG()
        backend_name = "SentenceTransformer"
    else:
        from qdrant.tfidf import TfidfRAG
        rag = TfidfRAG()
        backend_name = "TF-IDF"

    print(f"✅ {backend_name} backend initialized!")

    # Check collection status
    info = rag.get_collection_info()
    if 'error' in info:
        print(f"❌ Collection error: {info['error']}")
        return

    print(f"📊 Collection has {info.get('vectors_count', 0)} documents")

    # Example 1: General medical query
    print("\n" + "="*60)
    print("🧪 Example 1: General Medical Query")
    print("="*60)
    run_custom_query(rag, "What are the treatment options for respiratory infections?")

    # Example 2: PubMed only
    print("\n" + "="*60)
    print("🧪 Example 2: PubMed Only")
    print("="*60)
    run_custom_query(rag, "What are the clinical signs of hypoxaemia?", source_filter="pubmed")

    # Example 3: Semantic Scholar only
    print("\n" + "="*60)
    print("🧪 Example 3: Semantic Scholar Only")
    print("="*60)
    run_custom_query(rag, "What are the latest AI applications in healthcare?", source_filter="semantic_scholar")

    # Example 4: Recent papers (if year data available)
    print("\n" + "="*60)
    print("🧪 Example 4: Recent Papers (Year Filter)")
    print("="*60)
    run_custom_query(rag, "What are the latest developments in precision medicine?", year_filter=2024)

    # Example 5: COVID-19 research
    print("\n" + "="*60)
    print("🧪 Example 5: COVID-19 Research")
    print("="*60)
    run_custom_query(rag, "What are the latest treatments for COVID-19?")

    # Example 6: Pediatric medicine
    print("\n" + "="*60)
    print("🧪 Example 6: Pediatric Medicine")
    print("="*60)
    run_custom_query(rag, "What are the symptoms of hypoxaemia in children?")

    # Example 7: Specific medical topic with higher result count
    print("\n" + "="*60)
    print("🧪 Example 7: Broader Search (10 results)")
    print("="*60)
    run_custom_query(rag, "pneumonia treatment methods", limit=10)

    print(f"\n🎉 Example queries completed!")
    print(f"\n💡 To run your own queries, modify this script or use the CLI:")
    print(f"   python -m qdrant.cli --backend {backend}")


def custom_queries_interactive(backend='sentence_transformer'):
    """Interactive mode for custom queries."""
    print("🏥 Medical Research RAG System - Custom Query Mode")
    print("=" * 60)

    # Initialize RAG system
    if backend == 'sentence_transformer':
        from qdrant.sentence_transformer import SentenceTransformerRAG
        rag = SentenceTransformerRAG()
        backend_name = "SentenceTransformer"
    else:
        from qdrant.tfidf import TfidfRAG
        rag = TfidfRAG()
        backend_name = "TF-IDF"

    print(f"✅ {backend_name} backend initialized!")

    # Check collection status
    info = rag.get_collection_info()
    if 'error' in info:
        print(f"❌ Collection error: {info['error']}")
        return

    print(f"📊 Collection has {info.get('vectors_count', 0)} documents")

    print("\n💡 Example queries you can try:")
    print("- What are the symptoms of hypoxaemia in children?")
    print("- How is oxygen therapy administered?")
    print("- What are the risk factors for pneumonia?")
    print("\n🔧 Filter options:")
    print("- Add 'pubmed' to search only PubMed papers")
    print("- Add 'semantic' to search only Semantic Scholar papers")
    print("- Add 'year:2024' to filter by year")
    print("- Type 'quit' to exit")

    while True:
        print("\n" + "="*60)
        query = input("\n🔍 Enter your query (or 'quit' to exit): ").strip()

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
            query = query.replace('pubmed', '').replace('PubMed', '').strip()
        elif 'semantic' in query.lower():
            source_filter = 'semantic_scholar'
            query = query.replace('semantic', '').replace('Semantic', '').strip()

        if 'year:' in query:
            try:
                year_part = query.split('year:')[1].split()[0]
                year_filter = int(year_part)
                query = query.replace(f'year:{year_part}', '').strip()
            except Exception:
                pass

        if not query:
            print("Please enter a valid question.")
            continue

        run_custom_query(rag, query, source_filter=source_filter, year_filter=year_filter)


def main():
    """Main entry point for examples."""
    parser = argparse.ArgumentParser(
        description='Medical RAG System - Example Queries'
    )
    parser.add_argument(
        '--backend',
        choices=['sentence_transformer', 'tfidf'],
        default='sentence_transformer',
        help='Choose embedding backend (default: sentence_transformer)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )

    args = parser.parse_args()

    if args.interactive:
        custom_queries_interactive(backend=args.backend)
    else:
        run_examples(backend=args.backend)


if __name__ == "__main__":
    main()
