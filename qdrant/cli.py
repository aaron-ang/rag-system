"""
Interactive query interface for the Qdrant-based Medical RAG System.
Supports both SentenceTransformer and TF-IDF backends.
"""

import argparse
from dotenv import load_dotenv

load_dotenv()


def parse_filters(query: str):
    """
    Parse filter keywords from query string.

    Returns:
        tuple: (cleaned_query, source_filter, year_filter)
    """
    source_filter = None
    year_filter = None

    if "pubmed" in query.lower():
        source_filter = "pubmed"
        query = query.replace("pubmed", "").replace("PubMed", "").strip()
    elif "semantic" in query.lower():
        source_filter = "semantic_scholar"
        query = query.replace("semantic", "").replace("Semantic", "").strip()

    if "year:" in query:
        try:
            year_part = query.split("year:")[1].split()[0]
            year_filter = int(year_part)
            query = query.replace(f"year:{year_part}", "").strip()
        except Exception:
            pass

    return query, source_filter, year_filter


def print_results(result: dict):
    """Pretty print search results."""
    print(f"\n📄 Found {len(result['documents'])} relevant documents:")
    print("-" * 80)

    for i, doc in enumerate(result["documents"], 1):
        print(f"\n{i}. {doc['title']}")
        print(f"   📚 Source: {doc['source']}")
        print(f"   📊 Score: {doc['score']:.3f}")

        # Show original paper IDs and URLs
        if doc.get("pmid"):
            print(f"   🆔 PMID: {doc['pmid']}")
        if doc.get("paper_id"):
            print(f"   🆔 Paper ID: {doc['paper_id']}")
        if doc.get("url"):
            print(f"   🔗 URL: {doc['url']}")

        if doc["year"]:
            print(f"   📅 Year: {doc['year']}")
        if doc["citations"]:
            print(f"   📈 Citations: {doc['citations']}")
        if doc["authors"]:
            print(f"   👥 Authors: {', '.join(doc['authors'][:2])}")
        print(f"   📝 Abstract: {doc['abstract'][:200]}...")

    # Display generated answer
    if result.get("answer"):
        print("\n🤖 AI Answer:")
        print("-" * 80)
        print(result["answer"])
    else:
        print("\n💡 Tip: Add OpenAI API key to .env file for AI-generated answers")


def interactive_mode(rag_system, backend_name: str):
    """Interactive query interface."""
    print(f"\n{'=' * 80}")
    print(f"🏥 Medical Research RAG System ({backend_name})")
    print(f"{'=' * 80}")

    # Check collection status
    info = rag_system.get_collection_info()
    if "error" in info:
        print(f"❌ Collection error: {info['error']}")
        print(f"Please run the setup first: python -m qdrant.{backend_name.lower()}")
        return

    print(f"📊 Collection has {info.get('vectors_count', 0)} documents")

    print("\n💡 Example queries:")
    print("- What are the symptoms of hypoxaemia in children?")
    print("- How is oxygen therapy administered?")
    print("- What are the risk factors for pneumonia?")
    print("\n🔧 Filter options:")
    print("- Add 'pubmed' to search only PubMed papers")
    print("- Add 'semantic' to search only Semantic Scholar papers")
    print("- Add 'year:2024' to filter by year")
    print("- Type 'quit' to exit")

    while True:
        print(f"\n{'=' * 80}")
        query = input("\n🔍 Enter your question: ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break

        if not query:
            continue

        # Parse filters
        query, source_filter, year_filter = parse_filters(query)

        if not query:
            print("Please enter a valid question.")
            continue

        try:
            # Query the system
            result = rag_system.query(
                question=query,
                limit=5,
                source_filter=source_filter,
                year_filter=year_filter,
                generate_answer=True,
            )

            # Display results
            print_results(result)

        except Exception as e:
            print(f"❌ Error processing query: {e}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Medical RAG System - Interactive Query Interface"
    )
    parser.add_argument(
        "--backend",
        choices=["sentence_transformer", "tfidf"],
        default="sentence_transformer",
        help="Choose embedding backend (default: sentence_transformer)",
    )
    parser.add_argument("--query", type=str, help="Single query mode (non-interactive)")
    parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default: 5)"
    )

    args = parser.parse_args()

    # Initialize the appropriate backend
    print(f"🚀 Initializing {args.backend} backend...")

    if args.backend == "sentence_transformer":
        from qdrant.sentence_transformer import SentenceTransformerRAG

        rag_system = SentenceTransformerRAG()
        backend_name = "SentenceTransformer"
    else:
        from qdrant.tfidf import TfidfRAG

        rag_system = TfidfRAG()
        backend_name = "TF-IDF"

    print(f"✅ {backend_name} backend initialized!")

    # Single query mode or interactive mode
    if args.query:
        # Single query mode
        query, source_filter, year_filter = parse_filters(args.query)
        result = rag_system.query(
            question=query,
            limit=args.limit,
            source_filter=source_filter,
            year_filter=year_filter,
            generate_answer=True,
        )
        print_results(result)
    else:
        # Interactive mode
        interactive_mode(rag_system, backend_name)


if __name__ == "__main__":
    main()
