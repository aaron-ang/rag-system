"""
Main RAG system application.
Integrates SciNCL methodology for scientific document retrieval.
"""

from scincl import load_artifacts, create_artifacts


def interactive_query(retrieval):
    """Interactive query interface for the RAG system."""
    print("\n" + "=" * 60)
    print("SciNCL RAG System - Interactive Mode")
    print("=" * 60)
    print("Enter your queries below. Type 'quit' to exit.")
    print("Commands:")
    print("  - Type any question to search")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'help' for more information")
    print("=" * 60)

    while True:
        try:
            query = input("\nQuery: ").strip()

            if query.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            elif query.lower() == "help":
                print("\nAvailable commands:")
                print("  - Enter any question to search the document collection")
                print("  - 'quit' or 'exit' to stop the program")
                print("  - 'help' to show this help message")
                continue
            elif not query:
                print("Please enter a query.")
                continue

            print(f"\nSearching for: '{query}'")
            print("-" * 40)

            # Perform search
            results = retrieval.retrieve_similar_documents(query, k=5)

            if not results:
                print("No results found.")
                continue

            print(f"\nFound {len(results)} relevant documents:")
            print("=" * 60)

            for i, result in enumerate(results, 1):
                doc = result["document"]
                score = result["score"]

                print(f"\n{i}. {doc.get('title', 'No title')}...")
                print(f"   Score: {score:.3f}")
                print(f"   Source: {doc.get('source', 'unknown')}")
                print(f"   Abstract: {str(doc.get('abstract', 'No abstract'))}...")


        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")


def demo_queries(retrieval):
    """Run demo queries to showcase the system."""
    print("\n" + "=" * 60)
    print("SciNCL RAG System - Demo Queries")
    print("=" * 60)

    demo_queries = [
        "machine learning in medical diagnosis",
        "cancer treatment methods",
        "COVID-19 research",
        "neural networks in healthcare",
    ]

    for query in demo_queries:
        print(f"\nDemo Query: '{query}'")
        print("-" * 40)

        try:
            results = retrieval.retrieve_similar_documents(query, k=3)

            if results:
                print(f"Found {len(results)} relevant documents:")
                for i, result in enumerate(results, 1):
                    doc = result["document"]
                    score = result["score"]
                    print(
                        f"  {i}. {doc.get('title', 'No title')}... (Score: {score:.3f})"
                    )
            else:
                print("No results found.")

        except Exception as e:
            print(f"Error processing query: {e}")

    print("\n" + "=" * 60)


def main():
    """Main application entry point."""
    print("SciNCL-based RAG System")
    print("=" * 30)

    try:
        # Try to load existing artifacts first
        try:
            retrieval, documents = load_artifacts()
            print(f"System ready with {len(documents)} documents")
        except (FileNotFoundError, ValueError):
            print("No existing artifacts found. Creating new ones...")
            retrieval, documents = create_artifacts()
            print(f"System ready with {len(documents)} documents")

        # Run demo queries
        demo_queries(retrieval)

        # Start interactive mode
        interactive_query(retrieval)

    except Exception as e:
        print(f"Error: {e}")
        print("Please check your data files and try again.")


if __name__ == "__main__":
    main()
