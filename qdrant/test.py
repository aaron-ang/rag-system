"""
Test script for the Qdrant-based Medical RAG System.
Tests both SentenceTransformer and TF-IDF backends.
"""

import argparse


def test_rag_system(rag_system, backend_name: str):
    """Test the RAG system with sample queries."""
    print(f"🏥 Medical Research RAG System - Test Suite ({backend_name})")
    print("=" * 60)

    # Check collection status
    info = rag_system.get_collection_info()
    if "error" in info:
        print(f"❌ Collection error: {info['error']}")
        print(
            f"Please run the main setup first: python -m qdrant.{backend_name.lower()}"
        )
        return False

    print(f"📊 Collection has {info.get('vectors_count', 0)} documents")

    # Test queries
    test_queries = [
        {
            "description": "Basic medical query",
            "query": "What are the symptoms of hypoxaemia in children?",
            "filters": {},
        },
        {
            "description": "Treatment query",
            "query": "How is oxygen therapy administered?",
            "filters": {},
        },
        {
            "description": "Risk factors query",
            "query": "What are the risk factors for pneumonia in children?",
            "filters": {},
        },
        {
            "description": "PubMed-filtered query",
            "query": "What are the clinical signs of respiratory distress?",
            "filters": {"source_filter": "pubmed"},
        },
        {
            "description": "Semantic Scholar-filtered query",
            "query": "What are the latest treatments for COVID-19?",
            "filters": {"source_filter": "semantic_scholar"},
        },
        {
            "description": "Year-filtered query",
            "query": "What are recent advances in medical research?",
            "filters": {"year_filter": 2024},
        },
    ]

    passed = 0
    failed = 0

    for i, test in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"Test {i}/{len(test_queries)}: {test['description']}")
        print(f"Query: {test['query']}")
        if test["filters"]:
            print(f"Filters: {test['filters']}")
        print("=" * 60)

        try:
            # Query the system
            result = rag_system.query(
                question=test["query"],
                limit=3,
                generate_answer=False,  # Disable AI generation for testing
                **test["filters"],
            )

            # Check results
            if result["documents"]:
                print(f"✅ Test {i} PASSED")
                print(f"\n📄 Found {len(result['documents'])} relevant documents:")
                print("-" * 40)

                for j, doc in enumerate(result["documents"], 1):
                    print(f"\n{j}. {doc['title'][:70]}...")
                    print(f"   📚 Source: {doc['source']}")
                    print(f"   📊 Score: {doc['score']:.3f}")
                    if doc["year"]:
                        print(f"   📅 Year: {doc['year']}")
                    if doc["citations"]:
                        print(f"   📈 Citations: {doc['citations']}")
                    if doc["authors"]:
                        print(f"   👥 Authors: {', '.join(doc['authors'][:2])}")

                passed += 1
            else:
                print(f"❌ Test {i} FAILED: No results found")
                failed += 1

        except Exception as e:
            print(f"❌ Test {i} FAILED: {e}")
            failed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}")
    print(f"✅ Passed: {passed}/{len(test_queries)}")
    print(f"❌ Failed: {failed}/{len(test_queries)}")
    print("\n📊 Collection Info:")
    print(f"Documents in collection: {info.get('vectors_count', 'Unknown')}")

    if failed == 0:
        print("\n🎉 All tests passed! Your RAG system is working perfectly!")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return False


def test_backend(backend: str):
    """Test a specific backend."""
    print(f"🚀 Testing {backend} backend...")

    # Initialize RAG system
    if backend == "sentence_transformer":
        from qdrant.sentence_transformer import SentenceTransformerRAG

        rag_system = SentenceTransformerRAG()
        backend_name = "SentenceTransformer"
    else:
        from qdrant.tfidf import TfidfRAG

        rag_system = TfidfRAG()
        backend_name = "TF-IDF"

    print(f"✅ {backend_name} backend initialized!")

    # Run tests
    return test_rag_system(rag_system, backend_name)


def test_both_backends():
    """Test both backends sequentially."""
    print("🧪 Testing Both Backends")
    print("=" * 60)

    results = {}

    # Test SentenceTransformer
    print("\n📝 Part 1: Testing SentenceTransformer Backend")
    print("-" * 60)
    results["sentence_transformer"] = test_backend("sentence_transformer")

    # Test TF-IDF
    print("\n📝 Part 2: Testing TF-IDF Backend")
    print("-" * 60)
    results["tfidf"] = test_backend("tfidf")

    # Final summary
    print("\n" + "=" * 60)
    print("Final Test Summary")
    print("=" * 60)
    print(
        f"SentenceTransformer: {'✅ PASSED' if results['sentence_transformer'] else '❌ FAILED'}"
    )
    print(f"TF-IDF: {'✅ PASSED' if results['tfidf'] else '❌ FAILED'}")

    if all(results.values()):
        print("\n🎉 All backends passed testing!")
    else:
        print("\n⚠️  Some backends failed. Please check the errors above.")


def compare_backends():
    """Compare results between SentenceTransformer and TF-IDF."""
    print("📊 Comparing SentenceTransformer vs TF-IDF Backends")
    print("=" * 60)

    # Initialize both backends
    from qdrant.sentence_transformer import SentenceTransformerRAG
    from qdrant.tfidf import TfidfRAG

    st_rag = SentenceTransformerRAG()
    tfidf_rag = TfidfRAG()

    # Test query
    test_query = "What are the symptoms of hypoxaemia in children?"

    print(f"\nTest Query: {test_query}")
    print("=" * 60)

    # SentenceTransformer results
    print("\n📝 SentenceTransformer Results:")
    print("-" * 40)
    st_result = st_rag.query(test_query, limit=5, generate_answer=False)
    for i, doc in enumerate(st_result["documents"], 1):
        print(f"{i}. {doc['title'][:60]}... (Score: {doc['score']:.3f})")

    # TF-IDF results
    print("\n📝 TF-IDF Results:")
    print("-" * 40)
    tfidf_result = tfidf_rag.query(test_query, limit=5, generate_answer=False)
    for i, doc in enumerate(tfidf_result["documents"], 1):
        print(f"{i}. {doc['title'][:60]}... (Score: {doc['score']:.3f})")

    print("\n💡 Note: Scores are not directly comparable between backends.")
    print("SentenceTransformer typically provides more semantic understanding.")
    print("TF-IDF is faster but may miss semantic relationships.")


def main():
    """Main entry point for tests."""
    parser = argparse.ArgumentParser(description="Medical RAG System - Test Suite")
    parser.add_argument(
        "--backend",
        choices=["sentence_transformer", "tfidf", "both"],
        default="both",
        help="Choose which backend to test (default: both)",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare results between backends"
    )

    args = parser.parse_args()

    if args.compare:
        compare_backends()
    elif args.backend == "both":
        test_both_backends()
    else:
        test_backend(args.backend)


if __name__ == "__main__":
    main()
