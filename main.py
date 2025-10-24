"""
Main RAG system application.
"""

import textwrap
import pandas as pd
from scincl import SciNCLRetrieval, load_or_create_artifacts
from scincl.metrics import recall_at_k, precision_at_k, compute_ndcg_at_k



def interactive_query(retrieval: SciNCLRetrieval):
    """Enhanced interactive query interface with improved formatting."""
    print("\n" + "=" * 60)
    print("SciNCL RAG System - Interactive Mode")
    print("=" * 60)
    print("💡 Enter your queries below. Type 'quit' to exit or 'help' for info.")
    print("=" * 60)

    help_text = (
        "\n📋 Available commands:\n"
        "  🔍 Type any question to search the document collection\n"
        "  🚪 'quit' or 'exit' to stop the program\n"
        "  ❓ 'help' to show this help message\n"
        "  📊 'stats' to show system statistics\n"
    )

    while True:
        try:
            query = input("\n🔍 Query: ").strip()

            if query.lower() in ["quit", "exit"]:
                print("\n👋 Goodbye!")
                break
            if query.lower() == "help":
                print(help_text)
                continue
            if query.lower() == "stats":
                print("\n📊 System Statistics:")
                print(f"   📚 Total documents: {len(retrieval.documents)}")
                print("   🔍 Index type: FAISS")
                print("   🤖 Model: SciNCL")
                continue
            if not query:
                print("⚠️  Please enter a query.")
                continue

            print(f"\n🔍 Searching for: '{query}'")
            print("⏳ Processing...")

            results = retrieval.retrieve_similar_documents(query, k=5)

            if not results:
                print("❌ No results found.")
                continue

            print(f"\n✅ Found {len(results)} relevant document(s):")
            print("=" * 80)

            for i, result in enumerate(results, 1):
                doc = result.document
                score = result.sim_score
                title = doc.title
                source = doc.source
                abstract = doc.abstract
                score_indicator = _score_indicator(score)

                wrapped_title = textwrap.fill(
                    title,
                    width=75,
                    initial_indent=f"📄 {i}. ",
                    subsequent_indent="      ",
                )
                wrapped_abstract = textwrap.fill(
                    abstract[:500] + "...",
                    width=75,
                    initial_indent="   📝 ",
                    subsequent_indent="      ",
                )

                print(f"\n{wrapped_title}")
                print(f"   {score_indicator} Score: {score:.3f} | 📂 Source: {source}")
                print(wrapped_abstract)
                print("\n" + "-" * 80)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🔄 Please try again.")


def demo_queries(retrieval: SciNCLRetrieval):
    """Enhanced demo queries with improved formatting."""
    print("\n" + "=" * 60)
    print("SciNCL RAG System - Demo Queries")
    print("=" * 60)

    queries = [
        "machine learning in medical diagnosis",
        "cancer treatment methods",
        "COVID-19 research",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Demo Query {i}: '{query}'")
        print("⏳ Processing...")
        print("-" * 60)

        try:
            results = retrieval.retrieve_similar_documents(query, k=3)
            if results:
                print(f"✅ Found {len(results)} relevant document(s):")
                for idx, result in enumerate(results, 1):
                    doc = result.document
                    title = doc.title
                    score = result.sim_score
                    source = doc.source
                    score_indicator = _score_indicator(score)
                    doc_id = result.doc_id
                    print(f"   📄 {idx}. {title}")
                    print(f"      {score_indicator} Score: {score:.3f} | 📂 {source}")
            else:
                print("❌ No results found.")
        except Exception as e:
            print(f"❌ Error processing query: {e}")

    print("\n" + "=" * 60)

def _score_indicator(score: float):
    if score > 0.8:
        return "🟢"
    elif score > 0.6:
        return "🟡"
    else:
        return "🔴"
    
def test_set_queries(retrieval: SciNCLRetrieval):
    """
    Runs retrieval on test queries from CSV and returns retrieved and ground-truth docs.
    """
    df = pd.read_csv("data/retrieval_requests_with_ids.csv")

    retrieved_docs = []
    ground_truth_docs = []

    print("\n" + "=" * 60)
    print("SciNCL RAG System - Test Set Evaluation")
    print("=" * 60)

    for i, row in df.iterrows():
        query_id = row["query_id"]
        query = row["query_text"]
        gt_doc_ids = set(str(row["matching_paper_ids"]).split(","))  # assumes pipe-separated IDs


        print(f"\n🔍 Query {query_id}: '{query}'")
        print("⏳ Retrieving...")
        print("-" * 60)

        try:
            results = retrieval.retrieve_similar_documents(query, k=2)
            if results:
                retrieved_ids = [result.doc_id for result in results]
                formatted_retrieved_ids = [doc_id.split("_")[1] if "pubmed_" in doc_id else doc_id for doc_id in retrieved_ids]

                print(f"Expected documents: {gt_doc_ids}")
                print(f"✅ Retrieved documents: {formatted_retrieved_ids}")
            else:
                retrieved_ids = []
                print("❌ No results found.")

            retrieved_docs.append(formatted_retrieved_ids)
            ground_truth_docs.append(gt_doc_ids)

        except Exception as e:
            print(f"❌ Error processing query '{query}': {e}")
            retrieved_docs.append([])
            ground_truth_docs.append(gt_doc_ids)

    print("\n" + "=" * 60)
    print("✅ Retrieval completed.")
    print("=" * 60)

    print("Recall@3:", recall_at_k(retrieved_docs, ground_truth_docs, k=2))
    print("Precision@3:", precision_at_k(retrieved_docs, ground_truth_docs, k=2))
    print("Mean nDCG@3:", compute_ndcg_at_k(retrieved_docs, ground_truth_docs, k=2))


def main():
    """Main application entry point with enhanced formatting."""
    print("\n" + "=" * 60)
    print("🤖 SciNCL-based RAG System")
    print("=" * 60)
    print("📚 Medical Research Paper Retrieval System")
    print("=" * 60)

    try:
        retrieval = load_or_create_artifacts()

        # print("\n🎯 Running demo queries...")
        # demo_queries(retrieval)

        # print("\n🚀 Starting interactive mode...")
        # interactive_query(retrieval)

        test_set_queries(retrieval)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("🔧 Please check your data files and try again.")
        print("💡 Make sure to run 'uv run download.py' first if you haven't already.")


if __name__ == "__main__":
    main()
