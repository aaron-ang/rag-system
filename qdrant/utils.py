"""
Utility functions for inspecting and debugging the Qdrant database.
Includes functions to check original IDs, database structure, and display paper IDs.
"""

import json
from typing import Optional


def check_original_ids(rag_system, source: str = "both"):
    """
    Check if original paper IDs are preserved in the database.

    Args:
        rag_system: The RAG system instance (SentenceTransformerRAG or TfidfRAG)
        source: 'pubmed', 'semantic_scholar', or 'both' (default: 'both')
    """
    print("🔍 Checking Original Paper IDs in Document Structure")
    print("=" * 60)

    # Check collection info
    info = rag_system.get_collection_info()
    if "error" in info:
        print(f"❌ Collection error: {info['error']}")
        return

    print(f"📊 Collection: {info.get('name', 'Unknown')}")
    print(f"📊 Vectors count: {info.get('vectors_count', 'Unknown')}")

    # Check PubMed
    if source in ["pubmed", "both"]:
        print("\n🔍 Searching for PubMed sample...")
        try:
            pubmed_results = rag_system.search_similar_documents(
                "hypoxaemia children", limit=1, source_filter="pubmed"
            )

            if pubmed_results:
                doc = pubmed_results[0]
                print(f"\n📄 PubMed Sample:")
                print("-" * 40)
                print(f"🆔 Point ID: {doc['id']}")
                print(f"📚 Source: {doc['source']}")
                print(f"📝 Title: {doc['title'][:80]}...")

                # Check if PMID is in the payload
                print(f"\n🔍 Checking for PMID in payload...")
                if doc.get("pmid"):
                    print(f"✅ PMID found: {doc['pmid']}")
                    print(f"🔗 URL: https://pubmed.ncbi.nlm.nih.gov/{doc['pmid']}/")
                else:
                    print("❌ No PMID found in payload")

                print(f"\n📋 Available payload keys:")
                for key in doc.keys():
                    print(f"  - {key}")
            else:
                print("❌ No PubMed documents found")

        except Exception as e:
            print(f"❌ Error fetching PubMed sample: {e}")

    # Check Semantic Scholar
    if source in ["semantic_scholar", "both"]:
        print("\n" + "=" * 60)
        print("🔍 Searching for Semantic Scholar sample...")
        try:
            semantic_results = rag_system.search_similar_documents(
                "nanotechnology healthcare", limit=1, source_filter="semantic_scholar"
            )

            if semantic_results:
                doc = semantic_results[0]
                print(f"\n📄 Semantic Scholar Sample:")
                print("-" * 40)
                print(f"🆔 Point ID: {doc['id']}")
                print(f"📚 Source: {doc['source']}")
                print(f"📝 Title: {doc['title'][:80]}...")

                # Check if paper_id is in the payload
                print(f"\n🔍 Checking for paper_id in payload...")
                if doc.get("paper_id"):
                    print(f"✅ Paper ID found: {doc['paper_id']}")
                    print(
                        f"🔗 URL: https://www.semanticscholar.org/paper/{doc['paper_id']}"
                    )
                else:
                    print("❌ No paper_id found in payload")

                if doc.get("url"):
                    print(f"🌐 Direct URL: {doc['url']}")

                print(f"\n📋 Available payload keys:")
                for key in doc.keys():
                    print(f"  - {key}")
            else:
                print("❌ No Semantic Scholar documents found")

        except Exception as e:
            print(f"❌ Error fetching Semantic Scholar sample: {e}")

    print(f"\n✅ Original ID check completed!")
    print(
        f"\n💡 Note: Original paper IDs should be stored in the payload for easy access"
    )


def inspect_database_structure(rag_system):
    """
    Inspect the structure of data in the Qdrant database.

    Args:
        rag_system: The RAG system instance (SentenceTransformerRAG or TfidfRAG)
    """
    print("🔍 Checking Qdrant Database Structure")
    print("=" * 50)

    # Check collection info
    info = rag_system.get_collection_info()
    if "error" in info:
        print(f"❌ Collection error: {info['error']}")
        return

    print(f"📊 Collection: {info.get('name', 'Unknown')}")
    print(f"📊 Vectors count: {info.get('vectors_count', 'Unknown')}")

    # Search for a sample document
    print("\n🔍 Searching for a sample document...")
    try:
        # Get a sample document
        sample_results = rag_system.search_similar_documents(
            "medical research", limit=1
        )

        if sample_results:
            doc = sample_results[0]
            print("\n📄 Sample Document Structure:")
            print("-" * 40)
            print(f"🆔 Point ID: {doc['id']}")
            print(f"📊 Score: {doc['score']}")
            print(f"📚 Source: {doc['source']}")
            print(f"📝 Title: {doc['title'][:100]}...")
            print(f"📅 Year: {doc['year']}")
            print(f"📈 Citations: {doc['citations']}")
            print(f"👥 Authors: {doc['authors']}")

            print(f"\n📋 Full Payload Structure:")
            print("-" * 40)
            print(
                json.dumps(
                    {
                        "id": doc["id"],
                        "source": doc["source"],
                        "title": doc["title"][:50] + "...",
                        "text": doc["text"][:100] + "...",
                        "abstract": doc["abstract"][:100] + "...",
                        "year": doc["year"],
                        "citations": doc["citations"],
                        "authors": doc["authors"],
                        "pmid": doc.get("pmid"),
                        "paper_id": doc.get("paper_id"),
                        "url": doc.get("url"),
                        "metadata": doc["metadata"],
                    },
                    indent=2,
                )
            )

            # Check if original IDs are preserved
            print(f"\n🔍 Original Paper IDs:")
            print("-" * 40)
            if doc["source"] == "pubmed":
                if doc.get("pmid"):
                    print(f"📄 PubMed ID: {doc['pmid']}")
                else:
                    print("❌ No PMID found in payload")
            elif doc["source"] == "semantic_scholar":
                if doc.get("paper_id"):
                    print(f"📄 Semantic Scholar ID: {doc['paper_id']}")
                else:
                    print("❌ No paper_id found in payload")

            print(f"\n📊 Metadata Keys:")
            print("-" * 40)
            for key, value in doc["metadata"].items():
                print(f"  {key}: {type(value).__name__}")
                if key == "stats" and isinstance(value, dict):
                    print(f"    Stats keys: {list(value.keys())}")
        else:
            print("❌ No documents found in collection")

    except Exception as e:
        print(f"❌ Error fetching sample document: {e}")

    print(f"\n✅ Database structure check completed!")


def show_paper_ids(rag_system, queries: Optional[list] = None):
    """
    Show original paper IDs in search results.

    Args:
        rag_system: The RAG system instance (SentenceTransformerRAG or TfidfRAG)
        queries: Optional list of (source_filter, query) tuples. If None, uses default queries.
    """
    print("🔍 Showing Original Paper IDs in Search Results")
    print("=" * 60)

    if queries is None:
        queries = [
            ("pubmed", "What are the clinical signs of hypoxaemia?"),
            ("semantic_scholar", "What are the latest AI applications in healthcare?"),
        ]

    for source_filter, query in queries:
        print(f"\n{'=' * 80}")
        print(f"🔍 Query: {query}")
        print(f"📚 Source Filter: {source_filter}")
        print("=" * 80)

        try:
            results = rag_system.search_similar_documents(
                query, limit=3, source_filter=source_filter
            )

            if results:
                print(f"\n📄 Found {len(results)} documents:")
                print("-" * 50)

                for i, doc in enumerate(results, 1):
                    print(f"\n{i}. {doc['title'][:70]}...")
                    print(f"   📚 Source: {doc['source']}")
                    print(f"   📊 Score: {doc['score']:.3f}")

                    # Show original paper IDs
                    if doc["source"] == "pubmed" and doc.get("pmid"):
                        print(f"   🆔 PMID: {doc['pmid']}")
                        print(
                            f"   🔗 URL: https://pubmed.ncbi.nlm.nih.gov/{doc['pmid']}/"
                        )
                    elif doc["source"] == "semantic_scholar" and doc.get("paper_id"):
                        print(f"   🆔 Paper ID: {doc['paper_id']}")
                        print(
                            f"   🔗 URL: https://www.semanticscholar.org/paper/{doc['paper_id']}"
                        )

                    if doc.get("url"):
                        print(f"   🌐 Direct URL: {doc['url']}")

                    if doc["year"]:
                        print(f"   📅 Year: {doc['year']}")
                    if doc["citations"]:
                        print(f"   📈 Citations: {doc['citations']}")
                    if doc["authors"]:
                        print(f"   👥 Authors: {', '.join(doc['authors'][:2])}")
            else:
                print("❌ No documents found")

        except Exception as e:
            print(f"❌ Error processing query: {e}")

    print(f"\n🎉 Paper ID demonstration completed!")
    print(
        f"\n💡 Now you can access original paper IDs for referencing back to source databases!"
    )


def get_collection_stats(rag_system):
    """
    Get comprehensive statistics about the collection.

    Args:
        rag_system: The RAG system instance (SentenceTransformerRAG or TfidfRAG)
    """
    print("📊 Collection Statistics")
    print("=" * 50)

    info = rag_system.get_collection_info()
    if "error" in info:
        print(f"❌ Collection error: {info['error']}")
        return

    print(f"📚 Collection Name: {info.get('name', 'Unknown')}")
    print(f"📊 Total Documents: {info.get('vectors_count', 'Unknown')}")
    print(f"✅ Status: {info.get('status', 'Unknown')}")

    # Get counts by source
    try:
        pubmed_results = rag_system.qdrant_client.count(
            collection_name=rag_system.collection_name,
            count_filter={"must": [{"key": "source", "match": {"value": "pubmed"}}]},
        )
        semantic_results = rag_system.qdrant_client.count(
            collection_name=rag_system.collection_name,
            count_filter={
                "must": [{"key": "source", "match": {"value": "semantic_scholar"}}]
            },
        )

        print(f"\n📑 Documents by Source:")
        print(f"  - PubMed: {pubmed_results.count}")
        print(f"  - Semantic Scholar: {semantic_results.count}")
    except Exception as e:
        print(f"⚠️  Could not get source counts: {e}")

    print("\n✅ Statistics retrieved successfully!")


if __name__ == "__main__":
    print("This module provides utility functions for inspecting the Qdrant database.")
    print("\nUsage examples:")
    print("  from qdrant.sentence_transformer import SentenceTransformerRAG")
    print("  from qdrant.utils import check_original_ids, inspect_database_structure")
    print()
    print("  rag = SentenceTransformerRAG()")
    print("  check_original_ids(rag)")
    print("  inspect_database_structure(rag)")
    print("  show_paper_ids(rag)")
    print("  get_collection_stats(rag)")
