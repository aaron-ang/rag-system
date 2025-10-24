"""
CLI interface for SciNCL-based RAG system.
"""

import argparse
import os

from scincl.utils import load_artifacts, create_artifacts


def ingest_data(args):
    print("Starting SciNCL data ingestion...")

    try:
        # Check if artifacts exist and handle force flag
        if (
            os.path.exists(os.path.join(args.output_dir, "faiss_index.bin"))
            and not args.force
        ):
            print(f"Artifacts already exist in {args.output_dir}")
            print("Skipping ingestion. Use --force to re-ingest:")
            print("  python -m scincl.cli ingest --force")
            return

        if args.force and os.path.exists(args.output_dir):
            import shutil

            print(f"Removing existing artifacts from {args.output_dir}...")
            shutil.rmtree(args.output_dir)

        create_artifacts(
            model_name=args.model,
            index_type=args.index_type,
            artifacts_dir=args.output_dir,
        )

        print("Data ingestion completed successfully!")

    except Exception as e:
        print(f"Error during ingestion: {e}")
        return


def query_system(args):
    print(f"Loading artifacts from {args.artifacts_dir}...")

    try:
        retrieval, documents = load_artifacts(artifacts_dir=args.artifacts_dir)

        print(f"System ready with {len(documents)} documents")

        results = retrieval.retrieve_similar_documents(args.query, k=args.k)

        print(f"\nRetrieved {len(results)} documents:")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            doc = result.document
            print(f"\n{i}. {doc.title}")
            print(f"   Score: {result.sim_score:.3f}")
            print(f"   Source: {doc.source}")
            print(f"   Abstract: {str(doc.abstract)}")

    except Exception as e:
        print(f"Error loading artifacts: {e}")
        print("Please run 'python -m scincl.cli ingest' first to create artifacts.")


def main():
    parser = argparse.ArgumentParser(description="SciNCL-based RAG system")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data using SciNCL")
    ingest_parser.add_argument(
        "--model", default="malteos/scincl", help="SciNCL model name"
    )
    ingest_parser.add_argument(
        "--index-type",
        default="flat",
        choices=["flat", "ivf", "hnsw"],
        help="FAISS index type",
    )
    ingest_parser.add_argument(
        "--output-dir",
        default="data/scincl_artifacts",
        help="Output directory for artifacts",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if artifacts exist",
    )
    ingest_parser.set_defaults(func=ingest_data)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the retrieval system")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument(
        "--artifacts-dir",
        default="data/scincl_artifacts",
        help="Directory containing artifacts",
    )
    query_parser.add_argument(
        "--k", type=int, default=5, help="Number of documents to retrieve"
    )
    query_parser.set_defaults(func=query_system)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
