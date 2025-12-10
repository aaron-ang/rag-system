"""
CLI interface for SciNCL-based RAG system.
"""

import argparse
import os
import shutil

from scincl.utils import load_or_create_artifacts, create_artifacts


def ingest_data(args):
    print("Starting SciNCL data ingestion...")

    # v1: Milvus Lite + FLAT index; v2: Milvus server + IVF index (default)
    use_v1 = args.v1

    try:
        if os.path.exists(args.output_dir):
            print(f"Removing existing artifacts from {args.output_dir}...")
            shutil.rmtree(args.output_dir)

        create_artifacts(
            model_name=args.model,
            artifacts_dir=args.output_dir,
            use_v1=use_v1,
        )

        print("Data ingestion completed successfully!")

    except Exception as e:
        print(f"Error during ingestion: {e}")
        return


def query_system(args):
    use_v1 = args.v1
    enable_llm = args.llm

    print(f"Loading artifacts from {args.artifacts_dir}...")

    try:
        retrieval = load_or_create_artifacts(
            artifacts_dir=args.artifacts_dir,
            use_v1=use_v1,
            enable_llm=enable_llm,
        )

        retrieval_result = retrieval.retrieve(args.query, k=args.k)
        results = retrieval_result.retrieval_chunks

        if enable_llm and retrieval_result.llm_answer:
            print(f"LLM Answer: {retrieval_result.llm_answer}")

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
        print("Please run 'uv run -m scincl.cli ingest' first to create artifacts.")


def main():
    parser = argparse.ArgumentParser(description="SciNCL-based RAG system")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data using SciNCL")
    ingest_parser.add_argument(
        "--model", default="malteos/scincl", help="SciNCL model name"
    )
    ingest_parser.add_argument(
        "--output-dir",
        default="data/scincl_artifacts",
        help="Output directory for artifacts",
    )
    ingest_parser.add_argument(
        "--v1",
        action="store_true",
        help="Use v1 profile: Milvus Lite with FLAT index (default is v2: Milvus server with IVF index)",
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
        "--v1",
        action="store_true",
        help="Use v1 profile: Milvus Lite with FLAT index (default is v2: Milvus server with IVF index)",
    )
    query_parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-assisted rewrite/answering (requires Bedrock env vars)",
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
