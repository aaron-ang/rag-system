"""
Unified evaluation script for all RAG backends.
Supports SciNCL+Milvus (server IVF / v1 Lite FLAT), Qdrant+SentenceTransformer, and Qdrant+TF-IDF.
"""

import os
import sys
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from deepeval import evaluate
from deepeval.evaluate.configs import CacheConfig, DisplayConfig
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase
from deepeval.evaluate.types import EvaluationResult
from sklearn.metrics import ndcg_score

from qdrant.tfidf import TfidfRAG
from qdrant.sentence_transformer import SentenceTransformerRAG
from scincl import SciNCLRetrieval, load_or_create_artifacts


@dataclass
class QueryEvaluationResult:
    query_id: str | int
    query_text: str
    ground_truth_ids: set[str]
    retrieved_ids: list[str]
    retrieved_scores: list[float]
    contexts: list[str]
    llm_answer: str | None = None


def recall_at_k(
    retrieved_docs: list[list[str]], ground_truth_docs: list[set[str]], k: int
):
    """
    Compute Recall@k
    """
    recall_sum = 0
    for retrieved, truth in zip(retrieved_docs, ground_truth_docs):
        top_k = set(retrieved[:k])
        if len(truth) > 0:  # Avoid division by zero
            recall_sum += len(top_k & truth) / len(truth)
    return round(recall_sum / len(ground_truth_docs), 3)


def precision_at_k(
    retrieved_docs: list[list[str]], ground_truth_docs: list[set[str]], k: int
):
    """
    Compute Precision@k
    """
    precision_sum = 0
    for retrieved, truth in zip(retrieved_docs, ground_truth_docs):
        top_k = set(retrieved[:k])
        precision_sum += len(top_k & truth) / k
    return round(precision_sum / len(ground_truth_docs), 3)


def mean_average_precision(
    retrieved_docs: list[list[str]], ground_truth_docs: list[set[str]], k=None
):
    """
    Compute Mean Average Precision (MAP)
    MAP = average of Average Precision across all queries
    AP = (1/R) * Σ(Precision@k × rel(k)) for all k where rel(k)=1
    """
    ap_scores = []

    for retrieved, truth in zip(retrieved_docs, ground_truth_docs):
        if k is not None:
            retrieved = retrieved[:k]

        if len(truth) == 0:
            continue

        # Calculate Average Precision for this query
        num_relevant = 0
        precision_sum = 0.0

        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in truth:
                num_relevant += 1
                precision_at_i = num_relevant / i
                precision_sum += precision_at_i

        if num_relevant > 0:
            ap = precision_sum / len(truth)  # Normalize by total relevant docs
            ap_scores.append(ap)
        else:
            ap_scores.append(0.0)

    return round(np.mean(ap_scores) if ap_scores else 0.0, 3)


def ndcg_at_k(
    retrieved_docs: list[list[str]],
    ground_truth_docs: list[set[str]],
    retrieved_scores: list[list[float]],
    k=3,
):
    ndcg_scores = []
    for retrieved, truth, scores in zip(
        retrieved_docs, ground_truth_docs, retrieved_scores
    ):
        if len(retrieved) == 0:
            continue

        # Binary relevance: 1 if retrieved doc is in ground truth
        y_true = np.array([[1 if doc in truth else 0 for doc in retrieved]])
        # Use actual similarity scores
        y_score = np.array([scores]).reshape(1, -1)

        ndcg = ndcg_score(y_true, y_score, k=k)
        ndcg_scores.append(ndcg)

    return round(np.mean(ndcg_scores), 3) if ndcg_scores else 0.0


def mrr_at_k(
    retrieved_docs: list[list[str]], ground_truth_docs: list[set[str]], k: int
):
    """Mean Reciprocal Rank - position of first relevant document"""
    reciprocal_ranks = []
    for retrieved, truth in zip(retrieved_docs, ground_truth_docs):
        for i, doc in enumerate(retrieved[:k]):
            if doc in truth:
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0.0)
    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3)


def _metric_inputs(results: list[QueryEvaluationResult]):
    retrieved_docs = [result.retrieved_ids for result in results]
    ground_truth_docs = [result.ground_truth_ids for result in results]
    retrieved_scores = [result.retrieved_scores for result in results]
    return retrieved_docs, ground_truth_docs, retrieved_scores


def print_retrieval_metrics(
    results: list[QueryEvaluationResult], include_mrr: bool = False
):
    if not results:
        print("No evaluation data collected; skipping metric computation.")
        return

    retrieved_docs, ground_truth_docs, retrieved_scores = _metric_inputs(results)

    print("Recall@3:", recall_at_k(retrieved_docs, ground_truth_docs, 3))
    print("Recall@5:", recall_at_k(retrieved_docs, ground_truth_docs, 5))
    print("Recall@10:", recall_at_k(retrieved_docs, ground_truth_docs, 10))
    print("-" * 60)

    print("Precision@3:", precision_at_k(retrieved_docs, ground_truth_docs, 3))
    print("Precision@5:", precision_at_k(retrieved_docs, ground_truth_docs, 5))
    print("Precision@10:", precision_at_k(retrieved_docs, ground_truth_docs, 10))
    print("-" * 60)

    print("nDCG@3:", ndcg_at_k(retrieved_docs, ground_truth_docs, retrieved_scores, 3))
    print("nDCG@5:", ndcg_at_k(retrieved_docs, ground_truth_docs, retrieved_scores, 5))
    print(
        "nDCG@10:", ndcg_at_k(retrieved_docs, ground_truth_docs, retrieved_scores, 10)
    )
    print("-" * 60)

    if include_mrr:
        print("MRR@3:", mrr_at_k(retrieved_docs, ground_truth_docs, 3))
        print("MRR@5:", mrr_at_k(retrieved_docs, ground_truth_docs, 5))
        print("MRR@10:", mrr_at_k(retrieved_docs, ground_truth_docs, 10))
        print("-" * 60)

    print(
        "Mean Average Precision (MAP):",
        mean_average_precision(retrieved_docs, ground_truth_docs),
    )


def _summarize_deepeval(result: EvaluationResult):
    """Aggregate deepeval scores across test cases by metric name."""
    aggregated = {}
    for test_result in result.test_results:
        if not test_result.metrics_data:
            continue
        for metric_data in test_result.metrics_data:
            if metric_data.score is None:
                continue
            aggregated.setdefault(metric_data.name, []).append(metric_data.score)

    return {
        name: round(float(np.mean(scores)), 3)
        for name, scores in aggregated.items()
        if scores
    }


def run_deepeval_judge(
    results: list[QueryEvaluationResult],
    model: str = "gpt-5-mini",
):
    """Run deepeval metrics with LLM as the judge."""
    if not results:
        print("LLM Judge: no evaluation data to score.")
        return

    # Allow longer time budgets for judge LLM calls (5 minutes)
    os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "300")
    os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")

    judge = GPTModel(model=model, temperature=0)

    display_config = DisplayConfig(print_results=False)
    cache_config = CacheConfig(use_cache=True)

    context_cases = [
        LLMTestCase(
            input=result.query_text,
            actual_output=result.llm_answer or "",
            retrieval_context=result.contexts,
            name=f"Query {result.query_id}",
        )
        for result in results
        if result.contexts
    ]

    answer_cases = [
        case
        for case in context_cases
        if case.actual_output and case.actual_output.strip()
    ]
    if answer_cases:
        answer_metrics = [
            AnswerRelevancyMetric(model=judge),
            FaithfulnessMetric(model=judge),
        ]
        evaluate(
            answer_cases,
            answer_metrics,
            cache_config=cache_config,
            display_config=display_config,
        )
    elif context_cases:
        print("LLM Judge: no generated answers to evaluate for answer metrics.")


def evaluate_retriever(
    retrieval: SciNCLRetrieval,
    k=10,
    use_v1=False,
    enable_llm_judge: bool = False,
):
    """
    Evaluates the retrieval system on a test set of queries.
    """
    query_results: list[QueryEvaluationResult] = []
    queries_file = "eval/queries_v1.csv" if use_v1 else "eval/queries_latest.csv"
    df = pd.read_csv(queries_file)

    print("\n" + "=" * 60)
    print("SciNCL RAG System - Test Set Evaluation")
    print("=" * 60)

    for _, row in df.iterrows():
        query_id = row["query_id"]
        query = row["query_text"]
        gt_doc_ids = set(str(row["target_paper_ids"]).split(","))

        print(f"\n🔍 Query {query_id}: '{query}'")
        print("-" * 60)

        try:
            retrieval_result = retrieval.retrieve(query, k)
            chunks = retrieval_result.retrieval_chunks
            if chunks:
                retrieved_ids = [result.document.id for result in chunks]
                scores = [result.score for result in chunks]
                contexts = [
                    ctx
                    for ctx in (
                        chunk.chunk_text or chunk.document.abstract for chunk in chunks
                    )
                    if ctx
                ]
            else:
                retrieved_ids = []
                scores = []
                contexts = []
                print("❌ No results found.")

            query_results.append(
                QueryEvaluationResult(
                    query_id=query_id,
                    query_text=query,
                    ground_truth_ids=gt_doc_ids,
                    retrieved_ids=retrieved_ids,
                    retrieved_scores=scores,
                    contexts=contexts,
                    llm_answer=retrieval_result.llm_answer,
                )
            )

        except Exception as e:
            print(f"❌ Error processing query '{query}': {e}")
            query_results.append(
                QueryEvaluationResult(
                    query_id=query_id,
                    query_text=query,
                    ground_truth_ids=gt_doc_ids,
                    retrieved_ids=[],
                    retrieved_scores=[],
                    contexts=[],
                )
            )

    print("\n" + "=" * 60)
    print("✅ Retrieval completed.")
    print("=" * 60)

    print_retrieval_metrics(query_results)
    if enable_llm_judge:
        print("-" * 60)
        run_deepeval_judge(query_results)


def evaluate_qdrant_backend(
    rag_system: SentenceTransformerRAG | TfidfRAG,
    backend_name: str,
    k=10,
):
    """
    Evaluates the Qdrant-based RAG system on a test set of queries.
    Supports both SentenceTransformer and TF-IDF backends.
    """
    df = pd.read_csv("eval/queries_latest.csv")

    query_results = []

    print("\n" + "=" * 60)
    print(f"Qdrant ({backend_name}) RAG System - Test Set Evaluation")
    print("=" * 60)

    for _, row in df.iterrows():
        query_id = row["query_id"]
        query = row["query_text"]
        gt_doc_ids = set(str(row["target_paper_ids"]).split(","))

        print(f"\n🔍 Query {query_id}: '{query}'")
        print("-" * 60)

        try:
            # Use the RAG system's query method
            result = rag_system.query(query, limit=k, generate_answer=False)

            if result and result.get("documents"):
                # Extract SemanticScholar paper IDs from the retrieved documents
                retrieved_paper_ids = []
                retrieved_sim_scores = []
                contexts = []

                print(f"📄 Retrieved {len(result['documents'])} documents")
                for i, doc in enumerate(result["documents"][:3], 1):
                    print(f"  {i}. {doc['title'][:80]}... (Score: {doc['score']:.3f})")

                # Get the paper IDs from the payload
                for doc in result["documents"]:
                    # Try to get the paper_id from the payload
                    paper_id = doc.get("paper_id")
                    if paper_id:
                        retrieved_paper_ids.append(paper_id)
                        retrieved_sim_scores.append(doc["score"])
                    context_text = doc.get("text") or doc.get("abstract")
                    if context_text:
                        contexts.append(context_text)
                    else:
                        # For PubMed documents or documents without paper_id
                        # We can't match them against SemanticScholar IDs
                        pass
            else:
                retrieved_paper_ids = []
                retrieved_sim_scores = []
                contexts = []
                print("❌ No results found.")

            query_results.append(
                QueryEvaluationResult(
                    query_id=query_id,
                    query_text=query,
                    ground_truth_ids=gt_doc_ids,
                    retrieved_ids=retrieved_paper_ids,
                    retrieved_scores=retrieved_sim_scores,
                    contexts=contexts,
                    llm_answer=result.get("answer")
                    if isinstance(result, dict)
                    else None,
                )
            )

        except Exception as e:
            print(f"❌ Error processing query '{query}': {e}")
            import traceback

            traceback.print_exc()
            query_results.append(
                QueryEvaluationResult(
                    query_id=query_id,
                    query_text=query,
                    ground_truth_ids=gt_doc_ids,
                    retrieved_ids=[],
                    retrieved_scores=[],
                    contexts=[],
                )
            )

    print("\n" + "=" * 60)
    print("✅ Retrieval completed.")
    print("=" * 60)

    # Calculate metrics
    print("\n📊 EVALUATION METRICS")
    print("=" * 60)

    print_retrieval_metrics(query_results, include_mrr=True)
    print("=" * 60)

    # Detailed per-query results
    print("\n📋 DETAILED PER-QUERY RESULTS")
    print("=" * 60)
    for i, result in enumerate(query_results, 1):
        hits = len(set(result.retrieved_ids) & result.ground_truth_ids)
        total_relevant = len(result.ground_truth_ids)
        print(f"Query {i}: {hits}/{total_relevant} relevant docs found")
        if hits > 0:
            print(
                f"  Relevant docs found: {set(result.retrieved_ids) & result.ground_truth_ids}"
            )
        elif result.retrieved_ids:
            print(f"  Retrieved IDs: {result.retrieved_ids[:3]}... (showing first 3)")
            print(
                f"  Ground truth IDs: {list(result.ground_truth_ids)[:3]}... (showing first 3)"
            )


def main():
    """Main entry point with backend selection."""
    parser = argparse.ArgumentParser(
        description="RAG System Evaluation - Supports multiple backends"
    )
    parser.add_argument(
        "--backend",
        choices=["scincl", "qdrant_st", "qdrant_tfidf", "all"],
        default="scincl",
        help='Choose backend to evaluate (default: scincl). Use "all" to test all backends.',
    )
    parser.add_argument(
        "--v1",
        action="store_true",
        help="Use v1 profile: Milvus Lite (local milvus.db) with FLAT index and full-document embeddings",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM-powered query rewrite/answer generation for SciNCL backend.",
    )
    parser.add_argument(
        "-k", type=int, default=10, help="Number of documents to retrieve (default: 10)"
    )

    args = parser.parse_args()

    if args.backend == "scincl" or args.backend == "all":
        use_v1 = args.v1
        print("\n" + "=" * 80)
        print(
            "EVALUATING: SciNCL + Milvus "
            + (
                "V1 (Lite, full-document embeddings)"
                if use_v1
                else "V2 (Server, sliding-window)"
            )
            + " Backend"
        )
        print("=" * 80)
        try:
            retrieval = load_or_create_artifacts(use_v1=use_v1, enable_llm=args.llm)
            evaluate_retriever(
                retrieval,
                k=args.k,
                use_v1=use_v1,
                enable_llm_judge=args.llm,
            )
        except Exception as e:
            print(f"❌ Failed to evaluate SciNCL backend: {e}")
            if args.backend == "scincl":
                sys.exit(1)

    if args.backend == "qdrant_st" or args.backend == "all":
        print("\n" + "=" * 80)
        print("EVALUATING: Qdrant + SentenceTransformer Backend")
        print("=" * 80)
        try:
            evaluate_qdrant_backend(
                SentenceTransformerRAG(), "SentenceTransformer", k=args.k
            )
        except ImportError as e:
            print(f"❌ Qdrant backend not available: {e}")
            print("Make sure Qdrant is installed and running.")
            if args.backend == "qdrant_st":
                sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to evaluate Qdrant+SentenceTransformer backend: {e}")
            if args.backend == "qdrant_st":
                sys.exit(1)

    if args.backend == "qdrant_tfidf" or args.backend == "all":
        print("\n" + "=" * 80)
        print("EVALUATING: Qdrant + TF-IDF Backend")
        print("=" * 80)
        try:
            evaluate_qdrant_backend(TfidfRAG(), "TF-IDF", k=args.k)
        except ImportError as e:
            print(f"❌ Qdrant backend not available: {e}")
            print("Make sure Qdrant is installed and running.")
            if args.backend == "qdrant_tfidf":
                sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to evaluate Qdrant+TF-IDF backend: {e}")
            if args.backend == "qdrant_tfidf":
                sys.exit(1)

    if args.backend == "all":
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE FOR ALL BACKENDS")
        print("=" * 80)


if __name__ == "__main__":
    main()
