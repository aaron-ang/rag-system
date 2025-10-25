import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from scincl import SciNCLRetrieval, load_or_create_artifacts


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


def evaluate_retriever(retrieval: SciNCLRetrieval, k=10):
    """
    Evaluates the retrieval system on a test set of queries.
    """
    df = pd.read_csv("eval/retrieval_queries.csv")

    retrieved_docs = []
    retrieved_scores = []
    ground_truth_docs = []

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
            results = retrieval.retrieve_similar_documents(query, k)
            if results:
                retrieved_ids = [result.document.id for result in results]
                retrieved_sim_scores = [result.sim_score for result in results]
            else:
                retrieved_ids = []
                retrieved_sim_scores = []
                print("❌ No results found.")

            retrieved_docs.append(retrieved_ids)
            retrieved_scores.append(retrieved_sim_scores)
            ground_truth_docs.append(gt_doc_ids)

        except Exception as e:
            print(f"❌ Error processing query '{query}': {e}")
            retrieved_docs.append([])
            retrieved_scores.append([])
            ground_truth_docs.append(gt_doc_ids)

    print("\n" + "=" * 60)
    print("✅ Retrieval completed.")
    print("=" * 60)

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

    print(
        "Mean Average Precision (MAP):",
        mean_average_precision(retrieved_docs, ground_truth_docs),
    )
    print("-" * 60)


if __name__ == "__main__":
    retrieval = load_or_create_artifacts()
    evaluate_retriever(retrieval)
