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
    hits = 0
    for retrieved, truth in zip(retrieved_docs, ground_truth_docs):
        top_k = set(retrieved[:k])
        if len(top_k & truth) > 0:
            hits += 1
    return hits / len(ground_truth_docs)


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
    return precision_sum / len(ground_truth_docs)


def ndcg_at_k(retrieved_docs: list[list[str]], ground_truth_docs: list[set[str]], k=3):
    ndcg_scores = []
    for retrieved, truth in zip(retrieved_docs, ground_truth_docs):
        # Binary relevance: 1 if retrieved doc is in ground truth
        y_true = np.array([[1 if doc in truth else 0 for doc in retrieved]])
        # Assume retriever ranks them by order (highest score first)
        y_score = np.arange(len(retrieved), 0, -1).reshape(1, -1)

        ndcg = ndcg_score(y_true, y_score, k=k)
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores)


def evaluate_retriever(retrieval: SciNCLRetrieval, k=3):
    """
    Evaluates the retrieval system on a test set of queries.
    """
    df = pd.read_csv("eval/retrieval_requests.csv")

    retrieved_docs = []
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
                print(f"Expected documents: {gt_doc_ids}")
                print(f"✅ Retrieved documents: {retrieved_ids}")
            else:
                retrieved_ids = []
                print("❌ No results found.")

            retrieved_docs.append(retrieved_ids)
            ground_truth_docs.append(gt_doc_ids)

        except Exception as e:
            print(f"❌ Error processing query '{query}': {e}")
            retrieved_docs.append([])
            ground_truth_docs.append(gt_doc_ids)

    print("\n" + "=" * 60)
    print("✅ Retrieval completed.")
    print("=" * 60)

    print(f"Recall@{k}:", recall_at_k(retrieved_docs, ground_truth_docs, k))
    print(f"Precision@{k}:", precision_at_k(retrieved_docs, ground_truth_docs, k))
    print(f"Mean nDCG@{k}:", ndcg_at_k(retrieved_docs, ground_truth_docs, k))


if __name__ == "__main__":
    retrieval = load_or_create_artifacts()
    evaluate_retriever(retrieval)
