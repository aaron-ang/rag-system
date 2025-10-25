"""
Fixed Evaluation script for the Medical RAG System
Uses proper ID mapping between SemanticScholar paper IDs and stored documents
"""

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
import sys
import os

# Add parent directory to path to import RAG system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_with_persistence import PersistentMedicalRAG


def recall_at_k(retrieved_docs: list[list[str]], ground_truth_docs: list[set[str]], k: int):
    """
    Compute Recall@k
    """
    recall_sum = 0
    for retrieved, truth in zip(retrieved_docs, ground_truth_docs):
        top_k = set(retrieved[:k])
        if len(truth) > 0:  # Avoid division by zero
            recall_sum += len(top_k & truth) / len(truth)
    return round(recall_sum / len(ground_truth_docs), 3)


def precision_at_k(retrieved_docs: list[list[str]], ground_truth_docs: list[set[str]], k: int):
    """
    Compute Precision@k
    """
    precision_sum = 0
    for retrieved, truth in zip(retrieved_docs, ground_truth_docs):
        top_k = set(retrieved[:k])
        precision_sum += len(top_k & truth) / k
    return round(precision_sum / len(ground_truth_docs), 3)


def mean_average_precision(retrieved_docs: list[list[str]], ground_truth_docs: list[set[str]], k=None):
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


def ndcg_at_k(retrieved_docs: list[list[str]], ground_truth_docs: list[set[str]], retrieved_scores: list[list[float]], k=3):
    ndcg_scores = []
    for retrieved, truth, scores in zip(retrieved_docs, ground_truth_docs, retrieved_scores):
        if len(retrieved) == 0:
            continue

        # Binary relevance: 1 if retrieved doc is in ground truth
        y_true = np.array([[1 if doc in truth else 0 for doc in retrieved]])
        # Use actual similarity scores
        y_score = np.array([scores]).reshape(1, -1)

        ndcg = ndcg_score(y_true, y_score, k=k)
        ndcg_scores.append(ndcg)

    return round(np.mean(ndcg_scores), 3) if ndcg_scores else 0.0


def mrr_at_k(retrieved_docs: list[list[str]], ground_truth_docs: list[set[str]], k: int):
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


def evaluate_rag_system_fixed(rag_system: PersistentMedicalRAG, k=10):
    """
    Evaluates the RAG system on a test set of queries using proper ID mapping.
    """
    # Load test queries
    df = pd.read_csv("eval/retrieval_queries.csv")

    retrieved_docs = []
    retrieved_scores = []
    ground_truth_docs = []

    print("\n" + "=" * 60)
    print("Medical RAG System - FIXED Test Set Evaluation")
    print("=" * 60)

    for _, row in df.iterrows():
        query_id = row["query_id"]
        query = row["query_text"]
        gt_doc_ids = set(str(row["target_paper_ids"]).split(","))

        print(f"\n🔍 Query {query_id}: '{query}'")
        print("-" * 60)

        try:
            # Use the RAG system's query method
            result = rag_system.query(query, limit=k)
            
            if result and result['documents']:
                # Extract SemanticScholar paper IDs from the retrieved documents
                retrieved_paper_ids = []
                retrieved_sim_scores = []
                
                print(f"📄 Retrieved {len(result['documents'])} documents")
                for i, doc in enumerate(result['documents'][:3], 1):
                    print(f"  {i}. {doc['title'][:80]}... (Score: {doc['score']:.3f})")
                
                # Get the paper IDs from the payload
                for doc in result['documents']:
                    # Try to get the paper_id from the payload
                    paper_id = doc.get('paper_id')
                    if paper_id:
                        retrieved_paper_ids.append(paper_id)
                        retrieved_sim_scores.append(doc['score'])
                    else:
                        # If no paper_id, try to get it from the payload directly
                        # This would require accessing the full document from Qdrant
                        print(f"⚠️  No paper_id found for document: {doc['title'][:50]}...")
            else:
                retrieved_paper_ids = []
                retrieved_sim_scores = []
                print("❌ No results found.")

            retrieved_docs.append(retrieved_paper_ids)
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

    # Calculate metrics
    print("\n📊 EVALUATION METRICS")
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
    print("nDCG@10:", ndcg_at_k(retrieved_docs, ground_truth_docs, retrieved_scores, 10))
    print("-" * 60)

    print("MRR@3:", mrr_at_k(retrieved_docs, ground_truth_docs, 3))
    print("MRR@5:", mrr_at_k(retrieved_docs, ground_truth_docs, 5))
    print("MRR@10:", mrr_at_k(retrieved_docs, ground_truth_docs, 10))
    print("-" * 60)

    print("Mean Average Precision (MAP):", mean_average_precision(retrieved_docs, ground_truth_docs))
    print("=" * 60)

    # Detailed per-query results
    print("\n📋 DETAILED PER-QUERY RESULTS")
    print("=" * 60)
    for i, (retrieved, truth) in enumerate(zip(retrieved_docs, ground_truth_docs), 1):
        hits = len(set(retrieved) & truth)
        total_relevant = len(truth)
        print(f"Query {i}: {hits}/{total_relevant} relevant docs found")
        if hits > 0:
            print(f"  Relevant docs found: {set(retrieved) & truth}")
        else:
            print(f"  Retrieved IDs: {retrieved[:3]}... (showing first 3)")
            print(f"  Ground truth IDs: {list(truth)[:3]}... (showing first 3)")


if __name__ == "__main__":
    print("🏥 Medical RAG System Evaluation - FIXED VERSION")
    print("=" * 50)
    
    # Initialize the RAG system
    print("🔄 Initializing RAG system...")
    try:
        rag = PersistentMedicalRAG()
        print("✅ RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("Make sure Qdrant is running and the system is set up.")
        sys.exit(1)
    
    # Run evaluation
    evaluate_rag_system_fixed(rag)
