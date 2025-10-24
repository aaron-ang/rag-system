from typing import List, Set
import numpy as np
from sklearn.metrics import ndcg_score

def recall_at_k(retrieved_docs: List[List[str]], ground_truth_docs: List[Set[str]], k: int) -> float:
    """
    Compute Recall@k
    """
    hits = 0
    for retrieved, truth in zip(retrieved_docs, ground_truth_docs):
        top_k = set(retrieved[:k])
        if len(top_k & truth) > 0:
            hits += 1
    return hits / len(ground_truth_docs)

def precision_at_k(retrieved_docs: List[List[str]], ground_truth_docs: List[Set[str]], k: int) -> float:
    """
    Compute Precision@k
    """
    precision_sum = 0
    for retrieved, truth in zip(retrieved_docs, ground_truth_docs):
        top_k = set(retrieved[:k])
        precision_sum += len(top_k & truth) / k
    return precision_sum / len(ground_truth_docs)

def compute_ndcg_at_k(retrieved_docs, ground_truth_docs, k=3):
    ndcg_scores = []
    for retrieved, truth in zip(retrieved_docs, ground_truth_docs):
        # Binary relevance: 1 if retrieved doc is in ground truth
        y_true = np.array([[1 if doc in truth else 0 for doc in retrieved]])
        # Assume retriever ranks them by order (highest score first)
        y_score = np.arange(len(retrieved), 0, -1).reshape(1, -1)
        
        ndcg = ndcg_score(y_true, y_score, k=k)
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)

# Example usage
retrieved_docs = [
    ['doc1', 'doc2', 'doc3'],
    ['doc4', 'doc2', 'doc5'],
    ['doc6', 'doc7', 'doc8']
]

ground_truth_docs = [
    {'doc2'},       
    {'doc4', 'doc5'}, 
    {'doc9'} 
]

k = 2
print("Recall@2:", recall_at_k(retrieved_docs, ground_truth_docs, k))
print("Precision@2:", precision_at_k(retrieved_docs, ground_truth_docs, k))
print("Mean nDCG@3:", compute_ndcg_at_k(retrieved_docs, ground_truth_docs, k))
