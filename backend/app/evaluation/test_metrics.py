from backend.app.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank
)

# Simulated retrieval results
retrieved_ids = ["C1", "C2", "C3", "C4", "C5"]

# Chunks we know are relevant
relevant_ids = {"C2", "C4"}

precision = precision_at_k(retrieved_ids, relevant_ids, 5)
recall = recall_at_k(retrieved_ids, relevant_ids, 5)
mrr = reciprocal_rank(retrieved_ids, relevant_ids)

print("Precision@5:", precision)
print("Recall@5:", recall)
print("MRR:", mrr)