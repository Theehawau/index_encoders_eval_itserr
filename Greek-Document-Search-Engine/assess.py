import pandas as pd
from pathlib import Path
import os
from glob import glob
import argparse
import warnings
warnings.filterwarnings("ignore")

def mean_reciprocal_rank(rankings: list[list[int]]) -> float:
    """
    Args:
        rankings: A list of queries, each containing a list where
                  1 = relevant, 0 = not relevant (ordered by predicted rank).
    Returns:
        MRR score
    """
    reciprocal_ranks = []
    for ranked_list in rankings:
        try:
            rank = ranked_list.index(1) + 1  # first relevant item
            reciprocal_ranks.append(1 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)  # no relevant item found
    return sum(reciprocal_ranks) / len(rankings)

import math

def dcg(relevance: list[int], k: int) -> float:
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevance[:k]))

def ndcg(relevance: list[int], k: int) -> float:
    ideal = sorted(relevance, reverse=True)
    idcg = dcg(ideal, k)
    return dcg(relevance, k) / idcg if idcg > 0 else 0.0

def average_ndcg(relevance_lists: list[list[int]], k: int) -> float:
    scores = [ndcg(rel, k) for rel in relevance_lists]
    return sum(scores) / len(scores)

def recall_at_k(responses: list, references: list) -> float:
    retrieved = set(responses)
    relevant = set(references)
    intersection_count = len(retrieved.intersection(relevant))
    return intersection_count / len(relevant) if len(relevant) > 0 else 0.0

def average_recall_at_k(batch_responses: list[list], batch_references: list[list]) -> float:
    """
    Args:
        batch_responses: List of lists of responses (one per query)
        batch_references: List of lists of references (one per query)
    Returns:
        Average Recall@k over all queries
    """
    assert len(batch_responses) == len(batch_references), "Mismatch in number of queries"
    
    recalls = []
    for responses, references in zip(batch_responses, batch_references):
        recalls.append(recall_at_k(responses, references))
    return sum(recalls) / len(recalls)

 # Check matches
def check_matches(row):
    target_match = []
    for i in range(5):
        if row[targets[i]] in row[top_k].tolist():
            target_match.append(1)
        else:
            target_match.append(0)
    return target_match



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the performance of a search engine.")
    parser.add_argument("--model",
                        type=str, 
                        default=None, 
                        help="Model to use for the search")
    parser.add_argument("--index_model",
                        type=str, 
                        help="Type of the model to use for the search")
    parser.add_argument("--top_k", type=int, default=5, help="Top K results to consider for evaluation.")
    args = parser.parse_args()
    k = args.top_k
    data_path = f"target_data/{args.index_model}_index/Greek_benchmark_best_targets.txt"


    # Load the data
    df = pd.read_csv(data_path, sep="\t")
    
    # Define the target and top_k columns
    targets = [f"Target #{i}" for i in range(0, 5)]
    top_k = [f"best #{i}" for i in range(0, args.top_k)]
    top_250 = [f"best #{i}" for i in range(0, 250)]

    df['target_match'] = df.apply(check_matches, axis=1)

    print("Index Model:", args.index_model)
    if args.model is not None:
        print("-----------------------------------")
        print("Model:", args.model)
        print("-----------------------------------")

    # Calculate Recall@k
    recall_score = average_recall_at_k(df[top_k].values.tolist(), df[targets].values.tolist())
    print(f"Average Recall@{args.top_k}: {recall_score:.4f}")
    print("")

    # Calculate NDCG
    ndcg_score = average_ndcg(df['target_match'].tolist(), args.top_k)
    print(f"Average NDCG: {ndcg_score:.4f}")
    print("")

    # Calculate MRR
    mrr_score = mean_reciprocal_rank(df['target_match'].tolist())
    print(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}")
    print("")

    out_dir = f"target_data/{args.index_model}_index/_outputs_*"
    cos_similarity_path = f"{out_dir}/Greek_benchmark_best_cosine_similarities.txt"
    cross_dfs = list(glob(cos_similarity_path))

    for cross_df in cross_dfs:
        df_cross = pd.read_csv(cross_df, sep="\t")
        top5_columns_per_row = df_cross[top_250].apply(lambda row: row.nlargest(5).index.tolist(), axis=1)
        for index,row in enumerate(top5_columns_per_row):
            df_cross.loc[index,top_k] = df.loc[index,row].to_list()

        batch_responses = df[targets].values.tolist()
        batch_references = df_cross[top_k].values.tolist()

        avg_recall = average_recall_at_k(batch_responses, batch_references)
        print(f"Average Recall@5 for {cross_df.split('/')[2]}: {avg_recall:.4f}")



