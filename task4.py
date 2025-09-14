import pandas as pd
import numpy as np
import math
from typing import List, Tuple


def build_histogram(df: pd.DataFrame, score_col='fico_score', default_col='default'):
    """
    Aggregate counts by each integer FICO score (or unique score).
    Returns arrays of scores, counts, defaults aligned by index.
    """
    # ensure integer-ish scores (if floats, keep them)
    agg = df.groupby(score_col).agg(
        n=('customer_id', 'count'),
        k=(default_col, 'sum')
    ).reset_index().sort_values(by=score_col)
    scores = agg[score_col].to_numpy()
    counts = agg['n'].to_numpy(dtype=int)
    defaults = agg['k'].to_numpy(dtype=int)
    return scores, counts, defaults


def compute_prefix_sums(scores: np.ndarray, counts: np.ndarray):
    """For weighted points: compute prefix sums of weights, weighted values and weighted squares."""
    w = counts
    x = scores
    wx = w * x
    wx2 = w * (x * x)
    W = np.concatenate(([0], np.cumsum(w)))
    SX = np.concatenate(([0.0], np.cumsum(wx)))
    SX2 = np.concatenate(([0.0], np.cumsum(wx2)))
    return W, SX, SX2

def bucket_cost_mse(i: int, j: int, W, SX, SX2):
    """
    cost for indices i..j inclusive (1-based index for prefix arrays)
    uses prefix arrays where W[0]=0, SX[0]=0, ...
    returns SSE (sum squared errors) for approximating by weighted mean.
    """
    w = W[j] - W[i-1]
    if w == 0:
        return 0.0
    sx = SX[j] - SX[i-1]
    sx2 = SX2[j] - SX2[i-1]
    mean = sx / w
    sse = sx2 - (sx * sx) / w
    return sse

def optimal_mse_bins(scores: np.ndarray, counts: np.ndarray, K: int):
    """
    Returns K-1 boundaries (score values) that partition scores into K buckets
    minimizing weighted MSE. Uses DP O(K*N^2). N = number of unique score values.
    """
    N = len(scores)
    W, SX, SX2 = compute_prefix_sums(scores, counts)
    # DP arrays: dp[k][j] = minimal cost to partition first j points into k buckets
    dp = np.full((K+1, N+1), np.inf)
    prev = np.full((K+1, N+1), -1, dtype=int)
    dp[0,0] = 0.0

    # fill DP
    for k in range(1, K+1):
        for j in range(1, N+1):
            # try all possible split points i..j (i from 1..j)
            best_val = np.inf
            best_i = -1
            for i in range(1, j+1):
                val = dp[k-1, i-1] + bucket_cost_mse(i, j, W, SX, SX2)
                if val < best_val:
                    best_val = val
                    best_i = i
            dp[k, j] = best_val
            prev[k, j] = best_i

    # backtrack to get partitions in terms of indices
    boundaries_idx = []
    k = K
    j = N
    cuts = []
    while k > 0 and j > 0:
        i = prev[k, j]
        cuts.append((i-1, j-1))  # 0-based indices range for bucket
        j = i - 1
        k -= 1
    cuts.reverse()  # list of (start_idx, end_idx) for each bucket

    # boundaries: use upper endpoint of each bucket except last
    boundaries = []
    for _, end_idx in cuts[:-1]:
        boundaries.append(scores[end_idx])  # boundary is score at end of bucket
    return boundaries, cuts


def bucket_loglikelihood(i: int, j: int, counts_prefix, defaults_prefix, eps=1e-9):
    """
    i..j inclusive (1-based indexes in prefix arrays).
    returns negative log-lik? We'll return the *log-likelihood* value (higher better).
    """
    n = counts_prefix[j] - counts_prefix[i-1]
    k = defaults_prefix[j] - defaults_prefix[i-1]
    if n == 0:
        return 0.0
    # MLE p = k/n; avoid p==0 or p==1 by smoothing
    p = (k + eps) / (n + 2*eps)
    ll = k * math.log(p) + (n - k) * math.log(1 - p)
    return ll

def optimal_likelihood_bins(scores: np.ndarray, counts: np.ndarray, defaults: np.ndarray, K: int):
    """
    DP to maximize total log-likelihood across K buckets. O(K*N^2).
    Returns boundaries and bucket index ranges.
    """
    N = len(scores)
    # prefix sums
    counts_prefix = np.concatenate(([0], np.cumsum(counts)))
    defaults_prefix = np.concatenate(([0], np.cumsum(defaults)))

    # dp[k][j] = max log-likelihood using first j points in k buckets
    dp = np.full((K+1, N+1), -np.inf)
    prev = np.full((K+1, N+1), -1, dtype=int)
    dp[0,0] = 0.0

    for k in range(1, K+1):
        for j in range(1, N+1):
            best_val = -np.inf
            best_i = -1
            for i in range(1, j+1):
                val = dp[k-1, i-1] + bucket_loglikelihood(i, j, counts_prefix, defaults_prefix)
                if val > best_val:
                    best_val = val
                    best_i = i
            dp[k, j] = best_val
            prev[k, j] = best_i

    # backtrack
    cuts = []
    k = K
    j = N
    while k > 0 and j > 0:
        i = prev[k, j]
        cuts.append((i-1, j-1))
        j = i - 1
        k -= 1
    cuts.reverse()

    boundaries = [scores[end] for _, end in cuts[:-1]]
    return boundaries, cuts


def build_rating_map(csv_path: str,
                     K: int = 5,
                     method: str = 'likelihood',
                     score_col='fico_score', default_col='default',
                     customer_id_col='customer_id'):
    """
    Returns:
      - boundaries: list of K-1 score boundaries (ascending). Each boundary is inclusive upper bound of bucket.
      - rating_map: function(score) -> rating (1..K) where 1 is best (highest FICO)
      - bucket_ranges: list of (start_score, end_score) tuples for the K buckets
    method: 'mse' or 'likelihood'
    """
    df = pd.read_csv(csv_path)
    scores, counts, defaults = build_histogram(df, score_col=score_col, default_col=default_col)

    if method == 'mse':
        boundaries, cuts = optimal_mse_bins(scores, counts, K)
    elif method == 'likelihood':
        boundaries, cuts = optimal_likelihood_bins(scores, counts, defaults, K)
    else:
        raise ValueError("method must be 'mse' or 'likelihood'")

    # Build bucket ranges from cuts (each cut is (start_idx, end_idx))
    bucket_ranges = []
    for start_idx, end_idx in cuts:
        bucket_ranges.append((scores[start_idx], scores[end_idx]))

    # boundaries are upper bounds of buckets except last; ensure sorted ascending
    boundaries_sorted = sorted(boundaries)

    # rating: lower number = better credit. So highest FICO scores -> rating 1.
    # We have bucket_ranges sorted by ascending score; the last bucket has highest FICO.
    # We'll map such that rating = 1 for highest-FICO bucket.
    def rating_map(score_value):
        # find first boundary >= score_value
        for i, b in enumerate(boundaries_sorted):
            if score_value <= b:
                bucket_idx = i  # 0-based ascending bucket index
                break
        else:
            bucket_idx = len(boundaries_sorted)  # last bucket index
        # convert to rating: highest FICO -> rating 1
        # ascending bucket_idx -> rating = K - bucket_idx
        return int(K - bucket_idx)

    return boundaries_sorted, rating_map, bucket_ranges


if __name__ == "__main__":
    csv_path = "loan_data.csv"
    K = 5

    # run likelihood-based buckets 
    boundaries, rating_map, bucket_ranges = build_rating_map(csv_path, K=K, method='likelihood')
    print("Boundaries (upper bounds of buckets, ascending):", boundaries)
    print("Bucket ranges (low, high):", bucket_ranges)

    # show small sample mapping
    sample_scores = [300, 620, 680, 720, 780, 820]
    for s in sample_scores:
        print(s, "-> rating", rating_map(s))
