"""
BASELINES: Common Neighbors, Jaccard Coefficient and Adamic-Adar Index.

These three methods score a pair of users using ONLY the friendship graph
(no age, interests or location). They are the yardstick the GCN is compared to.

How each score works for a pair of users (i, j), where "neighbors" = friends:
  - Common Neighbors: how many friends do i and j share?
  - Jaccard:          shared friends / total distinct friends of i and j.
  - Adamic-Adar:      each shared friend z counts 1 / log(number of z's friends),
                      so a shared friend who knows few people counts more than
                      a shared friend who knows everyone.

References
  - Adamic & Adar, "Friends and neighbors on the Web," Social Networks, 2003.
  - Liben-Nowell & Kleinberg, "The link-prediction problem for social networks,"
    JASIST, 2007.
  - Also had coding/debugging assistance from Copilot
"""

import random
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score

METHODS = ["Common Neighbors", "Jaccard", "Adamic-Adar"]


# --------------------------------------------------------------------------
# 1. Turn an edge list into an adjacency matrix
#  IMPORTANT: pass in only the edges the model is allowed to see.
#    For the test evaluation that is test_data.edge_index (train + validation
#    friendships). The hidden test friendships are NOT in it.
# --------------------------------------------------------------------------
def build_adjacency(edge_index, num_nodes):
    row, col = edge_index
    A = sp.coo_matrix(
        (np.ones(len(row), dtype=np.float32), (row, col)),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    A.data[:] = 1.0          # if an edge was listed twice, count it once
    return A


class Scorer:
    """Holds the matrices needed to score pairs by each of the three methods."""

    def __init__(self, A):
        self.A = A
        self.deg = np.asarray(A.sum(axis=1)).ravel()

        # Adamic-Adar weight per user: 1 / log(degree). A user with 0 or 1
        # friends can never be a shared friend of two people, so weight 0.
        with np.errstate(divide="ignore"):
            w = 1.0 / np.log(self.deg)
        w[self.deg <= 1] = 0.0
        self.Aw = (A @ sp.diags(w)).tocsr()   # columns scaled by the weight
        self.AwT = self.Aw.T.tocsr()

    # ---- scores for a list of specific pairs (used for AUC / AP) ----------
    def pair_scores(self, pairs, method, chunk=50_000):
        src, dst = np.asarray(pairs[0]), np.asarray(pairs[1])
        out = np.zeros(len(src), dtype=np.float64)
        for s in range(0, len(src), chunk):
            i, j = src[s:s + chunk], dst[s:s + chunk]
            common = np.asarray(self.A[i].multiply(self.A[j]).sum(axis=1)).ravel()
            if method == "Common Neighbors":
                out[s:s + chunk] = common
            elif method == "Jaccard":
                denom = self.deg[i] + self.deg[j] - common
                out[s:s + chunk] = np.divide(common, denom,
                                             out=np.zeros_like(common, dtype=np.float64),
                                             where=denom > 0)
            elif method == "Adamic-Adar":
                out[s:s + chunk] = np.asarray(
                    self.A[i].multiply(self.Aw[j]).sum(axis=1)).ravel()
            else:
                raise ValueError(method)
        return out

    # ---- scores from ONE user to EVERY user (used for Precision@K) --------
    def row_scores(self, u, method):
        if method == "Adamic-Adar":
            return (self.A[u] @ self.AwT).toarray().ravel()
        common = (self.A[u] @ self.A).toarray().ravel()
        if method == "Common Neighbors":
            return common
        if method == "Jaccard":
            denom = self.deg[u] + self.deg - common
            return np.divide(common, denom, out=np.zeros_like(common, dtype=np.float64),
                             where=denom > 0)
        raise ValueError(method)


# ------------------------------------------------------------------
# 2. AUC and Average Precision on the test pairs
# --------------------------------------------------------------------------
def auc_ap(scorer, pos_pairs, neg_pairs, method):
    scores = np.concatenate([scorer.pair_scores(pos_pairs, method),
                             scorer.pair_scores(neg_pairs, method)])
    labels = np.concatenate([np.ones(len(pos_pairs[0])), np.zeros(len(neg_pairs[0]))])
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)


# --------------------------------------------------------------------------
# 3. Precision@K 
# ----------------------------------------------------------------------
def build_hidden_known(pos_pairs, edge_index):
    """hidden = the answer key (test friendships); known = friends already seen."""
    hidden, known = defaultdict(set), defaultdict(set)
    for a, b in zip(*pos_pairs):
        hidden[int(a)].add(int(b))
        hidden[int(b)].add(int(a))
    for a, b in zip(*edge_index):
        known[int(a)].add(int(b))
        known[int(b)].add(int(a))
    return hidden, known


def precision_at_k(row_score_fn, hidden, known, k=10, num_users=500, seed=42):
    rng = random.Random(seed)
    users = rng.sample(sorted(hidden), min(num_users, len(hidden)))
    tie_rng = np.random.default_rng(seed)

    precisions = []
    for u in users:
        s = np.asarray(row_score_fn(u), dtype=np.float64)
        # Many users tie at a score of 0, so break ties randomly. Otherwise the
        # lowest-numbered users would always be picked, which is not fair.
        s = s + 1e-9 * tie_rng.random(len(s))
        s[u] = -np.inf                       # don't recommend yourself
        for v in known[u]:
            s[v] = -np.inf                   # don't recommend current friends
        top = np.argpartition(-s, k)[:k]
        precisions.append(sum(int(v) in hidden[u] for v in top) / k)
    return float(np.mean(precisions))
print("done")