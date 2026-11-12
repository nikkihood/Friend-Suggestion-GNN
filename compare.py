"""
STEP 3: Run the baselines and compare them with the GCN.

Run order:   step 1 (imported)  ->  step 2 (run once)  ->  this file.

What this script does
  1. Rebuilds the exact same train/validation/test split as step 2
     (step1_build_graph.py sets the seed, so the split is identical).
  2. Builds the friendship graph the models are allowed to see at test time
     (train + validation friendships; the hidden test friendships are NOT in it).
  3. Scores the test pairs with Common Neighbors, Jaccard and Adamic-Adar.
  4. Computes AUC, Average Precision and Precision@K for each, using the same
     test pairs and the same sampled users as the GCN.
  5. Prints and saves a comparison table (results_comparison.csv) and a chart
     (results_comparison.png).

Files that must be in the same folder:
  step1_build_graph.py, baselines.py, the two CSV files, and gcn_results.json
  (gcn_results.json is created when you run step2_train_gcn.py).

Install once if needed:   pip install matplotlib
"""

import json
import os

import numpy as np
import pandas as pd

from baselineTests import METHODS, Scorer, auc_ap, build_adjacency, build_hidden_known, precision_at_k
from GNN import SEED, test_data

K = 10
NUM_EVAL_USERS = 500

# --------------------------------------------------------------------------
# 1. Get the test pairs and the graph the models may see at test time
# --------------------------------------------------------------------------
num_nodes = test_data.num_nodes
visible_edges = test_data.edge_index.cpu().numpy()          # train + validation edges
pos_pairs = test_data.pos_edge_label_index.cpu().numpy()    # hidden real friendships
neg_pairs = test_data.neg_edge_label_index.cpu().numpy()    # fake friendships

print(f"Visible edges: {visible_edges.shape[1]:,}   "
      f"Test positives: {pos_pairs.shape[1]:,}   Test negatives: {neg_pairs.shape[1]:,}")

A = build_adjacency(visible_edges, num_nodes)
scorer = Scorer(A)
hidden, known = build_hidden_known(pos_pairs, visible_edges)

# --------------------------------------------------------------------------
# 2. Score every baseline
# --------------------------------------------------------------------------
rows = []
for method in METHODS:
    auc, ap = auc_ap(scorer, pos_pairs, neg_pairs, method)
    p_at_k = precision_at_k(lambda u, m=method: scorer.row_scores(u, m),
                            hidden, known, k=K, num_users=NUM_EVAL_USERS, seed=SEED)
    rows.append({"Method": method, "AUC": auc, "AP": ap, f"Precision@{K}": p_at_k})
    print(f"{method:18s} AUC {auc:.4f} | AP {ap:.4f} | Precision@{K} {p_at_k:.4f}")

# Diagnostic for your write-up: the counting baselines can only give a pair a
# non-zero score if the two users already share at least one friend.
share_real = (scorer.pair_scores(pos_pairs, "Common Neighbors") > 0).mean()
share_fake = (scorer.pair_scores(neg_pairs, "Common Neighbors") > 0).mean()
print(f"\nReal test friendships with at least 1 common friend: {share_real:.1%}")
print(f"Fake test pairs with at least 1 common friend:       {share_fake:.1%}")

# --------------------------------------------------------------------------
# 3. Add the GCN row (saved by step 2)
# --------------------------------------------------------------------------
if os.path.exists("gcn_results.json"):
    g = json.load(open("gcn_results.json"))
    gcn = {"Method": "GCN (2-layer)", "AUC": g["AUC"], "AP": g["AP"],
           f"Precision@{K}": g[f"Precision@{K}"]}
else:
    # Fallback: the numbers from the first run of step 2. Re-run step 2 to refresh them.
    print("\n(gcn_results.json not found, using the numbers from the first step 2 run)")
    gcn = {"Method": "GCN (2-layer)", "AUC": 0.8139, "AP": 0.8257, f"Precision@{K}": 0.0004}
rows.append(gcn)

table = pd.DataFrame(rows).set_index("Method").round(4)
print("\n=== COMPARISON (same test split, same test users) ===")
print(table.to_string())
table.to_csv("results_comparison.csv")
print("\nSaved results_comparison.csv")

# --------------------------------------------------------------------------
# 4. Chart: AUC / AP side by side, Precision@K in its own panel
#    (Precision@K is much smaller than AUC, so it needs its own axis.)
# --------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(table.index)
    x = np.arange(len(methods))
    colors = ["#7f8c8d", "#7f8c8d", "#7f8c8d", "#2471a3"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=200,
                               gridspec_kw={"width_ratios": [1.6, 1]})
    w = 0.38
    a.bar(x - w / 2, table["AUC"], w, label="AUC", color="#2471a3")
    a.bar(x + w / 2, table["AP"], w, label="Average Precision", color="#1e8449")
    a.axhline(0.5, color="gray", ls="--", lw=1, label="Chance level (0.50)")
    for i, (u, v) in enumerate(zip(table["AUC"], table["AP"])):
        a.text(i - w / 2, u + 0.01, f"{u:.2f}", ha="center", fontsize=8)
        a.text(i + w / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    a.set_xticks(x); a.set_xticklabels(methods, fontsize=9)
    a.set_ylim(0, 1.05); a.set_ylabel("Score (higher is better)")
    a.set_title("Separating real from fake friendships", fontsize=11)
    a.legend(loc="upper left", fontsize=9); a.grid(axis="y", alpha=0.3)

    p = table[f"Precision@{K}"]
    b.bar(x, p, color=colors)
    for i, v in enumerate(p):
        b.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    b.set_xticks(x); b.set_xticklabels(methods, fontsize=8, rotation=20)
    b.set_ylabel(f"Precision@{K}")
    b.set_title(f"Quality of the top-{K} recommendations", fontsize=11)
    b.grid(axis="y", alpha=0.3)

    fig.suptitle("Baselines vs. GCN on the same hidden test friendships", fontsize=12.5)
    fig.tight_layout()
    fig.savefig("results_comparison.png")
    print("Saved results_comparison.png")
except ImportError:
    print("matplotlib not installed, skipped the chart (pip install matplotlib)")