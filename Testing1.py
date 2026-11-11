"""
Train a two-layer GCN for link prediction and evaluate it.
 
Run this AFTER step 1 works. It imports the graph objects that step 1 built
(train_data, val_data, test_data), so keep step1_build_graph.py in the same folder.
 
What this script does
  1. Defines a two-layer GCN that turns every user into an embedding.
  2. Scores a pair of users by multiplying their two embeddings (dot product).
  3. Trains the model: real friendships should score high, fake ones low.
  4. Evaluates with AUC and Average Precision on the hidden test friendships.
  5. Computes Precision@K: do the top K recommendations hit hidden friends?
 
References
  - GCNConv layer docs:
      https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html
  - Introduction by Example (two-layer GCN):
      https://pytorch-geometric.readthedocs.io/en/2.6.1/get_started/introduction.html
  - Kipf & Welling, GCN paper: https://arxiv.org/abs/1609.02907
"""
 
import random
import sys
from collections import defaultdict
import json

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
 
# Running step 1 (it executes when imported) gives us the three graph splits.
from GNN import SEED, test_data, train_data, val_data
 
# --------------------------------------------------------------------------
# 0. Settings 
# --------------------------------------------------------------------------
HIDDEN_DIM = 64      # size of the middle layer
EMBED_DIM = 32       # size of each user's final embedding
LEARNING_RATE = 0.01
EPOCHS = 100
K = 10               # for Precision@K
NUM_EVAL_USERS = 500 # how many users to average Precision@K over
 
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data, val_data, test_data = (d.to(device) for d in (train_data, val_data, test_data))
print("Using device:", device)
 
 
# --------------------------------------------------------------------------
# 1. The model: two GCN layers + a dot-product scorer
# --------------------------------------------------------------------------
class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, embed_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)     # layer 1: blend with direct friends
        self.conv2 = GCNConv(hidden_dim, embed_dim)  # layer 2: friends of friends
 
    def encode(self, x, edge_index):
        """Turn every user's features + friends into an embedding."""
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
 
    def decode(self, z, pairs):
        """Score each pair of users: bigger dot product = more likely friends."""
        return (z[pairs[0]] * z[pairs[1]]).sum(dim=-1)
 
 
model = GCNLinkPredictor(train_data.num_features, HIDDEN_DIM, EMBED_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
 
 
# --------------------------------------------------------------------------
# 2. One training step
# --------------------------------------------------------------------------
def train():
    model.train()
    optimizer.zero_grad()
 
    # Embeddings are built ONLY from the training graph (hidden friendships stay hidden).
    z = model.encode(train_data.x, train_data.edge_index)
 
    # Real friendships the model should score high...
    pos_pairs = train_data.pos_edge_label_index
    # ...and fresh fake non-friendships every epoch that it should score low.
    neg_pairs = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=pos_pairs.size(1),
    )
 
    pairs = torch.cat([pos_pairs, neg_pairs], dim=1)
    labels = torch.cat([torch.ones(pos_pairs.size(1)), torch.zeros(neg_pairs.size(1))]).to(device)
 
    loss = F.binary_cross_entropy_with_logits(model.decode(z, pairs), labels)
    loss.backward()
    optimizer.step()
    return loss.item()
 
 
# ----------------------------------------------------------------------------------
# 3. Evaluation with Areas under the Curve and Average Precision
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
 
    pairs = torch.cat([data.pos_edge_label_index, data.neg_edge_label_index], dim=1)
    labels = torch.cat([
        torch.ones(data.pos_edge_label_index.size(1)),
        torch.zeros(data.neg_edge_label_index.size(1)),
    ]).numpy()
    scores = model.decode(z, pairs).sigmoid().cpu().numpy()
 
    return roc_auc_score(labels, scores), average_precision_score(labels, scores)
 
 
# --------------------------------------------------------------------------
# 4. Precision@K on hidden friendships
#    For each sampled user: recommend the K highest-scoring non-friends and
#    count how many are friendships that were hidden in the test split.
# --------------------------------------------------------------------------
@torch.no_grad()
def precision_at_k(data, k=K, num_users=NUM_EVAL_USERS, seed=SEED):
    model.eval()
    z = model.encode(data.x, data.edge_index)
 
    # The answer key: real friendships the model never saw.
    hidden = defaultdict(set)
    for a, b in data.pos_edge_label_index.t().tolist():
        hidden[a].add(b)
        hidden[b].add(a)   # stored in one direction only, so add both
 
    # Friends the model already knew about (train + validation edges).
    known = defaultdict(set)
    for a, b in data.edge_index.t().tolist():
        known[a].add(b)
        known[b].add(a)
 
    # Only evaluate users who actually have at least one hidden friend.
    rng = random.Random(seed)
    users = rng.sample(sorted(hidden), min(num_users, len(hidden)))
 
    precisions = []
    for u in users:
        scores = z @ z[u]                     # similarity to every user
        scores[u] = float("-inf")             # don't recommend yourself
        for v in known[u]:
            scores[v] = float("-inf")         # don't recommend current friends
        top_k = torch.topk(scores, k).indices.tolist()
        precisions.append(sum(v in hidden[u] for v in top_k) / k)
 
    return sum(precisions) / len(precisions)
 
 
# --------------------------------------------------------------------------
# 5. Training loop (keeps the best model according to validation AUC)
# --------------------------------------------------------------------------
best_val_auc = 0.0
best_state = None
 
for epoch in range(1, EPOCHS + 1):
    loss = train()
    if epoch % 10 == 0 or epoch == 1:
        val_auc, val_ap = evaluate(val_data)
        print(f"Epoch {epoch:3d} | loss {loss:.4f} | validation Area Under the Curve {val_auc:.4f} | validation Average Precision {val_ap:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
 
# Reload the best version and score it ONCE on the test set.
model.load_state_dict(best_state)
test_auc, test_ap = evaluate(test_data)
test_prec = precision_at_k(test_data)
 
print("\n=== FINAL TEST RESULTS (GCN) ===")
print(f"AUC:            {test_auc:.4f}")
print(f"Average Prec.:  {test_ap:.4f}")
print(f"Precision@{K}:   {test_prec:.4f}  (averaged over up to {NUM_EVAL_USERS} users)")
 
# Save the final numbers so step3_compare.py can put them in the
# comparison table without retraining the model.

with open("gcn_results.json", "w") as f:
    json.dump({"AUC": test_auc, "AP": test_ap, f"Precision@{K}": test_prec,
               "K": K, "num_eval_users": NUM_EVAL_USERS}, f, indent=2)
print("Saved gcn_results.json")