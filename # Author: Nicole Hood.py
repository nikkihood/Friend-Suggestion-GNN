# Author: Nicole Hood
# Date Modified 10/26/2026
# GNN Graph Setup - Turn CSV Files into a graph a GNN can read.
# What this file does: 
# 1. Loads the users file and the friendship edge list 
# 2. Gives every user a node number (as required by the GNN)
# 3. Builds a feature vector for each users( age group + interests + country)
# 4. Builds the edge list in node-number form
# 5. Hides some real friendships (validations + tests) so we can later check whether the model can predict them. This helps avoid leakage



# Import Libraries Numpy, Pandas, PyTorch, Scikit

import re
import numpy as np 
import pandas as pd
import torch 
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import Data
import torch_geometic.transforms as T
from torch_geometric.utils import to_undirected 

#--------------------------------------------------------
#load in the files 
USERS_FILE = "SocialMediaUsersDataset_v2.csv"
EDGES_FILE = "friendship_edges.csv"
SEED= 42 # SEED: Fixes the starting point for all random choices in this project
# Using the same seed means every run produces the same split and results,
# so I can debug changes, compare the baselines and the GNN on identical
# data, and let others reproduce my numbers. The value 42 is arbitrary;
# any integer works.
torch.manual_seed(SEED)
#---------------------------------------------------------
#load the data into nodes and edges and verify result
users= pd.read_csv(USERS_FILE)
edges = pd.read_csv(EDGES_FILE)
print(f"Users: {len(users):,}    Friendships: {len(edges):,}")

#------------------------------------------------------------
#Give each user a node number 
# UserIDs are not 0...n-1 but the GNN need row numbers starting at 0
users ["node_id"] = np.arrange(len(users))
userid_to_node= dict(zip(users["UserID"], users["node_id"]))
#----------------------------------------------------------------
#build the node features ( one row per user)

#Age group -> one-hot ( a column of 0/1 for each group)
age_features = pd.get_dummies(users["AgeGroup"]).values.astype(np.float32)

#Country - One hot (big but fine for a prototype)
country_features= pd.get_dummies(users["cCountry"]).values.astype(np.float32)

#Intrests Multi-Hot vector( a 1 for every intrest the user has)
# for example the column may look like "'Music','Travel','Music'" so ewe pull out the duplicate words between the quotes and collapse to a single 1
interest_lists = users["Interests"].apply(lambda s: re.findall(r"'([^']+)'", s))
mlb = MultiLabelBinarizer()
interest_features = mlb.fit_transform(interest_lists).astype(np.float32)
