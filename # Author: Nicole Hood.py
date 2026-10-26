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