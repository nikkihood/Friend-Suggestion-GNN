# Graph Neural Network: Social Predictions

**Using Python to predict social media connections ("People You May Know")**

Algorithms Honors Project · Honors Category 6 (Research, Creative Exploration, and Practical Application)
Author: Nicole Hood

---

## Overview

A Graph Neural Network (GNN) is a deep learning model built for graph-structured data, where nodes represent entities and edges represent relationships. A GNN can learn from both a node's own features and the network around it.

This project applies those ideas in a Python prototype that simulates a social network and predicts potential friendships from:

- **Node features:** age, location, and shared interests
- **Graph structure:** mutual friends

The goal is a lightweight, explainable "People You May Know" (PYMK) recommender, plus a Flask web demo to explore the results.

> **Project status:** This README reflects the project plan and methodology as of **October 14, 2026**. Implementation, training, and evaluation are in progress. Features are listed as **Planned** until they are working in the demo, and **no accuracy results are claimed yet**.

---

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Roadmap](#project-roadmap)
- [Evaluation](#evaluation)
- [References](#references)
- [Acknowledgments](#acknowledgments)

---

## Features

| Feature | Status |
| --- | --- |
| Custom social network dataset (modified from a Kaggle dataset) | Done |
| Feature encoding (age bands, multi-hot interests, one-hot country) | Designed |
| Graph construction with NetworkX | Planned |
| Graph similarity baselines (Common Neighbors, Jaccard, Adamic-Adar, Random Walk with Restart) | Planned |
| 2-layer Graph Convolutional Network (PyTorch Geometric) | Planned |
| Link prediction via cosine similarity of embeddings | Planned |
| Evaluation with Precision@K on held-out edges | Planned |
| Flask demo with graph view, profile picker, and recommendations | Planned |
| Cold-start handling (users with no friends) | Planned |

---

## Dataset

A custom dataset was needed for this project, so an existing [Kaggle social media users dataset](https://www.kaggle.com/datasets/arindamsahoo/social-media-users) by A. Sahoo was modified with the help of Anthropic's Claude. The modified data is designed so that real friendships carry a measurable signal compared with random pairs:

| Signal | Friend pairs | Random pairs |
| --- | --- | --- |
| Average age gap | ~8 years | ~17 years |
| Average shared interests | ~0.9 | ~0.3 |
| Same country | ~61% | ~4% |

Roughly **11,900 users have no friends**. They are kept on purpose to test how the model recommends friends to someone with no existing connections (the cold-start case). Mutual friends already present in the data are used to support training and act as an additional matchmaking signal.

### Feature Encoding

Before the data goes into the GNN, it is transformed into a numerical feature matrix:

- **Age:** grouped into eight-year bands, then one-hot encoded
- **Interests:** each quoted string of interests is parsed into individual topics and encoded as a **multi-hot vector** (one binary slot per interest, since a user can have several)
- **Location:** **one-hot encoded** across 244 countries

The encoded columns are combined into a single node feature matrix.

---

## Methodology

The project uses a hybrid approach that combines classical graph similarity measures with a lightweight GNN.

### 1. Graph construction
Each user is a node with an attribute vector (age band, interests, country). Edges represent established friendships. The graph is built and manipulated with [NetworkX](https://networkx.org/en/).

### 2. Classical similarity baselines
These measures serve as baselines and candidate signals. Their exact contribution to prediction accuracy will be evaluated:

- Common Neighbors
- Jaccard Coefficient
- Adamic-Adar Index
- Random Walk with Restart

### 3. GNN and feature propagation
A two-layer Graph Convolutional Network built with [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) (`GCNConv`) aggregates information from first- and second-order neighborhoods. It combines graph structure (shared friends) with node features (matching interests, age, location) to produce latent user embeddings.

### 4. Link prediction and scoring
User embeddings are compared using **cosine similarity** to rank PYMK recommendations. This is a design choice that will be evaluated against test data.

### 5. Web demo
A lightweight [Flask](https://flask.palletsprojects.com/en/stable/) app with a route such as `/recommend/<user_id>` will run the recommendation logic and render results in an HTML template.

### Core pipeline

| Step | Description |
| --- | --- |
| **A. Data** | Load and encode the modified dataset; build the NetworkX graph |
| **B. Model** | Define a 2-layer GCN in PyTorch Geometric |
| **C. Train / Evaluate** | Split edges into train/test sets, predict held-out edges, compute Precision@K |
| **D. Serve** | Wrap the recommender in a Flask app |

---

## Tech Stack

- **Python**
- **NetworkX**: graph creation and analysis
- **PyTorch / PyTorch Geometric**: GNN models
- **pandas / NumPy**: data processing and encoding
- **Flask**: web demo

---

## Getting Started

> Setup is planned for Phase 1. The steps below are a template and will be finalized once the environment is built.

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

> Planned. Commands and file names are placeholders until the code is finalized.

```bash
# 1. Preprocess the data and build the graph
python preprocess.py

# 2. Train and evaluate the GCN
python train.py

# 3. Launch the Flask demo
flask run
```

Then open `http://127.0.0.1:5000/recommend/<user_id>` to see recommended connections and similarity scores for a chosen user.

---

## Project Roadmap

| Phase | Dates | Focus |
| --- | --- | --- |
| 1 | Oct 15 – Oct 21 | Environment setup and data modeling |
| 2 | Oct 22 – Oct 28 | Model architecture and training; link-prediction match logic; precision evaluation |
| 3 | Oct 29 – Nov 4 | Flask web app, real-time scoring, and graph visualization |
| 4 | Nov 5 – Nov 11 | Final report, slide deck, and demo dry runs |
| | **Nov 12** | **Class presentation and live demo** |

---

## Evaluation

Edges will be split into training and test sets. The model will predict the held-out edges and be scored with **Precision@K**. Results will be added to this section only after the model has been tested.

| Method | Precision@K |
| --- | --- |
| Common Neighbors | _TBD_ |
| Jaccard Coefficient | _TBD_ |
| Adamic-Adar | _TBD_ |
| Random Walk with Restart | _TBD_ |
| 2-layer GCN | _TBD_ |

---

## References

1. L. MacPherson and D. Gaule, "Graph Neural Networks: When Relationships Are the Signal," Snowflake. https://www.snowflake.com/en/artificial-intelligence/machine-learning/neural-network/graph-neural-network/
2. B. Sanchez-Lengeling, E. Reif, A. Pearce, and A. B. Wiltschko, "A Gentle Introduction to Graph Neural Networks," *Distill*, 2021. https://distill.pub/2021/gnn-intro/
3. Y. Yu, W. Qian, L. Zhang, and R. Gao, "A Graph-Neural-Network-Based Social Network Recommendation Algorithm Using High-Order Neighbor Information," *Sensors*, vol. 22, no. 19, Art. 7122, 2022. doi: 10.3390/s22197122
4. M. Moricz, Y. Dosbayev, and M. Berlyant, "PYMK: Friend Recommendation at MySpace," in *Proc. ACM SIGMOD '10*, 2010. doi: 10.1145/1807167.1807276
5. X. Wang, X. He, M. Wang, F. Feng, and T.-S. Chua, "Neural Graph Collaborative Filtering," in *Proc. 42nd Int. ACM SIGIR Conf.*, 2019, pp. 165–174. doi: 10.1145/3331184.3331267
6. X. He, K. Deng, X. Wang, Y. Li, Y.-D. Zhang, and M. Wang, "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation," in *Proc. 43rd Int. ACM SIGIR Conf.*, 2020, pp. 639–648. doi: 10.1145/3397271.3401063
7. NetworkX Developers, "NetworkX." https://networkx.org/en/
8. PyTorch Geometric, "PyG Documentation." https://pytorch-geometric.readthedocs.io/en/latest/
9. Pallets Projects, "Welcome to Flask." https://flask.palletsprojects.com/en/stable/
10. A. Sahoo, "Social-Media-Users-Dataset," Kaggle. https://www.kaggle.com/datasets/arindamsahoo/social-media-users
11. GeeksforGeeks, "One Hot Encoding in Machine Learning," 2026. https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/
12. Google Developers, "Categorical data: Vocabulary and one-hot encoding," *Machine Learning Crash Course*, 2025. https://developers.google.com/machine-learning/crash-course/categorical-data/one-hot-encoding

---

## Acknowledgments

- Anthropic's **Claude** was used to modify the Kaggle dataset to create the custom dataset for this project.
- **Gemini** was used for clarification of article wording and for methodology and resource recommendations. It is acknowledged as a tool used during preparation, not as academic evidence for the technical claims in this project.



