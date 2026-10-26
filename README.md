# Graph Neural Network: Social Connection Predictions

**Using Python to Predict Social Media Connections**

An Algorithms Honors Project exploring how Graph Neural Networks (GNNs) can predict potential social media connections by analyzing shared friends, user interests, and network structure.

## Overview

Social media platforms connect people through shared interests, mutual friends, and common activities. This project explores how graph-based machine learning can simulate a **"People You May Know" (PYMK)** recommendation system.

The application models a synthetic social network in Python, where users are represented as nodes and friendships are represented as edges. A proposed Graph Neural Network architecture combines user profile features with graph relationships to rank potential connections.

The goal is to investigate how graph analytics and machine learning can work together to identify meaningful connections in a simulated social network.

## Objectives

- Build a synthetic social network using Python.
- Represent users, interests, and friendships as graph data.
- Explore graph-based similarity algorithms for identifying potential connections.
- Develop a two-layer Graph Convolutional Network (GCN) to learn user representations.
- Investigate link prediction techniques for recommending potential friendships.
- Evaluate recommendation performance using metrics such as Precision@K.
- Develop a lightweight web interface to demonstrate the recommendation process.

## How It Works

The proposed system follows four main stages:

### 1. Synthetic Social Network Generation

Create a simulated network using NetworkX.

- **Nodes:** Individual users.
- **Edges:** Existing friendships.
- **Node features:** User interests, such as coding, gaming, music, and hiking.

Using synthetic data allows the project to explore recommendation behavior without requiring real social media user data.

### 2. Graph Analysis and Similarity

Analyze relationships between users using graph-based similarity measures:

- **Common Neighbors:** Counts the friends two users have in common.
- **Jaccard Coefficient:** Measures the overlap between their friend groups.
- **Adamic-Adar Index:** Gives greater weight to shared connections that are less common.
- **Random Walk with Restart:** Explores network proximity by traversing the graph while periodically returning to a starting user.

These measures are proposed signals for identifying candidate connections.

### 3. Graph Neural Network and Link Prediction

A planned two-layer Graph Convolutional Network (GCN), built with PyTorch Geometric, will aggregate information from users' neighborhoods.

The model is intended to combine:

- User interests and profile features.
- Existing friendships and shared connections.
- Learned user embeddings representing network context.

Cosine similarity is proposed as a scoring method for ranking potential connections. The system's actual predictive performance will be assessed through testing rather than assumed in advance.

### 4. Web-Based Demonstration

A planned Flask web application will provide an interactive way to explore the recommendation system.

The intended interface will allow users to:

- Select a simulated user profile.
- View
