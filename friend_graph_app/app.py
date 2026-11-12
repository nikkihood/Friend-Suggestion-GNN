"""
Friend Graph Explorer: a Flask app for Nicole's GNN project.

What it does
  - Reads the two CSV files (it never changes them).
  - Serves a web page with an interactive friendship graph.
  - Answers three small requests from the page:
        /api/sample          -> a starting sample of users and their links
        /api/user/<id>       -> one user's details and friend list
        /api/expand/<id>     -> that user's friends, plus mutual-friend counts

Run it:
    pip install flask pandas
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import re
import random

import pandas as pd
from flask import Flask, jsonify, render_template, abort

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# Put the CSVs in this folder, or in the folder above it. Either works.
def find_file(name):
    for folder in (HERE, os.path.dirname(HERE)):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {name}. Put it next to app.py.")

USERS_FILE = find_file("SocialMediaUsersDataset_v2.csv")
EDGES_FILE = find_file("friendship_edges.csv")
SAMPLE_SIZE = 150      # how many users show up on the homepage
SEED = 42              # same seed = same starting sample every time

app = Flask(__name__)

# --------------------------------------------------------------------------
# Load the data once when the server starts
# --------------------------------------------------------------------------
users = pd.read_csv(USERS_FILE)
edges = pd.read_csv(EDGES_FILE, usecols=["UserID_A", "UserID_B"])

# One dictionary of details per user, keyed by UserID.
profiles = {}
for r in users.itertuples(index=False):
    interests = sorted(set(re.findall(r"'([^']+)'", str(r.Interests))))
    profiles[int(r.UserID)] = {
        "id": int(r.UserID),
        "name": r.Name,
        "gender": r.Gender,
        "age": int(r.Age),
        "ageGroup": r.AgeGroup,
        "city": r.City,
        "country": r.Country,
        "interests": interests,
    }

# friends[u] = set of UserIDs who are friends with u
friends = {uid: set() for uid in profiles}
for a, b in zip(edges.UserID_A, edges.UserID_B):
    friends[int(a)].add(int(b))
    friends[int(b)].add(int(a))

print(f"Loaded {len(profiles):,} users and {len(edges):,} friendships.")


def pick_demo_seeds(count=6):
    """Users whose friends know each other a lot, so halos show up in the demo."""
    rng = random.Random(SEED)
    candidates = rng.sample([u for u in profiles if 8 <= len(friends[u]) <= 24], 4000)
    scored = []
    for u in candidates:
        circle = friends[u]
        haloed = sum(1 for f in circle if friends[f] & circle)
        scored.append((haloed, u))
    scored.sort(reverse=True)
    return [u for _, u in scored[:count]]


DEMO_SEEDS = pick_demo_seeds()


def node(uid):
    """The information the page needs to draw one person."""
    p = dict(profiles[uid])
    p["friendCount"] = len(friends[uid])
    return p


# --------------------------------------------------------------------------
# Pages and data requests
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", sample_size=SAMPLE_SIZE, page="graph")


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/results")
def results():
    return render_template("results.html", page="results")


@app.route("/api/sample")
def sample():
    """
    A starting sample that looks like a real network and shows off the halo
    effect right away. A random sample of 100,000 people would have almost no
    links between them, so we grow the sample outward from a few seed users
    who have lots of mutual friends, adding their friends until we reach
    SAMPLE_SIZE people.
    """
    rng = random.Random(SEED)
    chosen, chosen_set = [], set()

    def add(u):
        if u not in chosen_set and len(chosen) < SAMPLE_SIZE:
            chosen.append(u)
            chosen_set.add(u)

    for s in DEMO_SEEDS:
        add(s)
        for f in sorted(friends[s]):
            add(f)
    # If we are still short, top up with friends of friends of the seeds.
    pool = sorted({g for s in DEMO_SEEDS for f in friends[s] for g in friends[f]} - chosen_set)
    rng.shuffle(pool)
    for u in pool:
        add(u)

    nodes = [node(u) for u in chosen]
    links = [{"from": u, "to": v}
             for u in chosen for v in friends[u] if v in chosen_set and u < v]
    suggested = [{"id": u, "name": profiles[u]["name"],
                  "haloed": sum(1 for f in friends[u] if friends[f] & friends[u])}
                 for u in DEMO_SEEDS[:4] if u in chosen_set]
    return jsonify({"nodes": nodes, "links": links, "suggested": suggested})


@app.route("/api/user/<int:uid>")
def user(uid):
    if uid not in profiles:
        abort(404)
    data = node(uid)
    data["friends"] = sorted(friends[uid])
    return jsonify(data)


@app.route("/api/expand/<int:uid>")
def expand(uid):
    """
    Everything the page needs when you click person `uid`:
      - their friends (as nodes)
      - links between those friends (so you can see the circle they share)
      - for each friend, how many OTHER friends of `uid` they also know.
        A friend who knows at least one of them is a "mutual" and gets a halo.
    """
    if uid not in profiles:
        abort(404)
    circle = friends[uid]
    nodes = []
    for f in sorted(circle):
        shared = friends[f] & circle            # friends of uid that f also knows
        n = node(f)
        n["mutualCount"] = len(shared)
        n["mutualWith"] = sorted(shared)
        nodes.append(n)
    links = [{"from": uid, "to": f} for f in circle]
    links += [{"from": a, "to": b}
              for a in circle for b in friends[a] if b in circle and a < b]
    return jsonify({"center": uid, "nodes": nodes, "links": links})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)