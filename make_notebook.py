"""Builds Movie_Recommender_System.ipynb (EDA + analysis notebook).

Run once:  python make_notebook.py
Then open the notebook and Run All (requires generate_reports.py artifacts).
"""

import json


def md(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    md("""# Movie Recommender System: EDA and Model Analysis

Exploratory data analysis of the MovieLens 1M dataset, and analysis of the
SVD and PMF matrix factorization models (training curves, overfitting,
interpretability, and per-user recommendation quality).

Run `python generate_reports.py` before executing this notebook."""),

    code("""import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(".")
from utils.data_loader import load_movies, load_ratings, load_users, preprocess_ratings, split_ratings

ratings = preprocess_ratings(load_ratings())
movies = load_movies()
users = load_users()
train_df, test_df = split_ratings(ratings)
print(f"Ratings: {len(ratings)} | Users: {ratings.user_id.nunique()} | Movies: {ratings.movie_id.nunique()}")
print(f"Train: {len(train_df)} | Test: {len(test_df)} (random_state=42)")"""),

    md("""## 1. Exploratory Data Analysis

Key questions: how are ratings distributed, how sparse is the user-item
matrix, and how unevenly is activity spread across users and movies?"""),

    code("""sparsity = 1 - len(ratings) / (ratings.user_id.nunique() * ratings.movie_id.nunique())
print(f"Matrix sparsity: {sparsity:.2%} of user-movie pairs have NO rating")
print(f"Mean rating: {ratings.rating.mean():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
ratings.rating.value_counts().sort_index().plot.bar(ax=axes[0], color="steelblue")
axes[0].set_title("Rating distribution")
axes[0].set_xlabel("Rating"); axes[0].set_ylabel("Count")

ratings.user_id.value_counts().plot.hist(bins=60, ax=axes[1], color="seagreen")
axes[1].set_title("Ratings per user (min 20 by dataset design)")
axes[1].set_xlabel("Number of ratings")

ratings.movie_id.value_counts().plot.hist(bins=60, ax=axes[2], color="indianred")
axes[2].set_title("Ratings per movie (long tail)")
axes[2].set_xlabel("Number of ratings")
plt.tight_layout()"""),

    code("""# Genre popularity
genre_counts = movies.genres.str.split("|").explode().value_counts()
genre_counts.plot.barh(figsize=(8, 6), color="steelblue")
plt.title("Number of movies per genre")
plt.xlabel("Movies")
plt.tight_layout()"""),

    md("""**Insights:** ratings skew positive (4 is the most common value, mean
about 3.58) because people mostly watch films they expect to like. The
matrix is ~95.7% empty, which is exactly why matrix factorization is
needed: it generalizes from the few observed entries via latent factors.
Both users and movies follow long-tail distributions - a few very active
users / very popular movies and many rare ones - so regularization is
essential to avoid overfitting the rare rows and columns."""),

    md("""## 2. Model metrics and learning curves"""),

    code("""with open("reports/model_metrics.json") as f:
    metrics = json.load(f)
metrics"""),

    code("""with open("reports/pmf_history.json") as f:
    history = json.load(f)
epochs = np.arange(1, len(history["train_mse"]) + 1)
plt.figure(figsize=(9, 5))
plt.plot(epochs, np.sqrt(history["train_mse"]), marker="o", ms=3, label="Train RMSE")
plt.plot(epochs, history["test_rmse"], marker="s", ms=3, label="Test RMSE")
plt.xlabel("Epoch"); plt.ylabel("RMSE")
plt.title("PMF learning curves (first ensemble member)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()"""),

    md("""**Overfitting and when to stop training.** Train RMSE keeps falling
throughout, while test RMSE flattens out - the gap between the curves is
the amount of memorization. We prevent runaway overfitting with (1) L2
regularization on the factors (reg=0.07) and biases (0.01), the MAP
version of PMF's Gaussian priors, (2) learning-rate decay, and (3) an
epoch budget chosen by validation: during tuning, a 5% validation split
showed the validation RMSE reaching its minimum around epoch 60 and
degrading afterwards, so we stop at 65 epochs when training on the full
training set (slightly later because there is 5% more data). Training
longer would only increase the train/test gap. The final PMF averages 6
such models trained from different random initializations, which reduces
prediction variance without any additional risk of overfitting."""),

    md("""## 3. Global interpretability: what do the latent factors encode?

Each movie is a vector in V. Looking at which (well-known) movies sit at
the extremes of a latent dimension reveals what "taste axis" it captures."""),

    code("""V = np.load("reports/pmf_factors/V_seed42.npy")
movie_index = pd.Index(np.load("processed/movie_index.npy"))
titles = movies.set_index("movie_id")["title"]
pop = train_df.movie_id.value_counts()
popular = movie_index.isin(pop[pop >= 200].index)  # well-known movies only

for f in range(3):
    vals = V[:, f]
    order = np.argsort(vals)
    low = [titles[movie_index[i]] for i in order if popular[i]][:5]
    high = [titles[movie_index[i]] for i in order[::-1] if popular[i]][:5]
    print(f"--- Factor {f} ---")
    print("  low :", "; ".join(low))
    print("  high:", "; ".join(high))"""),

    code("""# Similarity analysis: nearest neighbours in latent space
def similar_movies(title_query, top_n=8):
    matches = movies[movies.title.str.contains(title_query, case=False)]
    mid = matches.movie_id.iloc[0]
    i = movie_index.get_loc(mid)
    v = V[i] / np.linalg.norm(V[i])
    sims = (V / np.linalg.norm(V, axis=1, keepdims=True)) @ v
    best = np.argsort(sims)[::-1][1:top_n + 1]
    return pd.DataFrame({"title": [titles[movie_index[j]] for j in best],
                         "cosine_similarity": sims[best].round(3)})

similar_movies("Toy Story")"""),

    md("""**Insights:** the factors are not hand-labeled, but extremes clearly
group movies by audience taste (e.g. mainstream blockbusters vs art-house,
family films vs adult drama). The cosine-similarity probe confirms the
latent space is meaningful: neighbours of an animated family film are
other animated/family films. This is the model's global logic: a user is
recommended movies whose latent vectors align with their taste vector."""),

    md("""## 4. Per-user analysis (2 train users + 1 test user)

`generate_reports.py` saved top-10 recommendations for three users: the
most active user, a median-activity user, and the user with the most
held-out (test) ratings. Here we quantify how accurate each model was for
them and explain why."""),

    code("""svd_preds = np.load("reports/svd_predictions.npy")
pmf_preds = np.load("reports/pmf_predictions.npy")
user_index = pd.Index(np.load("processed/user_index.npy"))

known = test_df[test_df.user_id.isin(user_index) & test_df.movie_id.isin(movie_index)]
train_counts = train_df.user_id.value_counts()
eval_users = [int(train_counts.index[0]),
              int(train_counts.index[len(train_counts) // 2]),
              int(known.user_id.value_counts().index[0])]

for uid in eval_users:
    u = known[known.user_id == uid]
    r = user_index.get_loc(uid)
    c = movie_index.get_indexer(u.movie_id)
    svd_rmse = np.sqrt(np.mean((svd_preds[r, c] - u.rating.values) ** 2))
    pmf_rmse = np.sqrt(np.mean((pmf_preds[r, c] - u.rating.values) ** 2))
    print(f"user {uid}: {len(u)} held-out ratings | "
          f"train ratings: {train_counts.get(uid, 0)} | "
          f"SVD RMSE {svd_rmse:.3f} | PMF RMSE {pmf_rmse:.3f} | "
          f"rating std {u.rating.std():.2f}")"""),

    md("""**Why one train user is predicted better than the other (local
interpretability).** The very active user has hundreds of training
ratings, so their latent taste vector is estimated from abundant evidence
and predictions for them are accurate. The median-activity user
contributes far fewer ratings: their factor vector is pulled toward zero
by regularization, so predictions fall back toward the user/item biases
(i.e. "popular average"), which costs accuracy - especially when the
user's actual ratings are highly variable (large rating std). In short:
recommendation accuracy grows with the amount and consistency of a user's
history, which is the classic cold-ish-start behaviour of matrix
factorization. For each user, a movie is recommended because the dot
product of their taste vector with that movie's latent vector (plus the
movie's bias) is high - the recommendation CSVs and the dashboard's
comparison chart make this visible per user."""),

    md("""## 5. Conclusions

- Both factorization models clearly beat the bias-only collaborative
  filtering benchmark (see `reports/rmse_comparison.png`).
- PMF (SGD ensemble) outperforms truncated SVD because it optimizes only
  over observed ratings instead of treating missing entries as zeros, and
  meets the required thresholds (see `reports/model_metrics.json`).
- Regularization + learning-rate decay + validation-chosen epoch budget
  keep the model from overfitting despite 95.7% sparsity."""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("Movie_Recommender_System.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Wrote Movie_Recommender_System.ipynb")
