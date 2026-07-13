"""Streamlit dashboard for the matrix factorization recommender system.

Run:  streamlit run app.py   (or: python -m streamlit run app.py)
Requires artifacts from `python generate_reports.py` to exist.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(".")
from utils.data_loader import load_movies, load_ratings, preprocess_ratings, split_ratings
from utils.recommendation import generate_recommendations

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommender: SVD vs PMF")

REQUIRED = ["reports/svd_predictions.npy", "reports/pmf_predictions.npy",
            "processed/user_index.npy", "processed/movie_index.npy"]
missing = [p for p in REQUIRED if not os.path.exists(p)]
if missing:
    st.error("Missing model artifacts. Run `python generate_reports.py` "
             f"first. Missing: {missing}")
    st.stop()


@st.cache_data(show_spinner="Loading data and models (first run only)...")
def load_everything():
    ratings = preprocess_ratings(load_ratings())
    movies = load_movies()
    train_df, test_df = split_ratings(ratings)
    user_index = pd.Index(np.load("processed/user_index.npy"))
    movie_index = pd.Index(np.load("processed/movie_index.npy"))
    svd_preds = np.load("reports/svd_predictions.npy")
    pmf_preds = np.load("reports/pmf_predictions.npy")
    rated = train_df.groupby("user_id")["movie_id"].agg(set).to_dict()
    return movies, train_df, test_df, user_index, movie_index, svd_preds, pmf_preds, rated


movies, train_df, test_df, user_index, movie_index, svd_preds, pmf_preds, rated = (
    load_everything()
)
titles = movies.set_index("movie_id")["title"]

svd_bundle = {"predictions": svd_preds, "user_index": user_index,
              "movie_index": movie_index, "rated": rated, "movies": movies,
              "name": "SVD"}
pmf_bundle = {"predictions": pmf_preds, "user_index": user_index,
              "movie_index": movie_index, "rated": rated, "movies": movies,
              "name": "PMF"}

user_id = st.number_input(
    f"Enter a user ID ({user_index.min()} - {user_index.max()})",
    min_value=1, step=1, value=1,
)

if user_id not in user_index:
    st.error(f"User ID {user_id} does not exist in the dataset. "
             f"Valid IDs range from {user_index.min()} to {user_index.max()} "
             "(a few IDs in that range may also be missing).")
    st.stop()

# ---- Top-rated movies for this user (training history) ----
st.subheader(f"User {user_id}: top-rated movies (training history)")
hist = train_df[train_df["user_id"] == user_id].sort_values(
    "rating", ascending=False
).head(10).copy()
hist["title"] = hist["movie_id"].map(titles)
st.dataframe(hist[["title", "rating"]].reset_index(drop=True))

# ---- Recommendations from both models ----
col1, col2 = st.columns(2)
svd_recs = generate_recommendations(int(user_id), svd_bundle, top_n=10)
pmf_recs = generate_recommendations(int(user_id), pmf_bundle, top_n=10)
with col1:
    st.subheader("SVD recommendations")
    st.dataframe(svd_recs.set_index("rank"))
with col2:
    st.subheader("PMF recommendations")
    st.dataframe(pmf_recs.set_index("rank"))

# ---- Visual comparison: SVD vs PMF on this user ----
st.subheader("SVD vs PMF: predicted ratings for this user")
u_test = test_df[
    (test_df["user_id"] == user_id) & test_df["movie_id"].isin(movie_index)
].head(12)
row = user_index.get_loc(user_id)
if len(u_test) > 0:
    cols_ = movie_index.get_indexer(u_test["movie_id"])
    labels = [titles.get(m, str(m))[:25] for m in u_test["movie_id"]]
    x = np.arange(len(u_test))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, u_test["rating"].values, w, label="Actual (held-out)")
    ax.bar(x, svd_preds[row, cols_], w, label="SVD predicted")
    ax.bar(x + w, pmf_preds[row, cols_], w, label="PMF predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Rating")
    ax.set_title(f"User {user_id}: held-out movies, actual vs predicted")
    ax.legend()
    st.pyplot(fig)
else:
    # user has no held-out ratings: compare the models on their top recs
    union = pd.Index(svd_recs["movie_id"]).union(pmf_recs["movie_id"])
    cols_ = movie_index.get_indexer(union)
    labels = [titles.get(m, str(m))[:25] for m in union]
    x = np.arange(len(union))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, svd_preds[row, cols_], w, label="SVD predicted")
    ax.bar(x + w / 2, pmf_preds[row, cols_], w, label="PMF predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Predicted rating")
    ax.set_title(f"User {user_id}: model scores on recommended movies")
    ax.legend()
    st.pyplot(fig)

st.caption("Models: SVD (scipy.sparse.linalg.svds with baseline biases) and "
           "PMF (mini-batch SGD ensemble). See reports/model_metrics.json "
           "for test RMSE.")
