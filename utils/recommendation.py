"""Movie recommendation generation from predicted rating matrices."""

import os

import numpy as np
import pandas as pd


def generate_recommendations(user_id, model, top_n=10):
    """Return the top_n recommended movies for a user from one model.

    Parameters
    ----------
    user_id : int
        The raw MovieLens user id.
    model : dict
        A model bundle with keys:
          "predictions"  - 2D numpy array (users x movies) of predicted ratings
          "user_index"   - pd.Index mapping row -> user_id
          "movie_index"  - pd.Index mapping column -> movie_id
          "rated"        - dict user_id -> set of movie_ids seen in training
          "movies"       - movies DataFrame (movie_id, title, genres)
          "name"         - model name string ("SVD" or "PMF")

    Returns
    -------
    DataFrame with columns: rank, movie_id, title, predicted_rating
    """
    user_index = model["user_index"]
    if user_id not in user_index:
        raise ValueError(f"Unknown user id: {user_id}")

    row = user_index.get_loc(user_id)
    scores = pd.Series(model["predictions"][row], index=model["movie_index"])

    # never recommend movies the user has already rated (training set)
    seen = model["rated"].get(user_id, set())
    scores = scores.drop(labels=list(seen), errors="ignore")

    top = scores.sort_values(ascending=False).head(top_n)
    recs = pd.DataFrame({
        "rank": np.arange(1, len(top) + 1),
        "movie_id": top.index,
        "predicted_rating": top.values.round(3),
    })
    titles = model["movies"].set_index("movie_id")["title"]
    recs["title"] = recs["movie_id"].map(titles)
    return recs[["rank", "movie_id", "title", "predicted_rating"]]


def save_user_recommendations(user_id, svd_model, pmf_model, top_n=10,
                              dir_path="reports"):
    """Save combined SVD + PMF top-n recommendations for one user as CSV."""
    os.makedirs(dir_path, exist_ok=True)
    svd = generate_recommendations(user_id, svd_model, top_n)
    pmf = generate_recommendations(user_id, pmf_model, top_n)
    svd = svd.rename(columns={"predicted_rating": "svd_predicted_rating"})
    pmf = pmf.rename(columns={"predicted_rating": "pmf_predicted_rating"})
    svd["model"] = "SVD"
    pmf["model"] = "PMF"
    combined = pd.concat([
        svd[["model", "rank", "movie_id", "title", "svd_predicted_rating"]]
        .rename(columns={"svd_predicted_rating": "predicted_rating"}),
        pmf[["model", "rank", "movie_id", "title", "pmf_predicted_rating"]]
        .rename(columns={"pmf_predicted_rating": "predicted_rating"}),
    ])
    path = f"{dir_path}/user_{user_id}_recommendations.csv"
    combined.to_csv(path, index=False)
    return path