"""User-item interaction matrix creation and normalization."""

import os

import pandas as pd


def create_user_item_matrix(train_df):
    """Create a user-item interaction matrix from training ratings.

    Rows are users, columns are movies, values are ratings.
    Unrated movies are NaN at this stage.
    """
    matrix = train_df.pivot(index="user_id", columns="movie_id", values="rating")
    return matrix


def normalize_matrix(matrix):
    """Normalize the matrix by subtracting each user's mean rating.

    Mean-centering removes per-user rating bias (some users rate
    everything high, others low). Missing entries are filled with 0,
    which after centering means "no deviation from this user's average".

    Returns the normalized matrix and the user means (needed later to
    convert predictions back to the 1-5 rating scale).
    """
    user_means = matrix.mean(axis=1)
    normalized = matrix.sub(user_means, axis=0).fillna(0)
    return normalized, user_means


def save_matrix(normalized, path="processed/user_item_matrix.csv"):
    """Save the normalized user-item matrix as a CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    normalized.round(3).to_csv(path)


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    from utils.data_loader import load_ratings, preprocess_ratings, split_ratings

    ratings = preprocess_ratings(load_ratings())
    train_df, _ = split_ratings(ratings)
    matrix = create_user_item_matrix(train_df)
    normalized, user_means = normalize_matrix(matrix)
    save_matrix(normalized)
    print(f"Matrix shape: {normalized.shape}")
    print(f"Saved to processed/user_item_matrix.csv")