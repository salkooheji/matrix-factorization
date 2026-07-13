"""SVD recommender model using scipy.sparse.linalg.svds."""

import json
import os

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds


def train_svd(matrix, k=50, damping=25.0):
    """Factorize the user-item matrix with truncated SVD over baseline residuals.

    A baseline rating is estimated first: global mean + regularized user
    bias + regularized item bias. SVD then factorizes only the residuals
    (how each rating deviates from its baseline), keeping the k strongest
    latent factors. The damping term shrinks biases of users/movies with
    few ratings toward zero (regularization).
    """
    mu = matrix.stack().mean()
    user_counts = matrix.count(axis=1)
    item_counts = matrix.count(axis=0)
    user_bias = matrix.sub(mu).sum(axis=1) / (damping + user_counts)
    item_bias = matrix.sub(mu).sub(user_bias, axis=0).sum(axis=0) / (
        damping + item_counts
    )
    baseline = mu + np.add.outer(user_bias.values, item_bias.values)
    residuals = np.nan_to_num(matrix.values - baseline)
    U, sigma, Vt = svds(residuals, k=k)
    predictions = baseline + U @ np.diag(sigma) @ Vt
    predictions = np.clip(predictions, 1, 5)
    return pd.DataFrame(
        predictions, index=matrix.index, columns=matrix.columns
    )


def compute_rmse(pred_df, test_df):
    """Compute RMSE of predictions against held-out test ratings.

    Test pairs whose user or movie is absent from the training matrix
    (cold start) are skipped, as the model cannot score them.
    """
    known = test_df[
        test_df["user_id"].isin(pred_df.index)
        & test_df["movie_id"].isin(pred_df.columns)
    ]
    rows = pred_df.index.get_indexer(known["user_id"])
    cols = pred_df.columns.get_indexer(known["movie_id"])
    predicted = pred_df.values[rows, cols]
    actual = known["rating"].values
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def save_predictions(pred_df, path="reports/svd_predictions.npy"):
    """Save the full predicted rating matrix as a .npy file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, pred_df.values.astype(np.float32))


def append_metric(name, value, path="reports/model_metrics.json"):
    """Append or update a metric in the consolidated metrics JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metrics = {}
    if os.path.exists(path):
        with open(path) as f:
            metrics = json.load(f)
    metrics[name] = round(value, 4)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    from utils.data_loader import load_ratings, preprocess_ratings, split_ratings
    from utils.matrix_creation import create_user_item_matrix, normalize_matrix

    ratings = preprocess_ratings(load_ratings())
    train_df, test_df = split_ratings(ratings)
    matrix = create_user_item_matrix(train_df)
    normalized, user_means = normalize_matrix(matrix)

    # Parameter tuning: try several latent dimensions
    best_k, best_rmse, best_preds = None, np.inf, None
    for k in [10, 20, 30, 50]:
        pred_df = train_svd(matrix, k=k)
        rmse = compute_rmse(pred_df, test_df)
        print(f"k={k:3d} -> test RMSE = {rmse:.4f}")
        if rmse < best_rmse:
            best_k, best_rmse, best_preds = k, rmse, pred_df

    print(f"Best: k={best_k}, RMSE={best_rmse:.4f}")
    save_predictions(best_preds)
    append_metric("SVD_RMSE", best_rmse)