"""Probabilistic Matrix Factorization trained with stochastic gradient descent."""

import json
import os

import matplotlib

matplotlib.use("Agg")  # render plots to files, no GUI needed
import matplotlib.pyplot as plt
import numpy as np


class PMF:
    """PMF with user/item biases, L2 regularization and SGD training.

    Ratings are modeled as: mu + user_bias + item_bias + U[u] . V[i]
    where U and V are latent factor matrices with Gaussian priors
    (the L2 penalty is the MAP estimate of those priors).
    """

    def __init__(self, n_factors=30, learning_rate=0.01, reg=0.05,
                 n_epochs=30, lr_decay=0.9, random_state=42):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.lr_decay = lr_decay
        self.random_state = random_state
        self.history = {"train_mse": [], "test_rmse": []}

    def fit(self, train_data, test_data=None, n_users=None, n_items=None):
        """Train with SGD on (user_idx, item_idx, rating) triples."""
        users, items, ratings = train_data
        rng = np.random.default_rng(self.random_state)

        self.mu = ratings.mean()
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.U = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.V = rng.normal(0, 0.1, (n_items, self.n_factors))

        lr = self.learning_rate
        n = len(ratings)
        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            sq_err_sum = 0.0
            for idx in order:
                u, i, r = users[idx], items[idx], ratings[idx]
                pred = self.mu + self.bu[u] + self.bi[i] + self.U[u] @ self.V[i]
                err = r - pred
                sq_err_sum += err * err
                self.bu[u] += lr * (err - self.reg * self.bu[u])
                self.bi[i] += lr * (err - self.reg * self.bi[i])
                u_row = self.U[u].copy()
                self.U[u] += lr * (err * self.V[i] - self.reg * u_row)
                self.V[i] += lr * (err * u_row - self.reg * self.V[i])

            train_mse = sq_err_sum / n
            self.history["train_mse"].append(train_mse)
            msg = f"Epoch {epoch + 1:2d}/{self.n_epochs} | train MSE = {train_mse:.4f}"
            if test_data is not None:
                test_rmse = self.evaluate(*test_data)
                self.history["test_rmse"].append(test_rmse)
                msg += f" | test RMSE = {test_rmse:.4f}"
            print(msg)
            lr *= self.lr_decay  # learning rate decay for stable convergence
        return self

    def predict(self, users, items):
        """Predict ratings for arrays of user/item indices, clipped to [1, 5]."""
        preds = (self.mu + self.bu[users] + self.bi[items]
                 + np.sum(self.U[users] * self.V[items], axis=1))
        return np.clip(preds, 1, 5)

    def evaluate(self, users, items, ratings):
        """RMSE against known ratings."""
        preds = self.predict(users, items)
        return float(np.sqrt(np.mean((preds - ratings) ** 2)))

    def full_matrix(self):
        """Dense predicted rating matrix for all users and items."""
        preds = self.mu + self.bu[:, None] + self.bi[None, :] + self.U @ self.V.T
        return np.clip(preds, 1, 5)

    def save_factors(self, dir_path="reports/pmf_factors"):
        """Save learned latent factor matrices U and V (and biases)."""
        os.makedirs(dir_path, exist_ok=True)
        np.save(f"{dir_path}/U.npy", self.U)
        np.save(f"{dir_path}/V.npy", self.V)
        np.save(f"{dir_path}/user_bias.npy", self.bu)
        np.save(f"{dir_path}/item_bias.npy", self.bi)

    def plot_convergence(self, path="reports/pmf_convergence.png"):
        """Save the MSE-vs-iteration convergence plot."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        epochs = range(1, len(self.history["train_mse"]) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.history["train_mse"], marker="o", label="Train MSE")
        if self.history["test_rmse"]:
            test_mse = [r ** 2 for r in self.history["test_rmse"]]
            plt.plot(epochs, test_mse, marker="s", label="Test MSE")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.title("PMF Training Convergence")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    from utils.data_loader import load_ratings, preprocess_ratings, split_ratings
    from utils.matrix_creation import create_user_item_matrix
    from models.svd_model import append_metric

    ratings = preprocess_ratings(load_ratings())
    train_df, test_df = split_ratings(ratings)
    matrix = create_user_item_matrix(train_df)
    user_index, movie_index = matrix.index, matrix.columns

    train = (
        user_index.get_indexer(train_df["user_id"]).astype(np.int32),
        movie_index.get_indexer(train_df["movie_id"]).astype(np.int32),
        train_df["rating"].values.astype(np.float64),
    )
    known = test_df[
        test_df["user_id"].isin(user_index) & test_df["movie_id"].isin(movie_index)
    ]
    test = (
        user_index.get_indexer(known["user_id"]).astype(np.int32),
        movie_index.get_indexer(known["movie_id"]).astype(np.int32),
        known["rating"].values.astype(np.float64),
    )

    model = PMF(n_factors=30, learning_rate=0.01, reg=0.05, n_epochs=30)
    model.fit(train, test, n_users=len(user_index), n_items=len(movie_index))

    pmf_rmse = model.evaluate(*test)
    model.save_factors()
    model.plot_convergence()
    append_metric("PMF_RMSE", pmf_rmse)

    with open("reports/model_metrics.json") as f:
        metrics = json.load(f)
    improvement = (metrics["SVD_RMSE"] - pmf_rmse) / metrics["SVD_RMSE"] * 100
    append_metric("PMF_vs_SVD_improvement_%", improvement)
    print(f"\nSVD RMSE: {metrics['SVD_RMSE']} | PMF RMSE: {pmf_rmse:.4f} "
          f"| Improvement: {improvement:.2f}%")