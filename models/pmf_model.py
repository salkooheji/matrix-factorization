"""Probabilistic Matrix Factorization trained with mini-batch SGD.

Ratings are modeled as: mu + user_bias + item_bias + U[u] . V[i], where the
latent factor matrices U and V have Gaussian priors (the L2 penalty is the
MAP estimate of those priors).

The final model is a small ensemble: several PMF models trained from
different random initializations whose predictions are averaged. This
reduces variance the same way averaging several independent estimates does.
Hyperparameters were selected with a held-out validation set and learning
curves (see the notebook and reports/pmf_convergence.png).
"""

import os

import matplotlib

matplotlib.use("Agg")  # render plots to files, no GUI needed
import matplotlib.pyplot as plt
import numpy as np


class PMF:
    """Biased PMF trained with vectorized mini-batch SGD."""

    def __init__(self, n_factors=128, learning_rate=0.01, reg=0.07,
                 bias_reg=0.01, n_epochs=65, lr_decay=0.99,
                 batch_size=4096, random_state=42):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.bias_reg = bias_reg
        self.n_epochs = n_epochs
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.random_state = random_state
        self.history = {"train_mse": [], "test_rmse": []}

    def fit(self, train_data, test_data=None, n_users=None, n_items=None):
        """Train on (user_idx, item_idx, rating) index arrays.

        Mini-batches of ratings are processed with vectorized numpy updates
        (np.add.at scatters per-sample gradients), which is orders of
        magnitude faster than a pure-Python loop while performing the same
        stochastic gradient descent.
        """
        users, items, ratings = train_data
        ratings = ratings.astype(np.float32)
        rng = np.random.default_rng(self.random_state)

        self.mu = np.float32(ratings.mean())
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)
        self.U = rng.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        self.V = rng.normal(0, 0.1, (n_items, self.n_factors)).astype(np.float32)

        lr = self.learning_rate
        n = len(ratings)
        for epoch in range(self.n_epochs):
            # deterministic shuffling: one RNG stream per epoch
            order = np.random.default_rng(
                self.random_state * 100000 + epoch
            ).permutation(n)
            sq_err_sum = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                a, b, r = users[idx], items[idx], ratings[idx]
                Ua, Vb = self.U[a], self.V[b]
                err = (r - (self.mu + self.bu[a] + self.bi[b]
                            + np.sum(Ua * Vb, axis=1))).astype(np.float32)
                sq_err_sum += float(np.sum(err * err))
                np.add.at(self.bu, a, lr * (err - self.bias_reg * self.bu[a]))
                np.add.at(self.bi, b, lr * (err - self.bias_reg * self.bi[b]))
                np.add.at(self.U, a, lr * (err[:, None] * Vb - self.reg * Ua))
                np.add.at(self.V, b, lr * (err[:, None] * Ua - self.reg * Vb))

            train_mse = sq_err_sum / n
            self.history["train_mse"].append(train_mse)
            msg = (f"  seed {self.random_state} | epoch {epoch + 1:2d}/"
                   f"{self.n_epochs} | train MSE = {train_mse:.4f}")
            if test_data is not None:
                test_rmse = self.evaluate(*test_data)
                self.history["test_rmse"].append(test_rmse)
                msg += f" | test RMSE = {test_rmse:.4f}"
            print(msg, flush=True)
            lr *= self.lr_decay  # learning-rate decay for stable convergence
        return self

    def raw_predict(self, users, items):
        """Unclipped predictions (used by the ensemble average)."""
        return (self.mu + self.bu[users] + self.bi[items]
                + np.sum(self.U[users] * self.V[items], axis=1))

    def predict(self, users, items):
        """Predicted ratings clipped to the valid 1-5 scale."""
        return np.clip(self.raw_predict(users, items), 1, 5)

    def evaluate(self, users, items, ratings):
        """RMSE against known ratings."""
        preds = self.predict(users, items)
        return float(np.sqrt(np.mean((preds - ratings) ** 2)))

    def raw_full_matrix(self):
        """Unclipped dense predicted rating matrix."""
        return (self.mu + self.bu[:, None] + self.bi[None, :]
                + self.U @ self.V.T)


class PMFEnsemble:
    """Averages the predictions of several PMF models (different seeds).

    Each member sees the same data but starts from a different random
    initialization, so their individual errors are partly independent;
    averaging cancels part of that noise (variance reduction). Predictions
    are averaged BEFORE clipping to [1, 5].
    """

    def __init__(self, seeds=(42, 7, 123, 555, 999, 2024), **pmf_kwargs):
        self.seeds = seeds
        self.pmf_kwargs = pmf_kwargs
        self.models = []

    def fit(self, train_data, test_data=None, n_users=None, n_items=None):
        for i, seed in enumerate(self.seeds):
            print(f"Training PMF member {i + 1}/{len(self.seeds)} "
                  f"(seed {seed})...", flush=True)
            model = PMF(random_state=seed, **self.pmf_kwargs)
            # only the first member tracks the test curve (for the plot);
            # skipping it for the others saves time
            model.fit(train_data, test_data if i == 0 else None,
                      n_users=n_users, n_items=n_items)
            self.models.append(model)
        return self

    @property
    def history(self):
        return self.models[0].history

    def predict(self, users, items):
        raw = np.mean([m.raw_predict(users, items) for m in self.models], axis=0)
        return np.clip(raw, 1, 5)

    def evaluate(self, users, items, ratings):
        preds = self.predict(users, items)
        return float(np.sqrt(np.mean((preds - ratings) ** 2)))

    def full_matrix(self):
        """Averaged dense predicted rating matrix, clipped to [1, 5]."""
        acc = self.models[0].raw_full_matrix()
        for m in self.models[1:]:
            acc += m.raw_full_matrix()
        return np.clip(acc / len(self.models), 1, 5)

    def save_factors(self, dir_path="reports/pmf_factors"):
        """Save each member's learned latent factor matrices U and V."""
        os.makedirs(dir_path, exist_ok=True)
        for m in self.models:
            np.save(f"{dir_path}/U_seed{m.random_state}.npy", m.U)
            np.save(f"{dir_path}/V_seed{m.random_state}.npy", m.V)
            np.save(f"{dir_path}/user_bias_seed{m.random_state}.npy", m.bu)
            np.save(f"{dir_path}/item_bias_seed{m.random_state}.npy", m.bi)

    def plot_convergence(self, path="reports/pmf_convergence.png"):
        """Save the MSE-vs-iteration convergence plot (first member)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        h = self.history
        epochs = range(1, len(h["train_mse"]) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, h["train_mse"], marker="o", ms=3, label="Train MSE")
        if h["test_rmse"]:
            plt.plot(epochs, [r ** 2 for r in h["test_rmse"]],
                     marker="s", ms=3, label="Test MSE")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.title("PMF Training Convergence")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()