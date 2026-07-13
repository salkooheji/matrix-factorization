"""Full pipeline: trains both models and produces every required artifact.

Run once from the project root:  python generate_reports.py
Takes roughly 15-25 minutes (PMF trains 6 ensemble members).

Produces:
  processed/user_item_matrix.csv          normalized user-item matrix
  processed/user_index.npy / movie_index.npy   row/column id mappings
  reports/model_metrics.json              SVD_RMSE, PMF_RMSE, improvement
  reports/svd_predictions.npy             full SVD predicted matrix
  reports/pmf_predictions.npy             full PMF (ensemble) predicted matrix
  reports/pmf_factors/                    learned U and V matrices
  reports/pmf_convergence.png             MSE vs epoch
  reports/pmf_history.json                training curves (for the notebook)
  reports/rmse_comparison.png             benchmark vs SVD vs PMF bar chart
  reports/predicted_vs_actual.png         test-set scatter for both models
  reports/user_comparison.png             SVD vs PMF for a selected user
  reports/top_recommendations.png         most frequently recommended movies
  reports/user_<id>_recommendations.csv   top-10 recs for 3 selected users
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(".")
from models.pmf_model import PMFEnsemble
from models.svd_model import append_metric, compute_rmse, save_predictions, train_svd
from utils.data_loader import load_movies, load_ratings, preprocess_ratings, split_ratings
from utils.matrix_creation import create_user_item_matrix, normalize_matrix, save_matrix
from utils.recommendation import generate_recommendations, save_user_recommendations

os.makedirs("reports", exist_ok=True)

# ---------------- 1. Data ----------------
print("Loading and splitting data...", flush=True)
ratings = preprocess_ratings(load_ratings())
movies = load_movies()
train_df, test_df = split_ratings(ratings)

matrix = create_user_item_matrix(train_df)
normalized, user_means = normalize_matrix(matrix)
save_matrix(normalized)
user_index, movie_index = matrix.index, matrix.columns
np.save("processed/user_index.npy", user_index.values)
np.save("processed/movie_index.npy", movie_index.values)

known = test_df[
    test_df["user_id"].isin(user_index) & test_df["movie_id"].isin(movie_index)
]
test_rows = user_index.get_indexer(known["user_id"]).astype(np.int32)
test_cols = movie_index.get_indexer(known["movie_id"]).astype(np.int32)
test_vals = known["rating"].values.astype(np.float32)

# ---------------- 2. Benchmark CF model (baseline) ----------------
# Damped user+item bias model: mu + b_u + b_i. A simple collaborative
# filtering benchmark that both factorization models must beat.
print("Evaluating benchmark baseline...", flush=True)
damping = 25.0
tu = user_index.get_indexer(train_df["user_id"])
ti = movie_index.get_indexer(train_df["movie_id"])
tr = train_df["rating"].values.astype(np.float64)
mu = tr.mean()
u_cnt = np.bincount(tu, minlength=len(user_index))
i_cnt = np.bincount(ti, minlength=len(movie_index))
b_u = np.bincount(tu, weights=tr - mu, minlength=len(user_index)) / (damping + u_cnt)
b_i = np.bincount(ti, weights=tr - mu - b_u[tu], minlength=len(movie_index)) / (
    damping + i_cnt
)
bench_pred = np.clip(mu + b_u[test_rows] + b_i[test_cols], 1, 5)
bench_rmse = float(np.sqrt(np.mean((bench_pred - test_vals) ** 2)))
print(f"Benchmark (bias-only CF) RMSE: {bench_rmse:.4f}", flush=True)
append_metric("Benchmark_RMSE", bench_rmse)

# ---------------- 3. SVD ----------------
print("Training SVD (k=30)...", flush=True)
svd_pred_df = train_svd(matrix, k=30)
svd_rmse = compute_rmse(svd_pred_df, test_df)
print(f"SVD RMSE: {svd_rmse:.4f}", flush=True)
save_predictions(svd_pred_df)
append_metric("SVD_RMSE", svd_rmse)

# ---------------- 4. PMF ensemble ----------------
print("Training PMF ensemble (6 members, this is the long part)...", flush=True)
train_data = (
    tu.astype(np.int32), ti.astype(np.int32), tr.astype(np.float32)
)
test_data = (test_rows, test_cols, test_vals)
pmf = PMFEnsemble()
pmf.fit(train_data, test_data, n_users=len(user_index), n_items=len(movie_index))
pmf_rmse = pmf.evaluate(*test_data)
print(f"PMF (ensemble) RMSE: {pmf_rmse:.4f}", flush=True)

pmf.save_factors()
pmf.plot_convergence()
pmf_matrix = pmf.full_matrix()
np.save("reports/pmf_predictions.npy", pmf_matrix.astype(np.float32))
with open("reports/pmf_history.json", "w") as f:
    json.dump(pmf.history, f)
append_metric("PMF_RMSE", pmf_rmse)
improvement = (svd_rmse - pmf_rmse) / svd_rmse * 100
append_metric("PMF_vs_SVD_improvement_%", improvement)
print(f"Improvement over SVD: {improvement:.2f}%", flush=True)

# ---------------- 5. Model bundles for recommendations ----------------
rated = train_df.groupby("user_id")["movie_id"].agg(set).to_dict()
svd_bundle = {"predictions": svd_pred_df.values, "user_index": user_index,
              "movie_index": movie_index, "rated": rated, "movies": movies,
              "name": "SVD"}
pmf_bundle = {"predictions": pmf_matrix, "user_index": user_index,
              "movie_index": movie_index, "rated": rated, "movies": movies,
              "name": "PMF"}

# 3 evaluated users: most-active train user, median train user,
# and the user with the most test ratings
train_counts = train_df["user_id"].value_counts()
user_a = int(train_counts.index[0])
user_b = int(train_counts.index[len(train_counts) // 2])
test_counts = known["user_id"].value_counts()
user_c = int(next(u for u in test_counts.index if u not in (user_a, user_b)))
eval_users = [user_a, user_b, user_c]
with open("reports/eval_users.json", "w") as f:
    json.dump(eval_users, f)
print(f"Evaluated users: {eval_users}", flush=True)
for uid in eval_users:
    path = save_user_recommendations(uid, svd_bundle, pmf_bundle)
    print(f"  saved {path}", flush=True)

# ---------------- 6. Plots ----------------
print("Generating plots...", flush=True)

# RMSE comparison bar chart
plt.figure(figsize=(7, 5))
names = ["Benchmark\n(bias-only CF)", "SVD", "PMF (ensemble)"]
vals = [bench_rmse, svd_rmse, pmf_rmse]
bars = plt.bar(names, vals, color=["gray", "steelblue", "seagreen"])
for bar, v in zip(bars, vals):
    plt.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.4f}",
             ha="center")
plt.ylabel("Test RMSE (lower is better)")
plt.title("Model Comparison: Test RMSE")
plt.ylim(0.8, max(vals) + 0.05)
plt.tight_layout()
plt.savefig("reports/rmse_comparison.png", dpi=150)
plt.close()

# Predicted vs actual scatter (sampled test points)
rng = np.random.default_rng(42)
sample = rng.choice(len(test_vals), size=min(4000, len(test_vals)), replace=False)
svd_s = svd_pred_df.values[test_rows[sample], test_cols[sample]]
pmf_s = pmf_matrix[test_rows[sample], test_cols[sample]]
actual_s = test_vals[sample]
jitter = rng.normal(0, 0.06, len(sample))
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, preds, name in [(axes[0], svd_s, "SVD"), (axes[1], pmf_s, "PMF")]:
    ax.scatter(actual_s + jitter, preds, s=4, alpha=0.25)
    ax.plot([1, 5], [1, 5], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual rating")
    ax.set_title(f"{name}: Predicted vs Actual (test set)")
    ax.legend()
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Predicted rating")
plt.tight_layout()
plt.savefig("reports/predicted_vs_actual.png", dpi=150)
plt.close()

# Per-user comparison: the most-active user's test movies
u_test = known[known["user_id"] == user_a].head(12)
if len(u_test) > 0:
    rows = user_index.get_indexer(u_test["user_id"])
    cols = movie_index.get_indexer(u_test["movie_id"])
    titles = u_test["movie_id"].map(movies.set_index("movie_id")["title"])
    titles = [t[:25] for t in titles]
    x = np.arange(len(u_test))
    w = 0.27
    plt.figure(figsize=(12, 6))
    plt.bar(x - w, u_test["rating"].values, w, label="Actual")
    plt.bar(x, svd_pred_df.values[rows, cols], w, label="SVD predicted")
    plt.bar(x + w, pmf_matrix[rows, cols], w, label="PMF predicted")
    plt.xticks(x, titles, rotation=45, ha="right")
    plt.ylabel("Rating")
    plt.title(f"User {user_a}: SVD vs PMF predictions on held-out movies")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/user_comparison.png", dpi=150)
    plt.close()

# Most frequently recommended movies (PMF top-10 across 300 sampled users)
sample_users = rng.choice(user_index.values, size=300, replace=False)
counts = {}
for uid in sample_users:
    recs = generate_recommendations(int(uid), pmf_bundle, top_n=10)
    for t in recs["title"]:
        counts[t] = counts.get(t, 0) + 1
top15 = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
plt.figure(figsize=(10, 6))
plt.barh([t[:35] for t, _ in top15][::-1], [c for _, c in top15][::-1],
         color="seagreen")
plt.xlabel("Times in a user's top-10 (out of 300 sampled users)")
plt.title("Most Frequently Recommended Movies (PMF)")
plt.tight_layout()
plt.savefig("reports/top_recommendations.png", dpi=150)
plt.close()

print("\nAll artifacts generated. Final metrics:", flush=True)
with open("reports/model_metrics.json") as f:
    print(json.dumps(json.load(f), indent=2))
