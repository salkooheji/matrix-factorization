# Matrix Factorization Recommender System

A movie recommender system built on the MovieLens 1M dataset using two
matrix factorization techniques: Singular Value Decomposition (SVD) and
Probabilistic Matrix Factorization (PMF). Includes an interactive
Streamlit dashboard for generating and comparing recommendations.

## Setup

1. Install dependencies:
```
   pip install -r requirements.txt
```
2. Download the [MovieLens 1M dataset](https://files.grouplens.org/datasets/movielens/ml-1m.zip),
   extract it, and place `ratings.dat`, `users.dat`, and `movies.dat` in the `data/` folder.

## Running the Dashboard

```
streamlit run app.py
```

Enter a user ID to see the user's top-rated movies, recommendations from
both models, and a visual comparison of SVD vs PMF predictions.

## Approach

1. Load and preprocess the dataset, split into train/test (`random_state=42`).
2. Build a normalized user-item interaction matrix from training data.
3. Train SVD (`scipy.sparse.linalg.svds`) and PMF (gradient descent with
   regularization), evaluate both with RMSE on the test set.
4. Generate top-10 recommendations per user and compare the models.