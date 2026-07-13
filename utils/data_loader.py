"""Data loading and preprocessing for the MovieLens 1M dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
RANDOM_STATE = 42


def load_ratings(path=f"{DATA_DIR}/ratings.dat"):
    """Load ratings.dat into a DataFrame (UserID::MovieID::Rating::Timestamp)."""
    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    return ratings


def load_users(path=f"{DATA_DIR}/users.dat"):
    """Load users.dat into a DataFrame (UserID::Gender::Age::Occupation::Zip)."""
    users = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
    )
    return users


def load_movies(path=f"{DATA_DIR}/movies.dat"):
    """Load movies.dat into a DataFrame (MovieID::Title::Genres)."""
    movies = pd.read_csv(
        path,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movie_id", "title", "genres"],
    )
    return movies


def preprocess_ratings(ratings):
    """Remove null values and duplicate user-movie pairs from the ratings."""
    ratings = ratings.dropna()
    ratings = ratings.drop_duplicates(subset=["user_id", "movie_id"])
    return ratings


def split_ratings(ratings, test_size=0.2):
    """Split ratings into train/test sets reproducibly (random_state=42)."""
    train_df, test_df = train_test_split(
        ratings, test_size=test_size, random_state=RANDOM_STATE
    )
    return train_df, test_df


if __name__ == "__main__":
    ratings = preprocess_ratings(load_ratings())
    train_df, test_df = split_ratings(ratings)
    print(f"Ratings: {len(ratings)} | Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Users: {ratings['user_id'].nunique()} | Movies: {ratings['movie_id'].nunique()}")