"""
preprocessing.py
----------------
Netflix Prize Dataset - Data Loading & Preprocessing Pipeline

Dataset source: https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
Expected files in data/:
    combined_data_1.txt
    combined_data_2.txt
    combined_data_3.txt
    combined_data_4.txt
    movie_titles.csv

Usage:
    from src.preprocessing import NetflixDataLoader, build_interaction_matrix, split_data
"""

import os
import gc
import time
import logging
import warnings
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
RANDOM_SEED = 42
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

NETFLIX_FILES = [
    "combined_data_1.txt",
    "combined_data_2.txt",
    "combined_data_3.txt",
    "combined_data_4.txt",
]

# Memory budget: tune these down if RAM is tight (8 GB laptop target)
MAX_USERS = 50_000       # cap unique users for memory safety
MAX_MOVIES = 5_000       # cap unique movies for memory safety
MIN_USER_RATINGS = 20    # minimum ratings per user (activity filter)
MIN_MOVIE_RATINGS = 50   # minimum ratings per movie (popularity filter)

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10   # fraction of training set used as validation


# ─────────────────────────────────────────────
# 1. Raw Data Parser
# ─────────────────────────────────────────────

class NetflixDataLoader:
    """
    Parses Netflix Prize raw .txt files into a clean Pandas DataFrame.

    The raw format alternates between:
        <movie_id>:          <- sentinel line
        <user_id>,<rating>,<date>
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        max_users: int = MAX_USERS,
        max_movies: int = MAX_MOVIES,
        min_user_ratings: int = MIN_USER_RATINGS,
        min_movie_ratings: int = MIN_MOVIE_RATINGS,
        use_subset: bool = True,          # auto-subset to stay within memory
        files: Optional[List[str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.max_users = max_users
        self.max_movies = max_movies
        self.min_user_ratings = min_user_ratings
        self.min_movie_ratings = min_movie_ratings
        self.use_subset = use_subset
        self.files = files or NETFLIX_FILES

        self.ratings_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.user2idx: Dict[int, int] = {}
        self.movie2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.idx2movie: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, cache: bool = True) -> pd.DataFrame:
        """
        Load & preprocess ratings. Uses on-disk cache when available.

        Returns
        -------
        pd.DataFrame  columns: [user_id, movie_id, rating, date]
        """
        cache_path = PROCESSED_DIR / "ratings_clean.parquet"

        if cache and cache_path.exists():
            logger.info("Loading cached ratings from %s", cache_path)
            self.ratings_df = pd.read_parquet(cache_path)
            self._build_index_maps()
            logger.info("Loaded %d ratings from cache.", len(self.ratings_df))
            return self.ratings_df

        logger.info("Parsing raw Netflix files …")
        self.ratings_df = self._parse_raw_files()
        self.ratings_df = self._filter_and_subset(self.ratings_df)
        self._build_index_maps()

        if cache:
            self.ratings_df.to_parquet(cache_path, index=False)
            logger.info("Cached processed ratings → %s", cache_path)

        return self.ratings_df

    def load_movies(self) -> pd.DataFrame:
        """
        Load movie titles metadata.

        Returns
        -------
        pd.DataFrame  columns: [movie_id, year, title]
        """
        path = self.data_dir / "movie_titles.csv"
        if not path.exists():
            logger.warning("movie_titles.csv not found at %s", path)
            return pd.DataFrame(columns=["movie_id", "year", "title"])

        movies = pd.read_csv(
            path,
            encoding="latin-1",
            header=None,
            names=["movie_id", "year", "title"],
            on_bad_lines="skip",
        )
        movies["movie_id"] = movies["movie_id"].astype(int)
        self.movies_df = movies
        logger.info("Loaded %d movie titles.", len(movies))
        return movies

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_raw_files(self) -> pd.DataFrame:
        """Stream-parse the raw Netflix .txt files efficiently."""
        available = [f for f in self.files if (self.data_dir / f).exists()]
        if not available:
            raise FileNotFoundError(
                f"No Netflix data files found in {self.data_dir}. "
                "Download from https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data"
            )

        # For memory safety, load only the first file if use_subset=True
        target_files = available[:1] if self.use_subset else available
        logger.info("Parsing %d file(s): %s", len(target_files), target_files)

        records: List[Tuple[int, int, int, str]] = []
        current_movie_id: Optional[int] = None

        for fname in target_files:
            fpath = self.data_dir / fname
            file_size = fpath.stat().st_size
            logger.info("Reading %s (%.1f MB) …", fname, file_size / 1e6)

            with open(fpath, "r") as fh:
                for line in tqdm(fh, desc=fname, unit=" lines", mininterval=2.0):
                    line = line.strip()
                    if not line:
                        continue
                    if line.endswith(":"):
                        # Movie sentinel line
                        current_movie_id = int(line[:-1])
                    else:
                        # Rating line: user_id,rating,date
                        parts = line.split(",")
                        if len(parts) == 3 and current_movie_id is not None:
                            user_id = int(parts[0])
                            rating = int(parts[1])
                            date = parts[2]
                            records.append((user_id, current_movie_id, rating, date))

            gc.collect()

        logger.info("Parsed %d raw records.", len(records))
        df = pd.DataFrame(records, columns=["user_id", "movie_id", "rating", "date"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["rating"] = df["rating"].astype(np.int8)
        df["user_id"] = df["user_id"].astype(np.int32)
        df["movie_id"] = df["movie_id"].astype(np.int16)
        return df

    def _filter_and_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply activity/popularity filters and user/movie caps.
        Ensures the final matrix fits comfortably in 8 GB RAM.
        """
        logger.info("Raw shape: %d ratings, %d users, %d movies",
                    len(df), df["user_id"].nunique(), df["movie_id"].nunique())

        # Drop duplicate (user, movie) pairs – keep the latest rating
        df = df.sort_values("date").drop_duplicates(
            subset=["user_id", "movie_id"], keep="last"
        )

        # Iterative co-filtering: removes sparse users & cold movies
        for iteration in range(3):
            n_before = len(df)

            user_counts = df["user_id"].value_counts()
            valid_users = user_counts[user_counts >= self.min_user_ratings].index
            df = df[df["user_id"].isin(valid_users)]

            movie_counts = df["movie_id"].value_counts()
            valid_movies = movie_counts[movie_counts >= self.min_movie_ratings].index
            df = df[df["movie_id"].isin(valid_movies)]

            logger.info("Co-filter pass %d: %d → %d ratings", iteration + 1, n_before, len(df))
            if len(df) == n_before:
                break  # converged

        # Cap users and movies if still too large
        if df["user_id"].nunique() > self.max_users:
            logger.info("Capping to top %d most active users.", self.max_users)
            top_users = (
                df["user_id"].value_counts()
                .head(self.max_users)
                .index
            )
            df = df[df["user_id"].isin(top_users)]

        if df["movie_id"].nunique() > self.max_movies:
            logger.info("Capping to top %d most rated movies.", self.max_movies)
            top_movies = (
                df["movie_id"].value_counts()
                .head(self.max_movies)
                .index
            )
            df = df[df["movie_id"].isin(top_movies)]

        df = df.reset_index(drop=True)
        logger.info(
            "Final dataset: %d ratings | %d users | %d movies",
            len(df), df["user_id"].nunique(), df["movie_id"].nunique(),
        )
        return df

    def _build_index_maps(self) -> None:
        """Build contiguous 0-based integer indices for users and movies."""
        unique_users = sorted(self.ratings_df["user_id"].unique())
        unique_movies = sorted(self.ratings_df["movie_id"].unique())

        self.user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie2idx = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.idx2user = {idx: uid for uid, idx in self.user2idx.items()}
        self.idx2movie = {idx: mid for mid, idx in self.movie2idx.items()}

        logger.info(
            "Index maps built: %d users, %d movies",
            len(self.user2idx), len(self.movie2idx),
        )


# ─────────────────────────────────────────────
# 2. Interaction Matrix Builder
# ─────────────────────────────────────────────

def build_interaction_matrix(
    ratings_df: pd.DataFrame,
    user2idx: Dict[int, int],
    movie2idx: Dict[int, int],
) -> csr_matrix:
    """
    Build a sparse User × Movie CSR matrix from the ratings DataFrame.

    Parameters
    ----------
    ratings_df : pd.DataFrame   must contain [user_id, movie_id, rating]
    user2idx   : dict           user_id  → row index
    movie2idx  : dict           movie_id → column index

    Returns
    -------
    csr_matrix  shape (n_users, n_movies), dtype float32
    """
    n_users = len(user2idx)
    n_movies = len(movie2idx)

    rows = ratings_df["user_id"].map(user2idx).values.astype(np.int32)
    cols = ratings_df["movie_id"].map(movie2idx).values.astype(np.int32)
    data = ratings_df["rating"].values.astype(np.float32)

    matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_movies), dtype=np.float32)

    sparsity = 1.0 - matrix.nnz / (n_users * n_movies)
    logger.info(
        "Interaction matrix: %d × %d | nnz=%d | sparsity=%.4f%%",
        n_users, n_movies, matrix.nnz, sparsity * 100,
    )
    return matrix


# ─────────────────────────────────────────────
# 3. Train / Validation / Test Split
# ─────────────────────────────────────────────

def split_data(
    ratings_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_SEED,
    stratify_users: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train / validation / test sets.

    Strategy: per-user temporal split — for each user, the most recent
    ratings go to test, the next most recent to validation, the rest to train.
    This prevents data leakage and mimics real recommendation scenarios.

    Parameters
    ----------
    ratings_df      : pd.DataFrame  with [user_id, movie_id, rating, date]
    test_size       : float         fraction for test
    val_size        : float         fraction of TRAIN for validation
    random_state    : int
    stratify_users  : bool          use temporal split (recommended)

    Returns
    -------
    (train_df, val_df, test_df) as DataFrames
    """
    if stratify_users and "date" in ratings_df.columns and ratings_df["date"].notna().sum() > 0:
        logger.info("Using per-user temporal split …")
        train_idx, val_idx, test_idx = [], [], []

        for uid, group in tqdm(
            ratings_df.groupby("user_id"), desc="Splitting users", unit=" users"
        ):
            group_sorted = group.sort_values("date")
            n = len(group_sorted)

            if n < 5:
                # Too few ratings — put everything in train
                train_idx.extend(group_sorted.index.tolist())
                continue

            n_test = max(1, int(np.floor(n * test_size)))
            n_val = max(1, int(np.floor(n * val_size)))

            test_idx.extend(group_sorted.index[-n_test:].tolist())
            val_idx.extend(group_sorted.index[-(n_test + n_val):-n_test].tolist())
            train_idx.extend(group_sorted.index[:-(n_test + n_val)].tolist())

        train_df = ratings_df.loc[train_idx].reset_index(drop=True)
        val_df = ratings_df.loc[val_idx].reset_index(drop=True)
        test_df = ratings_df.loc[test_idx].reset_index(drop=True)

    else:
        logger.info("Using random stratified split …")
        train_df, test_df = train_test_split(
            ratings_df,
            test_size=test_size,
            random_state=random_state,
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size,
            random_state=random_state,
        )

    logger.info(
        "Split sizes → train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ─────────────────────────────────────────────
# 4. Surprise-Compatible Dataset Builder
# ─────────────────────────────────────────────

def to_surprise_dataset(ratings_df: pd.DataFrame):
    """
    Convert a ratings DataFrame to a Surprise Dataset for SVD training.

    Parameters
    ----------
    ratings_df : pd.DataFrame  with [user_id, movie_id, rating]

    Returns
    -------
    surprise.Dataset
    """
    try:
        from surprise import Dataset, Reader
    except ImportError:
        raise ImportError("Install scikit-surprise: pip install scikit-surprise")

    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(
        ratings_df[["user_id", "movie_id", "rating"]],
        reader=reader,
    )
    return dataset


# ─────────────────────────────────────────────
# 5. Utility: Save / Load Processed Artefacts
# ─────────────────────────────────────────────

def save_matrix(matrix: csr_matrix, name: str = "interaction_matrix") -> Path:
    """Save sparse matrix to disk."""
    path = PROCESSED_DIR / f"{name}.npz"
    save_npz(str(path), matrix)
    logger.info("Saved %s → %s", name, path)
    return path


def load_matrix(name: str = "interaction_matrix") -> csr_matrix:
    """Load sparse matrix from disk."""
    path = PROCESSED_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")
    matrix = load_npz(str(path))
    logger.info("Loaded %s from %s", name, path)
    return matrix


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Persist train/val/test splits as parquet files."""
    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        path = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved %s split → %s (%d rows)", name, path, len(df))


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load persisted train/val/test splits from parquet files."""
    dfs = []
    for name in ["train", "val", "test"]:
        path = PROCESSED_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}. Run preprocessing first.")
        dfs.append(pd.read_parquet(path))
        logger.info("Loaded %s split (%d rows)", name, len(dfs[-1]))
    return tuple(dfs)  # type: ignore


# ─────────────────────────────────────────────
# 6. Quick Stats Helper
# ─────────────────────────────────────────────

def dataset_summary(ratings_df: pd.DataFrame) -> Dict:
    """Return a dictionary of key dataset statistics."""
    n_users = ratings_df["user_id"].nunique()
    n_movies = ratings_df["movie_id"].nunique()
    n_ratings = len(ratings_df)
    sparsity = 1.0 - n_ratings / (n_users * n_movies)

    summary = {
        "n_ratings": n_ratings,
        "n_users": n_users,
        "n_movies": n_movies,
        "sparsity_pct": round(sparsity * 100, 4),
        "avg_rating": round(ratings_df["rating"].mean(), 4),
        "std_rating": round(ratings_df["rating"].std(), 4),
        "min_rating": int(ratings_df["rating"].min()),
        "max_rating": int(ratings_df["rating"].max()),
        "avg_ratings_per_user": round(n_ratings / n_users, 2),
        "avg_ratings_per_movie": round(n_ratings / n_movies, 2),
    }

    logger.info("Dataset summary:\n%s",
                "\n".join(f"  {k}: {v}" for k, v in summary.items()))
    return summary


# ─────────────────────────────────────────────
# 7. Main — end-to-end preprocessing run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    # 7a. Load raw data
    loader = NetflixDataLoader(use_subset=True)   # set False to use all 4 files
    ratings_df = loader.load(cache=True)
    movies_df = loader.load_movies()

    # 7b. Dataset summary
    summary = dataset_summary(ratings_df)

    # 7c. Build interaction matrix
    matrix = build_interaction_matrix(ratings_df, loader.user2idx, loader.movie2idx)
    save_matrix(matrix, "interaction_matrix")

    # 7d. Train / val / test split
    train_df, val_df, test_df = split_data(ratings_df)
    save_splits(train_df, val_df, test_df)

    # 7e. Build and save train-only matrix (used by Item-CF)
    train_matrix = build_interaction_matrix(train_df, loader.user2idx, loader.movie2idx)
    save_matrix(train_matrix, "train_matrix")

    elapsed = time.time() - t0
    logger.info("Preprocessing complete in %.1f seconds.", elapsed)
    print("\n✅  Preprocessing done. Artefacts saved to data/processed/")
    print(f"   Ratings    : {summary['n_ratings']:,}")
    print(f"   Users      : {summary['n_users']:,}")
    print(f"   Movies     : {summary['n_movies']:,}")
    print(f"   Sparsity   : {summary['sparsity_pct']}%")
    print(f"   Avg rating : {summary['avg_rating']}")
